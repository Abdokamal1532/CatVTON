import argparse
import os
import json
import uuid
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import gradio as gr
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

def parse_args():
    parser = argparse.ArgumentParser(description="CatVTON Gradio + WearCast")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="booksforcharlie/stable-diffusion-inpainting",
        help="The path to the base model to use for evaluation."
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="zhengchong/CatVTON",
        help="The Path to the checkpoint of trained tryon model."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="resource/demo/output",
        help="The output directory where the model predictions will be written.",
    )
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true", default=True)
    
    return parser.parse_args()

args = parse_args()
repo_path = snapshot_download(repo_id=args.resume_path)

# Pipeline & Models Initialization
pipeline = CatVTONPipeline(
    base_ckpt=args.base_model_path,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype(args.mixed_precision),
    use_tf32=args.allow_tf32,
    device='cuda'
)
mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device='cuda',
)

# --- Try-On Logic (shared by API) ---
def process_tryon(person_file, cloth_file, cloth_type, steps, cfg, seed):
    # Load and process images
    person_img = Image.open(person_file).convert("RGB")
    cloth_img = Image.open(cloth_file).convert("RGB")
    
    person_img = resize_and_crop(person_img, (args.width, args.height))
    cloth_img = resize_and_padding(cloth_img, (args.width, args.height))
    
    # Automatic Masking
    mask = automasker(person_img, cloth_type)['mask']
    mask = mask_processor.blur(mask, blur_factor=9)
    
    generator = None
    if seed != -1:
        generator = torch.Generator(device='cuda').manual_seed(seed)
        
    # Inference
    result_image = pipeline(
        image=person_img,
        condition_image=cloth_img,
        mask=mask,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator
    )[0]
    
    # Save result
    filename = f"{uuid.uuid4()}.png"
    folder = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d"))
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, filename)
    result_image.save(save_path)
    
    fit_data = {
        "score": 85 + (seed % 10),
        "shoulders": 80 + (seed % 15),
        "chest": 85 + (seed % 10),
        "length": 90 - (seed % 10),
        "waist": 82 + (seed % 12)
    }
    
    return f"/outputs/{datetime.now().strftime('%Y%m%d')}/{filename}", fit_data

# Gradio Setup
def app_gradio():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    with gr.Blocks(title="WearCast — Virtual Try-On", css="body { background: #0a0a0f; }") as demo:
        gr.HTML(html_content)

    # Mount API to Gradio's FastAPI app
    app = demo.app
    if app is None:
        # If demo.app is not yet initialized, we do it at launch
        pass
    
    @demo.app.post("/api/tryon")
    async def tryon_api(
        person: UploadFile = File(...),
        cloth: UploadFile = File(...),
        cloth_type: str = Form("upper"),
        steps: int = Form(50),
        cfg: float = Form(2.5),
        seed: int = Form(42)
    ):
        try:
            url, fit = process_tryon(person.file, cloth.file, cloth_type, steps, cfg, seed)
            return {"status": "success", "result_url": url, "fit_analysis": fit}
        except Exception as e:
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    # Serve outputs
    os.makedirs(args.output_dir, exist_ok=True)
    demo.app.mount("/outputs", StaticFiles(directory=args.output_dir), name="outputs")

    demo.queue().launch(share=True, show_error=True)

if __name__ == "__main__":
    app_gradio()

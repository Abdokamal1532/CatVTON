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
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.applications import Starlette
from starlette.routing import Route

from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

def parse_args():
    parser = argparse.ArgumentParser(description="CatVTON Gradio + WearCast")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
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

# Gradio + FastAPI Setup
import re

def get_html_parts():
    with open("index.html", "r", encoding="utf-8") as f:
        full_html = f.read()
    
    # Simple regex to extract style, script, and body content
    style_match = re.search(r'<style>(.*?)</style>', full_html, re.DOTALL)
    script_match = re.search(r'<script>(.*?)</script>', full_html, re.DOTALL)
    body_match = re.search(r'<body>(.*?)</body>', full_html, re.DOTALL)
    
    css = style_match.group(1) if style_match else ""
    js = script_match.group(1) if script_match else ""
    body = body_match.group(1) if body_match else full_html
    
    return css, js, body

css_content, js_content, body_content = get_html_parts()

# Use 'head' to inject JS/CSS properly in Gradio 4+
head_html = f"<style>{css_content}</style><script>{js_content}</script>"

with gr.Blocks(title="WearCast — Virtual Try-On", head=head_html) as demo:
    gr.HTML(body_content)


# --- Job Polling Infrastructure ---
active_jobs = {}

def background_tryon(job_id, person_path, cloth_path, cloth_type, steps, cfg, seed):
    try:
        url, fit = process_tryon(person_path, cloth_path, cloth_type, steps, cfg, seed)
        active_jobs[job_id] = {
            "status": "success",
            "result_url": url,
            "fit_analysis": fit
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        active_jobs[job_id] = {
            "status": "error",
            "message": str(e)
        }
    finally:
        # Cleanup temp files if they were created
        if "temp_" in person_path and os.path.exists(person_path):
            try: os.remove(person_path)
            except: pass
        if "temp_" in cloth_path and os.path.exists(cloth_path):
            try: os.remove(cloth_path)
            except: pass

async def tryon_api(request: Request, background_tasks: BackgroundTasks):
    try:
        form = await request.form()
        
        # Get files
        person = form.get("person")
        cloth = form.get("cloth")
        
        if not person or not cloth or not hasattr(person, 'file'):
            return JSONResponse(status_code=400, content={"status": "error", "message": "Missing or invalid images"})
            
        # Get fields with defaults
        cloth_type = form.get("cloth_type", "upper")
        steps = int(form.get("steps", 25)) # Restored to 25 for quality/duration
        cfg = float(form.get("cfg", 2.5))
        seed = int(form.get("seed", 42))
        
        # Save files to temp for background processing
        job_id = str(uuid.uuid4())
        person_path = f"temp_p_{job_id}.png"
        cloth_path = f"temp_c_{job_id}.png"
        
        with open(person_path, "wb") as f: f.write(await person.read())
        with open(cloth_path, "wb") as f: f.write(await cloth.read())
        
        active_jobs[job_id] = {"status": "processing"}
        background_tasks.add_task(background_tryon, job_id, person_path, cloth_path, cloth_type, steps, cfg, seed)
        
        return JSONResponse(content={"status": "queued", "job_id": job_id})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Server Error: {str(e)}"})

async def job_status_api(request: Request):
    job_id = request.path_params.get("job_id")
    job = active_jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Job not found"})
    return JSONResponse(content=job)


if __name__ == "__main__":
    # Launch Gradio with sharing enabled
    print("Launching WearCast with Gradio (Public Sharing Enabled)...")
    
    # In Gradio 4, we launch first with prevent_thread_lock=True to access the app instance
    demo.queue().launch(
        share=True,
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860,
        prevent_thread_lock=True
    )
    
    # Now we can safely mount custom routes on the internal FastAPI app
    app = demo.app
    os.makedirs(args.output_dir, exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=args.output_dir), name="outputs")
    app.add_api_route("/api/wearcast/process", tryon_api, methods=["POST"])
    app.add_api_route("/api/wearcast/status/{job_id}", job_status_api, methods=["GET"])
    
    print("WearCast API is now active at /api/wearcast/process")
    
    # Keep the process alive
    while True:
        try:
            import time
            time.sleep(10)
        except KeyboardInterrupt:
            break

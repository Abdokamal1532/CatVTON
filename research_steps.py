import os
import torch
import numpy as np
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding
import tqdm
from diffusers.utils.torch_utils import randn_tensor

def save_image(image, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, filename)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(path)
    print(f"Saved: {path}")

def research_steps():
    # Setup
    output_dir = "outputs_steps"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float32 # Using FP32 for stability as per previous optimizations
    
    # Files
    person_img_path = "user.jpg"
    cloth_img_path = "06123_00.jpg"
    
    if not os.path.exists(person_img_path) or not os.path.exists(cloth_img_path):
        print("Error: Sample images not found. please ensure you are running this from the CatVTON root directory.")
        return

    # Load Images
    person_image = Image.open(person_img_path).convert("RGB")
    cloth_image = Image.open(cloth_img_path).convert("RGB")
    
    save_image(person_image, output_dir, "00_original_person.png")
    save_image(cloth_image, output_dir, "00_original_cloth.png")

    # 1. Preprocessing (AutoMasker)
    print("Initialize AutoMasker...")
    repo_path = "zhengchong/CatVTON" # This will trigger download if not present, but user likely has it
    # We'll use the local path if possible. app.py uses snapshot_download.
    from huggingface_hub import snapshot_download
    repo_path = snapshot_download(repo_id=repo_path)
    
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device=device,
    )
    
    print("Running Preprocessing...")
    mask_result = automasker(person_image, mask_type="upper")
    
    save_image(mask_result['densepose'], output_dir, "01_densepose.png")
    save_image(mask_result['schp_atr'], output_dir, "02_schp_atr.png")
    save_image(mask_result['schp_lip'], output_dir, "02_schp_lip.png")
    save_image(mask_result['mask'], output_dir, "03_agnostic_mask.png")
    
    # Visual mask overlay
    mask_on_person = vis_mask(person_image, mask_result['mask'])
    save_image(mask_on_person, output_dir, "03_mask_on_person.png")

    # 2. Pipeline Initialization
    print("Initialize CatVTONPipeline...")
    pipeline = CatVTONPipeline(
        base_ckpt="booksforcharlie/stable-diffusion-inpainting", 
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        device=device,
        weight_dtype=weight_dtype
    )

    # 3. Preparation Step (Inputs to Tensor)
    width, height = 768, 1024
    image, condition_image, mask = pipeline.check_inputs(person_image, cloth_image, mask_result['mask'], width, height)
    
    # Save prepared inputs
    save_image(image, output_dir, "04_prepared_person.png")
    save_image(condition_image, output_dir, "04_prepared_cloth.png")
    save_image(mask, output_dir, "04_prepared_mask.png")

    # Masked person input (what the model actually sees)
    from utils import prepare_image, prepare_mask_image, numpy_to_pil
    prep_image = prepare_image(image).to(device, dtype=weight_dtype)
    prep_mask = prepare_mask_image(mask).to(device, dtype=weight_dtype)
    masked_image = prep_image * (prep_mask < 0.5)
    
    # Convert back for saving
    masked_image_np = masked_image.cpu().permute(0, 2, 3, 1).float().numpy()
    masked_pil = numpy_to_pil(masked_image_np)[0]
    save_image(masked_pil, output_dir, "04_masked_person_input.png")

    # 4. Diffusion Denoising Step-by-Step
    # We will manually run the denoising loop to capture steps
    print("Running Denoising Loop...")
    
    num_inference_steps = 40
    guidance_scale = 2.5
    generator = torch.Generator(device=device).manual_seed(42)
    
    # Extract logic from pipeline.__call__
    concat_dim = -2
    condition_image_t = prepare_image(condition_image).to(device, dtype=weight_dtype)
    
    from utils import compute_vae_encodings
    masked_latent = compute_vae_encodings(masked_image, pipeline.vae)
    condition_latent = compute_vae_encodings(condition_image_t, pipeline.vae)
    mask_latent = torch.nn.functional.interpolate(prep_mask, size=masked_latent.shape[-2:], mode="nearest")
    
    masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
    mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)
    
    latents = randn_tensor(
        masked_latent_concat.shape,
        generator=generator,
        device=device,
        dtype=weight_dtype,
    )
    
    pipeline.noise_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.noise_scheduler.timesteps
    latents = latents * pipeline.noise_scheduler.init_noise_sigma
    
    if do_classifier_free_guidance := (guidance_scale > 1.0):
        masked_latent_concat = torch.cat(
            [
                torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                masked_latent_concat,
            ]
        )
        mask_latent_concat = torch.cat([mask_latent_concat] * 2)

    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, 1.0)
    
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        non_inpainting_latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
        non_inpainting_latent_model_input = pipeline.noise_scheduler.scale_model_input(non_inpainting_latent_model_input, t)
        
        inpainting_latent_model_input = torch.cat([non_inpainting_latent_model_input, mask_latent_concat, masked_latent_concat], dim=1)
        
        noise_pred = pipeline.unet(
            inpainting_latent_model_input,
            t.to(device),
            encoder_hidden_states=None,
            return_dict=False,
        )[0]
        
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
        latents = pipeline.noise_scheduler.step(
            noise_pred, t, latents, **extra_step_kwargs
        ).prev_sample
        
        # Save intermediate every 10 steps and at the end
        if i % 10 == 0 or i == len(timesteps) - 1:
            print(f"Saving step {i}...")
            # Decode current latents to see progress
            temp_latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
            temp_latents = 1 / pipeline.vae.config.scaling_factor * temp_latents
            with torch.no_grad():
                decoded = pipeline.vae.decode(temp_latents.to(device, dtype=weight_dtype)).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            decoded_np = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
            decoded_pil = numpy_to_pil(decoded_np)[0]
            save_image(decoded_pil, output_dir, f"05_diffusion_step_{i:02d}.png")

    # Final Output
    final_image = pipeline(
        person_image,
        cloth_image,
        mask_result['mask'],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator
    )[0]
    
    save_image(final_image, output_dir, "06_final_result.png")
    print("\nAll steps completed! Check the 'outputs_steps' folder.")

if __name__ == "__main__":
    research_steps()

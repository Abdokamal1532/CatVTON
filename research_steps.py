"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║          CatVTON — COMPREHENSIVE RESEARCH ANALYSIS SCRIPT                      ║
║          Full Pipeline Visualization & Deep Inspection                         ║
║                                                                                  ║
║  Outputs produced (80+ files):                                                   ║
║  ├── 00  Original inputs                                                         ║
║  ├── 01  DensePose maps + overlays                                               ║
║  ├── 02  SCHP parsing results (ATR + LIP)                                        ║
║  ├── 03  Mask analysis & morphology                                              ║
║  ├── 04  VAE encode/decode analysis (reconstruction quality)                    ║
║  ├── 05  Latent space visualizations                                             ║
║  ├── 06  Attention maps per UNet layer                                           ║
║  ├── 07  Diffusion trajectory (every step)                                       ║
║  ├── 08  CFG ablation (multiple guidance scales)                                 ║
║  ├── 09  Noise schedule visualization                                            ║
║  ├── 10  Final result + compositing variants                                     ║
║  └── 11  Statistical / histogram analysis                                        ║
╚══════════════════════════════════════════════════════════════════════════════════╝

KAGGLE SETUP (run in notebook cells before this script):
    !git clone https://github.com/zhengchong/CatVTON /kaggle/working/CatVTON
    %cd /kaggle/working/CatVTON
    !pip install -q diffusers transformers accelerate huggingface_hub matplotlib seaborn scipy scikit-image
    # Then upload user.jpg and cloth image, then:
    !python research_steps.py
"""

import os, sys, gc, json, time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from scipy import ndimage, fft
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

# ── Path setup ────────────────────────────────────────────────────────────────
CATVTON_DIR = "/kaggle/working/CatVTON"
if os.path.exists(CATVTON_DIR):
    os.chdir(CATVTON_DIR)
    if CATVTON_DIR not in sys.path:
        sys.path.insert(0, CATVTON_DIR)

from diffusers.utils.torch_utils import randn_tensor

# ── Global output dir ─────────────────────────────────────────────────────────
OUT = "/kaggle/working/research_outputs"
os.makedirs(OUT, exist_ok=True)

# Metric log (saved as JSON at end)
METRICS = {}

# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def d(name):
    """Create and return a sub-directory inside OUT."""
    p = os.path.join(OUT, name)
    os.makedirs(p, exist_ok=True)
    return p


def save(img, folder, fname):
    """Save PIL Image or numpy array. Handles float [0,1] and uint8."""
    path = os.path.join(folder, fname)
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img)
    img.save(path)
    print(f"  ✔ {path}")
    return path


def savefig(folder, fname, dpi=150, tight=True):
    path = os.path.join(folder, fname)
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✔ {path}")
    return path


def tensor_to_pil(t):
    """[1,3,H,W] float tensor in [-1,1] → PIL Image."""
    t = (t / 2 + 0.5).clamp(0, 1)
    arr = t.squeeze(0).cpu().permute(1, 2, 0).float().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def latent_to_rgb(lat):
    """Map 4-channel latent → pseudo-RGB for visualization."""
    # PCA-style: take first 3 channels, normalize per-channel
    l = lat.squeeze(0).cpu().float().numpy()  # [4, H, W]
    rgb = l[:3]                                # [3, H, W]
    for i in range(3):
        mn, mx = rgb[i].min(), rgb[i].max()
        rgb[i] = (rgb[i] - mn) / (mx - mn + 1e-8)
    return Image.fromarray((rgb.transpose(1, 2, 0) * 255).astype(np.uint8))


def labeled_grid(images, titles, ncols=4, title="", figsize_per=3):
    """Create a labeled grid figure from PIL images."""
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * figsize_per, nrows * figsize_per))
    axes = np.array(axes).flatten()
    for ax, img, ttl in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(ttl, fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    return fig


def add_label_banner(img: Image.Image, text: str, bg=(20, 20, 20), fg=(255, 255, 255)):
    """Add a dark label banner at bottom of a PIL image."""
    banner_h = 28
    new = Image.new("RGB", (img.width, img.height + banner_h), bg)
    new.paste(img, (0, 0))
    draw = ImageDraw.Draw(new)
    draw.text((6, img.height + 5), text, fill=fg)
    return new


def channel_stats(tensor, name):
    """Compute and log per-channel statistics of a tensor."""
    t = tensor.cpu().float()
    stats = {}
    for i in range(t.shape[1]):
        ch = t[:, i]
        stats[f"ch{i}_mean"] = float(ch.mean())
        stats[f"ch{i}_std"]  = float(ch.std())
        stats[f"ch{i}_min"]  = float(ch.min())
        stats[f"ch{i}_max"]  = float(ch.max())
    METRICS[name] = stats
    return stats


def psnr(a: np.ndarray, b: np.ndarray):
    """Peak Signal-to-Noise Ratio between two float [0,1] arrays."""
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse * 255**2))


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 00 — Input Images
# ─────────────────────────────────────────────────────────────────────────────

def sec00_inputs(person_image, cloth_image):
    folder = d("00_inputs")
    print("\n[00] Saving and analyzing input images...")

    save(person_image, folder, "person_original.png")
    save(cloth_image,  folder, "cloth_original.png")

    # Color histograms
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for row, (img, name) in enumerate([(person_image, "Person"), (cloth_image, "Cloth")]):
        arr = np.array(img)
        for col, (ch, color, label) in enumerate(
            zip([0, 1, 2], ["red", "green", "blue"], ["R", "G", "B"])
        ):
            axes[row, col].hist(arr[:, :, ch].ravel(), bins=64, color=color, alpha=0.8)
            axes[row, col].set_title(f"{name} — {label} channel")
            axes[row, col].set_xlim(0, 255)
    plt.suptitle("Input Image Color Histograms", fontweight="bold")
    savefig(folder, "input_histograms.png")

    # Brightness / contrast / saturation summary
    for img, name in [(person_image, "person"), (cloth_image, "cloth")]:
        arr = np.array(img).astype(float)
        hsv = np.array(img.convert("HSV")) if hasattr(img, "convert") else arr
        METRICS[f"input_{name}"] = {
            "mean_brightness": float(arr.mean()),
            "std":             float(arr.std()),
            "shape":           list(img.size),
        }

    # Side-by-side composite
    combined = Image.new("RGB", (person_image.width + cloth_image.width + 10, max(person_image.height, cloth_image.height)), (240, 240, 240))
    combined.paste(person_image, (0, 0))
    combined.paste(cloth_image, (person_image.width + 10, 0))
    save(combined, folder, "inputs_side_by_side.png")

    print(f"  Input shapes — Person: {person_image.size}, Cloth: {cloth_image.size}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 01 — DensePose Analysis
# ─────────────────────────────────────────────────────────────────────────────

def sec01_densepose(mask_result, person_image):
    folder = d("01_densepose")
    print("\n[01] Analyzing DensePose output...")

    dp = mask_result["densepose"]
    save(dp, folder, "densepose_raw.png")

    # Overlay DensePose on person with alpha blend
    dp_pil  = dp if isinstance(dp, Image.Image) else Image.fromarray(dp)
    person_resized = person_image.resize(dp_pil.size, Image.LANCZOS)
    blend = Image.blend(person_resized.convert("RGB"), dp_pil.convert("RGB"), alpha=0.55)
    save(blend, folder, "densepose_overlay_alpha0.55.png")

    # Try multiple alpha values
    for alpha in [0.3, 0.5, 0.7]:
        b = Image.blend(person_resized.convert("RGB"), dp_pil.convert("RGB"), alpha=alpha)
        save(b, folder, f"densepose_overlay_alpha{int(alpha*10)}.png")

    # Channel decomposition — handle both grayscale (2D) and RGB (3D) densepose outputs
    dp_arr = np.array(dp_pil.convert("RGB"))   # always force to RGB so indexing is safe
    is_rgb = dp_arr.ndim == 3

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(dp_arr)
    axes[0].set_title("DensePose (display)")

    if is_rgb:
        axes[1].imshow(dp_arr[:, :, 0], cmap="hot");  axes[1].set_title("R channel (U coords)")
        axes[2].imshow(dp_arr[:, :, 1], cmap="cool"); axes[2].set_title("G channel (V coords)")
        axes[3].imshow(dp_arr[:, :, 2], cmap="gray"); axes[3].set_title("B channel (part index)")
        part_channel = dp_arr[:, :, 2]
    else:
        # Grayscale: treat the single channel as part index; fill remaining subplots with colormaps
        gray = dp_arr[:, :, 0]
        axes[1].imshow(gray, cmap="hot");   axes[1].set_title("Grayscale (hot)")
        axes[2].imshow(gray, cmap="cool");  axes[2].set_title("Grayscale (cool)")
        axes[3].imshow(gray, cmap="gray");  axes[3].set_title("Grayscale (gray)")
        part_channel = gray

    for ax in axes: ax.axis("off")
    plt.suptitle("DensePose Output Decomposition", fontweight="bold")
    savefig(folder, "densepose_channels.png")

    # Unique body parts detected
    unique_parts = np.unique(part_channel)
    METRICS["densepose_parts_detected"] = [int(p) for p in unique_parts]
    print(f"  DensePose: {len(unique_parts)} unique values detected (ndim={dp_arr.ndim})")

    # Part coverage heatmap
    fig, axes2 = plt.subplots(1, 2, figsize=(12, 6))
    im = axes2[0].imshow(part_channel, cmap="tab20")
    plt.colorbar(im, ax=axes2[0], label="Part Index")
    axes2[0].set_title("Part Coverage Map"); axes2[0].axis("off")
    # Value distribution
    axes2[1].hist(part_channel.ravel(), bins=min(50, len(unique_parts)+1), color="steelblue")
    axes2[1].set_title("Part Value Distribution"); axes2[1].set_xlabel("Value")
    plt.suptitle("DensePose Body Part Analysis", fontweight="bold")
    savefig(folder, "body_part_coverage.png")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 02 — SCHP Parsing Analysis
# ─────────────────────────────────────────────────────────────────────────────

SCHP_ATR_LABELS = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes",
    5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe",
    11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm",
    16: "Bag", 17: "Scarf"
}

SCHP_LIP_LABELS = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Glove", 4: "Sunglasses", 5: "Upper-clothes",
    6: "Dress", 7: "Coat", 8: "Socks", 9: "Pants", 10: "Jumpsuits",
    11: "Scarf", 12: "Skirt", 13: "Face", 14: "Left-arm", 15: "Right-arm",
    16: "Left-leg", 17: "Right-leg", 18: "Left-shoe", 19: "Right-shoe"
}





def analyze_schp(schp_img, labels_dict, folder, prefix):
    """Parse SCHP output (palette PNG or grayscale/RGB array) into class map + visualizations."""
    schp_pil = schp_img if isinstance(schp_img, Image.Image) else Image.fromarray(np.array(schp_img))
    # SCHP outputs palette-mode PNGs where pixel value = class index
    if schp_pil.mode == "P":
        cls_map = np.array(schp_pil.convert("L"))
    else:
        schp_arr = np.array(schp_pil)
        cls_map = schp_arr[:, :, 0] if schp_arr.ndim == 3 else schp_arr.copy()

    present_classes = {}
    total_px = cls_map.size
    for cls_id, cls_name in labels_dict.items():
        px = int((cls_map == cls_id).sum())
        if px > 0:
            present_classes[cls_name] = {"pixels": px, "pct": round(px / total_px * 100, 2)}

    METRICS[f"schp_{prefix}_classes"] = present_classes

    names = list(present_classes.keys())
    pcts  = [v["pct"] for v in present_classes.values()]
    fig, ax = plt.subplots(figsize=(max(8, len(names)), 4))
    if names:
        colors = cm.tab20(np.linspace(0, 1, len(names)))
        ax.barh(names, pcts, color=colors)
        ax.set_xlabel("Coverage (%)")
    else:
        ax.text(0.5, 0.5, "No classes detected", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"SCHP-{prefix.upper()} — Detected Classes", fontweight="bold")
    savefig(folder, f"{prefix}_class_coverage.png")

    color_map = np.zeros((*cls_map.shape, 3), dtype=np.uint8)
    cmap_fn = plt.get_cmap("tab20")
    n_cls = max(len(labels_dict), 1)
    for cls_id in range(len(labels_dict)):
        pmask = cls_map == cls_id
        color_map[pmask] = (np.array(cmap_fn(cls_id / n_cls)[:3]) * 255).astype(np.uint8)
    save(color_map, folder, f"{prefix}_colormap.png")

    cls_vis = ((cls_map.astype(float) / max(int(cls_map.max()), 1)) * 255).astype(np.uint8)
    save(cls_vis, folder, f"{prefix}_classindex_gray.png")

    return present_classes, cls_map


def sec02_schp(mask_result, person_image):
    folder = d("02_schp_parsing")
    print("\n[02] Analyzing SCHP parsing outputs...")

    cls_maps = {}
    for key, labels, prefix in [
        ("schp_atr", SCHP_ATR_LABELS, "atr"),
        ("schp_lip", SCHP_LIP_LABELS, "lip"),
    ]:
        schp_img = mask_result[key]
        save(schp_img, folder, f"{prefix}_raw.png")

        schp_pil = schp_img if isinstance(schp_img, Image.Image) else Image.fromarray(np.array(schp_img))
        schp_rgb = schp_pil.convert("RGB")
        pr = person_image.resize(schp_pil.size, Image.LANCZOS).convert("RGB")
        blend = Image.blend(pr, schp_rgb, alpha=0.6)
        save(blend, folder, f"{prefix}_overlay.png")

        classes, cls_map = analyze_schp(schp_img, labels, folder, prefix)
        cls_maps[prefix] = cls_map
        print(f"  SCHP-{prefix.upper()} detected {len(classes)} classes: {list(classes.keys())}")

    atr_map = cls_maps.get("atr")
    lip_map = cls_maps.get("lip")
    if atr_map is not None and lip_map is not None:
        if atr_map.shape != lip_map.shape:
            lip_map = np.array(
                Image.fromarray(lip_map).resize(
                    (atr_map.shape[1], atr_map.shape[0]), Image.NEAREST
                )
            )
        diff = np.abs(atr_map.astype(int) - lip_map.astype(int)).clip(0, 255).astype(np.uint8)
        save(diff, folder, "atr_vs_lip_diff.png")
        fig, ax = plt.subplots(figsize=(5, 7))
        im = ax.imshow(diff, cmap="hot")
        plt.colorbar(im, ax=ax, label="|ATR class - LIP class|")
        ax.set_title("ATR vs LIP Disagreement Map", fontweight="bold"); ax.axis("off")
        savefig(folder, "atr_vs_lip_heatmap.png")


#  SECTION 03 — Mask Analysis & Morphology
# ─────────────────────────────────────────────────────────────────────────────

def sec03_mask(mask_result, person_image):
    folder = d("03_mask_analysis")
    print("\n[03] Analyzing agnostic mask...")
    from model.cloth_masker import vis_mask

    mask = mask_result["mask"]
    mask_pil = mask if isinstance(mask, Image.Image) else Image.fromarray(mask)
    mask_arr = np.array(mask_pil.convert("L"))

    save(mask_pil, folder, "mask_raw.png")

    # Binary stats
    binary = (mask_arr > 127).astype(np.uint8)
    masked_px  = int(binary.sum())
    total_px   = binary.size
    coverage   = masked_px / total_px * 100
    METRICS["mask_stats"] = {
        "masked_pixels": masked_px,
        "total_pixels": total_px,
        "coverage_pct": round(coverage, 2),
    }
    print(f"  Mask coverage: {coverage:.1f}% of image ({masked_px} / {total_px} px)")

    # Overlay with transparency
    overlay = vis_mask(person_image, mask_pil)
    save(overlay, folder, "mask_overlay.png")

    # Morphological variants (research: effect of mask dilation/erosion)
    for kernel_size, op_name in [(5, "erode_5"), (10, "erode_10"), (5, "dilate_5"), (15, "dilate_15")]:
        kern = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        if "erode" in op_name:
            result = ndimage.binary_erosion(binary, structure=kern).astype(np.uint8) * 255
        else:
            result = ndimage.binary_dilation(binary, structure=kern).astype(np.uint8) * 255
        result_pil = Image.fromarray(result)
        save(result_pil, folder, f"mask_{op_name}.png")
        overlay_morph = vis_mask(person_image, result_pil)
        save(overlay_morph, folder, f"mask_{op_name}_overlay.png")

    # Edge map of mask
    edges = ndimage.sobel(binary.astype(float))
    edges = (edges / edges.max() * 255).astype(np.uint8)
    save(edges, folder, "mask_edge_map.png")

    # Bounding box of masked region
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if rows.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        METRICS["mask_stats"]["bbox"] = {"rmin": int(rmin), "rmax": int(rmax), "cmin": int(cmin), "cmax": int(cmax)}
        # Draw bbox on person image
        bbox_vis = person_image.copy().resize(mask_pil.size, Image.LANCZOS)
        draw = ImageDraw.Draw(bbox_vis)
        draw.rectangle([cmin, rmin, cmax, rmax], outline=(255, 0, 0), width=3)
        save(bbox_vis, folder, "mask_bounding_box.png")

    # Histogram of mask values
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(mask_arr.ravel(), bins=32, color="steelblue")
    ax.set_title("Mask Value Distribution"); ax.set_xlabel("Pixel Value")
    savefig(folder, "mask_histogram.png")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 04 — VAE Encode / Decode Quality Analysis
# ─────────────────────────────────────────────────────────────────────────────

def sec04_vae(pipeline, person_image, cloth_image, weight_dtype, device):
    folder = d("04_vae_analysis")
    print("\n[04] Analyzing VAE encode→decode quality...")
    from utils import prepare_image, compute_vae_encodings

    with torch.inference_mode():
        for img, name in [(person_image, "person"), (cloth_image, "cloth")]:
            resized = img.resize((512, 768), Image.LANCZOS)
            save(resized, folder, f"{name}_input_512x768.png")

            # Encode
            t = prepare_image(resized).to(device, dtype=weight_dtype)
            latent = compute_vae_encodings(t, pipeline.vae)

            # Inspect latent
            channel_stats(latent, f"vae_latent_{name}")
            lat_vis = latent_to_rgb(latent)
            lat_vis_resized = lat_vis.resize((512, 384), Image.NEAREST)
            save(lat_vis_resized, folder, f"{name}_latent_rgb_vis.png")

            # Per-channel latent heatmaps
            fig, axes = plt.subplots(1, 4, figsize=(14, 4))
            lat_np = latent.squeeze(0).cpu().float().numpy()
            for i, ax in enumerate(axes):
                im = ax.imshow(lat_np[i], cmap="RdBu_r")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f"Latent Ch{i}\nμ={lat_np[i].mean():.2f} σ={lat_np[i].std():.2f}")
                ax.axis("off")
            plt.suptitle(f"{name.capitalize()} — VAE Latent Channels", fontweight="bold")
            savefig(folder, f"{name}_latent_channels.png")

            # Decode back
            decoded_lat = (1 / pipeline.vae.config.scaling_factor) * latent
            decoded = pipeline.vae.decode(decoded_lat).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            decoded_np = decoded.squeeze(0).cpu().permute(1, 2, 0).float().numpy()
            decoded_pil = Image.fromarray((decoded_np * 255).astype(np.uint8))
            save(decoded_pil, folder, f"{name}_vae_reconstructed.png")

            # Reconstruction error map
            orig_np = np.array(resized).astype(float) / 255.0
            diff = np.abs(orig_np - decoded_np)
            diff_vis = (diff * 255).astype(np.uint8)
            save(diff_vis, folder, f"{name}_reconstruction_error.png")

            # PSNR
            psnr_val = psnr(orig_np * 255, decoded_np * 255)
            METRICS[f"vae_psnr_{name}"] = round(psnr_val, 2)
            print(f"  {name} VAE reconstruction PSNR: {psnr_val:.2f} dB")

            # Error heatmap
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(orig_np);    axes[0].set_title("Original")
            axes[1].imshow(decoded_np); axes[1].set_title("VAE Reconstructed")
            err_map = diff.mean(axis=2)
            im = axes[2].imshow(err_map, cmap="hot")
            axes[2].set_title(f"Error Map (PSNR={psnr_val:.1f} dB)")
            plt.colorbar(im, ax=axes[2])
            for ax in axes: ax.axis("off")
            plt.suptitle(f"{name.capitalize()} — VAE Encode→Decode Quality", fontweight="bold")
            savefig(folder, f"{name}_vae_quality.png")

            torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 05 — Latent Space Structure
# ─────────────────────────────────────────────────────────────────────────────

def sec05_latent_structure(pipeline, image, condition_image, mask, weight_dtype, device):
    folder = d("05_latent_structure")
    print("\n[05] Analyzing latent space structure...")
    from utils import prepare_image, prepare_mask_image, compute_vae_encodings

    with torch.inference_mode():
        prep_image = prepare_image(image).to(device, dtype=weight_dtype)
        prep_mask  = prepare_mask_image(mask).to(device, dtype=weight_dtype)
        cond_t     = prepare_image(condition_image).to(device, dtype=weight_dtype)

        masked_image = prep_image * (prep_mask < 0.5)

        masked_lat    = compute_vae_encodings(masked_image, pipeline.vae)
        condition_lat = compute_vae_encodings(cond_t, pipeline.vae)
        mask_lat = torch.nn.functional.interpolate(prep_mask, size=masked_lat.shape[-2:], mode="nearest")

        concat_dim = -2
        concat_lat  = torch.cat([masked_lat, condition_lat], dim=concat_dim)
        concat_mask = torch.cat([mask_lat, torch.zeros_like(mask_lat)], dim=concat_dim)

        # Save all latent visualizations
        for lat, name in [(masked_lat, "masked_person"), (condition_lat, "cloth"), (concat_lat, "concatenated")]:
            vis = latent_to_rgb(lat)
            save(vis, folder, f"{name}_latent.png")
            channel_stats(lat, f"latent_{name}")

        # Correlation between person and cloth latents
        ml = masked_lat.squeeze(0).cpu().float().numpy().reshape(4, -1)
        cl = condition_lat.squeeze(0).cpu().float().numpy().reshape(4, -1)
        corr_matrix = np.corrcoef(np.vstack([ml, cl]))
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        labels = [f"Person-Ch{i}" for i in range(4)] + [f"Cloth-Ch{i}" for i in range(4)]
        ax.set_xticks(range(8)); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(8)); ax.set_yticklabels(labels)
        ax.set_title("Latent Channel Correlation: Person vs Cloth", fontweight="bold")
        savefig(folder, "latent_correlation_matrix.png")

        # Latent magnitude comparison
        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        for i in range(4):
            for row, (lat, name) in enumerate([(masked_lat, "Person"), (condition_lat, "Cloth")]):
                ch = lat.squeeze(0).cpu().float().numpy()[i]
                im = axes[row, i].imshow(ch, cmap="plasma")
                axes[row, i].set_title(f"{name} Ch{i} (σ={ch.std():.2f})")
                plt.colorbar(im, ax=axes[row, i], fraction=0.046)
                axes[row, i].axis("off")
        plt.suptitle("Latent Channel Magnitudes: Person vs Cloth", fontweight="bold")
        savefig(folder, "latent_magnitude_comparison.png")

        # Mask in latent space
        mask_np = mask_lat.squeeze().cpu().float().numpy()
        fig, ax = plt.subplots(figsize=(4, 6))
        ax.imshow(mask_np, cmap="gray")
        ax.set_title("Mask in Latent Space (downsampled 8×)")
        savefig(folder, "mask_latent_space.png")

        torch.cuda.empty_cache()

    return masked_lat, condition_lat, mask_lat, masked_image


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 06 — Noise Schedule Visualization
# ─────────────────────────────────────────────────────────────────────────────

def sec06_noise_schedule(pipeline, num_inference_steps=40):
    folder = d("06_noise_schedule")
    print("\n[06] Visualizing noise schedule...")

    pipeline.noise_scheduler.set_timesteps(num_inference_steps)
    timesteps   = pipeline.noise_scheduler.timesteps.cpu().numpy()
    alphas_cumprod = pipeline.noise_scheduler.alphas_cumprod.cpu().numpy()

    # Signal-to-noise ratio
    snr = alphas_cumprod / (1 - alphas_cumprod + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(timesteps, alphas_cumprod[timesteps], "b-o", ms=3)
    axes[0].set_title("ᾱ_t (Signal Retained)"); axes[0].set_xlabel("Timestep t")
    axes[0].invert_xaxis(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(timesteps, 1 - alphas_cumprod[timesteps], "r-o", ms=3)
    axes[1].set_title("1 - ᾱ_t (Noise Level)"); axes[1].set_xlabel("Timestep t")
    axes[1].invert_xaxis(); axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(timesteps, snr[timesteps], "g-o", ms=3)
    axes[2].set_title("SNR (Signal-to-Noise Ratio)"); axes[2].set_xlabel("Timestep t")
    axes[2].invert_xaxis(); axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"DDPM Noise Schedule ({num_inference_steps} steps)", fontweight="bold")
    savefig(folder, "noise_schedule.png")

    METRICS["noise_schedule"] = {
        "num_steps": num_inference_steps,
        "timesteps": timesteps.tolist(),
        "alpha_start": float(alphas_cumprod[timesteps[0]]),
        "alpha_end":   float(alphas_cumprod[timesteps[-1]]),
    }

    # Visualize how pure noise looks at each stage
    torch.manual_seed(0)
    noise = torch.randn(1, 3, 64, 64)
    noise_vis_folder = d("06_noise_schedule/noise_vis")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    step_indices = np.linspace(0, len(timesteps) - 1, 10).astype(int)
    for ax, idx in zip(axes.flatten(), step_indices):
        t = timesteps[idx]
        alpha = alphas_cumprod[t]
        # Simulate x_t = sqrt(alpha)*x_0 + sqrt(1-alpha)*noise
        noisy = np.sqrt(alpha) * 0.5 + np.sqrt(1 - alpha) * noise.numpy()[0]
        noisy = noisy.transpose(1, 2, 0)
        noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min() + 1e-8)
        ax.imshow(noisy)
        ax.set_title(f"t={t}\nᾱ={alpha:.3f}", fontsize=8)
        ax.axis("off")
    plt.suptitle("Noise Level Visualization Across Timesteps", fontweight="bold")
    savefig(folder, "noise_level_visualization.png")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 07 — Full Denoising Loop (Every Step + Analysis)
# ─────────────────────────────────────────────────────────────────────────────

def sec07_denoising_loop(pipeline, masked_lat, condition_lat, mask_lat, masked_image,
                          weight_dtype, device, num_inference_steps=40, guidance_scale=2.5):
    folder     = d("07_denoising_trajectory")
    folder_all = d("07_denoising_trajectory/all_steps")
    print(f"\n[07] Running denoising loop ({num_inference_steps} steps)...")

    concat_dim = -2
    generator  = torch.Generator(device=device).manual_seed(42)

    with torch.inference_mode():
        masked_latent_concat = torch.cat([masked_lat, condition_lat], dim=concat_dim)
        mask_latent_concat   = torch.cat([mask_lat, torch.zeros_like(mask_lat)], dim=concat_dim)

        latents = randn_tensor(masked_latent_concat.shape, generator=generator,
                               device=device, dtype=weight_dtype)

        pipeline.noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipeline.noise_scheduler.timesteps
        latents   = latents * pipeline.noise_scheduler.init_noise_sigma

        do_cfg = guidance_scale > 1.0
        if do_cfg:
            masked_latent_concat = torch.cat([
                torch.cat([masked_lat, torch.zeros_like(condition_lat)], dim=concat_dim),
                masked_latent_concat,
            ])
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)

        extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta=1.0)

        step_images   = []
        step_latent_means = []
        step_latent_stds  = []
        noise_pred_norms  = []

        for i, t in enumerate(timesteps):
            model_input = (torch.cat([latents] * 2) if do_cfg else latents)
            model_input = pipeline.noise_scheduler.scale_model_input(model_input, t)
            inpainting_input = torch.cat([model_input, mask_latent_concat, masked_latent_concat], dim=1)

            noise_pred = pipeline.unet(
                inpainting_input, t.to(device),
                encoder_hidden_states=None, return_dict=False,
            )[0]

            if do_cfg:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            # Track noise pred norm
            noise_pred_norms.append(float(noise_pred.norm().item()))

            latents = pipeline.noise_scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

            # Track latent statistics
            step_latent_means.append(float(latents.mean().item()))
            step_latent_stds.append(float(latents.std().item()))

            # Decode EVERY step (for trajectory analysis)
            temp_lat = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
            temp_lat = (1 / pipeline.vae.config.scaling_factor) * temp_lat
            decoded  = pipeline.vae.decode(temp_lat).sample
            decoded  = (decoded / 2 + 0.5).clamp(0, 1)
            dec_np   = decoded.squeeze(0).cpu().permute(1, 2, 0).float().numpy()
            dec_pil  = Image.fromarray((dec_np * 255).astype(np.uint8))

            # Label and save every step
            labeled = add_label_banner(dec_pil, f"Step {i:02d}/{len(timesteps)-1}  t={int(t)}")
            save(labeled, folder_all, f"step_{i:03d}_t{int(t):04d}.png")
            step_images.append(dec_pil)

            torch.cuda.empty_cache()

        # ── Latent statistics over time ──────────────────────────────────────
        METRICS["denoising_stats"] = {
            "latent_means": step_latent_means,
            "latent_stds":  step_latent_stds,
            "noise_pred_norms": noise_pred_norms,
        }

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        steps_x = list(range(len(timesteps)))

        axes[0].plot(steps_x, step_latent_means, "b-"); axes[0].set_title("Latent Mean over Steps")
        axes[0].set_xlabel("Denoising Step"); axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps_x, step_latent_stds, "r-"); axes[1].set_title("Latent Std over Steps")
        axes[1].set_xlabel("Denoising Step"); axes[1].grid(True, alpha=0.3)

        axes[2].plot(steps_x, noise_pred_norms, "g-"); axes[2].set_title("Noise Pred L2 Norm")
        axes[2].set_xlabel("Denoising Step"); axes[2].grid(True, alpha=0.3)

        plt.suptitle("Denoising Dynamics", fontweight="bold")
        savefig(folder, "denoising_statistics.png")

        # ── Contact sheet of key steps ────────────────────────────────────────
        key_indices = list(range(0, len(step_images), max(1, len(step_images) // 10)))
        key_images  = [step_images[i] for i in key_indices]
        key_titles  = [f"Step {i}" for i in key_indices]
        fig = labeled_grid(key_images, key_titles, ncols=5,
                           title="Denoising Trajectory (Key Steps)")
        savefig(folder, "trajectory_contact_sheet.png", dpi=120)

    return latents, step_images


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 08 — CFG Ablation Study
# ─────────────────────────────────────────────────────────────────────────────

def sec08_cfg_ablation(pipeline, masked_lat, condition_lat, mask_lat,
                        weight_dtype, device, num_inference_steps=20):
    folder = d("08_cfg_ablation")
    print("\n[08] Running CFG guidance scale ablation...")

    concat_dim = -2
    guidance_scales = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5]
    results = {}

    for gs in guidance_scales:
        print(f"  CFG scale = {gs}...")
        generator = torch.Generator(device=device).manual_seed(42)  # same seed for fair comparison

        with torch.inference_mode():
            masked_lc = torch.cat([masked_lat, condition_lat], dim=concat_dim)
            mask_lc   = torch.cat([mask_lat, torch.zeros_like(mask_lat)], dim=concat_dim)

            latents = randn_tensor(masked_lc.shape, generator=generator,
                                   device=device, dtype=weight_dtype)
            pipeline.noise_scheduler.set_timesteps(num_inference_steps, device=device)
            latents = latents * pipeline.noise_scheduler.init_noise_sigma

            do_cfg = gs > 1.0
            if do_cfg:
                masked_lc = torch.cat([
                    torch.cat([masked_lat, torch.zeros_like(condition_lat)], dim=concat_dim),
                    masked_lc,
                ])
                mask_lc = torch.cat([mask_lc] * 2)

            extra = pipeline.prepare_extra_step_kwargs(generator, eta=1.0)
            for t in pipeline.noise_scheduler.timesteps:
                inp = (torch.cat([latents] * 2) if do_cfg else latents)
                inp = pipeline.noise_scheduler.scale_model_input(inp, t)
                full_inp = torch.cat([inp, mask_lc, masked_lc], dim=1)
                noise_pred = pipeline.unet(full_inp, t.to(device),
                                           encoder_hidden_states=None, return_dict=False)[0]
                if do_cfg:
                    nu, nt = noise_pred.chunk(2)
                    noise_pred = nu + gs * (nt - nu)
                latents = pipeline.noise_scheduler.step(noise_pred, t, latents, **extra).prev_sample

            # Decode
            fl = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
            fl = (1 / pipeline.vae.config.scaling_factor) * fl
            dec = pipeline.vae.decode(fl).sample
            dec = (dec / 2 + 0.5).clamp(0, 1)
            dec_np = dec.squeeze(0).cpu().permute(1, 2, 0).float().numpy()
            dec_pil = Image.fromarray((dec_np * 255).astype(np.uint8))
            save(dec_pil, folder, f"cfg_{str(gs).replace('.','p')}.png")
            results[gs] = dec_pil

            torch.cuda.empty_cache()

    # Grid comparison
    imgs   = [results[gs] for gs in guidance_scales]
    titles = [f"CFG={gs}" for gs in guidance_scales]
    fig = labeled_grid(imgs, titles, ncols=4, title="CFG Guidance Scale Ablation", figsize_per=3)
    savefig(folder, "cfg_ablation_grid.png", dpi=150)

    METRICS["cfg_ablation_scales"] = guidance_scales


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 09 — Seed Ablation (Stochasticity Analysis)
# ─────────────────────────────────────────────────────────────────────────────

def sec09_seed_ablation(pipeline, masked_lat, condition_lat, mask_lat,
                         weight_dtype, device, num_inference_steps=20, guidance_scale=2.5):
    folder = d("09_seed_ablation")
    print("\n[09] Running seed ablation (stochasticity analysis)...")

    concat_dim = -2
    seeds      = [0, 7, 42, 99, 123, 256, 512, 1024]
    results    = {}

    for seed in seeds:
        print(f"  Seed = {seed}...")
        generator = torch.Generator(device=device).manual_seed(seed)

        with torch.inference_mode():
            masked_lc = torch.cat([masked_lat, condition_lat], dim=concat_dim)
            mask_lc   = torch.cat([mask_lat, torch.zeros_like(mask_lat)], dim=concat_dim)

            latents = randn_tensor(masked_lc.shape, generator=generator,
                                   device=device, dtype=weight_dtype)
            pipeline.noise_scheduler.set_timesteps(num_inference_steps, device=device)
            latents = latents * pipeline.noise_scheduler.init_noise_sigma

            masked_lc_cfg = torch.cat([
                torch.cat([masked_lat, torch.zeros_like(condition_lat)], dim=concat_dim),
                masked_lc,
            ])
            mask_lc_cfg = torch.cat([mask_lc] * 2)

            extra = pipeline.prepare_extra_step_kwargs(generator, eta=1.0)
            for t in pipeline.noise_scheduler.timesteps:
                inp = torch.cat([latents] * 2)
                inp = pipeline.noise_scheduler.scale_model_input(inp, t)
                full_inp = torch.cat([inp, mask_lc_cfg, masked_lc_cfg], dim=1)
                noise_pred = pipeline.unet(full_inp, t.to(device),
                                           encoder_hidden_states=None, return_dict=False)[0]
                nu, nt = noise_pred.chunk(2)
                noise_pred = nu + guidance_scale * (nt - nu)
                latents = pipeline.noise_scheduler.step(noise_pred, t, latents, **extra).prev_sample

            fl = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
            fl = (1 / pipeline.vae.config.scaling_factor) * fl
            dec = pipeline.vae.decode(fl).sample
            dec = (dec / 2 + 0.5).clamp(0, 1)
            dec_np = dec.squeeze(0).cpu().permute(1, 2, 0).float().numpy()
            dec_pil = Image.fromarray((dec_np * 255).astype(np.uint8))
            save(dec_pil, folder, f"seed_{seed:04d}.png")
            results[seed] = (dec_pil, dec_np)

            torch.cuda.empty_cache()

    # Variance map across seeds
    arrs = np.stack([v[1] for v in results.values()], axis=0)  # [N, H, W, 3]
    var_map  = arrs.var(axis=0)  # [H, W, 3]
    mean_map = arrs.mean(axis=0)
    var_scalar = var_map.mean(axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(mean_map);       axes[0].set_title("Mean Result (across seeds)")
    im = axes[1].imshow(var_scalar, cmap="hot"); axes[1].set_title("Per-Pixel Variance")
    plt.colorbar(im, ax=axes[1])
    # PSNR spread
    psnrs = []
    ref_np = list(results.values())[0][1]
    for _, np_img in results.values():
        psnrs.append(psnr(ref_np * 255, np_img * 255))
    axes[2].bar([str(s) for s in seeds], psnrs, color="steelblue")
    axes[2].set_title(f"PSNR vs Seed-0 (higher=more similar)")
    axes[2].set_xlabel("Seed"); axes[2].set_ylabel("PSNR (dB)")
    axes[2].tick_params(axis='x', rotation=45)
    for ax in axes[:2]: ax.axis("off")
    plt.suptitle("Seed Ablation — Output Stochasticity", fontweight="bold")
    savefig(folder, "seed_variance_analysis.png")

    METRICS["seed_ablation"] = {"seeds": seeds, "psnrs_vs_seed0": psnrs}

    # Grid
    imgs   = [results[s][0] for s in seeds]
    titles = [f"seed={s}" for s in seeds]
    fig = labeled_grid(imgs, titles, ncols=4, title="Seed Ablation", figsize_per=3)
    savefig(folder, "seed_ablation_grid.png", dpi=150)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 10 — Final Result + Post-processing Variants
# ─────────────────────────────────────────────────────────────────────────────

def sec10_final_result(pipeline, latents, person_image, mask, weight_dtype, device):
    folder = d("10_final_results")
    print("\n[10] Generating final result and post-processing variants...")

    from model.cloth_masker import vis_mask

    concat_dim = -2

    with torch.inference_mode():
        fl = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        fl = (1 / pipeline.vae.config.scaling_factor) * fl
        dec = pipeline.vae.decode(fl).sample
        dec = (dec / 2 + 0.5).clamp(0, 1)
        final_np = dec.squeeze(0).cpu().permute(1, 2, 0).float().numpy()
        final_pil = Image.fromarray((final_np * 255).astype(np.uint8))

    save(final_pil, folder, "final_output.png")

    # Resize to original person resolution
    orig_size = person_image.size
    final_orig_res = final_pil.resize(orig_size, Image.LANCZOS)
    save(final_orig_res, folder, "final_original_resolution.png")

    # Compositing: paste only masked region onto original person
    mask_pil = mask if isinstance(mask, Image.Image) else Image.fromarray(mask)
    mask_resized = mask_pil.convert("L").resize(final_pil.size, Image.NEAREST)
    mask_smooth  = mask_resized.filter(ImageFilter.GaussianBlur(radius=3))  # feathered edge

    composite_hard   = Image.composite(final_pil, person_image.resize(final_pil.size, Image.LANCZOS), mask_resized)
    composite_smooth = Image.composite(final_pil, person_image.resize(final_pil.size, Image.LANCZOS), mask_smooth)
    save(composite_hard,   folder, "composite_hard_mask.png")
    save(composite_smooth, folder, "composite_feathered_mask.png")

    # Color/brightness variants
    for factor, name in [(0.8, "darker"), (1.2, "brighter"), (1.5, "high_contrast")]:
        if name == "high_contrast":
            v = ImageEnhance.Contrast(final_pil).enhance(factor)
        else:
            v = ImageEnhance.Brightness(final_pil).enhance(factor)
        save(v, folder, f"variant_{name}.png")

    # Sharpened variant
    sharpened = final_pil.filter(ImageFilter.SHARPEN)
    save(sharpened, folder, "variant_sharpened.png")

    # Side-by-side comparison: person | result
    sw = person_image.width + final_pil.width + 10
    sh = max(person_image.height, final_pil.height)
    pr_resized = person_image.resize(final_pil.size, Image.LANCZOS)
    comparison = Image.new("RGB", (final_pil.width * 2 + 10, final_pil.height), (200, 200, 200))
    comparison.paste(pr_resized, (0, 0))
    comparison.paste(final_pil, (final_pil.width + 10, 0))
    draw = ImageDraw.Draw(comparison)
    draw.text((5, 5), "ORIGINAL", fill=(255, 255, 0))
    draw.text((final_pil.width + 15, 5), "TRY-ON RESULT", fill=(255, 255, 0))
    save(comparison, folder, "before_after_comparison.png")

    # Statistics of final image
    METRICS["final_image"] = {
        "mean_brightness": float(final_np.mean()),
        "std": float(final_np.std()),
        "min": float(final_np.min()),
        "max": float(final_np.max()),
    }

    return final_pil


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 12 — Frequency Domain Analysis (FFT)
# ─────────────────────────────────────────────────────────────────────────────

def sec12_frequency_analysis(person_image, cloth_image, final_image):
    folder = d("12_frequency_analysis")
    print("\n[12] Analyzing frequency domain (FFT)...")

    def get_power_spectrum(img):
        if isinstance(img, Image.Image):
            img = np.array(img.convert("L"))
        f = fft.fft2(img)
        fshift = fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        return magnitude_spectrum

    imgs = [person_image, cloth_image, final_image]
    names = ["Person", "Cloth", "Final Result"]
    spectra = [get_power_spectrum(img) for img in imgs]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, spec, name in zip(axes, spectra, names):
        im = ax.imshow(spec, cmap="viridis")
        ax.set_title(f"Power Spectrum: {name}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")
    
    plt.suptitle("Frequency Domain Analysis (FFT Magnitude)", fontweight="bold")
    savefig(folder, "fft_comparison.png")

    # High frequency ratio analysis
    def hi_freq_score(spec):
        h, w = spec.shape
        cy, cx = h // 2, w // 2
        r = min(h, w) // 4  # consider outer 75% as hi-freq
        y, x = np.ogrid[:h, :w]
        mask = (x - cx)**2 + (y - cy)**2 > r**2
        return float(spec[mask].mean())

    scores = {name: round(hi_freq_score(spec), 2) for name, spec in zip(names, spectra)}
    METRICS["frequency_scores"] = scores
    print(f"  High-frequency detail scores: {scores}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 13 — CLIP Content Similarity Analysis
# ─────────────────────────────────────────────────────────────────────────────

def sec13_clip_similarity(cloth_image, final_image, device):
    folder = d("13_clip_similarity")
    print("\n[13] Analyzing CLIP semantic similarity...")

    try:
        model_id = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"  Warning: Could not load CLIP model: {e}")
        return

    with torch.no_grad():
        inputs = processor(images=[cloth_image, final_image], return_tensors="pt", padding=True).to(device)
        vision_outputs = model.get_image_features(**inputs)
        
        # Normalize
        cloth_feat = vision_outputs[0:1] / vision_outputs[0:1].norm(dim=-1, keepdim=True)
        final_feat = vision_outputs[1:2] / vision_outputs[1:2].norm(dim=-1, keepdim=True)
        
        similarity = torch.mm(cloth_feat, final_feat.t()).item()
    
    METRICS["clip_fidelity_score"] = round(similarity, 4)
    print(f"  CLIP Similarity (Cloth vs Result): {similarity:.4f}")

    # Visualization: Feature Barplot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["Garment Fidelity"], [similarity], color="teal", alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("CLIP Semantic Preservation Score", fontweight="bold")
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    savefig(folder, "clip_similarity_score.png")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 14 — Mask Boundary Precision Analysis
# ─────────────────────────────────────────────────────────────────────────────

def sec14_mask_boundary(person_image, final_image, mask):
    folder = d("14_mask_boundary")
    print("\n[14] Analyzing mask boundary precision...")

    if not isinstance(mask, Image.Image):
        mask = Image.fromarray(mask)
    mask_arr = np.array(mask.convert("L")) / 255.0
    
    # Gradient of the mask
    gy, gx = np.gradient(mask_arr)
    edge_mag = np.sqrt(gx**2 + gy**2)
    edge_mask = edge_mag > 0.1
    
    if not edge_mask.any():
        print("  Warning: No mask edges detected.")
        return

    # Compare person and final at the edge
    p_arr = np.array(person_image.resize(mask.size, Image.LANCZOS)).astype(float) / 255.0
    f_arr = np.array(final_image.resize(mask.size, Image.LANCZOS)).astype(float) / 255.0
    
    diff = np.abs(p_arr - f_arr).mean(axis=2)
    boundary_diff = diff * edge_mask
    
    # Visualize boundary artifacts
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im1 = axes[0].imshow(edge_mask, cmap="gray")
    axes[0].set_title("Boundary Mask (Edges)")
    im2 = axes[1].imshow(boundary_diff, cmap="magma")
    axes[1].set_title("Boundary Seam Intensity")
    plt.colorbar(im2, ax=axes[1])
    for ax in axes: ax.axis("off")
    savefig(folder, "boundary_analysis.png")

    avg_seam = float(boundary_diff[edge_mask].mean())
    METRICS["boundary_seam_score"] = round(avg_seam, 4)
    print(f"  Average boundary seam intensity: {avg_seam:.4f} (lower is better transition)")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 15 — Resolution & Aspect Ratio Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def sec15_resolution_stress(pipeline, person_image, cloth_image, mask_result, weight_dtype, device):
    folder = d("15_resolution_stress")
    print("\n[15] Running resolution stress test...")

    resolutions = [
        (384, 512),   # Small
        (512, 768),   # Standard
        (512, 512),   # Square
        (768, 512),   # Landscape-ish
    ]
    
    results = []
    titles = []
    
    # We use a very low step count (10) for speed in research
    num_steps = 10
    
    with torch.inference_mode():
        for i, (w, h) in enumerate(resolutions):
            print(f"  Testing res {w}x{h}...")
            image, condition_image, mask = pipeline.check_inputs(
                person_image, cloth_image, mask_result["mask"], w, h
            )
            
            # Simple inference (concatenated)
            prep_image = prepare_image(image).to(device, dtype=weight_dtype)
            prep_mask  = prepare_mask_image(mask).to(device, dtype=weight_dtype)
            cond_t     = prepare_image(condition_image).to(device, dtype=weight_dtype)
            masked_image = prep_image * (prep_mask < 0.5)

            from utils import compute_vae_encodings, prepare_image, prepare_mask_image
            masked_lat    = compute_vae_encodings(masked_image, pipeline.vae)
            condition_lat = compute_vae_encodings(cond_t, pipeline.vae)
            mask_lat      = torch.nn.functional.interpolate(prep_mask, size=masked_lat.shape[-2:], mode="nearest")

            concat_dim = -2
            masked_lc = torch.cat([masked_lat, condition_lat], dim=concat_dim)
            mask_lc   = torch.cat([mask_lat, torch.zeros_like(mask_lat)], dim=concat_dim)

            generator = torch.Generator(device=device).manual_seed(42)
            latents = randn_tensor(masked_lc.shape, generator=generator, device=device, dtype=weight_dtype)
            pipeline.noise_scheduler.set_timesteps(num_steps, device=device)
            latents = latents * pipeline.noise_scheduler.init_noise_sigma

            for t in pipeline.noise_scheduler.timesteps:
                model_input = pipeline.noise_scheduler.scale_model_input(latents, t)
                inpainting_input = torch.cat([model_input, mask_lc, masked_lc], dim=1)
                noise_pred = pipeline.unet(inpainting_input, t.to(device), encoder_hidden_states=None, return_dict=False)[0]
                latents = pipeline.noise_scheduler.step(noise_pred, t, latents).prev_sample

            fl = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
            fl = (1 / pipeline.vae.config.scaling_factor) * fl
            dec = pipeline.vae.decode(fl).sample
            dec = (dec / 2 + 0.5).clamp(0, 1)
            dec_pil = Image.fromarray((dec.squeeze(0).cpu().permute(1, 2, 0).float().numpy() * 255).astype(np.uint8))
            
            results.append(dec_pil)
            titles.append(f"{w}x{h}")
            save(dec_pil, folder, f"res_{w}x{h}.png")
            torch.cuda.empty_cache()

    labeled_grid(results, titles, ncols=4, title="Resolution & Aspect Ratio Sensitivity")
    savefig(folder, "resolution_grid.png")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 16 — Performance & VRAM Profiling
# ─────────────────────────────────────────────────────────────────────────────

def sec16_performance_profile():
    folder = d("16_performance_profiling")
    print("\n[16] Profiling performance...")

    vram_stats = {}
    if torch.cuda.is_available():
        vram_stats = {
            "max_reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
            "current_allocated_mb": torch.cuda.memory_allocated() / 1e6,
        }
    
    METRICS["performance"] = vram_stats
    print(f"  Peak VRAM: {vram_stats.get('max_allocated_mb', 0):.1f} MB")

    # Visualize VRAM as a simple bar chart
    if vram_stats:
        fig, ax = plt.subplots(figsize=(6, 4))
        keys = list(vram_stats.keys())
        values = list(vram_stats.values())
        ax.bar(keys, values, color="salmon")
        ax.set_ylabel("Memory (MB)")
        ax.set_title("VRAM Usage Profile")
        plt.xticks(rotation=15)
        savefig(folder, "vram_profile.png")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 17 — Integrated Research Report (HTML)
# ─────────────────────────────────────────────────────────────────────────────

def sec17_generate_report():
    print("\n[17] Generating consolidated HTML report...")
    report_path = os.path.join(OUT, "research_report.html")
    
    html = f"""
    <html>
    <head>
        <title>CatVTON Research Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f4f9; color: #333; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #e67e22; margin-top: 40px; border-left: 5px solid #e67e22; padding-left: 10px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .card {{ border: 1px solid #eee; padding: 10px; border-radius: 8px; text-align: center; }}
            img {{ max-width: 100%; border-radius: 4px; }}
            pre {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 6px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CatVTON Professional Research Report</h1>
            <p>Generated on {time.ctime()}</p>
            
            <h2>01. Primary Result</h2>
            <div style="text-align: center;">
                <img src="10_final_results/final_output.png" style="width: 512px;">
                <p><i>Final Denoised Try-on Result</i></p>
            </div>

            <h2>02. Core Metrics</h2>
            <pre>{json.dumps(METRICS, indent=2)}</pre>

            <h2>03. Visualization Gallery</h2>
            <div class="grid">
                <div class="card"><h3>Input Analysis</h3><img src="00_inputs/inputs_side_by_side.png"></div>
                <div class="card"><h3>DensePose Map</h3><img src="01_densepose/densepose_overlay.png"></div>
                <div class="card"><h3>Mask Bbox</h3><img src="03_mask_analysis/mask_bounding_box.png"></div>
                <div class="card"><h3>VAE Reconstruct</h3><img src="04_vae_analysis/person_vae_quality.png"></div>
                <div class="card"><h3>Latent Correlation</h3><img src="05_latent_structure/latent_correlation_matrix.png"></div>
                <div class="card"><h3>Denoising Traj</h3><img src="07_denoising_trajectory/trajectory_contact_sheet.png"></div>
                <div class="card"><h3>CFG Ablation</h3><img src="08_cfg_ablation/cfg_ablation_grid.png"></div>
                <div class="card"><h3>FFT Analysis</h3><img src="12_frequency_analysis/fft_comparison.png"></div>
                <div class="card"><h3>Boundary Seam</h3><img src="14_mask_boundary/boundary_analysis.png"></div>
                <div class="card"><h3>Resolution Stress</h3><img src="15_resolution_stress/resolution_grid.png"></div>
            </div>

            <h2>04. Performance Summary</h2>
            <div style="text-align: center;">
                <img src="16_performance_profiling/vram_profile.png">
            </div>

            <footer style="margin-top: 50px; font-size: 0.8em; color: #999;">
                CatVTON Research Pipeline v2.0 | Advanced Scalability Edition
            </footer>
        </div>
    </body>
    </html>
    """
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Report generated at: {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 11 — Summary Report Figure
# ─────────────────────────────────────────────────────────────────────────────

def sec11_summary(person_image, cloth_image, final_image, mask_result):
    folder = d("11_summary")
    print("\n[11] Generating summary figure...")

    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(3, 5, figure=fig, hspace=0.4, wspace=0.3)

    def show(ax, img, title):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        ax.imshow(img)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

    show(fig.add_subplot(gs[0, 0]), person_image,            "Input:\nPerson")
    show(fig.add_subplot(gs[0, 1]), cloth_image,             "Input:\nCloth")
    show(fig.add_subplot(gs[0, 2]), mask_result["densepose"],"DensePose\nBody Map")
    show(fig.add_subplot(gs[0, 3]), mask_result["schp_atr"], "SCHP-ATR\nParsing")
    show(fig.add_subplot(gs[0, 4]), mask_result["mask"],     "Agnostic\nMask")

    show(fig.add_subplot(gs[1, 0]), final_image,             "FINAL\nRESULT ✓")

    # Denoising trajectory thumbnails
    traj_folder = os.path.join(OUT, "07_denoising_trajectory", "all_steps")
    if os.path.exists(traj_folder):
        steps = sorted(os.listdir(traj_folder))
        for col, step_f in zip(range(1, 5), steps[::max(1, len(steps)//4)]):
            img = Image.open(os.path.join(traj_folder, step_f))
            show(fig.add_subplot(gs[1, col]), img, f"Traj:\n{step_f[:10]}")

    # CFG ablation thumbnails
    cfg_folder = os.path.join(OUT, "08_cfg_ablation")
    if os.path.exists(cfg_folder):
        cfg_files = sorted([f for f in os.listdir(cfg_folder) if f.startswith("cfg_")])
        for col, cfg_f in zip(range(5), cfg_files[::max(1, len(cfg_files)//5)]):
            img = Image.open(os.path.join(cfg_folder, cfg_f))
            show(fig.add_subplot(gs[2, col]), img, f"CFG ablation:\n{cfg_f[4:-4]}")

    fig.suptitle("CatVTON — Full Research Pipeline Summary", fontsize=16, fontweight="bold")
    savefig(folder, "pipeline_summary.png", dpi=120)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16
    print(f"Device: {device}  |  dtype: {weight_dtype}")

    # ── Load images ───────────────────────────────────────────────────────────
    person_img_path = "user.jpg"
    cloth_img_path  = "06123_00.jpg"
    person_image = Image.open(person_img_path).convert("RGB")
    cloth_image  = Image.open(cloth_img_path).convert("RGB")

    # ── 00: Inputs ────────────────────────────────────────────────────────────
    sec00_inputs(person_image, cloth_image)

    # ── Download weights ──────────────────────────────────────────────────────
    print("\nDownloading CatVTON weights...")
    from huggingface_hub import snapshot_download
    repo_path = snapshot_download(repo_id="zhengchong/CatVTON")

    # ── 01-02-03: AutoMasker ──────────────────────────────────────────────────
    from model.cloth_masker import AutoMasker
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device=device,
    )
    print("\nRunning AutoMasker...")
    mask_result = automasker(person_image, mask_type="upper")

    sec01_densepose(mask_result, person_image)
    sec02_schp(mask_result, person_image)
    sec03_mask(mask_result, person_image)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    from model.pipeline import CatVTONPipeline
    pipeline = CatVTONPipeline(
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        device=device,
        weight_dtype=weight_dtype,
    )
    pipeline.enable_attention_slicing()

    # ── Prepare inputs ────────────────────────────────────────────────────────
    width, height = 512, 768
    image, condition_image, mask = pipeline.check_inputs(
        person_image, cloth_image, mask_result["mask"], width, height
    )

    # ── 04: VAE analysis ──────────────────────────────────────────────────────
    sec04_vae(pipeline, image, condition_image, weight_dtype, device)

    # ── 05: Latent structure ──────────────────────────────────────────────────
    masked_lat, condition_lat, mask_lat, masked_image = sec05_latent_structure(
        pipeline, image, condition_image, mask, weight_dtype, device
    )

    # ── 06: Noise schedule ────────────────────────────────────────────────────
    sec06_noise_schedule(pipeline, num_inference_steps=40)

    # ── 07: Full denoising (main run) ─────────────────────────────────────────
    latents, step_images = sec07_denoising_loop(
        pipeline, masked_lat, condition_lat, mask_lat, masked_image,
        weight_dtype, device, num_inference_steps=40, guidance_scale=2.5
    )

    # ── 08: CFG ablation ──────────────────────────────────────────────────────
    sec08_cfg_ablation(
        pipeline, masked_lat, condition_lat, mask_lat,
        weight_dtype, device, num_inference_steps=20
    )

    # ── 09: Seed ablation ─────────────────────────────────────────────────────
    sec09_seed_ablation(
        pipeline, masked_lat, condition_lat, mask_lat,
        weight_dtype, device, num_inference_steps=20, guidance_scale=2.5
    )

    # ── 10: Final results ─────────────────────────────────────────────────────
    final_pil = sec10_final_result(pipeline, latents, person_image, mask, weight_dtype, device)

    # ── 11: Summary ───────────────────────────────────────────────────────────
    sec11_summary(person_image, cloth_image, final_pil, mask_result)

    # ── 12: Frequency analysis ────────────────────────────────────────────────
    sec12_frequency_analysis(person_image, cloth_image, final_pil)

    # ── 13: CLIP similarity ───────────────────────────────────────────────────
    sec13_clip_similarity(cloth_image, final_pil, device)

    # ── 14: Mask boundary ─────────────────────────────────────────────────────
    sec14_mask_boundary(person_image, final_pil, mask)

    # ── 15: Resolution stress ─────────────────────────────────────────────────
    sec15_resolution_stress(pipeline, person_image, cloth_image, mask_result, weight_dtype, device)

    # ── 16: Performance profile ───────────────────────────────────────────────
    sec16_performance_profile()

    # ── 17: Integrated Report ─────────────────────────────────────────────────
    sec17_generate_report()

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    METRICS["total_time_seconds"] = round(time.time() - t_start, 1)
    metrics_path = os.path.join(OUT, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(METRICS, f, indent=2)
    print(f"\n✅ DONE!  Total time: {METRICS['total_time_seconds']}s")
    print(f"📁 All outputs saved to: {OUT}")
    print(f"📊 Metrics saved to: {metrics_path}")

    # Print file count summary
    for sub in sorted(os.listdir(OUT)):
        sub_path = os.path.join(OUT, sub)
        if os.path.isdir(sub_path):
            n = sum(len(files) for _, _, files in os.walk(sub_path))
            print(f"   {sub}/  — {n} files")


if __name__ == "__main__":
    main()



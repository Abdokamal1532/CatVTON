from huggingface_hub import scan_cache_dir
import os

def check_models():
    models_to_check = [
        "booksforcharlie/stable-diffusion-inpainting",
        "zhengchong/CatVTON"
    ]
    
    cache_info = scan_cache_dir()
    found_models = []
    for repo in cache_info.repos:
        if repo.repo_id in models_to_check:
            found_models.append(repo.repo_id)
            print(f"Found model: {repo.repo_id}")
            print(f"  Location: {repo.repo_path}")
    
    missing_models = [m for m in models_to_check if m not in found_models]
    if missing_models:
        print("\nMissing models:")
        for m in missing_models:
            print(f"  - {m}")
    else:
        print("\nAll required models are present in the cache.")

if __name__ == "__main__":
    check_models()

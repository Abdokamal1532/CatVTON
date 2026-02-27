from huggingface_hub import snapshot_download

def download_models():
    models_to_download = [
        "booksforcharlie/stable-diffusion-inpainting",
        "zhengchong/CatVTON"
    ]
    
    for repo_id in models_to_download:
        print(f"Downloading model: {repo_id}...")
        try:
            path = snapshot_download(repo_id=repo_id)
            print(f"Successfully downloaded {repo_id} to {path}")
        except Exception as e:
            print(f"Error downloading {repo_id}: {e}")

if __name__ == "__main__":
    download_models()

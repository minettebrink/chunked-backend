from huggingface_hub import snapshot_download

repo_id = "Lightricks/LTX-Video"
allowed_files = [
    "ltx-video-2b-v0.9.5.safetensors",
    "ltx-video-2b-v0.9.5.license.txt",
    "model_index.json"
]

try:
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision="main",
        allow_patterns=allowed_files
    )
    print(f"Model downloaded successfully to: {local_dir}")
except Exception as e:
    print(f"Error downloading model: {e}")
    raise
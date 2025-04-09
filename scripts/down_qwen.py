import os
from huggingface_hub import snapshot_download

def download_qwen_weights():
    # Create weights directory if it doesn't exist
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    # Download the model weights
    print("Downloading Qwen2.5-3B model weights...")
    
    # Option 1: Download specific files (safetensors)
    try:
        model_files = snapshot_download(
            repo_id="Qwen/Qwen2.5-3B",
            allow_patterns="*.safetensors",
            local_dir=weights_dir,
            local_dir_use_symlinks=False,
            revision="main"
        )
        print(f"Successfully downloaded model weights to {weights_dir}")
    except Exception as e:
        print(f"Error downloading weights: {e}")
        
if __name__ == "__main__":
    download_qwen_weights()
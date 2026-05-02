"""Quick download of SFT and DPO models from HuggingFace subfolder repo."""
from huggingface_hub import snapshot_download
import os

# Download SFT model
print("Downloading SFT model...")
sft_path = snapshot_download(
    repo_id="sirius5005/SFT-and-DPO",
    allow_patterns=["SFT_merged/**"],
    local_dir="./models_local/",
)
print(f"SFT downloaded to: ./models_local/SFT_merged/")

# Download DPO model  
print("Downloading DPO model...")
dpo_path = snapshot_download(
    repo_id="sirius5005/SFT-and-DPO",
    allow_patterns=["DPO_merged/**"],
    local_dir="./models_local/",
)
print(f"DPO downloaded to: ./models_local/DPO_merged/")

print("\nDone! Use these local paths in server.py:")
print("  python server.py --model ./models_local/SFT_merged/ --quantization none")
print("  python server.py --model ./models_local/DPO_merged/ --quantization none")

import torch
import os
from moshi.models import loaders
from huggingface_hub import hf_hub_download

os.makedirs("weights", exist_ok=True)
mimi_path = "weights/mimi"
moshi_path = "weights/moshi"

if not os.path.exists(mimi_path):
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME, local_dir=mimi_path)

if not os.path.exists(moshi_path):
    moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME, local_dir=moshi_path)

import torch
import os
from moshi.models import loaders
from huggingface_hub import hf_hub_download

os.makedirs("weights", exist_ok=True)
mimi_path = "weights/mimi"
moshi_path = "weights/moshi"

repo = loaders.DEFAULT_REPO
mimi_name = loaders.MIMI_NAME
moshi_name = loaders.MOSHI_NAME

if not os.path.exists(mimi_path):
    mimi_weight = hf_hub_download(repo, mimi_name, local_dir=mimi_path)

if not os.path.exists(moshi_path):
    moshi_weight = hf_hub_download(repo, moshi_name, local_dir=moshi_path)

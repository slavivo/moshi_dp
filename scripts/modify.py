import argparse
from typing import List, Optional, Tuple, Dict
import torch
import time
import random
import numpy as np
from torch import nn
from moshi.models import loaders, LMGen
from moshi.utils.sampling import sample_token
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_model
import sentencepiece
import psutil
import os
import threading

class MLPProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

parser = argparse.ArgumentParser()
parser.add_argument("--moshi-weight", type=str)
parser.add_argument("--qwen-weight", type=str)
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument(
    "--hf-repo",
    type=str,
    default=loaders.DEFAULT_REPO,
    help="HuggingFace repo to use (defaults to Moshiko)"
)
parser.add_argument("--logits", action="store_true")
args = parser.parse_args()

print("Loading Qwen2.5-3B tokenizer...")
qwen_tokenizer = AutoTokenizer.from_pretrained(args.qwen_weight)

print("Loading Qwen2.5-3B model...")
qwen = AutoModelForCausalLM.from_pretrained(
    args.qwen_weight,
    device_map="auto",
    torch_dtype=torch.float16,  # Use float16 for efficiency
    trust_remote_code=True,  # Needed for some Qwen-specific code
    output_hidden_states=True,  # Enable hidden states output
    output_attentions=True,  # Also enable attentions if needed
    return_dict_in_generate=True  # Return a model output object instead of just tokens
)

# Load the model
print("Loading LM...")
lm = loaders.get_moshi_lm(args.moshi_weight, args.device)
lm.text_projector = MLPProjector(2048+4096, 4096, 2048).to(lm.device)
lm.text_head = nn.Linear(
        in_features=qwen.get_output_embeddings().in_features,
        out_features=qwen.get_output_embeddings().out_features,
        bias=False,
    )
lm.text_head.load_state_dict(qwen.get_output_embeddings().state_dict())
lm.text_head.to(dtype=qwen.get_output_embeddings().weight.dtype, device=lm.device)
    
lm.depformer_projectors = nn.ModuleList(
    [MLPProjector(4096+2048+1024, 4096, 1024).to(lm.device) for _ in range(lm.dep_q)]
)
# TODO should this really be from scratch? Can't we instead somehow use emb of Qwen?
lm.depformer_text_emb = lm.EmbeddingFactory(qwen.config.vocab_size, lm.depformer_dim)
lm.depformer_in = None
lm.text_emb = None
lm.text_linear = None

# Save the model
print("Starting model save...")
save_model(lm, "weights/new/model.safetensors")

print("Save completed successfully!")
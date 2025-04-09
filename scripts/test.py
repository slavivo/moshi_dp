import argparse
from typing import List, Optional, Tuple, Dict
import torch
import time
import random
import numpy as np
from torch import nn
from moshi.models import loaders, QwenLMGen
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
lm = loaders.get_qwen_lm("weights/new/model.safetensors", qwen, qwen_tokenizer, args.device)
lm_gen = QwenLMGen(lm, temp=0.5, text_temp=0.5)
# dummy input of shape B, K, T where K = 17 and all the values are 0
dummy_input = torch.zeros(1, 17, 2048).to(lm.device)
text_tokens, audio_tokens = lm_gen(dummy_input, False)

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
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

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
    output_attentions=False,
    attn_implementation="flash_attention_2",
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
original_emb = qwen.get_input_embeddings()
embedding_matrix = original_emb.weight.data.float()
lm.depformer_text_emb = nn.Embedding(
    num_embeddings = original_emb.num_embeddings,
    embedding_dim= original_emb.embedding_dim,
    device=lm.device,
    dtype=lm.depth_dtype,
)
lm.depformer_text_emb.weight.data.copy_(embedding_matrix)
# TODO think if this should be linear or MLP
lm.depformer_text_proj = nn.Linear(qwen.config.hidden_size, lm.depformer_dim, bias=True).to(lm.device)
# Perform PCA for initialization
mean_embedding = embedding_matrix.mean(dim=0, keepdim=True)
centered_embeddings = embedding_matrix - mean_embedding
U, S, Vt = torch.linalg.svd(centered_embeddings, full_matrices=False)
truncated_vt = Vt[:lm.depformer_dim, :]

lm.depformer_text_proj.weight.data.copy_(truncated_vt.to(lm.device))
mean_projection = torch.matmul(mean_embedding, truncated_vt.t())
lm.depformer_text_proj.bias.data.copy_(-mean_projection.squeeze(0))

lm.depformer_in = None
lm.text_emb = None
lm.text_linear = None

# Save the model
print("Starting model save...")
save_model(lm, "weights/new/model.safetensors")

print("Save completed successfully!")
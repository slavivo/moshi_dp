#!/usr/bin/env python3

import argparse
from typing import List, Optional, Tuple, Dict
import torch
import time
import random
import numpy as np
from moshi.models import loaders, LMGen
from moshi.utils.sampling import sample_token
from huggingface_hub import hf_hub_download
import sentencepiece

parser = argparse.ArgumentParser()
parser.add_argument("--moshi-weight", type=str)
parser.add_argument("--context", type=str, required=True)
parser.add_argument("--steps", default=100, type=int)
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--tokenizer", type=str, default=None)
parser.add_argument(
    "--hf-repo",
    type=str,
    default=loaders.DEFAULT_REPO,
    help="HuggingFace repo to use (defaults to Moshiko)"
)
parser.add_argument("--logits", action="store_true")
args = parser.parse_args()

def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_model(device: str, moshi_weight: str, tokenizer_path: Optional[str], hf_repo: str) -> Tuple[LMGen, sentencepiece.SentencePieceProcessor]:
    if tokenizer_path is None:
        tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
    
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
    print("Loading language model...")
    lm = loaders.get_moshi_lm(moshi_weight, device)
    lm_gen = LMGen(lm, temp=0.5, temp_text=0.5)
    print("Language model loaded successfully")
    
    return lm_gen, text_tokenizer

def run_generation_step(
    input_tensor: torch.Tensor,
    generated_audio: Optional[torch.Tensor],
    generated_text: Optional[torch.Tensor],
    lm_gen: LMGen,
    text_tokenizer: sentencepiece.SentencePieceProcessor,
    iteration: int,
    main_text: List[str],
    main_audio: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    if generated_audio is not None:
        input_tensor[0, 0, -1].copy_(generated_text)
        if iteration == 1:
            input_tensor[0, 1, -1].copy_(generated_audio[0])
        else:
            input_tensor[0, 1:9, -1].copy_(generated_audio)
    start_time = time.time()
    text_tokens, audio_tokens = lm_gen.generate(input_tensor, verbose=True)
    generation_time = time.time() - start_time
    print(f"Generation time: {generation_time:.2f}s")

    generated_audio = audio_tokens[0, -1, :]
    generated_text = text_tokens[0, -1]
    print(f"generated text: {generated_text.item()}")
    if generated_text.item() not in (0, 3):
        text = text_tokenizer.id_to_piece(generated_text.item())
        text = text.replace("▁", " ")
        main_text.append(text)
    main_audio.append(generated_audio)
    text_logits, depth_logits = None, None
    return generated_audio, generated_text, text_logits, depth_logits

def main():
    seed_all(42424242)
    lm_gen, text_tokenizer = initialize_model(args.device, args.moshi_weight, args.tokenizer, args.hf_repo)
    
    print("Loading context...")
    context = torch.load(args.context, map_location=args.device)
    print("Context loaded successfully")

    data: List[Dict[str, torch.Tensor]] = []
    main_text: List[str] = []
    main_audio: List[torch.Tensor] = []
    generated_audio = None
    generated_text = None
    input_tensor = torch.empty(1, 17, 0, device=args.device, dtype=torch.long)

    for i in range(len(context)):
        input_tensor = torch.cat([input_tensor, context[i]['input']], dim=2)
        generated_audio, generated_text, text_logits, depth_logits = run_generation_step(
            input_tensor, generated_audio, generated_text, lm_gen, text_tokenizer,
            i, main_text, main_audio
        )
        
        if args.logits:
            data.append({
                "text_logits": text_logits.clone().detach(),
                "depth_logits": [tensor.clone().detach() for tensor in depth_logits],
            })

    print("".join(main_text))
    torch.save(main_text, "benchmark_text.pt")
    torch.save(main_audio, "benchmark_audio.pt")
    if args.logits:
        torch.save(data, "benchmark_data.pt")

if __name__ == "__main__":
    main()
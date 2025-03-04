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
parser.add_argument("--input", type=str, required=True)
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
    lm_gen = LMGen(lm, temp=0.0, temp_text=0.0)
    print("Language model loaded successfully")
    
    return lm_gen, text_tokenizer

def run_generation_step(
    input_tensor: torch.Tensor,
    lm_gen: LMGen,
    text_tokenizer: sentencepiece.SentencePieceProcessor,
    gen_text: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    text_tokens, audio_tokens = lm_gen.generate(input_tensor, verbose=True)

    generated_audio = audio_tokens[0, -1, :]
    generated_text = text_tokens[0, -1]
    print(f"generated text: {generated_text.item()}")
    if generated_text.item() not in (0, 3):
        text = text_tokenizer.id_to_piece(generated_text.item())
        text = text.replace("▁", " ")
        gen_text.append(text)

    return generated_audio, generated_text

def main():
    seed_all(42424242)
    lm_gen, text_tokenizer = initialize_model(args.device, args.moshi_weight, args.tokenizer, args.hf_repo)
    
    print("Loading input...")
    input_ = torch.load(args.input, map_location=args.device)
    print("Input loaded successfully")

    pad_token = text_tokenizer.pad_id()
    epad_token = text_tokenizer.unk_id()

    first_pad_tensor = torch.tensor([32000, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048]).to(args.device).unsqueeze(1)
    second_pad_tensor = torch.tensor([pad_token, 1031, 2048, 2048, 2048, 2048, 2048, 2048, 2048]).to(args.device).unsqueeze(1)
    pad_tensor = torch.tensor([pad_token, 1031, 243, 1178, 546, 1736, 1572, 1978, 1648]).to(args.device).unsqueeze(1)

    for i in range(len(input_)):
        gen_text = []
        input_dict = input_[i]
        name, user_input = list(input_dict.items())[0]
        length = user_input.size(-1)
        repeated_pad_tensor = pad_tensor.expand(9, length - 2)
        system_input = torch.cat([first_pad_tensor, second_pad_tensor, repeated_pad_tensor], dim=1)
        system_input = system_input.unsqueeze(0)
        print(f"System input shape: {system_input.shape}, User input shape: {user_input.shape}")
        input_tensor = torch.cat([system_input, user_input], dim=1)
        input_tensor[0, 0, -1] = epad_token
        pad_count = 0
        for j in range(500):
            generated_audio, generated_text = run_generation_step(input_tensor, lm_gen, text_tokenizer, gen_text)

            to_add = torch.empty(1,17,1, device=args.device, dtype=torch.long)
            to_add[0, 0, -1].copy_(generated_text)
            if j == 0:
                to_add[0, 1, -1].copy_(generated_audio[0])
                to_add[0, 2:9, -1] = 2048 * torch.ones(7, device=args.device, dtype=torch.long)
            else:
                to_add[0, 1:9, -1].copy_(generated_audio)
            to_add[0, 9:17, 0] = pad_tensor[1:, 0]
            if generated_text == pad_token:
                pad_count += 1
            else:
                pad_count = 0
            if pad_count == 10:
                break

            input_tensor = torch.cat([input_tensor, to_add], dim=2)
        
        print(f"Generated text for the sample {name}:")
        print("".join(gen_text))
        
        if i == 10:
            break

if __name__ == "__main__":
    main()
import argparse
import torch
import time
import random
import numpy as np
from moshi.models import loaders, LMGen
import os

parser = argparse.ArgumentParser()
parser.add_argument("--moshi-weight", type=str)
parser.add_argument("--steps", default=100, type=int)
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("-d", "--dir", type=str, default="samples")
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

def run_generation_step(input_, lm_gen):
    # First moshi audio then user audio
    start_time = time.time()
    text_tokens, audio_tokens, text_logits, depth_logits = lm_gen.generate(input_, verbose=True)
    data = {
        "text_logits": text_logits.clone().detach(),
        "depth_logits": [tensor.clone().detach() for tensor in depth_logits],
    }
    dt = time.time() - start_time
    print(f"Generation time: {dt:.2f}s")
    return text_tokens, audio_tokens, data

def main():
    seed_all(42424242)

    print("loading lm")
    lm = loaders.get_moshi_lm(args.moshi_weight, args.device)
    lm_gen = LMGen(lm, temp=0.0, temp_text=0.0)
    print("lm loaded")

    for root, dirs, files in os.walk(args.dir):
        for file in files:
            if file.endswith(".pt"):
                file = os.path.join(root, file)
                input_ = torch.load(file) # B, 8, T
                input_ = input_.to(args.device)
                input_ = input_.to(torch.long)
                _, _, data = run_generation_step(input_, lm_gen)

    torch.save(data, "benchmark_data.pt")

if __name__ == "__main__":
    main()
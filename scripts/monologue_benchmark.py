import argparse
import torch
import time
from moshi.models import loaders, LMGen
from moshi.utils.sampling import sample_token

parser = argparse.ArgumentParser()
parser.add_argument("--moshi-weight", type=str)
parser.add_argument("--context", type=str, required=True)
parser.add_argument("--steps", default=100, type=int)
parser.add_argument("--device", type=str, default='cuda')
args = parser.parse_args()

print("loading lm")
lm = loaders.get_moshi_lm(args.moshi_weight, args.device)
lm_gen = LMGen(lm)
print("lm loaded")

print("loading context")
context = torch.load(args.context)
print("context loaded")

first = context['tensor_1']
x = first['user_codes'].size(-1)

# Pad the text tokens to length x
text_tokens = first['text']
text_padded = [-3] * (x - len(text_tokens)) + text_tokens  # Prepend -3
text_tensor = torch.tensor(text_padded, dtype=torch.int64).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, x)

first['user_codes'] = first['user_codes'].to(dtype=torch.int64)
first['dummy_codes'] = first['dummy_codes'].to(dtype=torch.int64)

# Concatenate the tensors
result = torch.cat([text_tensor, first['user_codes'], first['dummy_codes']], dim=1)  # Shape (1, 17, x)

print("Result shape:", result.shape)

def benchmark_test():
    _input = result.to(args.device)
    start_time = time.time()
    transformer_out, text_logits = lm_gen.lm_model.forward_text(_input)
    text_token = sample_token(text_logits.float(), lm_gen.use_sampling, lm_gen.temp_text, lm_gen.top_k_text)
    text_token = text_token[:, :, 0] # TODO check this (should it be [:, 0, 0] or [:, :, 0]?)
    audio_token = lm_gen.depformer_step(text_token, transformer_out)
    dt = time.time() - start_time
    print(f"Generation time: {dt:.2f}s")
    return text_token, audio_token

with torch.no_grad():
    generated_tokens = benchmark_test()
import argparse
import random
import time

from huggingface_hub import hf_hub_download
import numpy as np
import sphn
import torch
from torch.profiler import profile, ProfilerActivity
from datasets import load_dataset, Audio
from moshi.models import loaders

parser = argparse.ArgumentParser()
parser.add_argument("--mimi-weight", type=str)
parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
parser.add_argument("--device", type=str,
                    default='cuda' if torch.cuda.device_count() else 'cpu')
parser.add_argument("--profile", action='store_true')
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


seed_all(42424242)


print("loading mimi")
if args.mimi_weight is None:
    args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
mimi = loaders.get_mimi(args.mimi_weight, args.device)
print("mimi loaded")

dataset_name = "hf-internal-testing/librispeech_asr_dummy"  # Replace with actual dataset name
config = "clean"
split = "validation"

def process_dataset(dataset_name, split, config):
    # Load the dataset
    if not config:
        dataset = load_dataset(dataset_name, split=split)
    else:
        dataset = load_dataset(dataset_name, config, split=split)

    dataset = load_dataset(dataset_name, split, config)
    tokens = dataset["tokens"]
    lengths = dataset["lengths"]
    sample_rate = mimi.sample_rate
    sample_sr = dataset["audio"]["sampling_rate"]

    for idx in range(10):
        audio_sample = dataset[idx]["audio"]["array"]
        sample_pcm = sphn.resample(
            audio_sample, src_sample_rate=sample_sr, dst_sample_rate=sample_rate
        )
        sample_pcm = torch.tensor(sample_pcm, device=args.device)

    return tokens, lengths, dataset

# Process dataset
print(f"Processing dataset: {dataset_name}")
tokens, lengths, dataset = process_dataset(dataset_name, split, config)

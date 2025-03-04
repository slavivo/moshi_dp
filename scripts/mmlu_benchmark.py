import argparse
import torch
import time
import random
import numpy as np
from moshi.models import loaders, LMGen
import os
import json
import sentencepiece
from huggingface_hub import hf_hub_download
import re


parser = argparse.ArgumentParser()
parser.add_argument("--moshi-weight", type=str)
parser.add_argument("--steps", default=100, type=int)
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("-f", "--file", type=str, required=True)
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
    text_tokens, text_logits = lm_gen.generate(input_, verbose=True)
    text_token = text_tokens[:, -1]
    return text_token, text_logits

def main():
    seed_all(42424242)

    print("loading lm")
    lm = loaders.get_moshi_lm(args.moshi_weight, args.device)
    lm_gen = LMGen(lm, temp=0.8, temp_text=0.8)
    print("lm loaded")

    with open(args.file, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]

    cnt_correct = 0
    cnt_total = 0

    # get the codes for A, B, C, D
    a_code = text_tokenizer.encode("A", out_type=int)[0]
    b_code = text_tokenizer.encode("B", out_type=int)[0]
    c_code = text_tokenizer.encode("C", out_type=int)[0]
    d_code = text_tokenizer.encode("D", out_type=int)[0]

    random.shuffle(data)
    for j in range(len(data)):
        item = data[j]
        prompt = "The following is a  multiple choice question with a correct answer\n\n"
        prompt += "Question: " + item["question"] + "\n\n"
        prompt += "A. " + item["answer_a"] + "\n"
        prompt += "B. " + item["answer_b"] + "\n"
        prompt += "C. " + item["answer_c"] + "\n"
        prompt += "D. " + item["answer_d"] + "\n\n"
        prompt += "Correct answer (A, B, C or D): "
        # print("Prompt: ", prompt)
        prompt =  text_tokenizer.encode(prompt, out_type=int)
        prompt.insert(0, 32000)
        prompt = torch.tensor(prompt).unsqueeze(0).unsqueeze(0).to(args.device)

        _, logits = run_generation_step(prompt, lm_gen)
        # get the logits for A, B, C, D
        a_logits = logits[0, 0, -1, a_code].float().cpu()
        b_logits = logits[0, 0, -1, b_code].float().cpu()
        c_logits = logits[0, 0, -1, c_code].float().cpu()
        d_logits = logits[0, 0, -1, d_code].float().cpu()

        logits = [a_logits, b_logits, c_logits, d_logits]
        pred_answer = chr(ord('A') + np.argmax(logits))

        if pred_answer == item["correct_answer"]:
            cnt_correct += 1
        cnt_total += 1

        if j == 150:
            print(f"Accuracy: {cnt_correct/cnt_total}")
            break


tokenizer = hf_hub_download("kyutai/moshiko-pytorch-bf16", "tokenizer_spm_32k_3.model")
text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)

if __name__ == "__main__":
    main()
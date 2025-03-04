from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import random
import json
import sentencepiece
from huggingface_hub import hf_hub_download
import re
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, required=True)
args = parser.parse_args()

def main():
    device = "cuda" # the device to load the model onto

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(args.file, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]

    cnt_correct = 0
    cnt_total = 0

    a_code = tokenizer.encode("A")[-1]
    b_code = tokenizer.encode("B")[-1]
    c_code = tokenizer.encode("C")[-1]
    d_code = tokenizer.encode("D")[-1]

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

        model_inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**model_inputs).logits

        a_logits = logits[0, -1, a_code].float().cpu()
        b_logits = logits[0, -1, b_code].float().cpu()
        c_logits = logits[0, -1, c_code].float().cpu()
        d_logits = logits[0, -1, d_code].float().cpu()

        logits = [a_logits, b_logits, c_logits, d_logits]
        pred_answer = chr(ord('A') + np.argmax(logits))

        print(f"Correct answer: {item['correct_answer']}, Predicted answer: {pred_answer}")
        if pred_answer == item["correct_answer"]:
            cnt_correct += 1
        cnt_total += 1

        if j == 100:
            print(f"Accuracy: {cnt_correct/cnt_total}")
            break

if __name__ == "__main__":
    main()
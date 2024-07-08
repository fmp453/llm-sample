import os
import sys

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import trange


def main(model_id):

    if "llama" in model_id.lower():
        llama()
        return 

    dataset = load_dataset("cais/mmlu", 'computer_security')["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        token=os.environ["HF_TOKEN"],
        torch_dtype=torch.float32 # default. if gemma2, torch.bfloat16 is recommended
    )

    cnt = 0
    for i in trange(100):
        prompt = f'''
        Answer the following multiple choice given questions. You must return **only** your choice.\n\n
        Question: {dataset[i]["question"]} \n 
        Choices: {dataset[i]["choices"]}
        Answer:
        '''

        messages = [
            {"role": "user", "content": prompt}
        ]

        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True,)

        input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        
        output_ids = model.generate(input_ids, max_new_tokens=300)
        output = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True) 
        
        if dataset[i]["choices"][dataset[i]["answer"]] in output:
            cnt += 1

    print(f"accuracy: {cnt / 100}")

def llama():
    dataset = load_dataset("cais/mmlu", 'computer_security')["test"]
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        token=os.environ["HF_TOKEN"],
        trust_remote_code=True
    )

    cnt = 0
    for i in trange(100):
        prompt = f'''
        Answer the following multiple choice given questions. You must return **only** your choice.\n\n
        Question: {dataset[i]["question"]} \n 
        Choices: {dataset[i]["choices"]}
        Answer:
        '''

        messages = [
            {"role": "user", "content": prompt},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        output_ids = model.generate(
            input_ids, 
            max_new_tokens=300,
            eos_token_id=terminators,
        )
        output = tokenizer.decode(output_ids[0][input_ids.shape[-1]:]).split("<|eot_id|>")[0]
        output = output.replace("\n", "")

        if dataset[i]["choices"][dataset[i]["answer"]] in output:
            cnt += 1

    print(f"accuracy: {cnt / 100}")


if __name__ == "__main__":
    models = [
        "microsoft/Phi-3-mini-128k-instruct",
        "microsoft/Phi-3-small-128k-instruct",
        "microsoft/Phi-3-medium-128k-instruct",
        "Qwen/Qwen2-7B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it"
    ]
    
    main(models[int(sys.argv[1])])
#!/usr/bin/env python


import argparse
import os
# remove GPU memory cap
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
# allow for using CPU when MPS is not available then use MPS later
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# silence progress bar
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
ap = argparse.ArgumentParser()
ap.add_argument("--base_model", type=str, default="models/Qwen2.5-7B-Instruct")
ap.add_argument("--adapter_dir", type=str, required=True)
ap.add_argument("--question", type=str, required=True)
ap.add_argument("--merge_output", type=str, help="If set, merge and save here instead of generating")
ap.add_argument("--max_new_tokens", type=int, default=16)
args = ap.parse_args()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# load tokenizer of mdoel with adapters
tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)

# load base model weights
base = AutoModelForCausalLM.from_pretrained(
    args.base_model, 
    dtype=torch.bfloat16, 
    low_cpu_mem_usage=True
)
# specific to Qwen model. None bc instead of random sampling, using greedy decoding. so same answer each time
base.generation_config.temperature = None
base.generation_config.top_p = None
base.generation_config.top_k = None

# load fine tuned model, unify (base model with trained adapter)
# forward pass through the base weights and adapter's learned low-rank updates
model = PeftModel.from_pretrained(base, args.adapter_dir)

# mode 1: merge base+adapters and save model to disk
if args.merge_output:
    # print(f"Merging adapter into base weights: {args.merge_output}")
    # load adapter info with base model (no PEFT wrapper)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.merge_output)
    tokenizer.save_pretrained(args.merge_output)
    # hard stop
    raise SystemExit(0) 

# mode 2: generate single answer
model.to(device)
# turn off dropout and put batch-norm-like layers in inference mode 
# (needed for running model for eval/generation) instead of just training
model.eval()

# format chat
messages = [
    {"role": "system", "content": "You are an expert clinical geneticist and variant curation officer."},
    {"role": "user", "content": args.question},
]
# convert message into toekn sequence
inputs = tokenizer.apply_chat_template( # return input ids and attention mask
    messages, 
    tokenize=True,  # return token ids, not raw string
    add_generation_prompt=True, # add token to signify it is the assistant/user turn
    return_tensors="pt", # return pytorch tensor isntead of plain list
    return_dict=True 
).to(device)

# only forward pass (no backward-pass graph) bc not training. reduces memory/compute
with torch.no_grad():
    out = model.generate(
        **inputs, # unpack the ids from the attention maks
        max_new_tokens=args.max_new_tokens, # stop after n new tokens
        do_sample=False, # greedy decoding (pick highest prob next token, aka deterministic output). for reproducibility
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, # safety if ran without eos: if no/0 pad token then use end of seq token
    )

# format answer to remove role/end of turn markers
answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(answer.strip())

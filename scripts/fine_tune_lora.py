#!/usr/bin/env python
import os

# macOS specific envr set up
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# universal envr set up
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import argparse
# ###############
# train_file='results/train.jsonl'
# base_model='models/qwen2.5-7b-instruct'
# output_dir='models/qwen2.5-7b-variant-lora'
# # max_len=512
# # epochs=3.0
# # lr=2e-4
# # per_device_batch_size=1
# # grad_accum=16
# # lora_r=16
# # lora_alpha=32
# # lora_dropout=0.05
# # save_steps=200
# # logging_steps=10
# # seed=3
# ###############
ap = argparse.ArgumentParser()
ap.add_argument("--train_file", type=str, required=True, help="Path to JSONL training file")
ap.add_argument("--eval_file", type=str, default=None, help="Optional JSONL eval file")
ap.add_argument("--base_model", type=str, default="models/Qwen2.5-7B-Instruct")
ap.add_argument("--output_dir", type=str, required=True)
ap.add_argument("--max_len", type=int, default=512)
ap.add_argument("--epochs", type=float, default=3.0)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--per_device_batch_size", type=int, default=1)
ap.add_argument("--grad_accum", type=int, default=16)
ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--lora_alpha", type=int, default=32)
ap.add_argument("--lora_dropout", type=float, default=0.05)
ap.add_argument("--save_steps", type=int, default=200)
ap.add_argument("--logging_steps", type=int, default=10)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()



def build_example(tokenizer, question, answer, max_len):
    # reformat into standardized role dictionaries (system context, query, assistant answer)
    system_msg = {"role": "system", "content": "You are an expert clinical geneticist and variant curation officer."}
    user_msg = {"role": "user", "content": question}
    assistant_msg = {"role": "assistant", "content": answer}

    # tokenize full conversation:
    # transform std role dictionaries into a token specific to chat model (ex. <|im_start|>user, <|im_start|>assistant)
    full_ids = tokenizer.apply_chat_template(
        [system_msg, user_msg, assistant_msg],
        tokenize=True,
        add_generation_prompt=False,
    )

    # find split point between user prompt end and assistant answer
    prompt_ids = tokenizer.apply_chat_template(
        [system_msg, user_msg],
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = full_ids[:max_len]
    prompt_len = min(len(prompt_ids), len(full_ids))

    # mask prompt (with -100 for any tokens related to prompt)
    labels = list(full_ids)
    for i in range(prompt_len):
        labels[i] = -100
    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}

def _map_fn(ex):
    '''only use question and answer. leaves out extra info'''
    return build_example(tokenizer, ex["question"], ex["answer"], args.max_len)


# speed up error message if machine doesn't have MPS support
if not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS is not available in this PyTorch build/runtime. "
        "Check `python -c \"import torch; print(torch.backends.mps.is_available())\"`."
    )

device = torch.device("mps")

tokenizer = AutoTokenizer.from_pretrained(args.base_model)

# handle for qwen2.5 that was not trained with pad token (reuse eos_token as pad_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# load pre-trained model
model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.bfloat16, # for machines with lower resources (VRAM)
    low_cpu_mem_usage=True, # handle for less RAM, weights loaded to uninitialized tensors
)
model.to(device)

# Reduce mem usage and others
model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)  # good for quantized and non-quantized models

# PEFT - LoRA
lora_config = LoraConfig(
    # higher more trainable capacity (but high mem)
    r=args.lora_r, 
    # scaling factor of adapter, higher adapter weights more influence to frozen base weights
    lora_alpha=args.lora_alpha,
    # help prevent overfitting
    lora_dropout=args.lora_dropout, 
    bias="none",
    task_type="CAUSAL_LM",
    # which layers to add adapter
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# base model + adapters
model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

# load and tokenize dataset
data_files = {"train": args.train_file}
if args.eval_file:
    data_files["validation"] = args.eval_file
raw_ds = load_dataset("json", data_files=data_files)

# runs once upfront instead of re-tokenizing each epoch
tokenized_ds = raw_ds.map(
    _map_fn,
    remove_columns=raw_ds["train"].column_names,
    desc="Tokenizing",
)

# Pads different length sequences to same size for each batch, pads marked as -100
collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)

# training. 
# for each batch, forward pass -> loss (cross-entropy on the un-masked answer tokens) ->
# backward pass -> (every grad_accum steps) optimizer step, updating only the LoRA adapter weights
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.per_device_batch_size,
    gradient_accumulation_steps=args.grad_accum,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=args.logging_steps,
    save_strategy="steps",
    save_steps=args.save_steps,
    save_total_limit=2,
    eval_strategy="steps" if args.eval_file else "no",
    eval_steps=args.save_steps if args.eval_file else None,
    bf16=True,               # matches the bf16 weights above
    fp16=False,              # MPS fp16 autograd is unreliable; use bf16 instead
    optim="adamw_torch",     # bitsandbytes optimizers are CUDA-only, use PyTorch Adamw instead
    dataloader_pin_memory=False,  # pinned memory is a CUDA concept; skip on MPS
    gradient_checkpointing=True,
    report_to=[], # no logging
    seed=args.seed,
    remove_unused_columns=False, # already trimmed
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds.get("validation"),
    data_collator=collator,
)
trainer.train()

# save
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
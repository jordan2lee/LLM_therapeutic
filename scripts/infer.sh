#!/bin/bash

INPUT_FILE="results/prompts.txt"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found."
    exit 1
fi

# Each line is a new prompt
while IFS= read -r prompt_text || [ -n "$prompt_text" ]; do

    # Account for if saved on windows
    prompt_text=$(echo "$prompt_text" | tr -d '\r')

    if [ -z "$prompt_text" ]; then
        continue
    fi
    python scripts/run_tuned_model.py \
        --adapter_dir models/qwen2.5-7b-variant-lora \
        --question "$prompt_text"
done < "$INPUT_FILE"


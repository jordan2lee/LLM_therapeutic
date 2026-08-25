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
    llama-cli \
        --hf-repo paultimothymooney/Qwen2.5-7B-Instruct-Q4_K_M-GGUF \
        --hf-file qwen2.5-7b-instruct-q4_k_m.gguf \
        -c 65536 \
        -sys "You are a concise classifier. Respond ONLY with the requested classification text. Do NOT include greetings, intro phrases, bullet points, rationales, headers, or markdown formatting." \
        -p "$prompt_text"
done < "$INPUT_FILE"

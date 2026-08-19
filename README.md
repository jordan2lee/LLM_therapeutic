# LLM_therapeutic

# Set up

Install llama.cpp by building from source
```bash
cd src
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

Then add this to PATH. Example `export PATH="$HOME/LLM_therapeutic/src/llama.cpp/build/bin:$PATH"` in `.zshrc`

> `llama-server` is used downstream instead of `llama server` because installed from source 

# Goal

Run a LLM (TxGemma) and systematically benchmark its ability to interpret cancer related genes and mutaitons using public data

# LLM (Fully Local Model and inference)

Using llama.cpp to run TxGemma (DevQuasar/google.txgemma-27b-predict-GGUF,  specifically `Q4_K_M` 16.6 GB). 

```bash
# Start server
llama-server -hf DevQuasar/google.txgemma-27b-predict-GGUF:Q4_K_M
```

> This set up has a fully local LLM with inference on-device so nothing is sent to external servers (proprietary code, patient identifiers, creds, etc)
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

Then copy and paste the URL displayed after "llama_server: listening on"

> This set up has a fully local LLM with inference on-device so nothing is sent to external servers (proprietary code, patient identifiers, creds, etc)

# Analysis Plan
Using an external public dataset, determine which genes are mutated and gene expression profile (differential gene expression). Feed these genes and several controls (not mutated, normal expression, not mutated high expression, etc) into LLM for predicitons on biological relevance. Then use the external dataset to benchmark predictions from LLM.

## Build Reference Dataset 
Use the public cleaned TCGA data that is described in the [Nature paper](https://www.cell.com/cancer-cell/fulltext/S1535-6108(24)00477-X?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS153561082400477X%3Fshowall%3Dtrue) (Ellrott et al 2025) and is referenced in the [Tumor Molecular Pathology Toolkit GitHub](https://github.com/NCICCGPO/gdan-tmp-models)
```bash
wget -P src https://api.gdc.cancer.gov/data/5116e86f-7646-4b7b-9d6e-dafddf2cc0f3
```
Then decompress file `tar -xf TMP_20230209.tar.gz`

```bash
# to change default files use -i and -o
python scripts/build_ref.py
```

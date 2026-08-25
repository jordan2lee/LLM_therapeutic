# LLM_therapeutic

# Set up
Git clone this repo

Then make sure to carry over the submodule
```bash
# populate submodule
git submodule init
git submodule update
```

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

Run a LLM and systematically benchmark its ability to interpret cancer related genes and mutaitons using public data

# Construct and Benchmark LLM 
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

Programatically query ClinVar (public literature and other sources) for clinical significance of these genes based on mutation status and gene expression profile
```
submodule/clinvar-genes/scripts/clinvar_genes.py submodule/clinvar-genes/tests/fixtures/gene_list.txt submodule/clinvar-genes/results.ndjson > results/clinical_true.tsv
```

> File clinical_true.tsv will be used to assess LLM performance

## Locally download the model
Using the Qwen family because it performs well on medical, scientific, and biological benchmarks
```bash
cd src/llama.cpp
curl -L -o models/qwen2.5-7b-instruct-q4_k_m.gguf "https://huggingface.co"
cd ../..
```

> This set up has a fully local LLM with inference on-device so nothing is sent to external servers (proprietary code, patient identifiers, creds, etc)


## Query LLM
Run LLM. Will initally connect to the internet to download, cache and save model on local drive. 
After that will run entirely locally

### Get prompts for LLM testing
By default it does not have permission to write to disk. You can enable this or simply have the outputs send to standard out and save those manually as a file. Shown is the second way.
```bash
bash scripts/get_prompts.sh
```
Then save the stdout to a file called `results/prompts.txt`
### Get LLM calls
This model is ran locally, so it is hardcoded to up the tokens. **DO NOT** modify to run on the cloud unless you decrease the tokens are check the projected cost to run.
```bash
bash scripts/llm_testing.sh
```
Then save the stdout to a file called `results/responses_LLM.txt`

> File responses_LLM.txt will be used to benchmark against public peer-reviewed literature and other data sources

### Consolidate into a single file
Combine results from different files into a single summary table
```bash
python scripts/build_summary.py --outfile results/summary.tsv
```

> output
## Benchmark Performance
Assess how well the model captures true clinical attributes.
```bash
python scripts/benchmark.py --inputfile results/summary.tsv
```

# Parameter-Efficient Fine-Tuning
Use LoRA to perform the PEFT to freeze original model weights and train low-rank adapters that will be put into the attention layers.

GGUF is a main inference format for llama.cpp and so will start with the original Qwen model (Qwen2.5-7B-Instruct in 16-bit/Safetensors form)

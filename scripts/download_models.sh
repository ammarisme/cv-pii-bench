#!/usr/bin/env bash
# Download detector weights into ./models (about 7 GB in total). Requires: pip install "huggingface_hub[cli]"
set -e
mkdir -p models && cd models
dl() { hf download "$1" --local-dir "$1" --exclude "onnx/*" --exclude "*.onnx" --exclude "original/*"; }
dl fastino/gliner2-privacy-filter-PII-multi          # GLiNER2-PII (205M)
dl openai/privacy-filter                             # OpenAI Privacy Filter (1.5B MoE)
dl OpenMed/privacy-filter-multilingual-v2            # OpenMed multilingual Privacy Filter
dl knowledgator/gliner-pii-base-v1.0                 # Knowledgator GLiNER-PII
dl <hf-user>/gliner2-pii-cv-lora                     # our LoRA adapter (see hf/model)
hf download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir Qwen/Qwen2.5-3B-Instruct-GGUF   # optional
cd .. && python -m spacy download en_core_web_trf && python -m spacy download en_core_web_lg

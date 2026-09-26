---
license: apache-2.0
base_model: fastino/gliner2-privacy-filter-PII-multi
library_name: gliner2
language: [en, de, fr, es, it, nl, pt, pl, sv, fi, da, "no"]
pipeline_tag: token-classification
tags: [pii, anonymization, redaction, cv, resume, gdpr, gliner, lora]
datasets: [<hf-user>/cv-pii-bench]
---

# GLiNER2-PII-CV (LoRA adapter)

A LoRA adapter for [fastino/gliner2-privacy-filter-PII-multi](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi) that adapts it to personal data in CVs. It is meant to be used with the CV policy layer from [cv-pii-bench](https://github.com/ammarisme/cv-pii-bench), which filters tools, employers and work-only locations and propagates confirmed names.

## Results (40-CV hold-out, never used for training or tuning)

| System | Token recall | Precision | CVs with ≥1 leak | FP / CV | CPU s/CV |
|---|---|---|---|---|---|
| Base model, CV labels, + CV layer | 0.956 | 0.908 | 13 / 40 | 2.75 | 0.8 |
| **This adapter (threshold 0.4) + CV layer** | 0.956 | **0.926** | 12 / 40 | 2.17 | 0.7 |
| This adapter ∪ OpenAI Privacy Filter + CV layer | 0.972 | 0.907 | 11 / 40 | 2.83 | 1.6 |
| This adapter ∪ Privacy Filter ∪ Knowledgator GLiNER-PII + CV layer | **0.974** | 0.905 | **10** / 40 | 2.92 | 2.1 |

Timings are on 2 CPU cores. Without the CV layer the model alone reaches 0.879 recall and 0.861 precision; the layer matters.

## Labels

Query the model with these labels; the right-hand side is the benchmark label.

| Query label | Maps to |
|---|---|
| `person` | NAME |
| `email` | EMAIL |
| `phone_number` | PHONE |
| `street_address` | ADDRESS |
| `city` | LOCATION (home) |
| `personal_website` | URL |
| `username` | HANDLE |
| `national_id_number` | ID |
| `date_of_birth` | DOB |
| `nationality` | NATIONALITY |
| `marital_status` | MARITAL |
| `university` | EDU_ORG |
| `gender` | GENDER |

## Usage

The adapter targets the encoder's attention and dense layers plus the span and classification heads (84 modules). The simplest route is to merge it into the base weights once (W = W0 + (α/r)·B·A, α/r = 2) and load the result as a normal GLiNER2 model:

```python
import torch
from gliner2 import GLiNER2
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

base = GLiNER2.from_pretrained("fastino/gliner2-privacy-filter-PII-multi")
lora = load_file(hf_hub_download("<hf-user>/gliner2-pii-cv-lora", "adapter_model.safetensors"))
sd = base.state_dict()
for k in [k for k in lora if ".lora_A." in k]:
    target = k.split(".lora_A.")[0].replace("base_model.model.", "") + ".weight"
    B = lora[k.replace(".lora_A.", ".lora_B.")]
    sd[target] = sd[target] + 2.0 * (B.float() @ lora[k].float()).to(sd[target].dtype)
base.load_state_dict(sd)

labels = ["person", "email", "phone_number", "street_address", "city", "personal_website", "username",
          "national_id_number", "date_of_birth", "nationality", "marital_status", "university", "gender"]
print(base.extract_entities(cv_text, labels, threshold=0.4))
```

Alternatively, `scripts/finetune_gliner2.py` in the repository reproduces the adapter and the merged checkpoint in about 15 minutes on a CPU.

## Training

- **Data:** `devsynth` (80) and `devhard` (60) splits of [CV-PII-Bench](https://huggingface.co/datasets/<hf-user>/cv-pii-bench), chunked to 1,500 characters. All CVs are fictional. The `holdout` split was never used.
- **LoRA:** r = 16, α = 32, on encoder attention/dense layers and the heads; learning rate 1e-4 for encoder and heads; 3 epochs; batch size 2; max length 512; seed 0.
- **Hardware:** 2 CPU cores, about 15 minutes.

## Limitations

The model was trained and evaluated on synthetic CVs only. Real CVs have OCR noise and layouts not seen here. About one CV in four still leaks at least one item even with the best stack, so keep a human review step and validate on your own consented data. Do not use it as the sole safeguard for sensitive data.

## Licence

Apache-2.0, same as the base model.

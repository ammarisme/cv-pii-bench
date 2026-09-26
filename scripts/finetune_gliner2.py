"""LoRA fine-tune GLiNER2-PII on DEV-SYNTH + DEVHARD (never on the hold-out), then merge the adapter.
CPU-only: about 15 minutes on 2 cores. Usage:  PII_MODELS=./models python scripts/finetune_gliner2.py"""
import os, sys, random, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.set_num_threads(os.cpu_count())
from gliner2 import GLiNER2
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
from gliner2.training.data import InputExample
from safetensors.torch import load_file
from piibench.runner import datasets
from piibench.detectors import _chunks

MODELS = os.environ.get("PII_MODELS", "./models")
BASE = os.path.join(MODELS, "fastino/gliner2-privacy-filter-PII-multi")
OUT = os.path.join(MODELS, "ft-gliner2")
L = {"NAME": "person", "EMAIL": "email", "PHONE": "phone_number", "ADDRESS": "street_address", "LOCATION": "city",
     "URL": "personal_website", "HANDLE": "username", "ID": "national_id_number", "DOB": "date_of_birth",
     "NATIONALITY": "nationality", "MARITAL": "marital_status", "EDU_ORG": "university", "GENDER": "gender"}

ds = datasets()
examples = []
for d in ds["dev"] + ds["h3"]:
    for off, chunk in _chunks(d["text"], 1500):
        ents = {}
        for g in d["gold"]:
            if g["required"] and g["label"] in L and g["start"] >= off and g["end"] <= off + len(chunk):
                ents.setdefault(L[g["label"]], []).append(g["text"])
        examples.append(InputExample(text=chunk, entities={k: sorted(set(v)) for k, v in ents.items()}))
random.Random(0).shuffle(examples)

model = GLiNER2.from_pretrained(BASE)
cfg = TrainingConfig(output_dir=OUT, experiment_name="cv-ft", num_epochs=3, batch_size=2, use_lora=True, lora_r=16,
                     lora_alpha=32.0, save_adapter_only=False, gradient_checkpointing=True, max_len=512,
                     encoder_lr=1e-4, task_lr=1e-4, eval_strategy="no", save_best=False, logging_steps=10,
                     num_workers=0, pin_memory=False, fp16=False, bf16=False, seed=0)
ExtractorTrainer(model=model, config=cfg, train_data=examples).train()
model.save_pretrained(os.path.join(OUT, "final"))

# merge LoRA weights (W = W0 + alpha/r * B @ A) into a plain GLiNER2 checkpoint
sd = load_file(os.path.join(OUT, "final", "model.safetensors"))
scale = 32.0 / 16
merged = {}
for k, v in sd.items():
    if ".lora_A." in k or ".lora_B." in k:
        continue
    if ".base_layer." in k:
        if k.endswith("weight"):
            pre = k[: -len("base_layer.weight")]
            A, B = sd[pre + "lora_A.default.weight"].float(), sd[pre + "lora_B.default.weight"].float()
            v = (v.float() + scale * (B @ A)).to(v.dtype)
        merged[k.replace(".base_layer.", ".")] = v
    else:
        merged[k] = v
base = GLiNER2.from_pretrained(BASE)
base.load_state_dict(merged, strict=True)
base.save_pretrained(os.path.join(OUT, "merged"))
print("saved", os.path.join(OUT, "merged"))

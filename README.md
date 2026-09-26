# CV-PII-Bench

**A benchmark and reference pipeline for masking personal data in CVs, with an independent hold-out set.**

Generic PII tools struggle on CVs. A CV is full of names that are *not* the candidate's (employers, universities, tools such as Jenkins or Julia, cities) and the candidate's own name often appears with no sentence context at all. This repository contains:

- **200 annotated CVs in 12 languages**, including a 40-CV hold-out written independently of the pipeline and never used for tuning.
- **A span-based masking pipeline**: pluggable detectors plus a *CV policy layer* of rules that understand CV structure. The text is never regenerated, so nothing can be hallucinated.
- **Every experiment we ran**: 17+ stacks and their parameter sweeps, with cached detector outputs so all tables reproduce in seconds on a laptop, no models required.

## Headline results (40-CV hold-out, unseen during development)

| Stack | Recall | Precision | CVs with ≥1 leak | CPU s/CV |
|---|---|---|---|---|
| Microsoft Presidio, default settings | 0.646 | 0.385 | 39 / 40 | 0.5 |
| spaCy transformer + CV policy layer | 0.913 | 0.862 | 24 / 40 | 0.5 |
| GLiNER2-PII (CV labels) + CV policy layer | 0.956 | 0.908 | 13 / 40 | 0.8 |
| GLiNER2-PII fine-tuned (LoRA) + CV policy layer | 0.956 | **0.926** | 12 / 40 | 0.7 |
| GLiNER2-ft ∪ OpenAI Privacy Filter ∪ Knowledgator GLiNER-PII + CV layer | **0.974** | 0.905 | **10 / 40** | 2.1 |

Recall and precision are token-level. "CVs with ≥1 leak" counts documents where at least one identifying token of a gold entity stayed visible; this is the metric that matters for deployment. All timings are on 2 CPU cores with no GPU.

**Main finding.** A rule layer tuned on the CVs you can see looks perfect on them. Ours scored 100% on the 20 probe CVs it was written against, yet only about 75% on independently written ones. Always report on a hold-out nobody tuning the system has read.

Full write-up: [`paper/PAPER.md`](paper/PAPER.md). How every experiment was run, with each stack, parameter grid and result file: [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Data

| Split | CVs | Required gold spans | Languages | Authorship | Role |
|---|---|---|---|---|---|
| `probe20` | 20 | 107 | en | Hand-written probes, one failure mode each | Seen during development; upper bound only |
| `devsynth` | 80 | 754 | en de fr es nl sv fi | Template generator (`piibench/gen_dev.py`) | Early development |
| `devhard` | 60 | 623 | 12 | Written by an LLM agent blind to the pipeline | Error analysis, tuning, fine-tuning |
| `holdout` | 40 | 477 | 10 | Written by a second LLM agent blind to the pipeline | **Final evaluation only** |

Every person, email, phone number and ID in the data is **fictional**. The `devhard` and `holdout` sets were written by AI agents (Claude) given only the masking policy and a list of hard phenomena: tool-like names, name particles, two-column PDF artefacts, Europass forms, referees, publication lists, and non-personal URLs. The same data is available on Hugging Face as `abamerdeen/cv-pii-bench`.

**Masking policy.** Mask names (candidate and third parties), email, phone, street address and postcode, home city/region/country, personal URLs and handles, ID/account/licence numbers, date of birth or age, nationality, marital status, gender and pronouns, and universities and schools. Keep employers, job titles, skills and tools, certification names, work-only cities, date ranges and metrics. Ambiguous items are marked `required: false` and are neither rewarded nor penalised.

## Quick start

```bash
git clone https://github.com/ammarisme/cv-pii-bench && cd cv-pii-bench
pip install -r requirements.txt
python scripts/reproduce.py          # rebuilds the results table from cached detector outputs
```

To run detectors live on new text, download the models into `./models` (see `scripts/download_models.sh`) and use:

```python
from piibench.pipeline import run_pipeline
from piibench.core import apply_mask
from piibench import detectors as D

g2 = D.gliner2_detector(threshold=0.4, labels=D.GLINER2_CV_LABELS)   # PII_MODELS=./models
spans, _ = run_pipeline(cv_text, {"gliner2": g2}, {"propagate": "surname"})
print(apply_mask(cv_text, spans))    # "[NAME] ... [EMAIL] ..."
```

## How the pipeline works

1. **Section tagging** (12 languages): header, personal, experience, education, skills, references, publications, footer.
2. **Pattern rules**: emails (including obfuscated forms), phones, IBANs, ORCIDs, around 90 keyed ID fields, and key–value personal fields (`Nationality: …`, `Zivilstand | …`, YAML).
3. **Model detectors** (any subset, unioned): spaCy, Presidio, GLiNER2-PII, OpenAI Privacy Filter, OpenMed Privacy Filter, Knowledgator GLiNER-PII, or a local LLM.
4. **CV policy filter**: drop tool and language names, employer names with company suffixes, degree names, and work-only locations. Keep locations only in contact zones.
5. **Identity propagation**: once a name is confirmed, mask its surname and name-bearing URLs everywhere.
6. **Safety net**: every masked string is masked at every other occurrence.

## Lessons

- **The CV policy layer is the biggest single gain.** It adds 9–43 recall points depending on the detector.
- **Label choice matters more than thresholds.** Asking GLiNER2 for five CV-specific labels at inference raised raw recall from 0.68 to 0.87 with no training.
- **Union detectors; don't vote.** Requiring agreement always let more CVs leak.
- **LLM judges: add, never veto.** Every judge allowed to remove detections leaked more CVs, both a local 3B model and gpt-4o-mini. An add-only gpt-4o-mini judge took the best stacks from 11–12 to 9 leak CVs (recall 0.980) for up to 1 extra false-positive token per CV; see `scripts/run_judge.py`.
- **Fine-tuning mostly buys precision.** LoRA on 140 CVs (15 min on CPU) cut false positives by about 20% at the same recall.

## Limitations

The data is synthetic and fictional. Real CVs bring OCR noise and layouts not simulated here, so validate on your own consented sample before deployment. The hold-out has 40 CVs, so differences under about 2 recall points are within noise.

## Citation

See [`CITATION.cff`](CITATION.cff).

## Licence

Code, data and results: Apache-2.0. Third-party models keep their own licences: GLiNER2-PII, Privacy Filter, OpenMed, Knowledgator and Qwen2.5 are Apache-2.0; spaCy models are MIT.

*Research and code were developed with the assistance of an AI model (Claude, Anthropic).*

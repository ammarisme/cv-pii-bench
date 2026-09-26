"""Reproduce the main results table from cached detector outputs (no models needed).
Usage:  python scripts/reproduce.py            # all stacks, splits h2,h3,test
        python scripts/reproduce.py --live     # recompute detector outputs (needs models in ./models)
"""
import os, sys, json, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from piibench.runner import run_stack, datasets, CACHE
from piibench.exp import QWEN3B
import piibench.exp  # registers detectors

CV = {"propagate": "surname"}
STACKS = [
 ("S1  Presidio default (en_core_web_lg)",           {"name": "S1-presidio-default-lg", "detectors": {"presidio:lg": 0.0}, "cfg": {"raw": True, "regex": False}}),
 ("S1  Presidio default (en_core_web_trf)",          {"name": "S1-presidio-default-trf", "detectors": {"presidio:trf": 0.0}, "cfg": {"raw": True, "regex": False}}),
 ("S4  Privacy Filter raw",                          {"name": "S4-pf-openai-raw", "detectors": {"pf:openai": 0.5}, "cfg": {"raw": True, "regex": False}}),
 ("S6  Knowledgator GLiNER-PII raw",                 {"name": "S6-knowledgator-raw", "detectors": {"gliner:knowledgator": 0.3}, "cfg": {"raw": True, "regex": False}}),
 ("S3  GLiNER2-PII raw (CV labels)",                 {"name": "S3-gliner2cv-raw", "detectors": {"gliner2cv": 0.5}, "cfg": {"raw": True, "regex": False}}),
 ("S0  CV layer, rules only",                        {"name": "S0-rules-only", "detectors": {}, "cfg": CV}),
 ("S5  OpenMed PF + CV layer (t=0.7)",               {"name": "S5-pf-openmed+cv", "detectors": {"pf:openmed": 0.7}, "cfg": CV}),
 ("S2b spaCy trf + CV layer",                        {"name": "S2b-spacy+cv-trf", "detectors": {"spacy:trf": 0.0}, "cfg": CV}),
 ("S4  Privacy Filter + CV layer (t=0.05)",          {"name": "S4-pf-openai+cv", "detectors": {"pf:openai": 0.05}, "cfg": CV}),
 ("S8  spaCy trf + Qwen2.5-3B + CV layer",           {"name": "S8-spacytrf+qwen3b+cv", "detectors": {"llm:qwen3b": 0.0, "spacy:trf": 0.0}, "cfg": CV}),
 ("S6  Knowledgator + CV layer (t=0.3)",             {"name": "S6-knowledgator+cv", "detectors": {"gliner:knowledgator": 0.3}, "cfg": CV}),
 ("S3  GLiNER2-PII + CV layer (t=0.4)",              {"name": "S3-gliner2cv+cv", "detectors": {"gliner2cv": 0.4}, "cfg": CV}),
 ("S3ft GLiNER2-PII fine-tuned + CV layer (t=0.4)",  {"name": "S3ft-gliner2ft+cv", "detectors": {"gliner2ft": 0.4}, "cfg": CV}),
 ("S12 GLiNER2 + Privacy Filter + CV (0.2/0.1)",     {"name": "S12-gliner2cv+pf+cv", "detectors": {"gliner2cv": 0.2, "pf:openai": 0.1}, "cfg": CV}),
 ("S12ft GLiNER2-ft + Privacy Filter + CV",          {"name": "S12ft-gliner2ft+pf+cv", "detectors": {"gliner2ft": 0.4, "pf:openai": 0.1}, "cfg": CV}),
 ("S17ft GLiNER2-ft + PF + Knowledgator + CV",       {"name": "S17ft-gliner2ft+pf+knowledgator+cv", "detectors": {"gliner2ft": 0.4, "pf:openai": 0.1, "gliner:knowledgator": 0.5}, "cfg": CV}),
 ("S9  spaCy trf + CV + Qwen judge (add-only)",      {"name": "S9-spacytrf+cv+qwen3b-judge", "detectors": {"spacy:trf": 0.0}, "cfg": CV, "adjudicator": {"kind": "llama", "model": QWEN3B, "mode": "add_only"}}),
]

def main():
    """Evaluate every stack in STACKS on h2/h3/test (skipping uncached ones) and write results/main_table.md.

    Example (illustrative, not executed):
      argv     = ["scripts/reproduce.py"]             # add "--live" to recompute detector outputs with ./models
      prints   = "S1  Presidio default (en_core_web_lg)            h2   R=0.723 P=0.369 F2=0.607 leakCVs=38/40 FP/CV=35.27" ...
      returns  = None                                 # writes results/main_table.md (one row per stack x split)
    """
    live = "--live" in sys.argv
    ds = datasets()
    rows = []
    for title, st in STACKS:
        for split in ("h2", "h3", "test"):
            if split == "h3" and any(k in st["detectors"] for k in ("gliner2ft", "llm:qwen3b")) or (split == "h3" and st.get("adjudicator")):
                continue  # gliner2ft was trained on H3; Qwen was only run on H2/TEST
            if not live:
                missing = [k for k in st["detectors"] if not (CACHE / f"{k.replace('/', '_').replace(':', '_')}__{split}.json").exists()]
                if missing: continue
            try:
                agg, _ = run_stack(st, split, ds[split], save=False)
            except (ValueError, OSError) as e:  # LLM output not cached and model not downloaded
                print(f"{title:<48} {split:<4} skipped: {e.__class__.__name__} (download the model to run live)"); continue
            rows.append((title, split, agg))
            print(f"{title:<48} {split:<4} R={agg['token_recall']:.3f} P={agg['token_precision']:.3f} F2={agg['f2']:.3f} "
                  f"leakCVs={agg['leak_docs']}/{agg['docs']} FP/CV={agg['fp_per_doc']:.2f}", flush=True)
    out = ["| Stack | Split | Recall | Precision | F2 | CVs with a leak | FP tokens / CV |", "|---|---|---|---|---|---|---|"]
    for t, s, a in rows:
        out.append(f"| {t} | {s} | {a['token_recall']:.3f} | {a['token_precision']:.3f} | {a['f2']:.3f} | {a['leak_docs']}/{a['docs']} | {a['fp_per_doc']:.2f} |")
    os.makedirs("results", exist_ok=True)
    open("results/main_table.md", "w").write("\n".join(out) + "\n")
    print("wrote results/main_table.md")

if __name__ == "__main__":
    main()

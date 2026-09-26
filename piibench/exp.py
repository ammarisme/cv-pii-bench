"""Stack registry + CLI. Usage: python -m piibench.exp <stack_group> [split]"""
import os
import json, sys, time
from .runner import register, run_stack, datasets, fmt, RES
from . import detectors as D

register("presidio:lg", lambda: D.presidio_detector("en_core_web_lg", 0.0))
register("presidio:trf", lambda: D.presidio_detector("en_core_web_trf", 0.0))
register("spacy:lg", lambda: D.spacy_detector("en_core_web_lg"))
register("spacy:trf", lambda: D.spacy_detector("en_core_web_trf"))
register("spacy:xx", lambda: D.spacy_detector("xx_ent_wiki_sm"))
register("spacy:auto", lambda: D.spacy_auto_detector())
register("gliner2", lambda: D.gliner2_detector(threshold=0.05))
register("gliner2ft", lambda: D.gliner2_detector(repo="ft-gliner2/merged", threshold=0.05, labels=D.GLINER2_FT_LABELS))
register("gliner2cv", lambda: D.gliner2_detector(threshold=0.05, labels=D.GLINER2_CV_LABELS))
register("gliner:knowledgator", lambda: D.gliner_detector("knowledgator/gliner-pii-base-v1.0", 0.05, labels=D.KNOWLEDGATOR_LABELS))
register("gliner:nvidia", lambda: D.gliner_detector("nvidia/gliner-PII", 0.05))
register("gliner:urchade", lambda: D.gliner_detector("urchade/gliner_multi_pii-v1", 0.05))
register("pf:openai", lambda: D.privacy_filter_tokens_detector("openai/privacy-filter", 0.05))
register("pf:openmed", lambda: D.privacy_filter_tokens_detector("OpenMed/privacy-filter-multilingual-v2", 0.05))

def _llm_det(kind, model):
    """Build an LLM extraction detector for the given backend kind and model.

    Example (illustrative, not executed):
      kind     = "openai"
      model    = "gpt-4o-mini"
      returns  = <fn(text) -> [{"start": ..., "end": ..., "label": "NAME", "score": ..., ...}, ...]>   # needs API key / local GGUF
    """
    from .llm import Backend, llm_extract_detector
    return llm_extract_detector(Backend(kind, model))


QWEN3B = os.path.join(os.environ.get("PII_MODELS", "./models"), "Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf")
register("llm:qwen3b", lambda: _llm_det("llama", QWEN3B))
register("llm:gpt-4o-mini", lambda: _llm_det("openai", "gpt-4o-mini"))
register("llm:gpt-4.1", lambda: _llm_det("openai", "gpt-4.1"))

LEADER = RES / "leaderboard.jsonl"


def log(stack, split, agg, group, note=""):
    """Append one result row (stack params + headline metrics) to LEADER.

    Example (LEADER pointed at a temp file):
      stack    = {"name": "demo", "detectors": {"gliner2": 0.3}, "cfg": {"vote_k": 1}}
      split    = "dev"
      agg      = {"token_recall": 1.0, "token_precision": 0.8, "f2": 0.952, ..., "fp_examples": ["Python"]}
      group    = "g1"
      returns  = None   # appends: {"ts": "2026-09-26 12:34", "group": "g1", "stack": "demo", "split": "dev",
                        #  "params": {"detectors": {"gliner2": 0.3}, "vote_k": 1}, "note": "", "token_recall": 1.0, ...}
    """
    row = {"ts": time.strftime("%Y-%m-%d %H:%M"), "group": group, "stack": stack["name"], "split": split,
           "params": {"detectors": stack.get("detectors", {}), **stack.get("cfg", {}), **({"adjudicator": stack["adjudicator"]["model"].split("/")[-1], "judge_mode": stack["adjudicator"].get("mode", "full"), "drop_labels": stack["adjudicator"].get("drop_labels")} if stack.get("adjudicator") else {})}, "note": note,
           **{k: agg[k] for k in ("token_recall", "token_precision", "f1", "f2", "entity_recall", "entities_leaked",
                                  "leak_docs", "docs", "fp_tokens", "fp_per_doc", "sec_per_doc", "per_label_recall", "fp_examples")}}
    with open(LEADER, "a") as f: f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(stacks, group, splits=("dev",), note=""):
    """Run each stack on each split, log to the leaderboard and print a summary line.

    Example (illustrative, not executed):
      stacks   = [{"name": "gliner2+re", "detectors": {"gliner2": 0.3}, "cfg": {}}]
      group    = "sweep1"
      splits   = ("dev",)
      returns  = None   # prints: [sweep1] gliner2+re  dev  R=0.9.. P=0.8.. F2=0.9.. entR=... leakDocs=../80 FP/doc=... s/doc=...
    """
    ds = datasets()
    for st in stacks:
        for sp in splits:
            agg, _ = run_stack(st, sp, ds[sp])
            log(st, sp, agg, group, note)
            print(f"[{group}] {st['name']:<34} {sp:<4} {fmt(agg)}", flush=True)

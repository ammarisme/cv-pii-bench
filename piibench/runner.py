"""Experiment runner with raw-detector caching so parameter sweeps are cheap."""
import json, time, hashlib
from pathlib import Path
from .core import load_test, load_holdout2, load_spec_json, load_jsonl, score_doc, aggregate, apply_mask, DATA, ROOT
from .pipeline import run_pipeline

CACHE = ROOT / "detector_outputs"; CACHE.mkdir(exist_ok=True)
RES = ROOT / "results"; RES.mkdir(exist_ok=True)
FACTORIES = {}   # det_key -> zero-arg factory returning fn(text)
_loaded = {}


def register(key, factory):
    """Register a zero-arg detector factory under det_key in FACTORIES.

    Example:
      key      = "fake:email"
      factory  = lambda: (lambda t: [{"start": 0, "end": 10, "label": "EMAIL", "score": 0.9, "src": "fake"}])
      returns  = None   # FACTORIES now has "fake:email"; the factory is only called lazily by raw_spans
    """
    FACTORIES[key] = factory


def datasets():
    """Load all benchmark splits keyed by split name.

    Example:
      (no args)
      returns  = {"dev": [...80 docs], "test": [...20 docs], "h2": [...40 docs], "h3": [...60 docs]}
    """
    return {"dev": load_jsonl(DATA / "dev_cvs.jsonl"), "test": load_test(), "h2": load_holdout2(), "h3": load_spec_json("devhard_spec.json")}


def raw_spans(det_key, split, docs):
    """Run (or load from CACHE) a detector's raw spans for every doc in a split.

    Example (CACHE pointed at a temp dir):
      det_key  = "fake:email"   # registered as in the register() example
      split    = "dev"
      docs     = [{"id": "d1", "text": "jane@x.com\\n"}]
      returns  = {"spans": {"d1": [{"start": 0, "end": 10, "label": "EMAIL", "score": 0.9, ...}]},
                  "seconds": 2e-06, "docs": 1}   # also writes CACHE/fake_email__dev.json; later calls read it
    """
    f = CACHE / f"{det_key.replace('/', '_').replace(':', '_')}__{split}.json"
    if f.exists():
        return json.loads(f.read_text())
    if det_key not in _loaded:
        _loaded[det_key] = FACTORIES[det_key]()
    fn = _loaded[det_key]
    t0 = time.time(); out = {}
    for d in docs:
        out[d["id"]] = fn(d["text"])
    el = time.time() - t0
    f.write_text(json.dumps({"spans": out, "seconds": el, "docs": len(docs)}))
    return json.loads(f.read_text())


def build_adjudicator(spec):
    """Build an LLM adjudicator from a stack's adjudicator spec, or None when absent.

    Example:
      spec     = None                  # or {} -> None
      returns  = None
      # spec = {"kind": "openai", "model": "gpt-4o-mini", "mode": "full"} -> make_adjudicator(...) callable (needs API)
    """
    if not spec: return None
    from .llm import Backend, make_adjudicator
    return make_adjudicator(Backend(spec["kind"], spec["model"]), add_missed=spec.get("add_missed", True),
                            mode=spec.get("mode", "full"), drop_labels=tuple(spec["drop_labels"]) if spec.get("drop_labels") else None)


def run_stack(stack, split, docs, adjudicator=None, save=True):
    """stack = {"name", "detectors": {det_key: min_score}, "cfg": {...}, "adjudicator": {kind, model}?}

    Example (default cfg, fake detector from the register() example):
      stack    = {"name": "demo", "detectors": {"fake:email": 0.5}, "cfg": {}}
      split    = "dev"
      docs     = [{"id": "d1", "text": "Jane Doe\\njane@x.com\\n",
                   "gold": spans_from_spec(text, [("Jane Doe", "NAME", True, None), ("jane@x.com", "EMAIL", True, None)])}]
      save     = False
      returns  = ({"token_recall": 1.0, "token_precision": 1.0, "f2": 1.0, "leak_docs": 0, ..., "sec_per_doc": 0.015},
                  [{"id": "d1", "masked": "[NAME]\\n[EMAIL]\\n", "leaks": [], "fp": []}])   # NAME from header-name regex
    """
    adjudicator = adjudicator or build_adjudicator(stack.get("adjudicator"))
    raws = {k: raw_spans(k, split, docs) for k in stack.get("detectors", {})}
    results, outputs = [], []
    t0 = time.time()
    for d in docs:
        dets = {k: (lambda _t, k=k, d=d: [s for s in raws[k]["spans"][d["id"]] if s["score"] >= stack["detectors"][k]])
                for k in raws}
        spans, _ = run_pipeline(d["text"], dets, stack.get("cfg", {}), adjudicator=adjudicator)
        r = score_doc(d, spans); r["id"] = d["id"]; results.append(r)
        outputs.append({"id": d["id"], "masked": apply_mask(d["text"], spans), "leaks": r["leaks"], "fp": r["fp_tokens"]})
    agg = aggregate(results)
    det_secs = sum(raws[k]["seconds"] for k in raws)
    agg["sec_per_doc"] = round((det_secs + time.time() - t0) / len(docs), 3)
    if save:
        key = hashlib.md5(json.dumps(stack, sort_keys=True).encode()).hexdigest()[:8]
        (RES / f"{stack['name']}__{split}__{key}.json").write_text(json.dumps({"stack": stack, "split": split, "agg": agg, "outputs": outputs}, indent=1, ensure_ascii=False))
    return agg, outputs


def fmt(agg):
    """One-line summary string of an aggregate() result.

    Example:
      agg      = run_stack(...)[0]   # from the run_stack() example
      returns  = "R=1.000 P=1.000 F2=1.000 entR=1.000 leakDocs=0/1 FP/doc=0.00 s/doc=0.015"
    """
    return (f"R={agg['token_recall']:.3f} P={agg['token_precision']:.3f} F2={agg['f2']:.3f} entR={agg['entity_recall']:.3f} "
            f"leakDocs={agg['leak_docs']}/{agg['docs']} FP/doc={agg['fp_per_doc']:.2f} s/doc={agg['sec_per_doc']}")

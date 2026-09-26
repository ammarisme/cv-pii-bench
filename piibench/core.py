"""Core data loading, gold span construction and scoring for the CV PII benchmark."""
import json, re, importlib.util
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

STRIP = ",.;:()[]{}|\"'<>!?"
# connector tokens inside a PII span that do not identify anyone on their own
CONNECTORS = {"dot", "at", "com", "org", "net", "io", "no", "flat", "of", "the", "and", "-", "&", "/",
              # field keywords that annotators sometimes include in a span but that identify no one
              "born", "age", "aged", "née", "né", "nata", "nato", "le", "il", "citizen", "years", "old", "ans", "anos", "años", "anni",
              "·", "•", "|", "(", ")", "–", "—"}


def parse_cv_file(path):
    """Split a CV dump file into docs on its `=====\\nCV-NNN | title\\n=====` banners.

    Example:
      path     = "cvs.txt"   # file contents: "=====\\nCV-001 | Data Engineer\\n=====\\nJane Doe\\njane@x.com\\n"
      returns  = [{"id": "CV-001", "title": "Data Engineer", "text": "Jane Doe\\njane@x.com\\n"}]
    """
    txt = Path(path).read_text()
    blocks = re.split(r"^=+\n(CV-\d+) \| (.*)\n=+\n", txt, flags=re.M)
    docs = []
    for i in range(1, len(blocks), 3):
        docs.append({"id": blocks[i], "title": blocks[i + 1].strip(), "text": blocks[i + 2].rstrip("\n") + "\n"})
    return docs


def spans_from_spec(text, spec):
    """spec: list of (surface, label, required, nth) -> list of dict spans with char offsets.

    Example:
      text     = "Jane Doe\\njane@x.com\\nJane Doe, Leeds"
      spec     = [("Jane Doe", "NAME", True, None),        # nth=None -> every occurrence
                  ("jane@x.com", "EMAIL", True, None)]
      returns  = [{"start": 0, "end": 8, "label": "NAME", "required": True, "text": "Jane Doe"},
                  {"start": 20, "end": 28, "label": "NAME", "required": True, "text": "Jane Doe"},
                  {"start": 9, "end": 19, "label": "EMAIL", "required": True, "text": "jane@x.com"}]
                 # nth=0 for NAME would keep only the first; a missing surface raises ValueError
    """
    out = []
    for surface, label, req, nth in spec:
        occ = [m.start() for m in re.finditer(re.escape(surface), text)]
        if not occ:
            raise ValueError(f"gold string not found: {surface!r}")
        if nth is not None:
            occ = [occ[nth]]
        for s in occ:
            out.append({"start": s, "end": s + len(surface), "label": label, "required": req, "text": surface})
    return out


def load_test():
    """Load the test split from data/test_cvs.txt and attach gold spans from gold_test_spec.py.

    Example:
      (no args)
      returns  = [{"id": "CV-01", "title": ..., "text": ..., "gold": [{"start": ..., "label": "NAME", ...}, ...]},
                  ...]   # 20 docs
    """
    docs = parse_cv_file(DATA / "test_cvs.txt")
    spec_mod = importlib.util.spec_from_file_location("gs", DATA / "gold_test_spec.py")
    m = importlib.util.module_from_spec(spec_mod); spec_mod.loader.exec_module(m)
    for d in docs:
        d["gold"] = spans_from_spec(d["text"], m.GOLD[d["id"]])
    return docs


def load_jsonl(path):
    """Read a JSONL file into a list of dicts, skipping blank lines.

    Example:
      path     = "a.jsonl"   # contents: '{"id": "d1", "text": "Hi"}\\n\\n{"id": "d2", "text": "Yo"}\\n'
      returns  = [{"id": "d1", "text": "Hi"}, {"id": "d2", "text": "Yo"}]
    """
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def tokens(text):
    """Whitespace tokens with surrounding STRIP punctuation trimmed, as (start, end, text).

    Example:
      text     = "Jane (Doe), jane@x.com | UK."
      returns  = [(0, 4, "Jane"), (6, 9, "Doe"), (12, 22, "jane@x.com"), (25, 27, "UK")]   # "|" is dropped entirely
    """
    out = []
    for m in re.finditer(r"\S+", text):
        s, e = m.start(), m.end()
        while s < e and text[s] in STRIP: s += 1
        while e > s and text[e - 1] in STRIP: e -= 1
        if s < e:
            out.append((s, e, text[s:e]))
    return out


def _mask_array(n, spans, pred=False):
    """Per-character mask: 2 for required/predicted spans, 1 for optional gold, 0 elsewhere.

    Example:
      n        = 10
      spans    = [{"start": 0, "end": 4, "required": True}, {"start": 6, "end": 8, "required": False}]
      pred     = False
      returns  = [2, 2, 2, 2, 0, 0, 1, 1, 0, 0]
      # _mask_array(6, [(1, 3)], pred=True) -> [0, 2, 2, 0, 0, 0]   (tuples allowed when pred=True)
    """
    arr = [0] * n
    for sp in spans:
        s, e = (sp["start"], sp["end"]) if isinstance(sp, dict) else (sp[0], sp[1])
        v = 2 if pred or sp.get("required", True) else 1
        for i in range(max(0, s), min(n, e)):
            arr[i] = max(arr[i], v)
    return arr


def score_doc(doc, pred_spans):
    """Score predicted spans against a doc's gold at token level plus an entity-level leak check.

    Example:
      doc        = {"text": "Jane Doe\\njane@x.com\\nPython\\n",
                    "gold": spans_from_spec(doc["text"], [("Jane Doe", "NAME", True, None), ("jane@x.com", "EMAIL", True, None)])}
      pred_spans = [{"start": 0, "end": 4, "label": "NAME"},     # "Jane" only -> "Doe" leaks
                    {"start": 20, "end": 26, "label": "NAME"}]   # "Python"    -> false positive
      returns    = {"req": 3, "req_hit": 1, "pred": 2, "pred_ok": 1, "ents": 2, "ents_leaked": 2,
                    "leaks": [{"label": "NAME", "text": "Jane Doe", "visible": ["Doe"]},
                              {"label": "EMAIL", "text": "jane@x.com", "visible": ["jane@x.com"]}],
                    "fp_tokens": ["Python"], "per_label": {"NAME": [1, 1], "EMAIL": [1, 1]}}
    """
    text = doc["text"]; n = len(text)
    gold = doc["gold"]
    garr = _mask_array(n, gold)                    # 2 required, 1 optional
    parr = _mask_array(n, pred_spans, pred=True)   # 2 masked
    toks = tokens(text)
    st = defaultdict(int)
    fp_tokens = []
    for s, e, t in toks:
        L = e - s
        g_req = sum(1 for i in range(s, e) if garr[i] == 2) / L >= 0.5
        g_opt = sum(1 for i in range(s, e) if garr[i] >= 1) / L >= 0.5
        p = sum(1 for i in range(s, e) if parr[i]) / L >= 0.5
        if g_req:
            st["req"] += 1
            st["req_hit"] += p
        if p:
            st["pred"] += 1
            if g_opt: st["pred_ok"] += 1
            else: fp_tokens.append(t)
    # entity-level leak check
    leaks = []
    ents = [g for g in gold if g["required"]]
    for g in ents:
        crit = [(s, e, t) for s, e, t in tokens(text[g["start"]:g["end"]])
                if t.lower() not in CONNECTORS]
        missed = []
        for s, e, t in crit:
            s += g["start"]; e += g["start"]
            if sum(1 for i in range(s, e) if parr[i]) / (e - s) < 0.5:
                missed.append(t)
        if missed:
            leaks.append({"label": g["label"], "text": g["text"], "visible": missed})
    return {"req": st["req"], "req_hit": st["req_hit"], "pred": st["pred"], "pred_ok": st["pred_ok"],
            "ents": len(ents), "ents_leaked": len(leaks), "leaks": leaks, "fp_tokens": fp_tokens,
            "per_label": _per_label(ents, leaks)}


def _per_label(ents, leaks):
    """Count gold entities and leaked entities per label as {label: [total, leaked]}.

    Example:
      ents     = [{"label": "NAME"}, {"label": "NAME"}, {"label": "EMAIL"}]
      leaks    = [{"label": "NAME"}]
      returns  = {"NAME": [2, 1], "EMAIL": [1, 0]}
    """
    d = defaultdict(lambda: [0, 0])
    for g in ents: d[g["label"]][0] += 1
    for l in leaks: d[l["label"]][1] += 1
    return {k: v for k, v in d.items()}


def aggregate(results):
    """Pool score_doc results into corpus-level recall/precision/F1/F2, leak and FP stats.

    Example:
      results  = [score_doc(doc, pred_spans),                          # the score_doc example above (2 leaks)
                  score_doc(doc, [{"start": 0, "end": 19, "label": "NAME"}])]   # masks name+email, no leaks
      returns  = {"token_recall": 0.667, "token_precision": 0.8, "f1": 0.727, "f2": 0.690, "entity_recall": 0.5,   # floats rounded here
                  "entities": 4, "entities_leaked": 2, "leak_docs": 1, "docs": 2, "fp_tokens": 1, ...,
                  "fp_per_doc": 0.5, "per_label_recall": {"EMAIL": 0.5, "NAME": 0.5}, "fp_examples": ["Python"]}
    """
    T = defaultdict(float); per_label = defaultdict(lambda: [0, 0])
    leak_docs = 0; fp_list = []
    for r in results:
        for k in ("req", "req_hit", "pred", "pred_ok", "ents", "ents_leaked"):
            T[k] += r[k]
        leak_docs += 1 if r["ents_leaked"] else 0
        fp_list += r["fp_tokens"]
        for k, (a, b) in r["per_label"].items():
            per_label[k][0] += a; per_label[k][1] += b
    rec = T["req_hit"] / T["req"] if T["req"] else 1.0
    prec = T["pred_ok"] / T["pred"] if T["pred"] else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    f2 = 5 * prec * rec / (4 * prec + rec) if prec + rec else 0
    return {"token_recall": rec, "token_precision": prec, "f1": f1, "f2": f2,
            "entity_recall": 1 - T["ents_leaked"] / T["ents"] if T["ents"] else 1.0,
            "entities": int(T["ents"]), "entities_leaked": int(T["ents_leaked"]),
            "leak_docs": leak_docs, "docs": len(results), "fp_tokens": len(fp_list),
            "gold_tokens": int(T["req"]), "tp_tokens": int(T["req_hit"]), "fn_tokens": int(T["req"] - T["req_hit"]),
            "pred_units": int(T["pred"]), "pred_ok_units": int(T["pred_ok"]),
            "fp_per_doc": len(fp_list) / len(results),
            "per_label_recall": {k: round(1 - b / a, 3) for k, (a, b) in sorted(per_label.items())},
            "fp_examples": sorted(set(fp_list))[:40]}


def apply_mask(text, spans, fmt="[{label}]"):
    """Replace (merged) spans in text with a label placeholder.

    Example:
      text     = "Jane Doe, jane@x.com"
      spans    = [{"start": 0, "end": 4, "label": "NAME"}, {"start": 3, "end": 8, "label": "NAME"},   # overlap -> merged
                  {"start": 10, "end": 20, "label": "EMAIL"}]
      fmt      = "[{label}]"
      returns  = "[NAME], [EMAIL]"
      # apply_mask("Jane Doe", [{"start": 0, "end": 8}], fmt="<{label}>") -> "<PII>"
    """
    spans = merge_spans(spans)
    out, last = [], 0
    for sp in spans:
        out.append(text[last:sp["start"]]); out.append(fmt.format(label=sp.get("label", "PII")))
        last = sp["end"]
    out.append(text[last:])
    return "".join(out)


def merge_spans(spans):
    """Sort spans and merge overlapping/touching ones, keeping the first span's label.

    Example:
      spans    = [{"start": 10, "end": 20, "label": "EMAIL"}, {"start": 0, "end": 4, "label": "NAME"},
                  {"start": 3, "end": 8, "label": "LOC"}]    # overlaps NAME -> absorbed, label stays NAME
      returns  = [{"start": 0, "end": 8, "label": "NAME"}, {"start": 10, "end": 20, "label": "EMAIL"}]
    """
    spans = sorted(({"start": s["start"], "end": s["end"], "label": s.get("label", "PII")} for s in spans),
                   key=lambda x: (x["start"], -x["end"]))
    out = []
    for s in spans:
        if out and s["start"] <= out[-1]["end"]:
            out[-1]["end"] = max(out[-1]["end"], s["end"])
        else:
            out.append(dict(s))
    return out


def load_spec_json(name):
    """Load a JSON spec file from data/ and expand each row's gold tuples into spans.

    Example:
      name     = "devhard_spec.json"
      returns  = [{"id": "H3-001", "lang": ..., "text": ..., "gold": [{"start": ..., "label": "NAME", ...}, ...]},
                  ...]   # 60 rows
    """
    rows = json.loads((DATA / name).read_text())
    for r in rows:
        r["gold"] = spans_from_spec(r["text"], [tuple(g) for g in r["gold"]])
    return rows


def load_holdout2():
    """Load data/holdout2_spec.json and expand each row's gold tuples into spans.

    Example:
      (no args)
      returns  = [{"id": "H2-001", "lang": ..., "text": ..., "gold": [{"start": ..., "label": "NAME", ...}, ...]},
                  ...]   # 40 rows
    """
    rows = json.loads((DATA / "holdout2_spec.json").read_text())
    for r in rows:
        r["gold"] = spans_from_spec(r["text"], [tuple(g) for g in r["gold"]])
    return rows

"""Phase 11 - Presidio-hosted GLiNER: threshold sweep on devhard, then frozen choices on holdout + probe20.
Groups: pg-sweep (h3), pg-union-sweep (h3), pg-eval (h2, test). Needs the outputs saved by 25_presidio_gliner_cache.py.
Every threshold, and the union partner, is chosen on devhard by F2; holdout is only run with the chosen values.
Writes results/presidio_gliner_table.md."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piibench.runner import run_stack, datasets, fmt, CACHE, RES
from piibench.exp import log
from piibench.presidio_gliner import PG_CONFIGS

TH = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
RAW, CV = {"raw": True, "regex": False}, {"propagate": "surname"}
ds = datasets()
cached = [n for n in PG_CONFIGS if all((CACHE / f"pg_{n}__{s}.json").exists() for s in ("h3", "h2", "test"))]


def go(stack, split, group):
    agg, _ = run_stack(stack, split, ds[split], save=False)
    log(stack, split, agg, group)
    print(f"[{group}] {stack['name']:<56} {split:<4} {fmt(agg)}", flush=True)
    return agg


def sweep(make, group="pg-sweep"):
    """Run make(th) for every th on h3; return (best F2 threshold, its agg); ties go to fewer leak CVs."""
    res = {th: go(make(th), "h3", group) for th in TH}
    th = max(res, key=lambda t: (res[t]["f2"], -res[t]["leak_docs"]))
    return th, res[th]


# 1. every Presidio config alone: Presidio output only ("raw"), and Presidio output fed into the CV layer
chosen = {}
for n in cached:
    raw_th, raw_agg = sweep(lambda th: {"name": f"PG-{n}-raw", "detectors": {f"pg:{n}": th}, "cfg": RAW})
    cv_th, cv_agg = sweep(lambda th: {"name": f"PG-{n}+cv", "detectors": {f"pg:{n}": th}, "cfg": CV})
    chosen[n] = {"raw": (raw_th, raw_agg), "cv": (cv_th, cv_agg)}
    print(f"== {n}: raw t={raw_th}  +cv t={cv_th}", flush=True)

# 2. unions with the existing detectors; every config with devhard F2 >= 0.94 (+cv) is a candidate partner
bases = {"gliner2cv": {"gliner2cv": 0.4}, "gliner2cv+pf": {"gliner2cv": 0.4, "pf:openai": 0.1}}
uni = {}
for n in [n for n in chosen if chosen[n]["cv"][1]["f2"] >= 0.94]:
    for b, base in bases.items():
        uni[(n, b)] = sweep(lambda th: {"name": f"PG-U-{b}+{n}+cv", "detectors": {**base, f"pg:{n}": th}, "cfg": CV}, "pg-union-sweep")
(u_n, u_b), (u_th, u_agg) = max(uni.items(), key=lambda kv: (kv[1][1]["f2"], -kv[1][1]["leak_docs"]))
print(f"== best union: {u_b} + pg:{u_n} t={u_th}", flush=True)

# 3. frozen choices on holdout and probe20
rows = [("Presidio sample as published (t=0.30, raw)", {"name": "PG-doc-urchade-raw@0.30", "detectors": {"pg:doc-urchade": 0.30}, "cfg": RAW}, None)]
for n, c in chosen.items():
    rows += [(f"pg:{n} raw (t={c['raw'][0]})", {"name": f"PG-{n}-raw", "detectors": {f"pg:{n}": c["raw"][0]}, "cfg": RAW}, c["raw"][1]),
             (f"pg:{n} + CV layer (t={c['cv'][0]})", {"name": f"PG-{n}+cv", "detectors": {f"pg:{n}": c["cv"][0]}, "cfg": CV}, c["cv"][1])]
rows.append((f"{u_b} ∪ pg:{u_n} (t={u_th}) + CV", {"name": f"PG-U-{u_b}+{u_n}+cv", "detectors": {**bases[u_b], f"pg:{u_n}": u_th}, "cfg": CV}, u_agg))
# the fine-tuned GLiNER2 was trained on devhard, so it only takes the slot chosen with gliner2cv and is not scored on devhard
ft = {("gliner2ft" if k == "gliner2cv" else k): v for k, v in bases[u_b].items()}
rows.append((f"{u_b.replace('gliner2cv', 'gliner2ft')} ∪ pg:{u_n} (t={u_th}) + CV",
             {"name": f"PG-U-{u_b.replace('gliner2cv', 'gliner2ft')}+{u_n}+cv", "detectors": {**ft, f"pg:{u_n}": u_th}, "cfg": CV}, None))
rows.append((f"S17ft ∪ pg:{u_n} (t={u_th}) + CV", {"name": f"PG-U-S17ft+{u_n}+cv",
             "detectors": {"gliner2ft": 0.4, "pf:openai": 0.1, "gliner:knowledgator": 0.5, f"pg:{u_n}": u_th}, "cfg": CV}, None))

out = ["| Stack | devhard R | devhard F2 | devhard leak | holdout R | holdout P | holdout F2 | holdout leak CVs | holdout FP/CV | probe20 R | probe20 leak | s/CV |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for title, st, h3 in rows:
    h2, te = go(st, "h2", "pg-eval"), go(st, "test", "pg-eval")
    d = f"{h3['token_recall']:.3f} | {h3['f2']:.3f} | {h3['leak_docs']}/60" if h3 else "– | – | –"
    out.append(f"| {title} | {d} | {h2['token_recall']:.3f} | {h2['token_precision']:.3f} | {h2['f2']:.3f} | {h2['leak_docs']}/40 | "
               f"{h2['fp_per_doc']:.2f} | {te['token_recall']:.3f} | {te['leak_docs']}/20 | {h2['sec_per_doc']:.2f} |")
(RES / "presidio_gliner_table.md").write_text("\n".join(out) + "\n")
print("wrote results/presidio_gliner_table.md")

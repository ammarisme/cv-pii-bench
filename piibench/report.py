import json, sys
from .runner import RES

def rows(split=None, group=None):
    """Read RES/leaderboard.jsonl, filtered by split and/or group.

    Example (leaderboard with 3 rows: 2 "dev"/g1, 1 "test"/g2):
      split    = "dev"
      group    = None
      returns  = [{"ts": ..., "group": "g1", "stack": "demo", "split": "dev", "params": {"detectors": {"gliner2": 0.3}, "vote_k": 1}, ...},
                  {"ts": ..., "group": "g1", "stack": "demo", "split": "dev", "params": {"detectors": {"gliner2": 0.1}, "vote_k": 1}, ...}]
      # rows() -> 3 rows, rows(group="g2") -> 1 row
    """
    out = [json.loads(l) for l in open(RES / "leaderboard.jsonl")]
    return [r for r in out if (split is None or r["split"] == split) and (group is None or r["group"] == group)]

def pstr(p):
    """Compact string for a leaderboard row's params: detectors@threshold | cfg key=value.

    Example:
      p        = {"detectors": {"gliner2": 0.3, "spacy:lg": 0.0}, "vote_k": 1, "regex": True}
      returns  = "gliner2@0.3,spacy:lg@0.0 | vote_k=1,regex=True"
      # pstr({"vote_k": 2}) -> " | vote_k=2"
    """
    d = dict(p); dets = d.pop("detectors", {})
    return ",".join(f"{k}@{v}" for k, v in dets.items()) + " | " + ",".join(f"{k}={v}" for k, v in d.items())

def table(rs, top=None):
    """Print leaderboard rows sorted by F2 desc, then FP/doc asc.

    Example:
      rs       = rows("dev")   # the 2 dev rows from the rows() example
      top      = 1
      returns  = None   # prints:
                 # demo                           dev  R=1.000 P=0.800 F2=0.952 leak=  0/ 0d FP/d=1.00  gliner2@0.3 | vote_k=1
    """
    rs = sorted(rs, key=lambda r: (-r["f2"], r["fp_per_doc"]))
    for r in rs[:top]:
        print(f"{r['stack']:<30} {r['split']:<4} R={r['token_recall']:.3f} P={r['token_precision']:.3f} F2={r['f2']:.3f} "
              f"leak={r['entities_leaked']:>3}/{r['leak_docs']:>2}d FP/d={r['fp_per_doc']:.2f}  {pstr(r['params'])}")

def best_per_stack(split):
    """Best leaderboard row per stack name for a split (max F2, ties -> lower FP/doc).

    Example:
      split    = "dev"   # leaderboard from the rows() example
      returns  = {"demo": {"stack": "demo", "f2": 0.952, "params": {"detectors": {"gliner2": 0.3}, ...}, ...}}   # beats F2=0.9 row
    """
    best = {}
    for r in rows(split):
        k = r["stack"]
        if k not in best or (r["f2"], -r["fp_per_doc"]) > (best[k]["f2"], -best[k]["fp_per_doc"]): best[k] = r
    return best

if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    group = sys.argv[2] if len(sys.argv) > 2 else None
    table(rows(split, group), top=int(sys.argv[3]) if len(sys.argv) > 3 else 60)

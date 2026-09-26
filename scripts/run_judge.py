"""LLM-judge experiments with a hosted model (OpenAI), run on any machine that can reach api.openai.com.
Uses cached detector outputs, so no local models are needed. Every LLM answer is cached in
detector_outputs/llm/, so re-runs are free and resumable.

Usage:
  export OPENAI_API_KEY=...                      # or: --env-file path/to/.env
  python scripts/run_judge.py                    # gpt-4o-mini, all stacks x modes
  python scripts/run_judge.py --model gpt-4.1-mini --splits h2 --modes add_only
Output: results/judge_results.jsonl and results/judge_table.md
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from piibench.runner import run_stack, datasets, CACHE
import piibench.exp  # registers detectors

CV = {"propagate": "surname"}
# base stacks; gliner2ft was trained on h3, so it is evaluated on h2/test only
BASES = [
    ("S3ft", {"gliner2ft": 0.4}, ("h2", "test")),
    ("S17ft", {"gliner2ft": 0.4, "pf:openai": 0.1, "gliner:knowledgator": 0.5}, ("h2", "test")),
    ("S3", {"gliner2cv": 0.4}, ("h3", "h2", "test")),
    ("S2b", {"spacy:trf": 0.0}, ("h3", "h2", "test")),
]
MODES = ["none", "add_only", "drop_only", "full"]


def main():
    """Run each base stack x split x judge mode with an OpenAI adjudicator and write jsonl/markdown results.

    Example (illustrative, not executed):
      argv     = ["scripts/run_judge.py", "--model", "gpt-4o-mini", "--stacks", "S3ft", "--splits", "h2", "--modes", "none,add_only"]
      prints   = "S3ft   h2   add_only  R=0.980 P=0.897 leak=9/40 FN=23 FP=128 entFN=12/477 (0.4s)" ...
      returns  = None                                 # writes results/judge_results.jsonl, judge_table.md, judge_errors__S3ft__h2__<mode>.jsonl
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--splits", default="h3,h2,test")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--stacks", default=",".join(b[0] for b in BASES))
    ap.add_argument("--env-file")
    a = ap.parse_args()
    if a.env_file:
        for line in open(a.env_file):
            if line.strip().startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY or pass --env-file")
    ds = datasets()
    splits, modes, stacks = a.splits.split(","), a.modes.split(","), a.stacks.split(",")
    os.makedirs("results", exist_ok=True)
    out = open("results/judge_results.jsonl", "w")
    rows = []
    for sid, dets, allowed in BASES:
        if sid not in stacks: continue
        for split in [s for s in splits if s in allowed]:
            missing = [k for k in dets if not (CACHE / f"{k.replace('/', '_').replace(':', '_')}__{split}.json").exists()]
            if missing:
                print(f"skip {sid} {split}: no cached outputs for {missing}"); continue
            for mode in modes:
                st = {"name": f"{sid}+judge-{a.model}-{mode}", "detectors": dets, "cfg": CV}
                if mode != "none":
                    st["adjudicator"] = {"kind": "openai", "model": a.model, "mode": mode}
                t0 = time.time()
                agg, outs = run_stack(st, split, ds[split], save=False)
                rec = {"ts": time.strftime("%Y-%m-%d %H:%M"), "stack": sid, "judge": a.model if mode != "none" else None,
                       "mode": mode, "split": split, "seconds": round(time.time() - t0, 1),
                       **{k: agg[k] for k in ("token_recall", "token_precision", "f2", "leak_docs", "docs", "fp_per_doc",
                                                "gold_tokens", "tp_tokens", "fn_tokens", "fp_tokens", "entities", "entities_leaked")}}
                # per-CV errors: which gold items leaked (FN) and which tokens were wrongly masked (FP)
                with open(f"results/judge_errors__{sid}__{split}__{mode}.jsonl", "w") as fe:
                    for o in outs:
                        fe.write(json.dumps({"id": o["id"], "fn_leaks": o["leaks"], "fp_tokens": o["fp"]}, ensure_ascii=False) + "\n")
                out.write(json.dumps(rec) + "\n"); out.flush(); rows.append(rec)
                print(f"{sid:<6} {split:<4} {mode:<9} R={rec['token_recall']:.3f} P={rec['token_precision']:.3f} "
                      f"leak={rec['leak_docs']}/{rec['docs']} FN={rec['fn_tokens']} FP={rec['fp_tokens']} entFN={rec['entities_leaked']}/{rec['entities']} ({rec['seconds']}s)", flush=True)
    md = ["| Stack | Split | Judge mode | Recall | Precision | FN tokens | FP tokens | Entities leaked | Leak CVs |",
          "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['stack']} | {r['split']} | {r['mode']} | {r['token_recall']:.3f} | {r['token_precision']:.3f} | "
           f"{r['fn_tokens']}/{r['gold_tokens']} | {r['fp_tokens']} | {r['entities_leaked']}/{r['entities']} | {r['leak_docs']}/{r['docs']} |" for r in rows]
    open("results/judge_table.md", "w").write("\n".join(md) + "\n")
    print("wrote results/judge_table.md and results/judge_results.jsonl")


if __name__ == "__main__":
    main()

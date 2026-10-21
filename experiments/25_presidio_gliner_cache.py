"""Phase 11 - Presidio-hosted GLiNER: compute and save raw detector outputs (no scoring).
Usage: PII_MODELS=./models python experiments/25_presidio_gliner_cache.py [config ...]   # default: every PG_CONFIGS entry"""
import os, sys, time; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piibench.runner import raw_spans, datasets
from piibench.presidio_gliner import PG_CONFIGS
import piibench.exp  # registers pg:* detectors

ds = datasets()
for name in sys.argv[1:] or PG_CONFIGS:
    for split in ("h3", "h2", "test"):
        t0 = time.time()
        r = raw_spans(f"pg:{name}", split, ds[split])
        print(f"pg:{name:<24} {split:<4} {r['seconds'] / r['docs']:.2f} s/CV  ({time.time() - t0:.0f}s wall)", flush=True)

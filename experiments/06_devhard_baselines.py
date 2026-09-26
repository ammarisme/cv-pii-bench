"""Phase 3 - rules/spaCy baselines on devhard (H3) after commissioning it. Groups: h3-iter0, h3-iter1.
Usage: python experiments/06_devhard_baselines.py h3-iter0
Original file: it_h3.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
from piibench.exp import run
S=[{"name":"S0-rules-only","detectors":{},"cfg":{"propagate":"surname"}},
   {"name":"S2b-spacy+cv-trf","detectors":{"spacy:trf":0.0},"cfg":{"propagate":"surname"}},
   {"name":"S2b-spacy+cv-lg","detectors":{"spacy:lg":0.0},"cfg":{"propagate":"surname"}},
   {"name":"S2b-spacy+cv-auto","detectors":{"spacy:auto":0.0},"cfg":{"propagate":"surname"}}]
run(S, sys.argv[1], splits=("h3",))

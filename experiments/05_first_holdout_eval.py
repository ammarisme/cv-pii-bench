"""Phase 2 - first look at probe20 + holdout: exposed the seen-vs-unseen gap. Group: iter1-eval.
Original file: eval_spacy.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piibench.exp import run
S=[{"name":"S1-presidio-default-lg","detectors":{"presidio:lg":0.0},"cfg":{"raw":True,"regex":False}},
   {"name":"S1-presidio-default-trf","detectors":{"presidio:trf":0.0},"cfg":{"raw":True,"regex":False}},
   {"name":"S0-rules-only","detectors":{},"cfg":{"propagate":"surname"}},
   {"name":"S2-presidio+cv-trf","detectors":{"presidio:trf":0.85},"cfg":{"propagate":"surname"}},
   {"name":"S2b-spacy+cv-trf","detectors":{"spacy:trf":0.0},"cfg":{"propagate":"none"}},
   {"name":"S2b-spacy+cv-lg","detectors":{"spacy:lg":0.0},"cfg":{"propagate":"none"}},
   {"name":"S2b-spacy+cv-auto","detectors":{"spacy:auto":0.0},"cfg":{"propagate":"none"}}]
run(S,"iter1-eval",splits=("test","h2"))

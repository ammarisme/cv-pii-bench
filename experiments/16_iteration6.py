"""Phase 5 - consolidated comparison. Groups: iter6-h3, iter6-eval.
Usage: python experiments/16_iteration6.py iter6-h3 h3
Original file: it6.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
from piibench.exp import run
S=[{"name":"S12-gliner2cv+pf+cv","detectors":{"gliner2cv":0.4,"pf:openai":0.3},"cfg":{"propagate":"surname"}},
   {"name":"S12-gliner2cv+pf+cv","detectors":{"gliner2cv":0.2,"pf:openai":0.1},"cfg":{"propagate":"surname"}},
   {"name":"S3-gliner2cv+cv","detectors":{"gliner2cv":0.4},"cfg":{"propagate":"surname"}},
   {"name":"S3-gliner2cv+cv","detectors":{"gliner2cv":0.2},"cfg":{"propagate":"surname"}},
   {"name":"S4-pf-openai+cv","detectors":{"pf:openai":0.05},"cfg":{"propagate":"surname"}},
   {"name":"S2b-spacy+cv-trf","detectors":{"spacy:trf":0.0},"cfg":{"propagate":"surname"}},
   {"name":"S0-rules-only","detectors":{},"cfg":{"propagate":"surname"}}]
run(S, sys.argv[1], splits=tuple(sys.argv[2].split(",")))

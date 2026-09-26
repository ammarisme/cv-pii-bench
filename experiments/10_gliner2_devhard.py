"""Phase 4 - selected GLiNER2 thresholds vs baselines on devhard. Group: iter5-h3.
Original file: it5.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piibench.exp import run
S=[{"name":"S3-gliner2cv+cv","detectors":{"gliner2cv":th},"cfg":{"propagate":"surname"}} for th in [0.2,0.3,0.4,0.5]]
S+=[{"name":"S2b-spacy+cv-trf","detectors":{"spacy:trf":0.0},"cfg":{"propagate":"surname"}},{"name":"S0-rules-only","detectors":{},"cfg":{"propagate":"surname"}}]
run(S,"iter5-h3",splits=("h3",))

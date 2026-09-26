"""Phase 4 - selected configs on holdout + probe20. Group: iter5-eval.
Original file: it5e.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piibench.exp import run
S=[{"name":"S3-gliner2cv+cv","detectors":{"gliner2cv":th},"cfg":{"propagate":"surname"}} for th in [0.2,0.4]]
S+=[{"name":"S10-gliner2cv+spacytrf+cv","detectors":{"gliner2cv":0.4,"spacy:trf":0.0},"cfg":{"propagate":"surname"}},
    {"name":"S11b-gliner2cv+qwen3b+cv","detectors":{"gliner2cv":0.4,"llm:qwen3b":0.0},"cfg":{"propagate":"surname"}},
    {"name":"S2b-spacy+cv-trf","detectors":{"spacy:trf":0.0},"cfg":{"propagate":"surname"}}]
run(S,"iter5-eval",splits=("h2","test"))

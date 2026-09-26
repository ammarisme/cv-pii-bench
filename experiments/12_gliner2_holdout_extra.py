"""Phase 4 - GLiNER2 variants and unions on holdout + probe20. Group: gliner2-eval.
Original file: eval_g2.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piibench.exp import run
S=[{"name":"S3-gliner2cv+cv","detectors":{"gliner2cv":0.4},"cfg":{"propagate":"surname"}},
   {"name":"S3-gliner2cv+cv","detectors":{"gliner2cv":0.1},"cfg":{"propagate":"surname"}},
   {"name":"S3-gliner2+cv","detectors":{"gliner2":0.5},"cfg":{"propagate":"surname"}},
   {"name":"S10-gliner2cv+spacytrf+cv","detectors":{"gliner2cv":0.4,"spacy:trf":0.0},"cfg":{"propagate":"surname"}},
   {"name":"S11-gliner2cv+spacytrf+qwen3b+cv","detectors":{"gliner2cv":0.4,"spacy:trf":0.0,"llm:qwen3b":0.0},"cfg":{"propagate":"surname"}},
   {"name":"S11b-gliner2cv+qwen3b+cv","detectors":{"gliner2cv":0.4,"llm:qwen3b":0.0},"cfg":{"propagate":"surname"}}]
run(S,"gliner2-eval",splits=("h2","test"))

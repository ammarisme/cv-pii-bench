"""Phase 7 - Knowledgator stacks and the S17ft union on holdout + probe20. Group: kn-eval.
Original file: eval_kn.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piibench.exp import run
S=[{"name":"S6-knowledgator+cv","detectors":{"gliner:knowledgator":0.3},"cfg":{"propagate":"surname"}},
   {"name":"S16-gliner2cv+knowledgator+cv","detectors":{"gliner2cv":0.4,"gliner:knowledgator":0.5},"cfg":{"propagate":"surname"}},
   {"name":"S17ft-gliner2ft+pf+knowledgator+cv","detectors":{"gliner2ft":0.4,"pf:openai":0.1,"gliner:knowledgator":0.5},"cfg":{"propagate":"surname"}}]
run(S,"kn-eval",splits=("h2","test"))

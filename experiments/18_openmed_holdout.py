"""Phase 6 - OpenMed stacks on holdout + probe20. Group: openmed-eval.
Original file: eval_om.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piibench.exp import run
S=[{"name":"S5-pf-openmed+cv","detectors":{"pf:openmed":0.7},"cfg":{"propagate":"surname"}},
   {"name":"S14-gliner2cv+openmed+cv","detectors":{"gliner2cv":0.4,"pf:openmed":0.3},"cfg":{"propagate":"surname"}},
   {"name":"S15ft-gliner2ft+pf+openmed+cv","detectors":{"gliner2ft":0.4,"pf:openai":0.1,"pf:openmed":0.7},"cfg":{"propagate":"surname"}}]
run(S,"openmed-eval",splits=("h2","test"))

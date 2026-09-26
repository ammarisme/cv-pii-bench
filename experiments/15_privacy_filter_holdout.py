"""Phase 5 - PF stacks, unions and a 2-vote on holdout + probe20. Group: pf-eval.
Original file: eval_pf.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piibench.exp import run
S=[{"name":"S4-pf-openai+cv","detectors":{"pf:openai":0.05},"cfg":{"propagate":"surname"}},
   {"name":"S12-gliner2cv+pf+cv","detectors":{"gliner2cv":0.4,"pf:openai":0.3},"cfg":{"propagate":"surname"}},
   {"name":"S12-gliner2cv+pf+cv","detectors":{"gliner2cv":0.2,"pf:openai":0.1},"cfg":{"propagate":"surname"}},
   {"name":"S12v-gliner2cv+pf+spacy+cv-vote2","detectors":{"gliner2cv":0.2,"pf:openai":0.1,"spacy:trf":0.0},"cfg":{"propagate":"surname","vote_k":2}},
   {"name":"S13-gliner2cv+pf+qwen3b+cv","detectors":{"gliner2cv":0.4,"pf:openai":0.3,"llm:qwen3b":0.0},"cfg":{"propagate":"surname"}}]
run(S,"pf-eval",splits=("h2","test"))

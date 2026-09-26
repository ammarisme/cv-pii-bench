"""Phase 5 - PF min-prob sweep and GLiNER2 x PF grid (devhard). Group: pf-sweep.
Original file: sweep_pf.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piibench.exp import run
S=[]
for th in [0.05,0.1,0.2,0.3,0.5,0.7,0.9]:
    S.append({"name":"S4-pf-openai+cv","detectors":{"pf:openai":th},"cfg":{"propagate":"surname"}})
for th in [0.1,0.3,0.5,0.7]:
    S.append({"name":"S12-gliner2cv+pf+cv","detectors":{"gliner2cv":0.4,"pf:openai":th},"cfg":{"propagate":"surname"}})
    S.append({"name":"S12-gliner2cv+pf+cv","detectors":{"gliner2cv":0.2,"pf:openai":th},"cfg":{"propagate":"surname"}})
run(S,"pf-sweep",splits=("h3",))

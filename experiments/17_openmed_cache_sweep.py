"""Phase 6 - OpenMed multilingual PF: cache, threshold sweep, unions (devhard). Groups: openmed-cache, openmed-sweep.
Original file: run_om.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
os.environ["HF_HUB_OFFLINE"]="1"
from piibench.exp import run
run([{"name":"S5-pf-openmed-raw","detectors":{"pf:openmed":0.5},"cfg":{"raw":True,"regex":False}}],"openmed-cache",splits=("h3","h2","test"))
S=[{"name":"S5-pf-openmed+cv","detectors":{"pf:openmed":th},"cfg":{"propagate":"surname"}} for th in [0.05,0.1,0.3,0.5,0.7,0.9]]
S+=[{"name":"S14-gliner2cv+openmed+cv","detectors":{"gliner2cv":g,"pf:openmed":o},"cfg":{"propagate":"surname"}} for g,o in [(0.4,0.3),(0.2,0.1),(0.4,0.1)]]
S+=[{"name":"S15-gliner2cv+pf+openmed+cv","detectors":{"gliner2cv":0.4,"pf:openai":0.3,"pf:openmed":0.3},"cfg":{"propagate":"surname"}}]
run(S,"openmed-sweep",splits=("h3",))

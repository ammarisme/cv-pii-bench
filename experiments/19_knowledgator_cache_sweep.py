"""Phase 7 - Knowledgator GLiNER-PII: cache and threshold sweep (devhard). Groups: kn-cache, kn-sweep.
Original file: run_kn.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
os.environ["HF_HUB_OFFLINE"]="1"
from piibench.exp import run
run([{"name":"S6-knowledgator-raw","detectors":{"gliner:knowledgator":0.3},"cfg":{"raw":True,"regex":False}}],"kn-cache",splits=("h3","h2","test"))
S=[{"name":"S6-knowledgator+cv","detectors":{"gliner:knowledgator":th},"cfg":{"propagate":"surname"}} for th in [0.1,0.2,0.3,0.4,0.5,0.7]]
S+=[{"name":"S16-gliner2cv+knowledgator+cv","detectors":{"gliner2cv":0.4,"gliner:knowledgator":th},"cfg":{"propagate":"surname"}} for th in [0.3,0.5]]
run(S,"kn-sweep",splits=("h3",))

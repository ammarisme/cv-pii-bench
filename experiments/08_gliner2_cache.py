"""Phase 4 - cache raw GLiNER2 outputs (default and CV labels) for all splits. Group: gliner2-cache.
Original file: run_g2.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
os.environ["HF_HUB_OFFLINE"]="1"
from piibench.exp import run
S=[]
for det in ["gliner2","gliner2cv"]:
    S.append({"name":f"S3-{det}-raw","detectors":{det:0.5},"cfg":{"raw":True,"regex":False}})
run(S,"gliner2-cache",splits=("h3","h2","test","dev"))

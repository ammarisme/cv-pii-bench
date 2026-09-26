"""Phase 4 - GLiNER2 threshold sweep 0.1-0.8, with/without spaCy union and 2-vote (devhard). Group: gliner2-sweep.
Original file: sweep_g2.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piibench.exp import run
S=[]
for det in ["gliner2","gliner2cv"]:
    for th in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
        S.append({"name":f"S3-{det}+cv","detectors":{det:th},"cfg":{"propagate":"surname"}})
for th in [0.2,0.3,0.4,0.5,0.6,0.7]:
    S.append({"name":"S10-gliner2cv+spacytrf+cv","detectors":{"gliner2cv":th,"spacy:trf":0.0},"cfg":{"propagate":"surname"}})
    S.append({"name":"S10-gliner2cv+spacytrf+cv-vote2","detectors":{"gliner2cv":th,"spacy:trf":0.0},"cfg":{"propagate":"surname","vote_k":2}})
run(S,"gliner2-sweep",splits=("h3",))

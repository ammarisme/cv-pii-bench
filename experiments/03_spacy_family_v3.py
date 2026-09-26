"""Phase 1 - re-run after CV-layer fixes (devsynth). Group: spacy-family-v3.
Original file: sweep_spacy3.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piibench.exp import run
S=[]
for be in ["lg","trf","auto","xx"]:
    for prop in ["none","surname","all"]:
        S.append({"name": f"S2b-spacy+cv-{be}", "detectors": {f"spacy:{be}": 0.0}, "cfg": {"propagate": prop}})
for be in ["lg","trf"]:
    for th in [0.0, 0.5, 0.85]:
        S.append({"name": f"S2-presidio+cv-{be}", "detectors": {f"presidio:{be}": th}, "cfg": {"propagate": "surname"}})
S.append({"name": "S0-rules-only", "detectors": {}, "cfg": {"propagate": "surname"}})
run(S, "spacy-family-v3", splits=("dev",))

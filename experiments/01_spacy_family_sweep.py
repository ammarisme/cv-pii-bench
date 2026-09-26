"""Phase 1 - Presidio/spaCy thresholds x name propagation on devsynth. Group: spacy-family.
Original file: sweep_spacy.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piibench.exp import run
S = []
for th in [0.0, 0.3, 0.5, 0.7, 0.85]:
    for be in ["lg", "trf"]:
        S.append({"name": f"S1-presidio-default-{be}", "detectors": {f"presidio:{be}": th}, "cfg": {"raw": True, "regex": False}})
for be in ["lg", "trf"]:
    for th in [0.0, 0.4, 0.6, 0.85]:
        for prop in ["none", "surname", "all"]:
            S.append({"name": f"S2-presidio+cv-{be}", "detectors": {f"presidio:{be}": th}, "cfg": {"propagate": prop}})
for be in ["lg", "trf", "auto", "xx"]:
    for prop in ["none", "surname", "all"]:
        S.append({"name": f"S2b-spacy+cv-{be}", "detectors": {f"spacy:{be}": 0.0}, "cfg": {"propagate": prop}})
run(S, "spacy-family", splits=("dev",))

"""Phase 1 - header-name and contact-line rules on/off (devsynth). Group: spacy-family-v2.
Original file: sweep_spacy2.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piibench.exp import run
S=[]
for be in ["lg","trf"]:
    for hn in [False, True]:
        for cl in [False, True]:
            S.append({"name": f"S2b-spacy+cv-{be}", "detectors": {f"spacy:{be}": 0.0}, "cfg": {"propagate": "surname", "header_name_rule": hn, "contact_line_rule": cl}})
run(S, "spacy-family-v2", splits=("dev",))

"""Phase 3 - Presidio thresholds and one-at-a-time CV-layer ablations on devhard. Group: sweep-iter2.
Original file: sweep_iter2.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piibench.exp import run
import itertools
S=[]
base={"propagate":"surname"}
for be in ["trf","lg"]:
    for th in [0.0,0.4,0.6,0.85,0.95]:
        S.append({"name":f"S2-presidio+cv-{be}","detectors":{f"presidio:{be}":th},"cfg":dict(base)})
abl={"propagate":["none","surname","all"],"header_name_rule":[False],"contact_line_rule":[False],"location_lines":[False],
     "edu_acronyms":[False],"mask_orgs_in_education":[True],"allowlist":[False],"org_suffix_rule":[False],"safety_net":[False],"url_policy":["all"]}
for k,vals in abl.items():
    for v in vals:
        S.append({"name":"S2b-spacy+cv-trf","detectors":{"spacy:trf":0.0},"cfg":{**base,k:v}})
run(S,"sweep-iter2",splits=("h3",))

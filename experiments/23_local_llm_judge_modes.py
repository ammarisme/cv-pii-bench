"""Phase 8 - judge modes: add_only, drop_only, full (restricted labels), drop_only NAME. Group: llm-qwen3b-judge-modes.
Original file: run_judge_modes.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piibench.exp import run, QWEN3B
A=[]
for mode,dl in [("add_only",None),("drop_only",None),("full",["ORG","EDU_ORG","LOCATION","NATIONALITY","DATE","OTHER"]),("drop_only",["NAME"])]:
    A.append({"name":f"S9-spacytrf+cv+qwen3b-judge","detectors":{"spacy:trf":0.0},"cfg":{"propagate":"surname"},"adjudicator":{"kind":"llama","model":QWEN3B,"mode":mode,"drop_labels":dl}})
run(A,"llm-qwen3b-judge-modes",splits=("test","h2"))

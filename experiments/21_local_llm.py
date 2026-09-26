"""Phase 8 - Qwen2.5-3B (llama.cpp, Q4_K_M) as extractor and as keep/drop judge. Groups: llm-qwen3b, llm-qwen3b-judge.
Original file: run_llm.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piibench.exp import run, QWEN3B
S=[{"name":"S7-llm-only-qwen3b","detectors":{"llm:qwen3b":0.0},"cfg":{"raw":True,"regex":False}},
   {"name":"S7b-qwen3b+cv","detectors":{"llm:qwen3b":0.0},"cfg":{"propagate":"surname"}},
   {"name":"S8-spacytrf+qwen3b+cv","detectors":{"llm:qwen3b":0.0,"spacy:trf":0.0},"cfg":{"propagate":"surname"}}]
run(S,"llm-qwen3b",splits=("h2","test","h3"))
A=[{"name":"S9-spacytrf+cv+qwen3b-judge","detectors":{"spacy:trf":0.0},"cfg":{"propagate":"surname"},"adjudicator":{"kind":"llama","model":QWEN3B}}]
run(A,"llm-qwen3b-judge",splits=("h2","test","h3"))

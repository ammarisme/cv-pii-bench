"""Phase 8 - re-run after prompt fixes, plus 2-vote. Groups: llm-qwen3b, llm-qwen3b-judge.
Original file: run_llm2.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piibench.exp import run, QWEN3B
S=[{"name":"S7b-qwen3b+cv","detectors":{"llm:qwen3b":0.0},"cfg":{"propagate":"surname"}},
   {"name":"S8-spacytrf+qwen3b+cv","detectors":{"llm:qwen3b":0.0,"spacy:trf":0.0},"cfg":{"propagate":"surname"}},
   {"name":"S8-spacytrf+qwen3b+cv-vote2","detectors":{"llm:qwen3b":0.0,"spacy:trf":0.0},"cfg":{"propagate":"surname","vote_k":2}}]
run(S,"llm-qwen3b",splits=("h2","test"))
A=[{"name":"S9-spacytrf+cv+qwen3b-judge","detectors":{"spacy:trf":0.0},"cfg":{"propagate":"surname"},"adjudicator":{"kind":"llama","model":QWEN3B}}]
run(A,"llm-qwen3b-judge",splits=("test","h2"))

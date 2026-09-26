"""Phase 5 - cache OpenAI Privacy Filter token probabilities. Group: pf-cache.
Original file: run_pf.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
os.environ["HF_HUB_OFFLINE"]="1"
from piibench.exp import run
run([{"name":"S4-pf-openai-raw","detectors":{"pf:openai":0.5},"cfg":{"raw":True,"regex":False}}],"pf-cache",splits=("h3","h2","test"))

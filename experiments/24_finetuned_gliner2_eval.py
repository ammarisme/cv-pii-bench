"""Phase 9 - evaluate the LoRA-fine-tuned GLiNER2 (train: scripts/finetune_gliner2.py) raw, +CV, and with PF. Group: gliner2-finetuned.
Original file: run_ft.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
os.environ["HF_HUB_OFFLINE"]="1"
from piibench.exp import run
S=[{"name":"S3ft-gliner2ft-raw","detectors":{"gliner2ft":0.5},"cfg":{"raw":True,"regex":False}}]
S+=[{"name":"S3ft-gliner2ft+cv","detectors":{"gliner2ft":th},"cfg":{"propagate":"surname"}} for th in [0.2,0.3,0.4,0.5,0.6]]
S+=[{"name":"S12ft-gliner2ft+pf+cv","detectors":{"gliner2ft":th,"pf:openai":0.1},"cfg":{"propagate":"surname"}} for th in [0.2,0.4]]
run(S,"gliner2-finetuned",splits=("h2","test"))

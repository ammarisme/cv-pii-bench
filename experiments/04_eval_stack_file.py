"""Phase 2 - run a JSON list of stacks on given splits. Used with stacks_spacy.json for groups iter1, iter2, iter3.
Usage: python experiments/04_eval_stack_file.py iter1 test,h3,h2 experiments/stacks_spacy.json
Original file: eval_all.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, json
from piibench.exp import run
group=sys.argv[1]; splits=tuple(sys.argv[2].split(","))
S=json.load(open(sys.argv[3]))
run(S, group, splits=splits)

"""Tool - per-split error analysis: FP by source/label and leaks by label.
Usage: python experiments/30_error_analysis.py h3 gliner2cv '{"propagate":"surname"}'
Original file: diag.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, json, collections
from piibench.runner import datasets
from piibench.pipeline import run_pipeline
from piibench.core import score_doc, aggregate
import piibench.exp
from piibench.runner import raw_spans
split=sys.argv[1]; det=sys.argv[2] if len(sys.argv)>2 and sys.argv[2]!='-' else None
cfg=json.loads(sys.argv[3]) if len(sys.argv)>3 else {"propagate":"surname"}
docs=datasets()[split]
raws=raw_spans(det,split,docs) if det else None
res=[]; fpsrc=collections.Counter(); leaklab=collections.Counter(); fpex=collections.defaultdict(list); leakex=collections.defaultdict(list)
for d in docs:
    dets={det:(lambda t,d=d:raws["spans"][d["id"]])} if det else {}
    sp,_=run_pipeline(d["text"],dets,cfg); r=score_doc(d,sp); res.append(r)
    gold_opt=[(g['start'],g['end']) for g in d['gold']]
    for x in sp:
        t=d['text'][x['start']:x['end']]
        if not any(x['start']<ge and x['end']>gs for gs,ge in gold_opt):
            fpsrc[(x['src'],x['label'])]+=1; fpex[(x['src'],x['label'])].append(t[:50])
    for l in r['leaks']: leaklab[l['label']]+=1; leakex[l['label']].append((d['id'],l['text'][:50]))
a=aggregate(res); print({k:round(v,3) if isinstance(v,float) else v for k,v in a.items() if k not in('fp_examples',)})
print("FP by source:"); [print(' ',k,v,fpex[k][:8]) for k,v in fpsrc.most_common(15)]
print("Leaks by label:"); [print(' ',k,v,leakex[k][:10]) for k,v in leaklab.most_common()]

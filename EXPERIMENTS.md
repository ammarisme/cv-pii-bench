# Experiments: what was run, how, and where the results are

This file is the review guide. It lists every detector, every stack, every parameter grid, the order the experiments were run in, and the file each result lives in. The paper (`paper/PAPER.md`) reports the conclusions; this file shows how they were reached.

## 1. Splits and their names in code

| Name in paper / HF | Name in code and logs | CVs | Used for |
|---|---|---|---|
| `probe20` | `test` | 20 | Hand-written probes; rules were written against them. Upper bound only. |
| `devsynth` | `dev` | 80 | Template-generated (`piibench/gen_dev.py`). Early development, Phase 1. |
| `devhard` | `h3` | 60 | Written by an agent blind to the pipeline. **All threshold and rule choices**, error analysis, fine-tuning data. |
| `holdout` | `h2` | 40 | Written by a second blind agent. **Final numbers only**, read in aggregate, never used to choose anything. |

Loaders: `piibench/runner.py:datasets()` → `data/test_cvs.txt` + `data/gold_test_spec.py`, `data/dev_cvs.jsonl`, `data/devhard_spec.json`, `data/holdout2_spec.json`.

## 2. Scoring (`piibench/core.py`)

- **Token recall** = gold personal-data tokens covered by a predicted span ÷ all required gold tokens. Connector tokens (`van`, `de`, `-`) and field keys are ignored.
- **Token precision** = predicted units that overlap a gold span by ≥ 50% of their characters ÷ all predicted units.
- **FN tokens** = required gold tokens left visible (`fn_tokens`).
- **FP tokens** = masked tokens outside every gold span (`fp_tokens`, examples in `fp_examples`).
- **Items leaked** = required gold entities with ≥ 1 identifying token visible (`entities_leaked`).
- **Leak CVs** = CVs with ≥ 1 leaked item (`leak_docs`). This is the deployment metric.
- **F2** (recall-weighted) selects the best configuration of each stack.
- Spans marked `required: false` are neither rewarded nor penalised.

## 3. Detectors (`piibench/exp.py`, `piibench/detectors.py`)

Every detector's raw output was computed **once at a very low threshold** and saved in `detector_outputs/<key>__<split>.json`. A threshold sweep then only filters that saved output, so all sweeps are deterministic and re-run in seconds.

| Key | Model | Labels asked / mapping | Saved at |
|---|---|---|---|
| `presidio:lg`, `presidio:trf` | Presidio Analyzer on spaCy `en_core_web_lg` / `_trf` | Presidio defaults | score ≥ 0 |
| `spacy:lg`, `spacy:trf`, `spacy:xx` | spaCy NER (`xx_ent_wiki_sm` = multilingual) | PERSON/GPE/LOC/ORG → NAME/LOCATION/ORG | — |
| `spacy:auto` | per-CV language ID (lingua) → matching spaCy model | as above | — |
| `gliner2` | `fastino/gliner2-privacy-filter-PII-multi` | `GLINER2_LABELS` (model defaults) | 0.05 |
| `gliner2cv` | same model | `GLINER2_CV_LABELS`: defaults + university, school, nationality, marital_status, personal_website | 0.05 |
| `gliner2ft` | same model + our LoRA adapter, merged (`models/ft-gliner2/merged`) | `GLINER2_FT_LABELS` (13 labels, see model card) | 0.05 |
| `gliner:knowledgator` | `knowledgator/gliner-pii-base-v1.0` | `KNOWLEDGATOR_LABELS` | 0.05 |
| `pf:openai` | `openai/privacy-filter` | token classifier; span score = min over tokens of P(not O) | 0.05 |
| `pf:openmed` | `OpenMed/privacy-filter-multilingual-v2` | same decoding, `OPENMED_MAP` | 0.05 |
| `llm:qwen3b` | Qwen2.5-3B-Instruct Q4_K_M via llama.cpp, temperature 0, JSON output | extraction prompt `piibench/llm.py` | — |

Registered but not run (weights not downloaded): `gliner:nvidia`, `gliner:urchade`, `llm:gpt-4o-mini`, `llm:gpt-4.1` as extractors.

## 4. CV policy layer switches (`piibench/pipeline.py`, `cfg` of each stack)

| Switch | Default | What it does |
|---|---|---|
| `raw` | False | True = no CV layer at all (model output only); used for the "raw" rows |
| `regex` | True | email, phone, IBAN, ORCID, ~90 keyed ID fields, key–value personal fields |
| `policy` | True | the CV filter below as a whole |
| `allowlist` | True | drop tool, framework and language names (Jenkins, Julia, Ruby…) |
| `org_suffix_rule` | True | drop ORG-like spans with company suffixes (GmbH, Ltd, AB, Oy…) |
| `drop_names_in_skills` | True | drop NAME spans inside skills sections |
| `mask_orgs_in_education` | True | mask ORG spans inside education sections as universities |
| `edu_acronyms` | True | mask university acronyms (TU/e, NTNU, EPFL…), case-sensitive |
| `header_name_rule` | True | treat the first name-shaped line as the candidate's name |
| `contact_line_rule` | True | mask cities/regions only on contact lines |
| `location_lines` | True | `City, Country` lines in the header zone |
| `kv_address` | True | `Address:`-style key–value lines |
| `url_policy` | `personal` | mask personal URLs only (`all` = every URL) |
| `propagate` | `surname` | after a confirmed name, mask its surname and name-bearing URLs everywhere (`none`, `all`) |
| `safety_net` | True | any string masked once is masked at every occurrence |
| `vote_k` | 1 | require k detectors to agree (1 = union) |
| `name_min_score` | 0 | extra score floor on NAME spans |

## 5. Stack catalogue (final configurations)

Thresholds per detector were chosen on `devhard` by F2, then frozen and run once on `holdout`.

| ID | Detectors {key: threshold} | CV layer | Judge | Script |
|---|---|---|---|---|
| S0 | none (rules only) | on | – | 06, 16 |
| S1 | `presidio:lg` 0 / `presidio:trf` 0 | **off** (`raw`) | – | 01, 05 |
| S2 | `presidio:trf` 0.85 | on | – | 05, 07 |
| S2b | `spacy:trf` (also `lg`, `auto`, `xx`) | on | – | 01–07 |
| S3 | `gliner2` 0.5 / `gliner2cv` 0.4 | on | – | 09–12 |
| S3ft | `gliner2ft` 0.4 | on | – | 24 |
| S4 | `pf:openai` 0.05 | on | – | 14–16 |
| S5 | `pf:openmed` 0.7 | on | – | 17–18 |
| S6 | `gliner:knowledgator` 0.3 | on | – | 19–20 |
| S7 / S7b | `llm:qwen3b` | off / on | – | 21–22 |
| S8 | `llm:qwen3b` ∪ `spacy:trf` (and 2-vote) | on | – | 21–22 |
| S9 | `spacy:trf` | on | Qwen2.5-3B: full, add_only, drop_only, drop NAME only | 21–23 |
| S10 | `gliner2cv` 0.4 ∪ `spacy:trf` (and 2-vote) | on | – | 09, 11 |
| S11 / S11b | `gliner2cv` 0.4 ∪ (`spacy:trf`) ∪ `llm:qwen3b` | on | – | 11–12 |
| S12 | `gliner2cv` 0.2 ∪ `pf:openai` 0.1 (S12v = + spaCy, 2-vote) | on | – | 14–16 |
| S12ft | `gliner2ft` 0.4 ∪ `pf:openai` 0.1 | on | – | 24 |
| S13 | `gliner2cv` 0.4 ∪ `pf:openai` 0.3 ∪ `llm:qwen3b` | on | – | 15 |
| S14 | `gliner2cv` 0.4 ∪ `pf:openmed` 0.3 | on | – | 17–18 |
| S15 / S15ft | + `pf:openai` + `pf:openmed` | on | – | 17–18 |
| S16 | `gliner2cv` 0.4 ∪ `gliner:knowledgator` 0.5 | on | – | 19–20 |
| **S17ft** | `gliner2ft` 0.4 ∪ `pf:openai` 0.1 ∪ `gliner:knowledgator` 0.5 | on | – | 20 |
| J-* | S3ft, S17ft, S3, S2b | on | gpt-4o-mini: none / add_only / drop_only / full | `scripts/run_judge.py` |

## 6. Experiment log, in the order it was run

Every run appended one row to `results/leaderboard_full_log.jsonl` (366 rows): timestamp, `group`, `stack`, `split`, full `params`, all metrics, per-label recall and FP examples. The `group` column matches the groups below. Console output of each phase is in `experiments/logs/`.

| Phase | Script(s) | Group(s) | Split | Grid | Outcome |
|---|---|---|---|---|---|
| 1 | 01, 02, 03 | spacy-family, -v2, -v3 | devsynth | Presidio lg/trf × score {0, .3, .5, .7, .85}; spaCy lg/trf/auto/xx × propagate {none, surname, all}; header/contact rules on/off | CV layer adds most of the gain; `propagate=surname` best; 100% on probes |
| 2 | 04 (+ `stacks_spacy.json`), 05 | iter1, iter1-eval, iter2, iter3 | probe20, holdout, devsynth | S0, S1, S2, S2b | Holdout recall ~0.75 vs 1.00 on probes → commissioned `devhard` |
| 3 | 06, 07 | h3-iter0, h3-iter1, sweep-iter2 | devhard | Presidio thresholds; one-at-a-time ablation of every CV switch | Error analysis (`30_error_analysis.py`) drove rule fixes; each switch kept only if it helped devhard |
| 4 | 08–12 | gliner2-cache, gliner2-sweep, iter5-h3, iter5-eval, gliner2-eval | devhard → holdout | GLiNER2 default vs CV labels × {0.1…0.8}; union and 2-vote with spaCy | CV labels: raw recall 0.68 → 0.87; 0.4 chosen |
| 5 | 13–16 | pf-cache, pf-sweep, pf-eval, iter6-h3, iter6-eval | devhard → holdout | PF min-prob {0.05…0.9}; GLiNER2 {0.2, 0.4} × PF {0.1…0.7}; 2-vote | Union 0.2/0.1 best recall; voting leaks more |
| 6 | 17–18 | openmed-cache, openmed-sweep, openmed-eval | devhard → holdout | OpenMed {0.05…0.9}; unions | Precise but lower recall; small union gain |
| 7 | 19–20 | kn-cache, kn-sweep, kn-eval | devhard → holdout | Knowledgator {0.1…0.7}; union with GLiNER2 {0.3, 0.5} | Best third detector for the union |
| 8 | 21–23 | llm-qwen3b, llm-qwen3b-judge, llm-qwen3b-judge-modes | probe20, holdout | Qwen as extractor, in unions, 2-vote, and judge in 4 modes | Extractor helps a little at 50× cost; any veto loses 8–26 recall points |
| 9 | `scripts/finetune_gliner2.py`, 24 | gliner2-finetuned | probe20, holdout | LoRA r16/α32, 3 epochs on devsynth + devhard; threshold {0.2…0.6}; with PF | Same recall, ~20% fewer FPs; S17ft best overall |
| 10 | `scripts/run_judge.py` (run on the author's machine) | `results/judge_results.jsonl` | devhard, holdout, probe20 | 4 base stacks × {none, add_only, drop_only, full}, gpt-4o-mini, T = 0 | Add-only: 11 → 9 leak CVs; every veto mode leaks more |

The fine-tuned model is never scored on `devhard`, because it was trained on it.

## 7. Where each result lives

| What | File |
|---|---|
| Every run with parameters and metrics | `results/leaderboard_full_log.jsonl` |
| Main table (regenerated) | `results/main_table.md` ← `scripts/reproduce.py` |
| Hosted-judge runs with FN/FP counts | `results/judge_results.jsonl`, `results/judge_table.md` |
| Per-CV leaks and FPs for each judge run | `results/judge_errors__<stack>__<split>__<mode>.jsonl` |
| Saved detector outputs | `detector_outputs/*.json` |
| Saved LLM answers (Qwen + gpt-4o-mini), keyed by prompt hash | `detector_outputs/llm/*.json` |
| Console logs per phase, fine-tuning loss curves | `experiments/logs/` |
| Prompts (extractor, judge) | `piibench/llm.py` (`POLICY`, `EXTRACT_SYS`, `ADJ_SYS`) |

## 8. Re-running

```bash
python scripts/reproduce.py                 # main table from saved outputs, no models
python experiments/14_privacy_filter_sweep.py   # any sweep, from saved outputs
python experiments/30_error_analysis.py h2 gliner2cv '{"propagate":"surname"}'
python scripts/run_judge.py --env-file .env # hosted judge; free when answers are saved
PII_MODELS=./models python scripts/finetune_gliner2.py   # re-train the adapter (~15 min CPU)
```

Scripts append to `results/leaderboard.jsonl` and write per-run masked outputs to `results/<stack>__<split>__<hash>.json`; both are git-ignored so the published log stays as run.

## 9. Known deviations

- **Rule changes between phases.** The CV layer was improved between phases, so an early row in the full log can differ from the same stack re-run with today's code. The paper's tables are re-runs with the final code; the log keeps the historical values.
- **Environment.** The judge runs were done on a different machine (Python 3.9, older spaCy/lingua). Its no-judge baselines differ from the main table by ≤ 0.001 recall and 1 leak CV; judge rows are compared only with baselines from the same machine.
- **Qwen judge (S9).** 9 of 60 saved answers were written under an older cache key, so `reproduce.py` skips S9 unless the Qwen model is downloaded; its numbers are in the full log.
- **Fine-tuning reruns.** The adapter was trained twice with seed 0; hold-out results were identical to three decimals.

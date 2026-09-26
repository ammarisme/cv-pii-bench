---
license: apache-2.0
language: [en, de, fr, es, it, nl, pt, pl, sv, fi, da, "no"]
task_categories: [token-classification]
tags: [pii, anonymization, redaction, cv, resume, gdpr, ner, synthetic]
pretty_name: CV-PII-Bench
size_categories: [n<1K]
configs:
  - config_name: default
    data_files:
      - {split: probe20, path: probe20.jsonl}
      - {split: devsynth, path: devsynth.jsonl}
      - {split: devhard, path: devhard.jsonl}
      - {split: holdout, path: holdout.jsonl}
---

# CV-PII-Bench

200 fictional CVs in 12 languages with character-level annotations of personal data, built to evaluate PII masking in recruitment. It includes a 40-CV hold-out written independently of any masking system.

Code, pipeline, all experiment results and the paper: https://github.com/ammarisme/cv-pii-bench

## Splits

| Split | CVs | Required spans | Authorship | Intended use |
|---|---|---|---|---|
| `probe20` | 20 | 107 | Hand-written, one failure mode per CV | Smoke test only; the reference pipeline was developed against it |
| `devsynth` | 80 | 754 | Template generator | Training / development |
| `devhard` | 60 | 623 | LLM agent blind to the pipeline | Tuning, error analysis, training |
| `holdout` | 40 | 477 | Second LLM agent blind to the pipeline | **Final evaluation only — please do not train or tune on it** |

## Schema

```json
{"id": "H2-001", "split": "holdout", "lang": "en", "text": "LIN Xiaoyu\nData Engineer | Manchester, UK\n...",
 "spans": [{"start": 0, "end": 10, "label": "NAME", "required": true, "text": "LIN Xiaoyu"}, ...]}
```

Labels: `NAME`, `EMAIL`, `PHONE`, `ADDRESS`, `LOCATION` (home city/region/country), `URL` (personal), `HANDLE`, `ID`, `DOB`, `NATIONALITY`, `MARITAL`, `GENDER`, `EDU_ORG`.

`required: false` marks ambiguous items (for example a city that is both home and workplace). Scorers should neither reward nor penalise them.

## Masking policy

Mask the candidate's and third parties' names and contact details, home location, personal URLs, IDs, date of birth/age, nationality, marital status, gender/pronouns and educational institutions. Keep employers, job titles, skills, tools, certifications, work-only cities, dates and metrics. Text that looks like a name but isn't — tools like Jenkins or Julia, eponyms, company names — must stay visible.

## Hard phenomena covered

Two-column PDF artefacts, Europass and German key–value forms, name particles and compound surnames, CJK and transliterated names, tool names that are also personal names, referees and supervisors, publication author lists, obfuscated emails, and non-personal URLs next to personal ones.

## Baseline results on `holdout`

| System | Token recall | Precision | CVs with ≥1 leak |
|---|---|---|---|
| Presidio default | 0.646 | 0.385 | 39 / 40 |
| GLiNER2-PII + CV policy layer | 0.956 | 0.908 | 13 / 40 |
| Best union (see repo) | 0.974 | 0.905 | 10 / 40 |

## Provenance and limitations

All people, emails, phone numbers, addresses and IDs are **fictional**; any resemblance to real persons is coincidental. Phone numbers use reserved fictional ranges where possible, and emails use `example.*` domains. The `devhard` and `holdout` CVs were written by AI agents (Claude, Anthropic), so they lack real OCR noise and may share the blind spots of one model family. Use the hold-out as an independent check, not as proof of performance on real CVs.

## Citation

```bibtex
@misc{cvpiibench2026,
  title  = {Masking Personal Data in CVs: A Span-Based Pipeline and an Honest Hold-Out},
  author = {Ammar Ameeruddeen},
  year   = {2026},
  url    = {https://github.com/ammarisme/cv-pii-bench}
}
```

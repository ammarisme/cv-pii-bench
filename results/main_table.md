| Stack | Split | Recall | Precision | F2 | CVs with a leak | FP tokens / CV |
|---|---|---|---|---|---|---|
| S1  Presidio default (en_core_web_lg) | h2 | 0.723 | 0.369 | 0.607 | 38/40 | 35.27 |
| S1  Presidio default (en_core_web_lg) | h3 | 0.727 | 0.392 | 0.621 | 50/60 | 29.50 |
| S1  Presidio default (en_core_web_lg) | test | 0.804 | 0.464 | 0.702 | 9/20 | 13.15 |
| S1  Presidio default (en_core_web_trf) | h2 | 0.646 | 0.385 | 0.569 | 39/40 | 29.18 |
| S1  Presidio default (en_core_web_trf) | h3 | 0.671 | 0.415 | 0.597 | 55/60 | 24.75 |
| S1  Presidio default (en_core_web_trf) | test | 0.779 | 0.525 | 0.710 | 12/20 | 10.00 |
| S4  Privacy Filter raw | h2 | 0.500 | 0.739 | 0.535 | 39/40 | 5.03 |
| S4  Privacy Filter raw | h3 | 0.466 | 0.781 | 0.507 | 57/60 | 3.35 |
| S4  Privacy Filter raw | test | 0.678 | 0.875 | 0.710 | 18/20 | 1.35 |
| S6  Knowledgator GLiNER-PII raw | h2 | 0.812 | 0.713 | 0.790 | 37/40 | 9.28 |
| S6  Knowledgator GLiNER-PII raw | h3 | 0.835 | 0.728 | 0.811 | 51/60 | 8.12 |
| S6  Knowledgator GLiNER-PII raw | test | 0.888 | 0.857 | 0.881 | 7/20 | 2.10 |
| S3  GLiNER2-PII raw (CV labels) | h2 | 0.866 | 0.769 | 0.845 | 34/40 | 7.38 |
| S3  GLiNER2-PII raw (CV labels) | h3 | 0.873 | 0.788 | 0.854 | 46/60 | 6.10 |
| S3  GLiNER2-PII raw (CV labels) | test | 0.909 | 0.857 | 0.898 | 5/20 | 2.15 |
| S0  CV layer, rules only | h2 | 0.849 | 0.933 | 0.864 | 31/40 | 1.73 |
| S0  CV layer, rules only | h3 | 0.855 | 0.924 | 0.868 | 41/60 | 1.82 |
| S0  CV layer, rules only | test | 1.000 | 0.993 | 0.999 | 0/20 | 0.10 |
| S5  OpenMed PF + CV layer (t=0.7) | h2 | 0.904 | 0.918 | 0.906 | 26/40 | 2.30 |
| S5  OpenMed PF + CV layer (t=0.7) | h3 | 0.926 | 0.896 | 0.920 | 34/60 | 2.77 |
| S5  OpenMed PF + CV layer (t=0.7) | test | 1.000 | 0.993 | 0.999 | 0/20 | 0.10 |
| S2b spaCy trf + CV layer | h2 | 0.913 | 0.862 | 0.903 | 24/40 | 4.15 |
| S2b spaCy trf + CV layer | h3 | 0.941 | 0.875 | 0.927 | 24/60 | 3.45 |
| S2b spaCy trf + CV layer | test | 1.000 | 0.993 | 0.999 | 0/20 | 0.10 |
| S4  Privacy Filter + CV layer (t=0.05) | h2 | 0.925 | 0.906 | 0.921 | 22/40 | 2.73 |
| S4  Privacy Filter + CV layer (t=0.05) | h3 | 0.939 | 0.889 | 0.928 | 27/60 | 3.03 |
| S4  Privacy Filter + CV layer (t=0.05) | test | 1.000 | 0.982 | 0.996 | 0/20 | 0.25 |
| S8  spaCy trf + Qwen2.5-3B + CV layer | h2 | 0.950 | 0.812 | 0.918 | 18/40 | 6.28 |
| S8  spaCy trf + Qwen2.5-3B + CV layer | test | 1.000 | 0.986 | 0.997 | 0/20 | 0.20 |
| S6  Knowledgator + CV layer (t=0.3) | h2 | 0.950 | 0.888 | 0.937 | 19/40 | 3.40 |
| S6  Knowledgator + CV layer (t=0.3) | h3 | 0.947 | 0.873 | 0.931 | 25/60 | 3.55 |
| S6  Knowledgator + CV layer (t=0.3) | test | 1.000 | 0.982 | 0.996 | 0/20 | 0.25 |
| S3  GLiNER2-PII + CV layer (t=0.4) | h2 | 0.956 | 0.908 | 0.946 | 13/40 | 2.75 |
| S3  GLiNER2-PII + CV layer (t=0.4) | h3 | 0.967 | 0.891 | 0.951 | 15/60 | 3.05 |
| S3  GLiNER2-PII + CV layer (t=0.4) | test | 1.000 | 0.986 | 0.997 | 0/20 | 0.20 |
| S3ft GLiNER2-PII fine-tuned + CV layer (t=0.4) | h2 | 0.956 | 0.926 | 0.950 | 12/40 | 2.17 |
| S3ft GLiNER2-PII fine-tuned + CV layer (t=0.4) | test | 1.000 | 0.986 | 0.997 | 0/20 | 0.20 |
| S12 GLiNER2 + Privacy Filter + CV (0.2/0.1) | h2 | 0.971 | 0.874 | 0.950 | 12/40 | 4.00 |
| S12 GLiNER2 + Privacy Filter + CV (0.2/0.1) | h3 | 0.978 | 0.843 | 0.948 | 11/60 | 4.68 |
| S12 GLiNER2 + Privacy Filter + CV (0.2/0.1) | test | 1.000 | 0.986 | 0.997 | 0/20 | 0.20 |
| S12ft GLiNER2-ft + Privacy Filter + CV | h2 | 0.972 | 0.907 | 0.958 | 11/40 | 2.83 |
| S12ft GLiNER2-ft + Privacy Filter + CV | test | 1.000 | 0.986 | 0.997 | 0/20 | 0.20 |
| S17ft GLiNER2-ft + PF + Knowledgator + CV | h2 | 0.974 | 0.905 | 0.960 | 10/40 | 2.92 |
| S17ft GLiNER2-ft + PF + Knowledgator + CV | test | 1.000 | 0.986 | 0.997 | 0/20 | 0.20 |

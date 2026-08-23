# VidProM Cluster Carnator Noun-Pair Audit

`stage2_candidate_pairs` is the usable noun-pair count: CLIP Top-20 candidates with cosine similarity >= 0.65, followed by Carnator's noun prefilter and per-target Top-5 cap. These are pre-signature candidates only; no step-5 signature comparison or video/cache reuse was run.

| dataset | sampled prompts | noun-valid prompts | stage-1 recalled pairs | noun-compatible before cap | usable noun pairs | targets with a pair | same-noun pairs | one-noun-substitution pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cluster_00_2968 | 2000 | 2000 | 224 | 14 | 14 | 14 | 6 | 8 |
| cluster_02_2170 | 2000 | 1999 | 14248 | 2359 | 1563 | 483 | 78 | 1485 |
| cluster_03_3500 | 2000 | 1996 | 1544 | 80 | 80 | 36 | 14 | 66 |
| cluster_04_3227 | 2000 | 1998 | 17392 | 993 | 881 | 351 | 28 | 853 |
| cluster_07_4250 | 2000 | 1997 | 17503 | 1916 | 1272 | 434 | 66 | 1206 |
| cluster_08_4235 | 2000 | 1999 | 20854 | 202 | 199 | 111 | 16 | 183 |
| cluster_11_2414 | 2000 | 1999 | 18008 | 2225 | 1553 | 514 | 199 | 1354 |
| cluster_14_2547 | 2000 | 2000 | 17933 | 16 | 16 | 16 | 2 | 14 |
| cluster_15_7565 | 2000 | 1995 | 35442 | 1535 | 1154 | 414 | 50 | 1104 |
| cluster_16_2878 | 2000 | 1998 | 2656 | 49 | 49 | 44 | 16 | 33 |
| cluster_18_2505 | 2000 | 1999 | 667 | 54 | 54 | 47 | 14 | 40 |

The samples are in `samples/`, the exact capped candidate pairs are in `pairs/`, and every target's retrieval outcome is in `target_audits/`. Prompts with no `NOUN`/`PROPN` are in `nounless_prompts/` and are excluded from both source and target retrieval because Carnator cannot form their noun set.

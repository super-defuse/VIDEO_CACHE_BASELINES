# Source Provenance

The repository-local implementations were migrated from the following frozen
development sources. Their code is now maintained here; they are not runtime
dependencies.

| Component | Frozen source revision | Repository-local destination |
| --- | --- | --- |
| CHAI Wan/CogVideo Cache Attention | `e90eb5887f6b8666cb85c86b416f954c71ab322d` | `methods/chai/` |
| Carnator native implementations | `07e604a87243d8dfa7b053a85ffa80c6a073f6a4` plus the working-tree fixes used for acceptance | `carnator/`, `scripts/run/carnator_*.py` |
| Quality-runner protocol | `e90eb5887f6b8666cb85c86b416f954c71ab322d` | `scripts/benchmark/quality_runner.py` |

Third-party model and evaluator sources are separately listed in
`external/lock.json` and are intentionally kept out of these implementation
directories.

## VidProM dataset boundary

`artifacts/vidprom_1000cache_300input_noun_match/` is the active 300-request
result set based on `datasets/1000cache_300input_noun_match/`; its Carnator
retrieval is v2: noun filter, Early Signature, then CLIP top-1. The live
`artifacts/shared_cross_request_cache/vidprom_cluster_02_2170_first1000/`
cache uses the same dataset boundary. The cache and result artifacts are not
interchangeable with other datasets.

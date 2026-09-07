# Carnator Early Signature quality--hit-rate optimisation

## Scope

This is an offline replay of the production four-gate Early Signature rule over the completed, intentionally unbounded Carnator run. It does not regenerate videos.

- Requests: 300
- Searched hit-rate interval: 60.0%--100.0% (180--300 requests)
- Grid candidates: 5308416; interval-eligible candidates: 634706
- Full-population CLIP-T / VBench-Q: 0.261571 / 0.640110

## Distance Audit

- All 297 single-noun-substitution requests have four finite Early Signature distances.
- Same-noun-set requests: [93, 105, 108]. Their `log_area_ratio`, `shape_one_minus_dice`, and `changed_color_ciede2000` are inapplicable and were set to zero only for the offline replay, matching the runtime comparator which ignores those gates for same-noun matches.
- No required Early Signature distance or quality score is missing.

## Recommended Balanced Gate

The default recommendation maximises the equal-weight mean of the accepted-set CLIP-T and VBench-Q improvements, each normalised by its full-population standard deviation, subject to the searched hit-rate interval.

- Accepted requests / hit rate: 183 / 61.0000%
- Mean CLIP-T: 0.268080 (+0.1716 population stddev)
- Mean VBench-Q: 0.641357 (+0.0369 population stddev)
- Balanced quality gain: +0.1042 population stddev
- `max_context_color_distance`: 2.83160103778
- `max_log_area_ratio`: 2.79028828291
- `max_shape_distance`: 0.71875
- `max_changed_color_distance`: 1.95349600803

## Single-Metric Bounds

### Highest CLIP-T candidate

- Accepted requests / hit rate: 181 / 60.3333%
- Mean CLIP-T: 0.268121 (+0.1727 population stddev)
- Mean VBench-Q: 0.641187 (+0.0319 population stddev)
- Balanced quality gain: +0.1023 population stddev
- `max_context_color_distance`: 2.83160103778
- `max_log_area_ratio`: 2.79028828291
- `max_shape_distance`: 0.716535433071
- `max_changed_color_distance`: 1.95349600803

### Highest VBench-Q candidate

- Accepted requests / hit rate: 188 / 62.6667%
- Mean CLIP-T: 0.263521 (+0.0514 population stddev)
- Mean VBench-Q: 0.642650 (+0.0751 population stddev)
- Balanced quality gain: +0.0633 population stddev
- `max_context_color_distance`: 2.53093143899
- `max_log_area_ratio`: 1.34553404052
- `max_shape_distance`: 0.769230769231
- `max_changed_color_distance`: 1.97353891767

## Interpretation

`--min-hit-rate` is the operational speed constraint. Raising it reruns the same offline sweep with a stricter throughput requirement; lowering it permits more selective, potentially higher-quality reuse. The accompanying CSV files give the best balanced candidate at every reachable accepted-request count and the CLIP-T/VBench-Q/balanced choices within each coverage band.

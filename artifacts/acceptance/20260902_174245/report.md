# Acceptance experiment

- Run root: `/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260902_174245`
- Target GPUs: `0`
- Shared cache `pair-000`: `e0217002edecd1c6cc615e35b594673b1e55aa381abc22728d9c597f71b3a58f` (/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/20260902_174245/shared_cache/cache/shared_dense/manifest.json)

| Baseline | Status | GPU | Denoising (s) | CLIP-T | VBench-Q |
| --- | --- | ---: | ---: | ---: | ---: |
| dense_baseline | ok | 0 | 265.22910683997907 | 0.337158203125 | 0.6330714069844214 |
| nirvana_vid | ok | 0 | 229.35375702707097 | 0.355712890625 | 0.6362940693916224 |
| teacache | ok | 0 | 135.56259147590026 | 0.3330078125 | 0.6219078426602153 |
| chorus | ok | 0 | 195.7610147523228 | 0.35546875 | 0.6112559992677009 |
| chorus2 | ok | 0 | 196.1223408030346 | 0.33447265625 | 0.6269709150927067 |
| carnator_boundary_b5 | ok | 0 | 122.78039575996809 | 0.354736328125 | 0.6326169480667867 |

## Reference validation

| Baseline | Status | Denoising ref | Denoising actual | Delta | CLIP-T ref | CLIP-T actual | VBench-Q ref | VBench-Q actual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_baseline | compared | 274.0134920409764 | 265.22910683997907 | -8.784385200997349 | 0.337158203125 | 0.337158203125 | 0.6330714069844214 | 0.6330714069844214 |
| nirvana_vid | compared | 227.39453633700032 | 229.35375702707097 | 1.9592206900706515 | 0.355712890625 | 0.355712890625 | 0.6362940693916224 | 0.6362940693916224 |
| teacache | compared | 135.67041324696038 | 135.56259147590026 | -0.10782177106011659 | 0.3330078125 | 0.3330078125 | 0.6219078426602153 | 0.6219078426602153 |
| chorus | compared | 195.76877891202457 | 195.7610147523228 | -0.007764159701764584 | 0.35546875 | 0.35546875 | 0.6112559992677009 | 0.6112559992677009 |
| chorus2 | protocol_changed | 147.51219222904183 | 196.1223408030346 |  | 0.35595703125 | 0.33447265625 | 0.6352634557365161 | 0.6269709150927067 |
| carnator_boundary_b5 | compared | 124.08330051100347 | 122.78039575996809 | -1.3029047510353848 | 0.3547 | 0.354736328125 | 0.6326 | 0.6326169480667867 |

```json
{
  "reference_root": "/home/tangxuebin/project/VIDEO_CACHE_BASELINES/artifacts/acceptance/wan22-ti2v-5b-dog-to-cat",
  "checks": [
    {
      "baseline": "dense_baseline",
      "denoising_reference": 274.0134920409764,
      "denoising_actual": 265.22910683997907,
      "clip_t_reference": 0.337158203125,
      "clip_t_actual": 0.337158203125,
      "vbench_q_reference": 0.6330714069844214,
      "vbench_q_actual": 0.6330714069844214,
      "status": "compared",
      "denoising_delta": -8.784385200997349,
      "clip_t_delta": 0.0,
      "vbench_q_delta": 0.0
    },
    {
      "baseline": "nirvana_vid",
      "denoising_reference": 227.39453633700032,
      "denoising_actual": 229.35375702707097,
      "clip_t_reference": 0.355712890625,
      "clip_t_actual": 0.355712890625,
      "vbench_q_reference": 0.6362940693916224,
      "vbench_q_actual": 0.6362940693916224,
      "status": "compared",
      "denoising_delta": 1.9592206900706515,
      "clip_t_delta": 0.0,
      "vbench_q_delta": 0.0
    },
    {
      "baseline": "teacache",
      "denoising_reference": 135.67041324696038,
      "denoising_actual": 135.56259147590026,
      "clip_t_reference": 0.3330078125,
      "clip_t_actual": 0.3330078125,
      "vbench_q_reference": 0.6219078426602153,
      "vbench_q_actual": 0.6219078426602153,
      "status": "compared",
      "denoising_delta": -0.10782177106011659,
      "clip_t_delta": 0.0,
      "vbench_q_delta": 0.0
    },
    {
      "baseline": "chorus",
      "denoising_reference": 195.76877891202457,
      "denoising_actual": 195.7610147523228,
      "clip_t_reference": 0.35546875,
      "clip_t_actual": 0.35546875,
      "vbench_q_reference": 0.6112559992677009,
      "vbench_q_actual": 0.6112559992677009,
      "status": "compared",
      "denoising_delta": -0.007764159701764584,
      "clip_t_delta": 0.0,
      "vbench_q_delta": 0.0
    },
    {
      "baseline": "chorus2",
      "denoising_reference": 147.51219222904183,
      "denoising_actual": 196.1223408030346,
      "clip_t_reference": 0.35595703125,
      "clip_t_actual": 0.33447265625,
      "vbench_q_reference": 0.6352634557365161,
      "vbench_q_actual": 0.6269709150927067,
      "status": "protocol_changed",
      "note": "current Chorus2 is the 50-step sparse-mask protocol; the historical acceptance artifact used a 40-step source-latent restart"
    },
    {
      "baseline": "carnator_boundary_b5",
      "denoising_reference": 124.08330051100347,
      "denoising_actual": 122.78039575996809,
      "clip_t_reference": 0.3547,
      "clip_t_actual": 0.354736328125,
      "vbench_q_reference": 0.6326,
      "vbench_q_actual": 0.6326169480667867,
      "status": "compared",
      "denoising_delta": -1.3029047510353848,
      "clip_t_delta": 3.632812499998472e-05,
      "vbench_q_delta": 1.6948066786648397e-05
    }
  ]
}
```

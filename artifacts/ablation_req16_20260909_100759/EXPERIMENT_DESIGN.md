# Carnator Ablation Study - Corrected Design

## Experiment Overview

This ablation study tests the individual and combined contributions of two key optimization techniques in CARNATOR:
- **Latent Injection**: Partial computation in middle layers (10-20) using cached latents
- **Sparse Attention**: Sparse attention masks in outer layers (0-9, 21-29)

Additionally, W8A8 FFN quantization is applied to all optimized variants.

## Variants

### 1. Full Computation (Baseline)
- **Middle Layers (10-20)**: Dense computation, no latent reuse
- **Outer Layers (0-9, 21-29)**: Dense attention
- **FFN**: Full precision (fp)
- **Expected Speedup**: 1.00x (baseline)

### 2. Latent Reuse Only
- **Middle Layers (10-20)**: Latent injection enabled, partial computation in Medit region
- **Middle Attention Mask**: **DISABLED** (no attention mask in middle layers)
- **Outer Layers (0-9, 21-29)**: Dense attention (normal computation)
- **FFN**: W8A8 quantization
- **Expected Speedup**: ~1.3-1.4x

Configuration:
```
--reuse-start-step 10
--mask-layers 15
--sparse-layers 10-20
--no-middle-attention-mask  # Key: disable attention mask
--outer-attn-backend dense  # Key: outer layers use dense
--ffn-backend w8a8
```

### 3. Attention Reuse Only
- **Middle Layers (10-20)**: No latent injection, **FULL computation**
- **Outer Layers (0-9, 21-29)**: Sparse attention (sparge-sage2)
- **FFN**: W8A8 quantization
- **Expected Speedup**: ~1.2-1.3x

Configuration:
```
--no-latent-injection  # Key: disable latent injection, run full 50 steps
--outer-attn-backend sparge-sage2  # Key: sparse attention in outer layers
--outer-sparse-layers 0-9,21-29
--ffn-backend w8a8
```

### 4. Joint Latent-Attention Reuse
- **Middle Layers (10-20)**: Latent injection + attention mask
- **Outer Layers (0-9, 21-29)**: Sparse attention
- **FFN**: W8A8 quantization
- **Expected Speedup**: ~1.5-1.7x (highest)

Configuration:
```
--reuse-start-step 10
--mask-layers 15
--sparse-layers 10-20
--middle-attention-mask  # Key: attention mask in middle layers
--outer-attn-backend sparge-sage2  # Key: sparse attention in outer layers
--outer-sparse-layers 0-9,21-29
--ffn-backend w8a8
```

## Key Differences from Previous Incorrect Design

### Previous (Incorrect):
- All variants used `FFN=fp` (no quantization)
- All variants used `outer-attn-backend=dense` (no outer sparse)
- Attention-only variant incorrectly enabled middle attention mask

### Corrected:
- ✅ All non-baseline variants use `FFN=w8a8`
- ✅ Attention-only and Joint use `outer-attn-backend=sparge-sage2`
- ✅ Latent-only has `--no-middle-attention-mask`
- ✅ Attention-only has `--no-latent-injection`

## Expected Results

The joint method should achieve the highest speedup by combining both optimizations:

| Variant | Middle Optimization | Outer Optimization | FFN | Expected Speedup |
|---------|--------------------|--------------------|-----|------------------|
| Full Computation | None | None | fp | 1.00x |
| Latent Only | Latent injection | Dense | w8a8 | ~1.3-1.4x |
| Attention Only | None | Sparse | w8a8 | ~1.2-1.3x |
| Joint | Latent injection + Attention mask | Sparse | w8a8 | ~1.5-1.7x |

**Key expectation**: Joint > Latent Only ≈ Attention Only > Full Computation

## Prompts

- **Source/Cache Prompt**: "a little dog and a penguin dancing"
- **Target/Request Prompt**: "A cute little lion and a wobbly penguin appear"
- **Seed**: 20260311
- **Steps**: 50

## Source Cache

Using pre-generated source cache from:
`artifacts/ablation_req16_20260909_032326/source_gen/source_gen`

## Running Configuration

- **GPUs**: 3, 4, 5, 6 (parallel execution)
- **Conda Environment**: wan
- **Model**: Wan2.2-TI2V-5B
- **Device**: cuda:0 (mapped via CUDA_VISIBLE_DEVICES)

## Output Structure

```
artifacts/ablation_req16_20260909_100759/
├── full_computation/
│   ├── run.log
│   └── ablation/
├── latent_only/
│   ├── run.log
│   └── ablation/
├── attention_only/
│   ├── run.log
│   └── ablation/
├── joint/
│   ├── run.log
│   └── ablation/
└── summary.json
```

Results will be written to:
```
results/ti2v_5b/ablation/
```

## Verification Checklist

After completion, verify:
- [ ] Full Computation: No optimization applied, FFN=fp
- [ ] Latent Only: Middle latent injection, no middle attention mask, outer dense, FFN=w8a8
- [ ] Attention Only: No latent injection (50 steps), outer sparse, FFN=w8a8
- [ ] Joint: Middle latent+attention, outer sparse, FFN=w8a8
- [ ] Joint speedup > Latent Only speedup
- [ ] Joint speedup > Attention Only speedup

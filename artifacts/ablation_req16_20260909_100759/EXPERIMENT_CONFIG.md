# Carnator Ablation Study - Corrected Configuration

**实验时间**: 2026-09-09 10:07:59  
**实验目录**: `artifacts/ablation_req16_20260909_100759/`  
**结果目录**: `results/ti2v_5b/ablation/`

## 实验目标

完成 Table 3: Ablation of latent and attention reuse in CARNATOR 的数据采集，正确测试各个优化组件的独立和联合效果。

## 变体配置

### 1. Full Computation (基线)
- **GPU**: 3
- **配置**: 无任何优化
- **参数**:
  - `--target-dense-only`
  - `--ffn-backend fp` (全精度FFN)
- **预期**: 最慢，1.00x基线

### 2. Latent Reuse Only
- **GPU**: 4
- **配置**: 仅中间层(10-20)局部计算，头尾层正常
- **参数**:
  - `--reuse-start-step 10`
  - `--sparse-layers 10-20`
  - `--no-middle-attention-mask` (中间层不用attention mask)
  - `--outer-attn-backend dense` (头尾层密集计算)
  - `--ffn-backend w8a8` (FFN量化)
- **预期**: ~1.2-1.3x加速

### 3. Attention Reuse Only
- **GPU**: 5
- **配置**: 头尾层(0-9,21-29)稀疏attention，中间层正常
- **参数**:
  - `--no-latent-injection` (禁用latent注入)
  - `--outer-attn-backend sparge-sage2` (头尾层稀疏)
  - `--outer-sparse-layers 0-9,21-29`
  - `--ffn-backend w8a8` (FFN量化)
- **预期**: ~1.2-1.3x加速

### 4. Joint Latent-Attention Reuse
- **GPU**: 6
- **配置**: 中间层局部计算 + 头尾层稀疏attention
- **参数**:
  - `--reuse-start-step 10`
  - `--sparse-layers 10-20`
  - `--middle-attention-mask` (中间层用attention mask)
  - `--outer-attn-backend sparge-sage2` (头尾层稀疏)
  - `--outer-sparse-layers 0-9,21-29`
  - `--ffn-backend w8a8` (FFN量化)
- **预期**: ~1.4-1.5x加速（最高）

## 与之前实验的主要差异

### 之前的错误配置：
1. ❌ 所有变体FFN都是 `fp`
2. ❌ 所有变体 `outer_sparse` 都是 `false`
3. ❌ Attention-only 错误地启用了 `middle_attention_mask`

### 现在的正确配置：
1. ✅ 除Full Computation外，其他变体FFN都用 `w8a8`
2. ✅ Attention-only和Joint启用了头尾层稀疏 `sparge-sage2`
3. ✅ Latent-only禁用了中间层attention mask
4. ✅ Attention-only禁用了latent injection

## 实验数据

**Request Prompt**: "A cute little lion and a wobbly penguin appear"  
**Cache Prompt**: "a little dog and a penguin dancing"  
**Source Cache**: `artifacts/ablation_req16_20260909_032326/source_gen/source_gen`  
**Seed**: 20260311  
**Steps**: 50

## 运行命令

```bash
python3 scripts/ablation_index43_corrected.py --gpus 3 4 5 6
```

## 监控进度

```bash
bash scripts/check_ablation_progress.sh
```

## 预期结果

正确配置后，应该观察到：
- Full Computation: 最慢（基线）
- Latent Only ≈ Attention Only: 中等加速
- Joint: 最快（两种优化叠加效果）

即: **Joint > (Latent Only, Attention Only) > Full Computation**

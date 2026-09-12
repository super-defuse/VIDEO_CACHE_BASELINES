from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from adapters.ltx_sparse import (
    LtxBlockMaskCache,
    LtxBlockMaskConfig,
    LtxMaskCaptureProcessor,
    LtxSparseExecutor,
    LtxSparseProcessor,
    branches_for,
)

# A 2x2 block grid over six tokens in three heads: small enough to run on CPU,
# large enough that the mask is a real routing problem.
CONFIG = LtxBlockMaskConfig(q_block_size=2, k_block_size=2, top_p=0.6, min_top_k=0.0)
# [tokens, heads, head_dim] -- block_attention_mass wants the NHD layout, which
# is what the capture processor hands it after transposing one branch.
QKV_SHAPE = (6, 3, 4)
MASK_SHAPE = (3, 3, 3)
LAYERS = [0, 1, 2, 3]
START_STEP = 3


def _qk(seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(QKV_SHAPE, generator=generator),
        torch.randn(QKV_SHAPE, generator=generator),
    )


def _source_cache(tmp_path, config: LtxBlockMaskConfig = CONFIG, layers: list[int] | None = None):
    return LtxBlockMaskCache(
        tmp_path, config, layers=LAYERS if layers is None else layers, start_step=START_STEP, create=True
    )


def _captured(tmp_path, *, steps=(START_STEP,), branches=("cond",), layers=(0,)):
    cache = _source_cache(tmp_path, layers=list(LAYERS))
    query, key = _qk()
    for step in steps:
        for branch in branches:
            for layer in layers:
                cache.capture(step, branch, layer, query, key)
    cache.finalize()
    return cache


class _RecordingKernel:
    """Stands in for ``block_sparse_sage2_attn_cuda``: records, echoes q back."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, query, key, value, *, mask_id, smooth_k, pvthreshd, tensor_layout):
        self.calls.append(
            {
                "query": query, "key": key, "value": value, "mask": mask_id,
                "smooth_k": smooth_k, "pvthreshd": pvthreshd, "tensor_layout": tensor_layout,
            }
        )
        return query


class _StubExecutor:
    """Executor stand-in for the processor tests: records the calls it receives."""

    def __init__(self, cache) -> None:
        self.cache = cache
        self.calls: list[tuple[int, str, int]] = []

    def run(self, query, key, value, *, step, branch, layer):
        self.calls.append((step, branch, layer))
        return value


# --------------------------------------------------------------------- config


def test_block_mask_config_rejects_invalid_geometry_and_thresholds():
    with pytest.raises(ValueError, match="block sizes must be positive"):
        LtxBlockMaskConfig(q_block_size=0)
    with pytest.raises(ValueError, match="outside their valid range"):
        LtxBlockMaskConfig(top_p=0.0)
    with pytest.raises(ValueError, match="outside their valid range"):
        LtxBlockMaskConfig(min_top_k=1.5)
    # The Wan2.1 geometry the LTX replay is aligned to.
    assert (LtxBlockMaskConfig().q_block_size, LtxBlockMaskConfig().k_block_size) == (64, 128)


def test_branches_for_only_guesses_the_batch_size_it_can_see():
    assert branches_for(1) == ("cond",)
    assert branches_for(2) == ("uncond", "cond")
    assert branches_for(3) == ("uncond", "cond", "cond_perturbed")
    with pytest.raises(ValueError, match="got batch 4"):
        branches_for(4)


# ----------------------------------------------------------------- the cache


def test_captured_mask_round_trips_through_the_packed_file(tmp_path):
    cache = _captured(tmp_path)
    entry = cache.entries[(START_STEP, "cond", 0)]
    assert entry["mask_shape"] == list(MASK_SHAPE)
    assert entry["qkv_shape"] == list(QKV_SHAPE)

    # One bit per element, so the file is a fraction of the boolean mask.
    stored = (tmp_path / entry["path"]).stat().st_size
    assert stored == -(-(MASK_SHAPE[0] * MASK_SHAPE[1] * MASK_SHAPE[2]) // 8)

    replay = LtxBlockMaskCache(tmp_path, CONFIG)
    mask, replayed_entry = replay.get(START_STEP, "cond", 0, torch.zeros(3, 6, 4))
    assert mask.dtype == torch.bool
    assert tuple(mask.shape) == MASK_SHAPE
    assert replayed_entry["sha256"] == entry["sha256"]
    # The decoded mask is the one whose statistics the entry recorded.
    assert float(mask.float().mean().item()) == pytest.approx(entry["density"])


def test_replaying_under_a_different_block_geometry_is_rejected(tmp_path):
    _captured(tmp_path)
    # The 128/128 geometry the LTX baselines used before they were aligned to
    # Wan2.1: the same directory must not silently serve it.
    wrong = LtxBlockMaskConfig(q_block_size=128, k_block_size=128, top_p=0.6, min_top_k=0.0)
    with pytest.raises(ValueError, match="does not match the target"):
        LtxBlockMaskCache(tmp_path, wrong)


def test_corrupted_mask_fails_its_checksum(tmp_path):
    cache = _captured(tmp_path)
    entry = cache.entries[(START_STEP, "cond", 0)]
    (tmp_path / entry["path"]).write_bytes(b"\xff\xff\xff\xff")

    replay = LtxBlockMaskCache(tmp_path, CONFIG)
    with pytest.raises(ValueError, match="checksum mismatch"):
        replay.get(START_STEP, "cond", 0, torch.zeros(3, 6, 4))


def test_get_rejects_a_query_whose_shape_does_not_match_the_source(tmp_path):
    cache = _captured(tmp_path)
    with pytest.raises(ValueError, match="does not match source"):
        cache.get(START_STEP, "cond", 0, torch.zeros(3, 7, 4))
    with pytest.raises(KeyError, match="missing LTX source mask"):
        cache.get(START_STEP, "cond", 1, torch.zeros(3, 6, 4))


def test_prepare_decodes_everything_once_and_get_then_does_no_io(tmp_path):
    _captured(tmp_path, branches=("uncond", "cond"))
    replay = LtxBlockMaskCache(tmp_path, CONFIG)
    query = torch.zeros(3, 6, 4)

    first = replay.prepare()
    assert first["entries"] == 2
    assert first["decoded_entries"] == 2
    assert first["file_bytes"] > 0
    # Preparing again decodes nothing: this is the memoisation the timed loop
    # depends on.
    second = replay.prepare()
    assert second["decoded_entries"] == 0

    # Nothing on the hot path may touch the file: prepare() placed the mask.
    for entry in replay.entries.values():
        (tmp_path / entry["path"]).unlink()
    mask, _entry = replay.get(START_STEP, "cond", 0, query)
    assert mask.dtype == torch.bool


def test_validate_coverage_names_the_first_missing_entry(tmp_path):
    cache = _captured(tmp_path, branches=("cond",), layers=(0,))
    cache.validate_coverage(steps=range(START_STEP, START_STEP + 1), branches=("cond",), layers=[0])
    with pytest.raises(ValueError, match=r"first missing entry is \(3, 'cond', 1\)"):
        cache.validate_coverage(steps=range(START_STEP, START_STEP + 1), branches=("cond",), layers=[0, 1])


# ---------------------------------------------------------------- the SpargeAttn guards


def test_executor_rejects_an_execution_dtype_the_kernel_cannot_take(tmp_path):
    cache = _source_cache(tmp_path)
    with pytest.raises(ValueError, match="must be FP16 or BF16"):
        LtxSparseExecutor(cache, execution_dtype=torch.float32)
    assert LtxSparseExecutor(cache).execution_dtype is torch.float16


def test_sparge_kernel_requires_cuda_and_sm90(tmp_path, monkeypatch):
    executor = LtxSparseExecutor(_source_cache(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        executor._sparge_kernel()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    with pytest.raises(RuntimeError, match=r"requires sm90, got capability=\(8, 0\)"):
        executor._sparge_kernel()


def test_sparge_kernel_requires_the_wan_block_geometry(tmp_path, monkeypatch):
    # SpargeAttn's kernel is compiled for exactly Q=64/K=128. The LTX baselines
    # ran 128/128 while they were on flex attention, and that geometry must not
    # be able to reach the kernel.
    wrong = LtxBlockMaskConfig(q_block_size=128, k_block_size=128)
    executor = LtxSparseExecutor(_source_cache(tmp_path, config=wrong))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(ValueError, match="requires Q=64 and K=128"):
        executor._sparge_kernel()


def test_sparge_kernel_resolves_the_packed_attn_entry_point(tmp_path, monkeypatch):
    executor = LtxSparseExecutor(_source_cache(tmp_path, config=LtxBlockMaskConfig()))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))

    sentinel = object()
    module = types.ModuleType("spas_sage_attn")
    module.block_sparse_sage2_attn_cuda = sentinel
    monkeypatch.setitem(sys.modules, "spas_sage_attn", module)
    assert executor._sparge_kernel() is sentinel

    monkeypatch.setitem(sys.modules, "spas_sage_attn", None)
    with pytest.raises(RuntimeError, match="SpargeAttn is unavailable"):
        executor._sparge_kernel()


def test_prepare_reports_the_backend_the_masks_and_the_missing_entry_point(tmp_path, monkeypatch):
    cache = _captured(tmp_path, branches=("uncond", "cond"))
    executor = LtxSparseExecutor(cache)
    monkeypatch.setattr(LtxSparseExecutor, "_sparge_kernel", lambda self, device=None: _RecordingKernel())

    prepared = executor.prepare()
    assert prepared["backend"] == "sparge-sage2"
    assert prepared["masks"]["decoded_entries"] == 2
    assert prepared["device"] is None


# ------------------------------------------------------------- the timed path


def test_run_hands_the_kernel_hnd_batches_and_returns_the_input_dtype(tmp_path):
    _captured(tmp_path)
    cache = LtxBlockMaskCache(tmp_path, CONFIG)
    executor = LtxSparseExecutor(cache, layer_refresh_size=0)
    kernel = _RecordingKernel()
    executor._sparge_kernel = lambda device=None: kernel  # type: ignore[method-assign]

    # The processor indexes one guidance branch out of ltx_qkv's [B, H, S, D],
    # so run() receives [heads, sequence, head_dim] and must add only the batch.
    query = torch.randn(3, 6, 4, dtype=torch.bfloat16)
    output = executor.run(query, query, query, step=START_STEP, branch="cond", layer=0)

    assert output.dtype is torch.bfloat16
    assert tuple(output.shape) == (3, 6, 4)
    assert len(kernel.calls) == 1
    call = kernel.calls[0]
    assert call["tensor_layout"] == "HND"
    assert tuple(call["query"].shape) == (1, 3, 6, 4)
    # INT8 QK and FP8 PV: the kernel only ever sees the reduced dtype.
    assert call["query"].dtype is torch.float16
    assert call["mask"].dtype is torch.bool
    assert tuple(call["mask"].shape) == (1, *MASK_SHAPE)
    assert (call["smooth_k"], call["pvthreshd"]) == (True, 50.0)

    assert executor.runs == 1
    assert 0.0 <= executor.summary()["average_runtime_density"] <= 1.0


def test_refresh_gives_the_group_last_layer_every_block_missed_earlier(tmp_path):
    # Wan's ``_apply_layer_refresh`` semantics: the final layer of a group
    # receives all block positions not selected by the preceding layers, so the
    # group as a whole can never drop a block an earlier member routed.
    cache = _source_cache(tmp_path, layers=LAYERS)
    executor = LtxSparseExecutor(cache, layer_refresh_size=2)
    step, branch = START_STEP, "cond"

    first = torch.zeros(3, 2, 2, dtype=torch.bool)
    first[0, 0, 0] = True
    assert torch.equal(executor._refresh(first, step=step, branch=branch, layer=0), first)
    assert executor.refresh_events == 0

    second = torch.zeros_like(first)
    refreshed = executor._refresh(second, step=step, branch=branch, layer=1)
    assert torch.equal(refreshed, ~first)
    assert (first | refreshed).all()
    assert refreshed.sum().item() == first.numel() - 1
    assert executor.refresh_events == 1

    executor._refresh(torch.zeros_like(first), step=step, branch=branch, layer=2)
    executor._refresh(torch.zeros_like(first), step=step, branch=branch, layer=3)
    assert executor.refresh_events == 2


def test_refresh_is_a_no_op_when_every_layer_keeps_its_own_mask(tmp_path):
    executor = LtxSparseExecutor(_source_cache(tmp_path, layers=LAYERS), layer_refresh_size=0)
    mask = torch.zeros(3, 2, 2, dtype=torch.bool)
    assert torch.equal(executor._refresh(mask, step=1, branch="cond", layer=3), mask)
    assert executor.refresh_events == 0


def test_summary_declares_the_sparge_backend_and_its_quantisation(tmp_path):
    cache = _captured(tmp_path)
    executor = LtxSparseExecutor(cache, layer_refresh_size=8)
    summary = executor.summary()

    assert summary["backend"] == "sparge-sage2"
    assert summary["kernel"] == "block-sparse-sage2-attn-cuda"
    assert summary["quantization"] == "QK INT8 and PV FP8"
    assert summary["runs"] == 0
    assert summary["average_runtime_density"] is None
    assert summary["mask_cache"]["format"] == LtxBlockMaskCache.FORMAT
    # No second backend may reappear: flex attention could not run this geometry.
    assert "flex" not in str(summary).lower()


# ---------------------------------------------------------------- processors


def test_processor_stays_dense_before_the_sparse_window(tmp_path, monkeypatch):
    cache = _captured(tmp_path)
    delegate = _StubExecutor(cache)
    from adapters import ltx_sparse

    dense_calls: list[tuple] = []
    monkeypatch.setattr(ltx_sparse, "dense_self_attention", lambda *a, **k: dense_calls.append(a) or "dense")

    processor = LtxSparseProcessor(delegate, state=SimpleNamespace(computing_step=START_STEP - 1), layer=0)
    assert processor(attention=None, hidden_states=None) == "dense"
    assert len(dense_calls) == 1
    assert delegate.calls == []


def test_processor_runs_one_sparse_call_per_guidance_branch(tmp_path, monkeypatch):
    cache = _captured(tmp_path, branches=("uncond", "cond"))
    delegate = _StubExecutor(LtxBlockMaskCache(tmp_path, CONFIG))
    from adapters import ltx_sparse

    query = torch.randn(2, 3, 6, 4)
    monkeypatch.setattr(ltx_sparse, "ltx_qkv", lambda attention, hidden, rotary=None: (query, query, query))
    monkeypatch.setattr(ltx_sparse, "ltx_project", lambda attention, flat: flat)

    processor = LtxSparseProcessor(delegate, state=SimpleNamespace(computing_step=START_STEP), layer=0)
    processor(attention=None, hidden_states=None)

    # branches_for(2) -> ("uncond", "cond"): both rows go through their own mask.
    assert delegate.calls == [(START_STEP, "uncond", 0), (START_STEP, "cond", 0)]


def test_capture_processor_records_one_entry_per_branch(tmp_path, monkeypatch):
    cache = _source_cache(tmp_path, layers=LAYERS)
    from adapters import ltx_sparse

    query = torch.randn(2, 3, 6, 4)
    monkeypatch.setattr(ltx_sparse, "ltx_qkv", lambda attention, hidden, rotary=None: (query, query, query))
    monkeypatch.setattr(ltx_sparse, "ltx_project", lambda attention, flat: flat)

    processor = LtxMaskCaptureProcessor(cache, state=SimpleNamespace(computing_step=START_STEP), layer=0)
    processor(attention=None, hidden_states=None)

    assert set(cache.entries) == {(START_STEP, "uncond", 0), (START_STEP, "cond", 0)}
    for entry in cache.entries.values():
        assert entry["mask_shape"] == list(MASK_SHAPE)


def test_capture_processor_runs_dense_outside_the_sparse_window(tmp_path, monkeypatch):
    cache = _source_cache(tmp_path, layers=LAYERS)
    from adapters import ltx_sparse

    monkeypatch.setattr(ltx_sparse, "dense_self_attention", lambda *a, **k: "dense")
    processor = LtxMaskCaptureProcessor(cache, state=SimpleNamespace(computing_step=START_STEP - 1), layer=0)

    assert processor(attention=None, hidden_states=None) == "dense"
    assert cache.entries == {}

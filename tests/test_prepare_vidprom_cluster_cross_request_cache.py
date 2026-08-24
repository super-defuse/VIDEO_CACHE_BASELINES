from __future__ import annotations

import json

import pytest
import torch

from adapters.wan22 import Wan22Adapter
from carnator.outer_sparse import BlockMaskConfig, SourceBlockMaskCache
from scripts.benchmark import prepare_vidprom_cluster_cross_request_cache as prepare


def test_read_prompt_rows_preserves_csv_order_and_handles_bom(tmp_path):
    source = tmp_path / "requests.csv"
    source.write_text("\ufeffuuid,prompt\nfirst,first prompt\nsecond,second prompt\n", encoding="utf-8")

    rows = prepare.read_prompt_rows(source, 1)

    assert rows == [prepare.PromptRow(request_index=0, uuid="first", prompt="first prompt")]


def test_read_prompt_rows_rejects_duplicate_uuid(tmp_path):
    source = tmp_path / "requests.csv"
    source.write_text("uuid,prompt\nfirst,one\nfirst,two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate UUID"):
        prepare.read_prompt_rows(source, 2)


def test_canonical_bitpacked_mask_round_trip_and_backend_views():
    mask = torch.rand((3, 5, 7)) > 0.4

    payload, elements = prepare.pack_bool_mask(mask)
    restored = prepare.unpack_bool_mask(payload, list(mask.shape))

    assert elements == mask.numel()
    assert torch.equal(restored, mask)
    assert prepare.flashinfer_mask_view(restored).data_ptr() == restored.data_ptr()
    sparge = prepare.spargeattn_mask_view(restored)
    assert sparge.shape == (1, *mask.shape)
    assert torch.equal(sparge[0], prepare.flashinfer_mask_view(restored))


def test_load_canonical_mask_rejects_bad_checksum(tmp_path):
    mask = torch.tensor([[[True, False, True]]])
    payload, elements = prepare.pack_bool_mask(mask)
    path = tmp_path / "mask.bin"
    path.write_bytes(payload)
    entry = {
        "path": "mask.bin",
        "mask_shape": list(mask.shape),
        "packed_elements": elements,
        "sha256": "not-the-checksum",
    }

    with pytest.raises(ValueError, match="checksum"):
        prepare.load_canonical_mask(tmp_path, entry)


def test_attention_capture_writes_one_bitpacked_payload_for_both_backends(tmp_path):
    capture = prepare.SharedAttentionCapture(
        tmp_path / "attention",
        source_uuid="source",
        config={"q_block_size": 2, "k_block_size": 2, "top_p": 0.9, "min_top_k": 0.0},
        cfg_branch_count=2,
        max_pending_writes=1,
    )
    generator = torch.Generator().manual_seed(7)
    q = torch.randn((4, 2, 8), generator=generator)
    k = torch.randn((4, 2, 8), generator=generator)

    capture.capture(11, "cond", 0, q, k)
    metadata = capture.close()

    assert metadata["entries"][0]["from_step"] == 10
    mask = prepare.load_canonical_mask(tmp_path / "attention", metadata["entries"][0])
    assert torch.equal(prepare.flashinfer_mask_view(mask), prepare.spargeattn_mask_view(mask)[0])
    assert len(list((tmp_path / "attention" / "maskbits").rglob("*.bin"))) == 1


def test_chorus2_reads_the_single_carnator_raw_mask_cache(tmp_path):
    raw_root = tmp_path / "carnator" / "source_dense" / "outer_block_masks"
    capture = prepare.SharedAttentionCapture(
        raw_root,
        source_uuid="source",
        config={"q_block_size": 2, "k_block_size": 2, "top_p": 0.9, "min_top_k": 0.0},
        cfg_branch_count=1,
        max_pending_writes=1,
    )
    generator = torch.Generator().manual_seed(11)
    q = torch.randn((4, 2, 8), generator=generator)
    k = torch.randn((4, 2, 8), generator=generator)
    capture.capture(11, "cond", 0, q, k)
    capture.capture(11, "cond", 10, q, k)
    capture.close()

    raw_metadata = json.loads((raw_root / "metadata.json").read_text(encoding="utf-8"))
    by_layer = {entry["layer"]: entry for entry in raw_metadata["entries"]}
    assert raw_metadata["format"] == prepare.ATTENTION_FORMAT
    assert by_layer[10]["path"] == "maskbits/from_step_010/cond/layer_10.bin"
    assert not (tmp_path / "chorus2").exists()

    cache = SourceBlockMaskCache(raw_root, BlockMaskConfig(q_block_size=2, k_block_size=2, top_p=0.9, min_top_k=0.0))
    loaded, entry, _elapsed = cache.get(11, "cond", 10, q)
    assert entry["layer"] == 10
    assert loaded.dtype == torch.bool
    assert tuple(loaded.shape) == (2, 2, 2)


def test_chorus_reads_native_carnator_trajectory_without_a_sidecar(tmp_path):
    source_cache = tmp_path / "carnator" / "cache" / "source_dense"
    source_cache.mkdir(parents=True)
    torch.save(torch.zeros((2, 3)), source_cache / "latents_initial.pt")
    trajectory = []
    for step in range(2):
        path = source_cache / f"latents_step_{step:03d}.pt"
        torch.save(torch.full((2, 3), step, dtype=torch.float32), path)
        trajectory.append({"kind": "post_scheduler", "step_index": step, "path": str(path)})
    (source_cache / "manifest.json").write_text(json.dumps({
        "steps": 2,
        "seed": 7,
        "latent_shape": [2, 3],
        "trajectory": trajectory,
    }), encoding="utf-8")

    payload = Wan22Adapter({}).load_latent(source_cache)

    assert payload["format"] == "chorus-trajectory-v2"
    assert payload["source_format"] == "native-carnator-source-cache-v1"
    assert payload["trajectory_paths"] == [
        source_cache / "latents_initial.pt",
        source_cache / "latents_step_000.pt",
        source_cache / "latents_step_001.pt",
    ]


def test_estimate_includes_one_canonical_cache_without_baseline_copies(tmp_path):
    args = prepare.build_parser().parse_args([])
    estimate = prepare.estimate_space(args, selected_count=1_000, available_bytes=0)

    assert estimate["assumptions"]["cached_mask_steps"] == 40
    assert estimate["assumptions"]["chorus2_reuses_carnator_raw_masks"] is True
    assert estimate["per_entry"]["carnator_raw_block_masks"] > 0
    assert "chorus2_middle_raw_mask_copy" not in estimate["per_entry"]
    assert "nirvana_vid_step_10_payload" not in estimate["per_entry"]
    assert estimate["assumptions"]["trajectory_latents"] == 51
    assert estimate["fits_without_headroom"] is False


def test_dry_run_writes_frozen_run_manifest_without_creating_entries(tmp_path):
    source = tmp_path / "requests.csv"
    source.write_text("uuid,prompt\nfirst,first prompt\nsecond,second prompt\n", encoding="utf-8")
    output = tmp_path / "output"

    assert prepare.main([
        "--input", str(source), "--output", str(output), "--limit", "2", "--gpus", "0", "--dry-run",
    ]) == 0

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "dry_run"
    assert manifest["selected_requests"] == 2
    assert not (output / "entries").exists()


def test_full_run_requires_explicit_confirmation_after_pilot(tmp_path):
    source = tmp_path / "requests.csv"
    source.write_text("uuid,prompt\nfirst,first prompt\nsecond,second prompt\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="confirm-full-run"):
        prepare.main(["--input", str(source), "--output", str(tmp_path / "output"), "--gpus", "0"])

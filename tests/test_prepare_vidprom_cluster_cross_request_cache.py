from __future__ import annotations

import json

import pytest
import torch

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
    mask = prepare.load_canonical_mask(tmp_path, metadata["entries"][0])
    assert torch.equal(prepare.flashinfer_mask_view(mask), prepare.spargeattn_mask_view(mask)[0])
    assert len(list((tmp_path / "attention" / "maskbits").rglob("*.bin"))) == 1


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

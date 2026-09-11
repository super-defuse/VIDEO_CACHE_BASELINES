from __future__ import annotations

import json
import math

import pytest
import torch

from carnator.outer_sparse import (
    BlockMaskCacheError,
    BlockMaskConfig,
    SourceBlockMaskCache,
    pack_bool_mask,
    unpack_bool_mask,
)

CONFIG = BlockMaskConfig(q_block_size=2, k_block_size=2, top_p=0.6, min_top_k=0.0)
# 27 elements: three heads x three q_blocks x three k_blocks, which is not a
# multiple of 8 and so exercises the padding byte.
QKV_SHAPE = (6, 3, 4)


def _qk(seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(QKV_SHAPE, generator=generator)
    k = torch.randn(QKV_SHAPE, generator=generator)
    return q, k


def _entries(root) -> dict:
    return json.loads((root / "metadata.json").read_text(encoding="utf-8"))


def test_pack_bool_mask_round_trips_and_shrinks_eightfold():
    mask = torch.tensor([[[True, False, True], [False, False, True], [True, True, False]]])
    payload, element_count = pack_bool_mask(mask)

    assert element_count == mask.numel() == 9
    assert len(payload) == math.ceil(9 / 8) == 2
    assert torch.equal(unpack_bool_mask(payload, list(mask.shape)), mask)


def test_pack_bool_mask_pads_a_partial_final_byte():
    mask = torch.zeros((1, 3, 3), dtype=torch.bool)
    mask[0, 2, 2] = True  # final element, flat index 8

    payload, element_count = pack_bool_mask(mask)

    assert element_count == 9
    assert len(payload) == 2
    # LSB-first: flat index 8 lands at bit 0 of byte 1, leaving 7 padding bits.
    assert payload == bytes([0b00000000, 0b00000001])
    assert torch.equal(unpack_bool_mask(payload, list(mask.shape)), mask)


def test_packed_and_plain_caches_write_identical_masks_and_metadata(tmp_path):
    q, k = _qk()
    plain = SourceBlockMaskCache(tmp_path / "plain", CONFIG, source_run_id="s", cfg_branch_count=1, create=True)
    packed = SourceBlockMaskCache(
        tmp_path / "packed", CONFIG, source_run_id="s", cfg_branch_count=1, create=True,
        pack_format=SourceBlockMaskCache.PACKED_FORMAT,
    )

    plain.capture(11, "cond", 0, q, k)
    packed.capture(11, "cond", 0, q, k)

    assert _entries(tmp_path / "plain")["format"] == SourceBlockMaskCache.FORMAT
    packed_meta = _entries(tmp_path / "packed")
    assert packed_meta["format"] == SourceBlockMaskCache.PACKED_FORMAT

    plain_entry = plain.entries[(11, "cond", 0)]
    packed_entry = packed.entries[(11, "cond", 0)]
    # Every reported statistic must survive the storage change untouched.
    for field in ("density", "coverage", "head_density_min", "head_density_max", "head_density_cv"):
        assert plain_entry[field] == packed_entry[field], field

    elements = math.prod(packed_entry["mask_shape"])
    assert packed_entry["packed_elements"] == elements
    assert packed_entry["mask_shape"] == [3, 3, 3]
    packed_bytes = (tmp_path / "packed" / packed_entry["path"]).stat().st_size
    plain_bytes = (tmp_path / "plain" / plain_entry["path"]).stat().st_size
    # The packed file is exactly one bit per element; the plain file is a
    # torch.save container whose payload is one byte per element plus framing.
    assert packed_bytes == math.ceil(elements / 8)
    assert plain_bytes > packed_bytes

    # Reopen the packed cache the way a target stage does and confirm the mask
    # decodes bit-for-bit to what the plain cache stored.  Entries only reach
    # metadata.json on flush(), which the capture context manager calls at exit.
    packed.flush()
    replay = SourceBlockMaskCache(tmp_path / "packed", CONFIG)
    replay_mask, _, _ = replay.get(11, "cond", 0, q)
    plain_mask, _, _ = plain.get(11, "cond", 0, q)
    assert replay_mask.dtype == torch.bool
    assert torch.equal(replay_mask, plain_mask)


def test_packed_capture_verifies_its_own_checksum(tmp_path):
    q, k = _qk()
    cache = SourceBlockMaskCache(
        tmp_path, CONFIG, source_run_id="s", cfg_branch_count=1, create=True,
        pack_format=SourceBlockMaskCache.PACKED_FORMAT,
    )
    cache.capture(11, "cond", 0, q, k)
    cache.flush()
    entry = cache.entries[(11, "cond", 0)]
    assert entry["sha256"]

    (tmp_path / entry["path"]).write_bytes(b"\xff\xff\xff\xff")
    replay = SourceBlockMaskCache(tmp_path, CONFIG)
    with pytest.raises(BlockMaskCacheError, match="checksum mismatch"):
        replay.get(11, "cond", 0, q)


def test_metadata_is_written_once_per_flush_not_per_capture(tmp_path):
    q, k = _qk()
    cache = SourceBlockMaskCache(
        tmp_path, CONFIG, source_run_id="s", cfg_branch_count=1, create=True,
        pack_format=SourceBlockMaskCache.PACKED_FORMAT,
    )
    assert _entries(tmp_path)["entries"] == []

    cache.capture(11, "cond", 0, q, k)
    cache.capture(11, "cond", 1, q, k)
    # capture() must not re-serialize the growing metadata on every call.
    assert _entries(tmp_path)["entries"] == []

    cache.flush()
    assert [entry["layer"] for entry in _entries(tmp_path)["entries"]] == [0, 1]


def test_flush_is_a_no_op_for_a_cache_opened_for_replay(tmp_path):
    q, k = _qk()
    cache = SourceBlockMaskCache(
        tmp_path, CONFIG, source_run_id="s", cfg_branch_count=1, create=True,
        pack_format=SourceBlockMaskCache.PACKED_FORMAT,
    )
    cache.capture(11, "cond", 0, q, k)
    cache.flush()
    before = (tmp_path / "metadata.json").read_bytes()

    replay = SourceBlockMaskCache(tmp_path, CONFIG)
    replay.flush()
    replay.summary()

    assert (tmp_path / "metadata.json").read_bytes() == before


def test_pack_format_rejects_unknown_formats(tmp_path):
    with pytest.raises(ValueError, match="unsupported packed source block-mask format"):
        SourceBlockMaskCache(
            tmp_path, CONFIG, source_run_id="s", cfg_branch_count=1, create=True, pack_format="nonsense-v1"
        )


def test_cache_without_pack_format_keeps_the_unpacked_layout(tmp_path):
    q, k = _qk()
    cache = SourceBlockMaskCache(tmp_path, CONFIG, source_run_id="s", cfg_branch_count=1, create=True)
    cache.capture(11, "cond", 0, q, k)

    assert _entries(tmp_path)["format"] == SourceBlockMaskCache.FORMAT
    assert "packed_elements" not in cache.entries[(11, "cond", 0)]

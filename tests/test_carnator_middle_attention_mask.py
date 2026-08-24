from __future__ import annotations

import torch

from carnator.adapters import wan_ti2v_5b as native
from carnator.outer_sparse import BlockMaskConfig, SourceBlockMaskCache, select_region_tokens_from_block_mask


def test_middle_attention_mask_keeps_only_medit_queries_reaching_msee_context():
    # Five tokens with Q/K blocks of two tokens.  The Medit token in query
    # block 1 can only see key block 1, which contains no Msee token.
    block_mask = torch.tensor(
        [[[True, False, False], [False, True, False], [False, False, True]]],
        dtype=torch.bool,
    )
    medit = torch.tensor([True, False, True, False, True])
    msee = torch.tensor([True, True, False, False, True])

    query_mask, context_mask, stats = select_region_tokens_from_block_mask(
        block_mask,
        medit,
        msee,
        q_block_size=2,
        k_block_size=2,
    )

    assert query_mask.tolist() == [True, False, False, False, True]
    assert context_mask.tolist() == [True, True, False, False, True]
    assert stats == {
        "input_medit_count": 3,
        "input_msee_count": 3,
        "query_count": 2,
        "context_count": 3,
        "active_query_blocks": 2,
        "active_key_blocks": 2,
        "query_ratio": 0.4,
        "context_ratio": 0.6,
    }


def test_middle_attention_mask_is_enabled_by_default_and_captures_all_layers():
    args = native.build_parser().parse_args([])

    assert args.middle_attention_mask is True
    assert native.block_mask_capture_layers(args) == list(range(30))

    legacy = native.build_parser().parse_args(["--no-middle-attention-mask"])
    assert native.block_mask_capture_layers(legacy) == list(range(10)) + list(range(21, 30))


def test_shared_dense_export_references_one_canonical_source_cache(tmp_path):
    args = native.build_parser().parse_args([
        "--prepare-shared-dense-cache",
        "--steps", "11",
        "--reuse-start-step", "10",
        "--guidance-scale", "1",
        "--artifact-root", str(tmp_path / "artifacts"),
        "--results-root", str(tmp_path / "results"),
        "--run-id", "shared",
    ])
    source_root = tmp_path / "source_dense"
    source_root.mkdir()
    torch.save(torch.zeros((2, 3)), source_root / "latents_initial.pt")
    trajectory = {}
    for step in range(args.steps):
        path = source_root / f"latents_step_{step:03d}.pt"
        torch.save(torch.full((2, 3), step, dtype=torch.float32), path)
        trajectory[step] = path
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"video")

    config = BlockMaskConfig(q_block_size=64, k_block_size=128, top_p=0.95, min_top_k=0.1)
    source_masks = SourceBlockMaskCache(
        source_root / "outer_block_masks", config, source_run_id="shared", cfg_branch_count=1, create=True
    )
    q = torch.ones((5, 1, 2), dtype=torch.float16)
    k = torch.ones_like(q)
    for layer in range(30):
        source_masks.capture(11, "cond", layer, q, k)

    shared = native.export_shared_dense_cache(
        args,
        shared_root=tmp_path / "shared_dense",
        source={"cache_dir": str(source_root), "latent_shape": [2, 3], "output": str(source_video), "manifest_path": None},
        source_trajectory=trajectory,
        source_mask_cache=source_masks,
    )

    assert (tmp_path / "shared_dense" / "manifest.json").is_file()
    assert shared["export"]["baseline_cache_copies"] == 0
    assert shared["export"]["chai_exported"] is False
    assert shared["consumers"]["nirvana_vid"]["latent"] == str(trajectory[9])
    assert shared["consumers"]["chorus"]["source_cache"] == str(source_root)
    assert shared["consumers"]["chorus"]["source_video"] == str(source_video)
    assert shared["consumers"]["chorus2"]["source_mask_cache"] == str(source_masks.root)
    assert shared["consumers"]["chorus2"]["normal_sparse_raw_layers"] == list(range(10, 21))
    assert not any((tmp_path / "shared_dense" / name).exists() for name in ("nirvana_vid", "chai", "chorus", "chorus2"))

    loaded, _entry, _elapsed = source_masks.get(11, "cond", 15, torch.empty_like(q))
    assert loaded.dtype is torch.bool

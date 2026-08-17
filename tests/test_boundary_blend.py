from __future__ import annotations

import torch

from carnator.masks import boundary_blend_latent, boundary_blend_weights, get_canonical_latent_masks


def _masks(size: int = 9) -> dict[str, torch.Tensor]:
    base = torch.zeros((1, 1, 1, size, size), dtype=torch.bool)
    base[..., size // 2, size // 2] = True
    edit = torch.nn.functional.max_pool2d(base[:, :, 0].float(), 5, 1, 2).bool()[:, :, None]
    return {"latent_mbase": base, "latent_medit": edit}


def test_boundary_weights_are_zero_outside_and_one_at_outer_band():
    masks = _masks()
    alpha, band = boundary_blend_weights(masks, function="smoothstep")

    assert alpha.shape == masks["latent_mbase"].shape
    assert band.shape == alpha.shape
    assert not bool((band & masks["latent_mbase"]).any())
    assert float(alpha[band].min()) >= 0.0
    assert float(alpha[band].max()) == 1.0
    assert bool((alpha[~band] == 0).all())


def test_boundary_blend_keeps_core_target_and_stable_background_source():
    masks = _masks()
    target = torch.ones((1, 1, 9, 9))
    source = torch.zeros_like(target)

    output, stats = boundary_blend_latent(target, source, masks)

    assert float(output[0, 0, 4, 4]) == 1.0
    assert float(output[0, 0, 0, 0]) == 0.0
    assert 0.0 < stats["band_fraction"] < stats["edit_fraction"] < 1.0
    assert 0.0 <= stats["alpha_min"] <= stats["alpha_mean"] <= stats["alpha_max"] <= 1.0


def test_single_cell_band_remains_blended():
    base = torch.zeros((1, 1, 1, 5, 5), dtype=torch.bool)
    base[..., 2, 2] = True
    edit = torch.nn.functional.max_pool2d(base[:, :, 0].float(), 3, 1, 1).bool()[:, :, None]
    alpha, band = boundary_blend_weights({"latent_mbase": base, "latent_medit": edit})

    assert int(band.sum().item()) == 8
    assert float(alpha[band].mean()) == 0.5


def test_old_bundle_without_mbase_preserves_legacy_mask():
    medit = torch.ones((1, 3, 3), dtype=torch.bool)
    bundle = {"Medit": medit, "Msee": medit, "source_completed_step": 5}
    latents = torch.zeros((1, 1, 1, 3, 3))

    masks = get_canonical_latent_masks(bundle, latents)

    assert torch.equal(masks["latent_mbase"], masks["latent_medit"])

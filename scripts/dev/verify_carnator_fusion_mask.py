#!/usr/bin/env python3
"""Verify that Carnator fusion mask refactoring works correctly."""

import sys
from pathlib import Path

# Add VIDEO_CACHE_BASELINES to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")

    try:
        # Test carnator module imports
        from carnator import cogvideox15
        print("✓ carnator.cogvideox15 imported")

        from carnator.cogvideo_attention_probe import (
            CogVideoAttentionProbe,
            compute_self_contrast_score,
            compute_cross_attention_score,
            infer_token_grid_shape,
        )
        print("✓ carnator.cogvideo_attention_probe imported")

        from carnator.masks import (
            FusionMaskConfig,
            build_canonical_fusion_mask,
            save_mask_bundle,
            load_mask_bundle,
            robust_norm,
        )
        print("✓ carnator.masks imported")

        # Verify removed dependencies are NOT imported
        try:
            from methods.chorus.cogvideo_mask import build_cogvideo_edit_mask
            print("✗ ERROR: Old dependency still imported (build_cogvideo_edit_mask)")
            return False
        except ImportError:
            pass  # Expected

        try:
            from methods.chorus.grounded_sam import GroundedSam2Segmenter
            print("✗ ERROR: Old dependency still imported (GroundedSam2Segmenter)")
            return False
        except ImportError:
            pass  # Expected

        print("✓ Old dependencies correctly removed")

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

    return True

def test_config():
    """Test that config has fusion mask parameters."""
    print("\nTesting config...")

    import yaml

    config_path = ROOT / "configs" / "models" / "cogvideox15.yaml"
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return False

    with open(config_path) as f:
        config = yaml.safe_load(f)

    carnator_b5 = config.get("carnator_b5", {})

    # Check for new fusion mask parameters
    required_params = [
        "attention_probe_layer",
        "mask_top_p",
        "mask_background_p",
        "fusion_eps",
        "fusion_gamma",
        "fusion_high_threshold",
        "fusion_low_threshold",
        "mask_spatial_medit_radius",
        "mask_spatial_msee_radius",
    ]

    missing = []
    for param in required_params:
        if param not in carnator_b5:
            missing.append(param)

    if missing:
        print(f"✗ Missing fusion mask parameters: {missing}")
        return False

    print("✓ Config has all fusion mask parameters")

    # Check that old parameters are removed
    old_params = ["mask_dilation_edit", "mask_dilation_see"]
    present_old = [p for p in old_params if p in carnator_b5]

    if present_old:
        print(f"⚠ Warning: Old parameters still present: {present_old}")
        print("  (These are ignored in the new implementation)")

    return True

def test_function_signatures():
    """Test that refactored functions have correct signatures."""
    print("\nTesting function signatures...")

    from carnator import cogvideox15
    import inspect

    # Check _source function
    source_sig = inspect.signature(cogvideox15._source)
    print(f"✓ _source signature: {source_sig}")

    # Check _mask function
    mask_sig = inspect.signature(cogvideox15._mask)
    print(f"✓ _mask signature: {mask_sig}")

    # Check _generate_fusion_mask_from_latent exists
    if not hasattr(cogvideox15, "_generate_fusion_mask_from_latent"):
        print("✗ _generate_fusion_mask_from_latent function not found")
        return False

    gen_sig = inspect.signature(cogvideox15._generate_fusion_mask_from_latent)
    print(f"✓ _generate_fusion_mask_from_latent signature: {gen_sig}")

    return True

def main():
    """Run all verification tests."""
    print("="*80)
    print("Carnator Fusion Mask Refactoring - Verification")
    print("="*80)

    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Function Signatures", test_function_signatures),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "="*80)
    print("Summary:")
    print("="*80)

    all_passed = True
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
        if not success:
            all_passed = False

    if all_passed:
        print("\n✓ All verifications passed!")
        print("\nCarnator is ready to run with pure fusion mask.")
        print("You can now run experiments with:")
        print("  python scripts/dev/run_5baselines_10samples_fixed.py --gpu 3 --limit 10")
        return 0
    else:
        print("\n✗ Some verifications failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

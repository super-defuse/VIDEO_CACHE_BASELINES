#!/usr/bin/env python3
"""Test WAN2.1 baselines initialization"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_masks_like_import():
    """Test that masks_like can be imported from WAN2.1"""
    print("Testing masks_like import...")
    sys.path.insert(0, str(PROJECT_ROOT / "external/Wan2.1"))
    try:
        from wan.utils.utils import masks_like
        print("✓ masks_like imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import masks_like: {e}")
        return False

def test_adapter_load():
    """Test loading WAN2.1 adapter"""
    print("\nTesting WAN2.1 adapter initialization...")
    try:
        from adapters.wan21 import Wan21Adapter
        from core.request import VideoGenerationRequest

        adapter = Wan21Adapter()
        print("✓ Wan21Adapter initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize adapter: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_carnator_validation():
    """Test carnator model validation"""
    print("\nTesting carnator model validation...")
    try:
        import argparse
        from pathlib import Path
        from carnator.adapters.wan_ti2v_5b import validate_native_model

        args = argparse.Namespace()
        args.model_dir = Path("/home/tangxuebin/models/Wan2.1-T2V-1.3B")

        validate_native_model(args)
        print("✓ Carnator model validation passed")
        return True
    except Exception as e:
        print(f"✗ Carnator validation failed: {e}")
        return False

def main():
    print("=" * 60)
    print("WAN2.1 Baselines Pre-flight Check")
    print("=" * 60)

    results = []
    results.append(("masks_like import", test_masks_like_import()))
    results.append(("Adapter initialization", test_adapter_load()))
    results.append(("Carnator validation", test_carnator_validation()))

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)
    print("=" * 60)
    if all_passed:
        print("All checks passed! Ready to run experiments.")
        return 0
    else:
        print("Some checks failed. Fix issues before running experiments.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

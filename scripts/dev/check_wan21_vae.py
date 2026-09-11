#!/usr/bin/env python3
"""测试 WAN2.1 VAE 是否能正确加载"""

import sys
from pathlib import Path

# Add Wan2.1 to path
wan21_path = Path(__file__).parent / "external" / "Wan2.1"
sys.path.insert(0, str(wan21_path.resolve()))

import torch
from wan.configs import WAN_CONFIGS
from wan.modules.vae import WanVAE

def test_wan21_vae():
    """测试 WAN2.1 VAE 加载"""
    print("=" * 60)
    print("测试 WAN2.1 VAE 加载")
    print("=" * 60)

    # 获取配置
    config = WAN_CONFIGS['t2v-1.3B']
    print(f"\n✓ 配置加载成功: t2v-1.3B")
    print(f"  - dim: {config.dim}")
    print(f"  - num_layers: {config.num_layers}")
    print(f"  - vae_checkpoint: {config.vae_checkpoint}")
    print(f"  - vae_stride: {config.vae_stride}")

    # 检查模型文件
    model_dir = Path("models/Wan2.1-T2V-1.3B")
    vae_path = model_dir / config.vae_checkpoint

    if not vae_path.exists():
        print(f"\n✗ VAE 文件不存在: {vae_path}")
        return False

    print(f"\n✓ VAE 文件存在: {vae_path}")
    print(f"  大小: {vae_path.stat().st_size / 1024 / 1024:.1f} MB")

    # 尝试加载 VAE
    try:
        print(f"\n正在加载 WAN2.1 VAE...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        vae = WanVAE(vae_pth=str(vae_path), device=device)
        print(f"✓ WAN2.1 VAE 加载成功!")
        print(f"  - z_dim: {vae.model.z_dim}")
        print(f"  - device: {device}")

        # 检查 VAE 的通道数
        encoder_conv1_weight_shape = vae.model.encoder.conv1.weight.shape
        print(f"  - encoder.conv1.weight.shape: {encoder_conv1_weight_shape}")
        print(f"    输入通道数: {encoder_conv1_weight_shape[1]} (应该是 3)")
        print(f"    输出通道数: {encoder_conv1_weight_shape[0]}")

        return True

    except Exception as e:
        print(f"\n✗ VAE 加载失败:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_wan21_full_pipeline():
    """测试完整的 WAN2.1 pipeline"""
    print("\n" + "=" * 60)
    print("测试 WAN2.1 完整 Pipeline")
    print("=" * 60)

    try:
        from wan import WanT2V
        from wan.configs import WAN_CONFIGS

        config = WAN_CONFIGS['t2v-1.3B']
        model_dir = "models/Wan2.1-T2V-1.3B"

        print(f"\n正在初始化 WanT2V pipeline...")
        device_id = 0 if torch.cuda.is_available() else -1

        if device_id < 0:
            print("✗ 需要 GPU 才能完整测试 pipeline")
            return False

        pipeline = WanT2V(
            config=config,
            checkpoint_dir=model_dir,
            device_id=device_id,
        )

        print(f"✓ WanT2V pipeline 初始化成功!")
        print(f"  - device: {pipeline.device}")
        print(f"  - vae.model.z_dim: {pipeline.vae.model.z_dim}")
        print(f"  - model.num_layers: {len(pipeline.model.blocks)}")

        return True

    except Exception as e:
        print(f"\n✗ Pipeline 初始化失败:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_wan21_vae()

    if success and torch.cuda.is_available():
        test_wan21_full_pipeline()

    print("\n" + "=" * 60)
    if success:
        print("✓ WAN2.1 VAE 测试通过")
        sys.exit(0)
    else:
        print("✗ WAN2.1 VAE 测试失败")
        sys.exit(1)

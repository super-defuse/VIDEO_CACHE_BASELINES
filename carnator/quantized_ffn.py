"""Runtime W8A8 replacements for Wan FFNs.

Weights are quantized once per output channel. Activations are quantized per
token row with Triton, then an INT8 Tensor Core GEMM fuses the INT32
accumulator dequantization, bias, and FP16/BF16 output conversion.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def _quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    weight_fp = weight.detach().float()
    scale = weight_fp.abs().amax(dim=1).clamp_min(1e-8).div(127.0)
    quantized = torch.round(weight_fp / scale[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
    error = float((weight_fp - quantized.float() * scale[:, None]).abs().max().item())
    return quantized, scale, error


def w8a8_kernel_available() -> bool:
    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _int8_quantize_per_token_kernel(
        x_ptr,
        output_ptr,
        scale_ptr,
        stride_x_row,
        stride_x_col,
        hidden_dim,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + row * stride_x_row + offsets * stride_x_col,
            mask=offsets < hidden_dim,
            other=0.0,
        ).to(tl.float32)
        finite = (x == x) & (tl.abs(x) <= 3.402823466e38)
        x = tl.where(finite, x, 0.0)
        absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-8)
        inverse_scale = 127.0 / absmax
        scaled = x * inverse_scale
        rounded = tl.where(scaled >= 0.0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
        rounded = tl.maximum(tl.minimum(rounded, 127.0), -127.0)
        tl.store(output_ptr + row * hidden_dim + offsets, rounded.to(tl.int8), mask=offsets < hidden_dim)
        tl.store(scale_ptr + row, 1.0 / inverse_scale)


    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=8),
            triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=8),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _w8a8_linear_bias_kernel(
        a_ptr,
        b_ptr,
        bias_ptr,
        a_scale_ptr,
        b_scale_ptr,
        output_ptr,
        M,
        N,
        K,
        stride_a_row,
        stride_a_col,
        stride_b_k,
        stride_b_n,
        stride_output_row,
        stride_output_col,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)
        group_width = GROUP_M * grid_n
        group_id = pid // group_width
        first_pid_m = group_id * GROUP_M
        group_size_m = min(grid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % group_width) // group_size_m

        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offsets_m[:, None] * stride_a_row + offsets_k[None, :] * stride_a_col
        b_ptrs = b_ptr + offsets_k[:, None] * stride_b_k + offsets_n[None, :] * stride_b_n
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        for block_k in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = offsets_k < K - block_k * BLOCK_K
            a = tl.load(a_ptrs, mask=(offsets_m[:, None] < M) & k_mask[None, :], other=0).to(tl.int8)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & (offsets_n[None, :] < N), other=0).to(tl.int8)
            accumulator = tl.dot(a, b, accumulator, out_dtype=tl.int32)
            a_ptrs += BLOCK_K * stride_a_col
            b_ptrs += BLOCK_K * stride_b_k

        a_scale = tl.load(a_scale_ptr + offsets_m, mask=offsets_m < M, other=0.0).to(tl.float32)
        b_scale = tl.load(b_scale_ptr + offsets_n, mask=offsets_n < N, other=0.0).to(tl.float32)
        output = accumulator.to(tl.float32) * a_scale[:, None] * b_scale[None, :]
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offsets_n, mask=offsets_n < N, other=0.0).to(tl.float32)
            output += bias[None, :]

        output_ptrs = output_ptr + offsets_m[:, None] * stride_output_row + offsets_n[None, :] * stride_output_col
        output_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
        tl.store(output_ptrs, output, mask=output_mask)


def _quantize_activation_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("W8A8 requires Triton; no eager fallback is used")
    flat = x.reshape(-1, x.shape[-1])
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    rows, hidden_dim = flat.shape
    quantized = torch.empty((rows, hidden_dim), dtype=torch.int8, device=flat.device)
    scales = torch.empty((rows,), dtype=torch.float32, device=flat.device)
    block_k = triton.next_power_of_2(hidden_dim)
    _int8_quantize_per_token_kernel[(rows,)](
        flat,
        quantized,
        scales,
        flat.stride(0),
        flat.stride(1),
        hidden_dim,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return quantized, scales


def _w8a8_linear(
    activation: torch.Tensor,
    qweight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("W8A8 requires Triton; no eager fallback is used")
    rows, input_features = activation.shape
    output_features, weight_input_features = qweight.shape
    if input_features != weight_input_features:
        raise ValueError(f"activation width {input_features} != quantized weight width {weight_input_features}")
    output = torch.empty((rows, output_features), dtype=output_dtype, device=activation.device)
    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]) * triton.cdiv(output_features, meta["BLOCK_N"]),)
    _w8a8_linear_bias_kernel[grid](
        activation,
        qweight,
        bias if bias is not None else output,
        activation_scale,
        weight_scale,
        output,
        rows,
        output_features,
        input_features,
        activation.stride(0),
        activation.stride(1),
        qweight.stride(1),
        qweight.stride(0),
        output.stride(0),
        output.stride(1),
        HAS_BIAS=bias is not None,
        GROUP_M=8,
    )
    return output


class QuantizedLinear(nn.Module):
    """W8A8 Linear with Triton activation quantization and fused epilogue."""

    kernel_name = "triton-w8a8-v1"

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        if not w8a8_kernel_available():
            raise RuntimeError("W8A8 requires Triton and CUDA; no floating-point fallback is used")
        qweight, weight_scale, weight_max_abs_error = _quantize_weight(linear.weight)
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.output_dtype = linear.weight.dtype
        self.register_buffer("qweight", qweight, persistent=True)
        self.register_buffer("weight_scale", weight_scale, persistent=True)
        if linear.bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", linear.bias.detach().float().clone(), persistent=True)
        self.weight_max_abs_error = weight_max_abs_error

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected last dim {self.in_features}, got {x.shape[-1]}")
        qactivation, activation_scale = _quantize_activation_per_token(x)
        out = _w8a8_linear(
            qactivation,
            self.qweight,
            activation_scale,
            self.weight_scale,
            self.bias,
            self.output_dtype,
        )
        return out.reshape(*x.shape[:-1], self.out_features)

    def metadata(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel_name,
            "weight_dtype": str(self.qweight.dtype),
            "weight_layout": "output-input",
            "weight_scale_granularity": "per-output-channel",
            "activation_dtype": "torch.int8",
            "activation_scale_granularity": "per-token-row",
            "accumulator_dtype": "torch.int32",
            "epilogue": "scale + bias + output cast fused in Triton GEMM",
            "output_dtype": str(self.output_dtype),
            "weight_max_abs_error": self.weight_max_abs_error,
        }


class QuantizedFFN(nn.Module):
    """Drop-in replacement for ``Linear -> GELU(tanh) -> Linear`` Wan FFNs."""

    def __init__(self, first: nn.Linear, activation: nn.Module, second: nn.Linear) -> None:
        super().__init__()
        if not isinstance(activation, nn.GELU) or activation.approximate != "tanh":
            raise ValueError("Wan W8A8 FFN requires GELU(approximate='tanh')")
        self.fc1 = QuantizedLinear(first)
        self.activation = nn.GELU(approximate="tanh")
        self.fc2 = QuantizedLinear(second)

    @classmethod
    def from_module(cls, module: nn.Module) -> "QuantizedFFN":
        if not isinstance(module, nn.Sequential) or len(module) != 3:
            raise ValueError("expected Wan FFN nn.Sequential(Linear, GELU, Linear)")
        first, activation, second = module
        if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
            raise ValueError("Wan FFN must start and end with nn.Linear")
        return cls(first, activation, second)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))

    def metadata(self) -> dict[str, Any]:
        return {
            "kernel": QuantizedLinear.kernel_name,
            "format": "W8A8 dynamic activation / per-output-channel weight",
            "fc1": self.fc1.metadata(),
            "fc2": self.fc2.metadata(),
        }


def compare_ffn_outputs(reference: nn.Module, quantized: QuantizedFFN, sample: torch.Tensor) -> dict[str, float]:
    """Numerical probe kept out of timed inference paths."""

    with torch.no_grad():
        expected = reference(sample)
        actual = quantized(sample)
    delta = (actual.float() - expected.float()).reshape(1, -1)
    return {
        "max_abs_error": float(delta.abs().max().item()),
        "mean_abs_error": float(delta.abs().mean().item()),
        "cosine": float(torch.nn.functional.cosine_similarity(actual.float().reshape(1, -1), expected.float().reshape(1, -1)).item()),
    }


@dataclass
class QuantizedFFNState:
    layers: list[int]
    originals: list[tuple[Any, nn.Module]] = field(default_factory=list)
    modules: dict[int, QuantizedFFN] = field(default_factory=dict)
    output_errors: dict[int, dict[str, float]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "backend": "w8a8",
            "kernel": QuantizedLinear.kernel_name,
            "layers": self.layers,
            "layer_metadata": {str(layer): module.metadata() for layer, module in self.modules.items()},
            "layer_output_error": {str(layer): error for layer, error in self.output_errors.items()},
        }


class QuantizedFFNHooks(AbstractContextManager):
    """Temporarily replace Wan block FFNs and restore the original modules."""

    def __init__(self, model: Any, layers: list[int] | None = None) -> None:
        layer_count = len(model.blocks)
        self.state = QuantizedFFNState(layers=list(range(layer_count)) if layers is None else list(layers))
        self.model = model

    def __enter__(self) -> QuantizedFFNState:
        for layer in self.state.layers:
            block = self.model.blocks[layer]
            original = block.ffn
            quantized = QuantizedFFN.from_module(original).to(device=next(original.parameters()).device)
            self.state.originals.append((block, original))
            self.state.modules[layer] = quantized
            block.ffn = quantized
        return self.state

    def verify(self, sample_tokens: int = 0) -> dict[int, dict[str, float]]:
        """Optionally record small FP-vs-W8A8 probes without contaminating timing."""

        if sample_tokens <= 0:
            return self.state.output_errors
        for (block, reference), layer in zip(self.state.originals, self.state.layers):
            quantized = self.state.modules[layer]
            device = quantized.fc1.qweight.device
            dtype = next(reference.parameters()).dtype
            sample = torch.randn((sample_tokens, quantized.fc1.in_features), device=device, dtype=dtype)
            self.state.output_errors[layer] = compare_ffn_outputs(reference, quantized, sample)
        return self.state.output_errors

    def attach_profiler(self, profiler: Any) -> None:
        for layer, module in self.state.modules.items():
            profiler.add_module(module, "ffn", f"blocks.{layer}.ffn.w8a8")

    def __exit__(self, exc_type, exc, tb):
        for block, original in reversed(self.state.originals):
            block.ffn = original
        self.state.originals.clear()
        return False

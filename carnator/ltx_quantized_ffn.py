"""W8A8 FFN for LTX-Video's ``FeedForward``.

LTX's block FFN is the same ``output = fc2(act(fc1(x)))`` shape as Wan's, but
diffusers builds it as ``ModuleList([GELU, Dropout, Linear])`` rather than
``Sequential(Linear, GELU, Linear)``: the first projection lives *inside* the
activation module as ``GELU.proj``.  ``final_dropout`` defaults to false in
``LTXVideoTransformerBlock``, so the trailing dropout is absent and the list is
three long.  The body below is therefore ``CogVideoQuantizedFFN`` with the
trailing dropout dropped.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any


class LtxQuantizedFFN:
    def __new__(cls, module: Any) -> Any:
        import torch.nn as nn

        from carnator.quantized_ffn import QuantizedLinear

        if len(module.net) != 3 or not hasattr(module.net[0], "proj") or not isinstance(module.net[2], nn.Linear):
            raise ValueError("unsupported LTX FeedForward layout for W8A8")

        class QuantizedModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = QuantizedLinear(module.net[0].proj)
                self.approximate = module.net[0].approximate
                self.dropout = module.net[1]
                self.fc2 = QuantizedLinear(module.net[2])

            def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
                import torch.nn.functional as functional

                hidden_states = functional.gelu(self.fc1(hidden_states), approximate=self.approximate)
                return self.fc2(self.dropout(hidden_states))

        return QuantizedModule()


class LtxQuantizedFFNHooks(AbstractContextManager):
    def __init__(self, transformer: Any, layers: list[int]) -> None:
        self.transformer = transformer
        self.layers = list(layers)
        self.originals: list[tuple[Any, Any]] = []
        self.modules: dict[int, Any] = {}

    def __enter__(self) -> "LtxQuantizedFFNHooks":
        for layer in self.layers:
            block = self.transformer.transformer_blocks[layer]
            original = block.ff
            quantized = LtxQuantizedFFN(original).to(next(original.parameters()).device)
            self.originals.append((block, original))
            self.modules[layer] = quantized
            block.ff = quantized
        return self

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "backend": "triton-w8a8-v1",
            "layers": self.layers,
            "linear_count": 2 * len(self.modules),
        }

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        for block, original in reversed(self.originals):
            block.ff = original
        self.originals.clear()
        return False

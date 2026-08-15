from __future__ import annotations

from typing import Any, Callable


class TeaCacheController:
    """A small, model-agnostic TeaCache forward wrapper.

    It keeps the previous transformer input/output for each CFG branch.  A
    denoising step is skipped only when the relative input change remains
    under the configured threshold; ``max_skip_steps`` bounds drift.
    """

    def __init__(self, *, steps: int, threshold: float, max_skip_steps: int) -> None:
        self.steps = steps
        self.threshold = threshold
        self.max_skip_steps = max_skip_steps
        self.executed_steps: set[int] = set()
        self._previous_inputs: dict[int, Any] = {}
        self._previous_outputs: dict[int, Any] = {}
        self._consecutive_skips = 0
        self._decisions: dict[int, bool] = {}

    @staticmethod
    def _input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any | None:
        value = args[0] if args else kwargs.get("hidden_states")
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        return value

    def _should_execute(self, step: int, value: Any, branches: int) -> bool:
        if step == 0 or step >= self.steps - 1 or self._consecutive_skips >= self.max_skip_steps:
            return True
        if value is None or any(branch not in self._previous_outputs for branch in range(branches)):
            return True
        previous = self._previous_inputs.get(0)
        if previous is None:
            return True
        change = (value.float() - previous.float()).abs().mean()
        scale = previous.float().abs().mean().clamp_min(1e-6)
        return bool((change / scale).item() > self.threshold)

    def patch(self, module: Any, *, branches: int) -> Callable[[], None]:
        original = module.forward
        calls = 0

        def forward(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            step, branch = divmod(calls, branches)
            calls += 1
            if branch == 0:
                value = self._input(args, kwargs)
                execute = self._should_execute(step, value, branches)
                self._decisions[step] = execute
                if execute:
                    self.executed_steps.add(step)
                    self._consecutive_skips = 0
                    if value is not None:
                        self._previous_inputs[0] = value.detach()
                else:
                    self._consecutive_skips += 1
            execute = self._decisions[step]
            if not execute:
                return self._previous_outputs[branch]
            output = original(*args, **kwargs)
            self._previous_outputs[branch] = output
            return output

        module.forward = forward

        def restore() -> None:
            module.forward = original

        return restore

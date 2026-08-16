"""Small OpenAI-compatible client for Chorus prompt-difference extraction."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.request import Request as URLRequest
from urllib.request import urlopen


DEFAULT_BASE_URL = "https://www.dmxapi.cn/v1/chat/completions"
DEFAULT_MODEL = "qwen3.5-plus-2026-02-15"


def _load_key_from_dotenv() -> str | None:
    key = os.environ.get("LLM_API_KEY")
    if key:
        return key
    # Read only the requested variable; never print or return the file content.
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        try:
            for line in candidate.read_text(encoding="utf-8").splitlines():
                if line.startswith("LLM_API_KEY="):
                    value = line.partition("=")[2].strip().strip('"').strip("'")
                    return value or None
        except OSError:
            continue
    return None


def _fallback(source: str, target: str, error: str | None = None) -> dict[str, Any]:
    source_words = re.findall(r"[A-Za-z0-9_'-]+", source.lower())
    target_words = re.findall(r"[A-Za-z0-9_'-]+", target.lower())
    source_set, target_set = set(source_words), set(target_words)
    removed = [word for word in source_words if word not in target_set]
    added = [word for word in target_words if word not in source_set]
    payload: dict[str, Any] = {
        "source_entities": removed,
        "target_entities": added,
        "removed_tokens": removed,
        "added_tokens": added,
        "edit_summary": f"replace {', '.join(removed) or 'source phrase'} with {', '.join(added) or 'target phrase'}",
        "backend": "lexical_fallback",
    }
    if error:
        payload["error"] = error[:240]
    return payload


def _parse_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("LLM response is not a JSON object")
    return value


def extract_prompt_diff(source: str, target: str, *, timeout_s: float = 45.0) -> dict[str, Any]:
    """Ask the configured DMXAPI model for a compact, structured prompt diff."""
    key = _load_key_from_dotenv()
    if not key:
        return _fallback(source, target, "LLM_API_KEY is not configured")
    base_url = os.environ.get("LLM_API_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Compare source and target video prompts. Return JSON only with keys "
                    "source_entities, target_entities, removed_tokens, added_tokens, edit_summary. "
                    "Keep arrays short and use literal words from the prompts."
                ),
            },
            {"role": "user", "content": json.dumps({"source": source, "target": target}, ensure_ascii=False)},
        ],
    }
    request = URLRequest(
        base_url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        result = _parse_json(content)
        result["backend"] = "dmxapi"
        result["model"] = model
        result["latency_s"] = round(time.perf_counter() - started, 3)
        return result
    except Exception as exc:  # Keep an experiment reproducible if the endpoint is transient.
        result = _fallback(source, target, f"{type(exc).__name__}: {exc}")
        result["model"] = model
        result["latency_s"] = round(time.perf_counter() - started, 3)
        return result

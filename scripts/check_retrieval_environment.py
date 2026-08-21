#!/usr/bin/env python3
"""Validate the Python environment used by a real retrieval run."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import site
import sys
from pathlib import Path


def _module_version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "installed"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("wan", "cogvideo"), required=True)
    args = parser.parse_args()

    required = ["spacy", "faiss", "sklearn"]
    if args.runtime == "wan":
        required.append("chromadb")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    print(f"python={sys.executable}")
    print(f"prefix={sys.prefix}")
    print(f"base_prefix={sys.base_prefix}")
    print(f"user_site_enabled={site.ENABLE_USER_SITE}")
    if site.ENABLE_USER_SITE:
        raise SystemExit("user site is enabled; rerun with PYTHONNOUSERSITE=1")
    if missing:
        raise SystemExit("missing retrieval modules: " + ", ".join(missing))
    for name in required:
        print(f"{name}={_module_version(name)}")

    import spacy

    model = "en_core_web_sm"
    if not spacy.util.is_package(model):
        raise SystemExit(
            f"missing spaCy model {model}; run: {sys.executable} -m spacy download {model}"
        )
    print(f"{model}=installed")
    if Path(sys.prefix) != Path(sys.base_prefix):
        raise SystemExit("environment is not an isolated Conda Python environment")
    print("retrieval environment check passed")


if __name__ == "__main__":
    main()

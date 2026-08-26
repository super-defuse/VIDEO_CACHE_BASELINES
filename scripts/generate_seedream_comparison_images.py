#!/usr/bin/env python3
"""Generate four controlled, photorealistic comparison images with Seedream 4.5.

The first pair depicts the same mountainside in two conditions: generate the
snowy apple scene first, then use it as a reference to create the snow-free
peach scene while preserving the camera and terrain geometry. The second pair
keeps the football field composition similar while changing the object's
geometry. The barbell request uses the football image as an input reference so
the generated field, camera, and placement stay aligned.

Set OPENROUTER_API_KEY in the environment, or add it to a gitignored ``.env``
file in the repository root, before running this script.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "seedream-comparison-images"
OPENROUTER_IMAGES_URL = "https://openrouter.ai/api/v1/images"
MODEL = "bytedance-seed/seedream-4.5"
SMOKE_TEST_RESOLUTION = "1K"
DEFAULT_RESOLUTION = "4K"
DEFAULT_CANVAS_SIZE = (4096, 3072)


@dataclass(frozen=True)
class ImageSpec:
    filename: str
    prompt: str
    reference_filename: str | None = None


IMAGE_SPECS = (
    ImageSpec(
        filename="apple_snowy_mountainside.png",
        prompt=(
            "Photorealistic 4:3 landscape photo of one ripe, glossy red apple on a snowy alpine mountainside. "
            "Use a balanced medium-wide view with the camera level with the apple, not an oversized foreground product close-up. "
            "Place the fully visible apple in the upper-middle central area of the frame at 50 percent image width and 42 percent "
            "image height, about 18 percent of the image height, with clear space around it, especially below it. "
            "The snow-covered slope rises from lower left to upper right, leading to a rocky ridgeline and distant jagged peaks. "
            "Keep the apple, snow, rocks, and distant mountains sharp: deep depth of field, no background blur or bokeh. "
            "Cold diffuse daylight and natural apple, snow, and rock textures. Exactly one apple; no people, animals, buildings, "
            "labels, logos, text, watermark, basket, or other objects."
        ),
    ),
    ImageSpec(
        filename="peach_grassy_mountainside.png",
        reference_filename="apple_snowy_mountainside.png",
        prompt=(
            "Use the input reference for the same mountainside in a 4:3 composition: keep the camera view, framing, diagonal slope, "
            "rocky ridgeline, distant peaks, and subject placement unchanged. Replace the apple with one ripe peach at 50 percent "
            "image width and 42 percent image height, in the upper-middle central area of the frame, at the same scale, about "
            "18 percent of the image height, with clear space around it, especially below it. "
            "Turn the winter scene into the same snow-free mountainside with green grass and exposed gray rock. "
            "There must be no snow, ice, glaciers, snowcaps, or white patches. Keep the peach, grass, rocks, and distant mountains sharp: "
            "deep depth of field, no background blur or bokeh. Warm diffuse daylight and natural peach, grass, and stone textures. "
            "Exactly one peach; no apple remains; no people, animals, buildings, labels, logos, text, watermark, basket, or other objects."
        ),
    ),
    ImageSpec(
        filename="american_football_field.png",
        prompt=(
            "Use case: photorealistic-natural. Asset type: controlled visual comparison. "
            "Landscape 4:3 framing. One brown leather American football with visible white laces, lying horizontally near the exact center "
            "of an outdoor stadium practice field. The entire football is visible and occupies approximately 34 percent "
            "of the image width. Scene/backdrop: fresh green artificial turf with subtle white yard lines, low metal "
            "bleachers far behind the subject, an overcast bright sky, no crowd. "
            "Style/medium: realistic sports editorial photograph, 50 mm lens, low eye-level camera, centered composition, "
            "soft background bokeh, neutral overcast daylight, accurate leather and turf texture. "
            "Constraints: exactly one American football; no people, animals, helmets, hands, other sports equipment, labels, "
            "logos, text, watermark, or extra objects."
        ),
    ),
    ImageSpec(
        filename="olympic_barbell_field.png",
        reference_filename="american_football_field.png",
        prompt=(
            "Use case: photorealistic-natural. Asset type: controlled visual comparison. "
            "Landscape 4:3 framing. Use the input reference as a strict reference for the stadium field, camera position, lighting, "
            "background, depth of field, and centered object placement. Replace only the American football with one "
            "Olympic barbell: a straight black steel bar with two large round black weight plates, lying horizontally "
            "on the turf. Keep the entire barbell visible and make its overall width, including plates, approximately "
            "34 percent of the image width, matching the apparent scale and central placement of the football in the reference. "
            "Preserve the same green artificial turf, subtle white yard lines, low distant metal bleachers, and neutral overcast daylight. "
            "Style/medium: realistic sports editorial photograph, 50 mm lens, low eye-level camera, accurate rubber, steel, and turf textures. "
            "Constraints: exactly one barbell; no football remains; no people, animals, hands, other sports equipment, labels, logos, text, watermark, or extra objects."
        ),
    ),
)


class OpenRouterError(RuntimeError):
    """An OpenRouter response could not produce a usable image."""


def parse_canvas_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("canvas size must be WIDTHxHEIGHT, for example 2048x2048") from exc
    if width < 1 or height < 1:
        raise argparse.ArgumentTypeError("canvas size dimensions must be positive")
    return width, height


def dotenv_value(path: Path, key: str) -> str | None:
    """Read one simple KEY=value entry without importing a dotenv dependency."""

    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if separator and name.strip() == key:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            return value or None
    return None


def api_key_from_environment(env_name: str, dotenv_path: Path) -> str | None:
    return os.environ.get(env_name) or dotenv_value(dotenv_path, env_name)


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_payload(spec: ImageSpec, *, resolution: str, seed: int, reference_path: Path | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL,
        "prompt": spec.prompt,
        "resolution": resolution,
        "seed": seed,
    }
    if reference_path is not None:
        payload["input_references"] = [
            {"type": "image_url", "image_url": {"url": image_data_url(reference_path)}}
        ]
    return payload


def response_body(error: HTTPError) -> str:
    try:
        return error.read().decode("utf-8", errors="replace")
    except OSError:
        return "<could not read response body>"


def generate_image(payload: dict[str, Any], api_key: str, timeout_seconds: float, retries: int) -> tuple[bytes, dict[str, Any] | None]:
    request_body = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = Request(
            OPENROUTER_IMAGES_URL,
            data=request_body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw_response = response.read()
            decoded = json.loads(raw_response)
            data = decoded.get("data")
            if not isinstance(data, list) or not data or not isinstance(data[0], dict):
                raise OpenRouterError("OpenRouter response has no image data")
            encoded_image = data[0].get("b64_json")
            if not isinstance(encoded_image, str) or not encoded_image:
                raise OpenRouterError("OpenRouter response has no base64 image")
            try:
                return base64.b64decode(encoded_image, validate=True), decoded.get("usage")
            except (binascii.Error, ValueError) as exc:
                raise OpenRouterError("OpenRouter returned invalid base64 image data") from exc
        except HTTPError as exc:
            detail = response_body(exc)
            retryable = exc.code == 429 or 500 <= exc.code < 600
            last_error = OpenRouterError(f"OpenRouter returned HTTP {exc.code}: {detail}")
        except (URLError, TimeoutError, json.JSONDecodeError, OpenRouterError) as exc:
            retryable = isinstance(exc, (URLError, TimeoutError))
            last_error = exc
        if attempt < retries and retryable:
            time.sleep(2**attempt)
            continue
        break
    assert last_error is not None
    raise last_error


def save_normalized_png(image_bytes: bytes, output_path: Path, canvas_size: tuple[int, int]) -> tuple[int, int]:
    """Center-crop only when needed, so every comparison output has equal pixels."""

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            normalized = ImageOps.fit(
                image,
                canvas_size,
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = output_path.with_suffix(".tmp")
            normalized.save(temporary_path, format="PNG")
            temporary_path.replace(output_path)
    except Exception as exc:
        raise OpenRouterError(f"response was not a readable raster image: {exc}") from exc
    return canvas_size


def existing_output_paths(output_dir: Path) -> list[Path]:
    return [output_dir / spec.filename for spec in IMAGE_SPECS if (output_dir / spec.filename).exists()]


def selected_image_specs(filenames: list[str] | None) -> tuple[ImageSpec, ...]:
    if filenames is None:
        return IMAGE_SPECS
    requested = set(filenames)
    return tuple(spec for spec in IMAGE_SPECS if spec.filename in requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resolution", choices=("1K", "2K", "4K"), default=DEFAULT_RESOLUTION)
    parser.add_argument(
        "--canvas-size",
        type=parse_canvas_size,
        default=DEFAULT_CANVAS_SIZE,
        help="Final WIDTHxHEIGHT PNG size; all four images are normalized to this canvas (default: 4096x3072, 4:3).",
    )
    parser.add_argument("--seed", type=int, default=20260825, help="Shared Seedream seed for reproducible comparison runs.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout in seconds (default: 300).")
    parser.add_argument("--retries", type=int, default=2, help="Retries for network, 429, and 5xx failures (default: 2).")
    parser.add_argument("--dotenv", type=Path, default=ROOT / ".env", help="Optional dotenv file (default: repository-root .env).")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY", help="Environment variable holding the OpenRouter key.")
    parser.add_argument(
        "--images",
        choices=tuple(spec.filename for spec in IMAGE_SPECS),
        nargs="+",
        help="Generate only these filenames. A dependent peach or barbell rerun requires its reference PNG to already exist.",
    )
    parser.add_argument("--force", action="store_true", help="Allow this run to replace existing outputs in --output-dir.")
    parser.add_argument("--dry-run", action="store_true", help="Print the requests without contacting OpenRouter or requiring an API key.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Request and validate one disposable 1K apple image before a 4K production run; no image is saved.",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.retries < 0:
        parser.error("--retries must be non-negative")
    if args.dry_run and args.smoke_test:
        parser.error("--dry-run and --smoke-test cannot be used together")
    return args


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    specs = selected_image_specs(args.images)

    if args.dry_run:
        preview = [
            {
                "filename": spec.filename,
                "payload": build_payload(spec, resolution=args.resolution, seed=args.seed, reference_path=None),
                "has_input_reference": spec.reference_filename is not None,
            }
            for spec in specs
        ]
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0

    api_key = api_key_from_environment(args.api_key_env, args.dotenv)
    if not api_key:
        raise SystemExit(
            f"Missing {args.api_key_env}. Set it in the environment or add {args.api_key_env}=... to {args.dotenv}."
        )

    if args.smoke_test:
        print(f"Running disposable {SMOKE_TEST_RESOLUTION} API smoke test...", flush=True)
        payload = build_payload(
            IMAGE_SPECS[0], resolution=SMOKE_TEST_RESOLUTION, seed=args.seed, reference_path=None
        )
        image_bytes, usage = generate_image(payload, api_key, args.timeout, args.retries)
        try:
            with Image.open(io.BytesIO(image_bytes)) as response_image:
                response_image.verify()
        except Exception as exc:
            raise OpenRouterError(f"smoke-test response was not a valid image: {exc}") from exc
        print(f"Smoke test succeeded: received {len(image_bytes)} bytes at {SMOKE_TEST_RESOLUTION}; usage={usage}")
        return 0

    existing = [output_dir / spec.filename for spec in specs if (output_dir / spec.filename).exists()]
    if (existing or (args.images is None and manifest_path.exists())) and not args.force:
        names = ", ".join(path.name for path in [*existing, manifest_path] if path.exists())
        raise SystemExit(f"Refusing to replace existing output(s): {names}. Re-run with --force to replace them.")

    previous_entries: dict[str, dict[str, Any]] = {}
    if args.images is not None and manifest_path.is_file():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in previous_manifest.get("images", []):
                if isinstance(entry, dict) and isinstance(entry.get("filename"), str):
                    previous_entries[entry["filename"]] = entry
        except json.JSONDecodeError:
            pass

    outputs: list[dict[str, Any]] = []
    for spec in specs:
        reference_path = output_dir / spec.reference_filename if spec.reference_filename else None
        if reference_path is not None and not reference_path.is_file():
            raise SystemExit(f"Missing required reference image: {reference_path}")
        payload = build_payload(spec, resolution=args.resolution, seed=args.seed, reference_path=reference_path)
        print(f"Generating {spec.filename}...", flush=True)
        image_bytes, usage = generate_image(payload, api_key, args.timeout, args.retries)
        size = save_normalized_png(image_bytes, output_dir / spec.filename, args.canvas_size)
        outputs.append(
            {
                "filename": spec.filename,
                "prompt": spec.prompt,
                "input_reference": spec.reference_filename,
                "size": {"width": size[0], "height": size[1]},
                "usage": usage,
            }
        )

    entries_by_filename = {entry["filename"]: entry for entry in previous_entries.values()}
    entries_by_filename.update({entry["filename"]: entry for entry in outputs})
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "endpoint": OPENROUTER_IMAGES_URL,
        "model": MODEL,
        "resolution": args.resolution,
        "seed": args.seed,
        "canvas_size": {"width": args.canvas_size[0], "height": args.canvas_size[1]},
        "images": [entries_by_filename[spec.filename] for spec in IMAGE_SPECS if spec.filename in entries_by_filename],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(outputs)} image(s) and manifest to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

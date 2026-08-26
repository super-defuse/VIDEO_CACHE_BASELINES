from __future__ import annotations

import base64
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts.generate_seedream_comparison_images import (
    DEFAULT_CANVAS_SIZE,
    DEFAULT_RESOLUTION,
    IMAGE_SPECS,
    ImageSpec,
    SMOKE_TEST_RESOLUTION,
    build_payload,
    dotenv_value,
    main,
    parse_canvas_size,
    save_normalized_png,
    selected_image_specs,
)


class GenerateSeedreamComparisonImagesTest(unittest.TestCase):
    def test_parse_canvas_size(self) -> None:
        self.assertEqual(parse_canvas_size("2048x1024"), (2048, 1024))

    def test_smoke_and_production_resolution_defaults_are_deliberately_different(self) -> None:
        self.assertEqual(SMOKE_TEST_RESOLUTION, "1K")
        self.assertEqual(DEFAULT_RESOLUTION, "4K")
        self.assertEqual(DEFAULT_CANVAS_SIZE, (4096, 3072))

    def test_dotenv_value_accepts_export_and_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dotenv_path = Path(directory) / ".env"
            dotenv_path.write_text("OTHER=value\nexport OPENROUTER_API_KEY='test-key'\n", encoding="utf-8")
            self.assertEqual(dotenv_value(dotenv_path, "OPENROUTER_API_KEY"), "test-key")

    def test_build_payload_uses_a_png_data_url_for_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference_path = Path(directory) / "reference.png"
            reference_path.write_bytes(b"png-bytes")
            payload = build_payload(
                ImageSpec(filename="barbell.png", prompt="replace the object"),
                resolution="2K",
                seed=17,
                reference_path=reference_path,
            )
        url = payload["input_references"][0]["image_url"]["url"]
        self.assertEqual(url, "data:image/png;base64," + base64.b64encode(b"png-bytes").decode("ascii"))

    def test_normalized_output_has_the_requested_equal_dimensions(self) -> None:
        source = Image.new("RGB", (300, 150), color=(70, 110, 150))
        image_bytes = io.BytesIO()
        source.save(image_bytes, format="PNG")
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "output.png"
            self.assertEqual(save_normalized_png(image_bytes.getvalue(), output_path, (256, 256)), (256, 256))
            with Image.open(output_path) as output:
                self.assertEqual(output.size, (256, 256))

    def test_expected_comparison_specs_are_present(self) -> None:
        self.assertEqual([spec.filename for spec in IMAGE_SPECS], [
            "apple_snowy_mountainside.png",
            "peach_grassy_mountainside.png",
            "american_football_field.png",
            "olympic_barbell_field.png",
        ])

    def test_fruit_prompts_keep_the_subject_centered(self) -> None:
        for spec in IMAGE_SPECS[:2]:
            self.assertIn("central", spec.prompt)
            self.assertIn("upper-middle", spec.prompt)
            self.assertIn("central area of the frame", spec.prompt)
            self.assertNotIn("lower center", spec.prompt)

    def test_fruit_prompts_pin_the_subject_to_the_upper_middle(self) -> None:
        for spec in IMAGE_SPECS[:2]:
            self.assertIn("50 percent image width", spec.prompt)
            self.assertIn("42 percent image height", spec.prompt)
            self.assertIn("especially below it", spec.prompt)
        self.assertIn("not an oversized foreground product close-up", IMAGE_SPECS[0].prompt)

    def test_peach_uses_the_apple_as_a_reference_for_the_same_mountainside(self) -> None:
        apple, peach = IMAGE_SPECS[:2]
        self.assertIsNone(apple.reference_filename)
        self.assertEqual(peach.reference_filename, apple.filename)
        self.assertIn("same mountainside", peach.prompt)
        self.assertIn("no apple remains", peach.prompt)
        self.assertIn("no snow, ice", peach.prompt)

    def test_selected_image_specs_keeps_requested_canonical_order(self) -> None:
        selected = selected_image_specs([
            "olympic_barbell_field.png",
            "peach_grassy_mountainside.png",
        ])
        self.assertEqual([spec.filename for spec in selected], [
            "peach_grassy_mountainside.png",
            "olympic_barbell_field.png",
        ])

    def test_dry_run_does_not_require_force_when_outputs_already_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / IMAGE_SPECS[0].filename).write_bytes(b"existing")
            stdout = io.StringIO()
            with patch.object(sys, "argv", ["seedream", "--output-dir", str(output_dir), "--dry-run"]):
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(main(), 0)
            self.assertIn("apple_snowy_mountainside.png", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.external import ROOT as REPOSITORY_ROOT

# Set this before transformers/huggingface_hub is imported by Wan/CHAI.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class GroundedSam2Segmenter:
    """Grounding DINO + SAM2.1 video mask generation used by Chorus SRD."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._models: tuple[Any, Any, Any] | None = None

    def _paths(self) -> tuple[Path, Path, Path, Path]:
        root = Path(self.config.get("root", "external/grounded-sam2")).expanduser()
        if not root.is_absolute():
            root = (REPOSITORY_ROOT / root).resolve()
        checkpoint_dir = Path(self.config.get("checkpoint_dir", "models/Grounded-SAM-2-v1.0"))
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = (REPOSITORY_ROOT / checkpoint_dir).resolve()
        sam_config = root / "sam2" / self.config.get("sam_config", "configs/sam2.1/sam2.1_hiera_l.yaml")
        sam_checkpoint = checkpoint_dir / "checkpoints" / self.config.get("sam_checkpoint", "sam2.1_hiera_large.pt")
        gdino_config = root / "grounding_dino" / "groundingdino" / "config" / self.config.get(
            "groundingdino_config", "GroundingDINO_SwinT_OGC.py"
        )
        gdino_checkpoint = checkpoint_dir / "gdino_checkpoints" / self.config.get(
            "groundingdino_checkpoint", "groundingdino_swint_ogc.pth"
        )
        return sam_config, sam_checkpoint, gdino_config, gdino_checkpoint

    def _load(self):
        if self._models is not None:
            return self._models
        sam_config, sam_checkpoint, gdino_config, gdino_checkpoint = self._paths()
        root = sam_config.parents[3]
        missing = [str(path) for path in (sam_config, sam_checkpoint, gdino_config, gdino_checkpoint) if not path.exists()]
        if missing:
            raise FileNotFoundError("Grounded-SAM-2 files are missing: " + ", ".join(missing))
        import sys
        # Grounded-SAM-2 ships GroundingDINO as a nested project.  Put both
        # import roots first so an unrelated editable ``groundingdino``
        # installation cannot supply Python modules while this pinned checkout
        # supplies (or lacks) its matching CUDA extension.
        for import_root in (root, root / "grounding_dino"):
            value = str(import_root)
            if value in sys.path:
                sys.path.remove(value)
            sys.path.insert(0, value)
        hf_endpoint = self.config.get("hf_endpoint")
        if hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", str(hf_endpoint))
            try:
                import huggingface_hub.constants as hf_constants
                hf_constants.ENDPOINT = str(hf_endpoint)
            except ImportError:
                pass
        if self.config.get("hf_offline", True):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            try:
                import huggingface_hub.constants as hf_constants
                hf_constants.HF_HUB_OFFLINE = True
                import transformers.utils.hub as transformers_hub
                transformers_hub._is_offline_mode = True
            except ImportError:
                pass
        local_text_encoder = self.config.get("text_encoder_path")
        if local_text_encoder:
            text_path = Path(local_text_encoder).expanduser()
            if not text_path.is_absolute():
                text_path = (REPOSITORY_ROOT / text_path).resolve()
        else:
            cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            snapshots = sorted((cache_root / "hub" / "models--bert-base-uncased" / "snapshots").glob("*"))
            text_path = snapshots[-1] if snapshots else None
        if text_path is None or not text_path.is_dir():
            raise FileNotFoundError("Grounding DINO requires a local bert-base-uncased snapshot")
        os.environ["GROUNDINGDINO_TEXT_ENCODER"] = str(text_path)
        import torch
        from groundingdino.util.inference import load_model
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = str(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        grounding_model = load_model(str(gdino_config), str(gdino_checkpoint), device=device)
        sam_config_name = str(sam_config.relative_to(root / "sam2"))
        video_predictor = build_sam2_video_predictor(sam_config_name, str(sam_checkpoint), device=device)
        image_predictor = SAM2ImagePredictor(build_sam2(sam_config_name, str(sam_checkpoint), device=device))
        self._models = grounding_model, video_predictor, image_predictor
        return self._models

    @staticmethod
    def _write_frames(video_path: Path, directory: Path) -> tuple[list[Path], int, int]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open source video for Grounded-SAM-2: {video_path}")
        paths: list[Path] = []
        width = height = 0
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            path = directory / f"{index:05d}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"Cannot write Grounded-SAM-2 frame: {path}")
            paths.append(path)
            index += 1
        capture.release()
        if not paths:
            raise RuntimeError(f"Source video has no frames: {video_path}")
        return paths, height, width

    def segment(self, video_path: Path, phrase: str) -> tuple[np.ndarray, dict[str, Any]]:
        import sys
        sam_config_path = self._paths()[0]
        repo_root = sam_config_path.parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import torch
        from torchvision.ops import box_convert
        from groundingdino.util.inference import load_image, predict

        grounding_model, video_predictor, image_predictor = self._load()
        phrase = phrase.strip().rstrip(".")
        if not phrase:
            raise ValueError("Grounded-SAM-2 requires a non-empty source object phrase")
        device = str(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        box_threshold = float(self.config.get("box_threshold", 0.35))
        text_threshold = float(self.config.get("text_threshold", 0.25))
        with tempfile.TemporaryDirectory(prefix="chorus-grounded-sam2-") as temp_dir:
            frame_dir = Path(temp_dir) / "frames"
            frame_dir.mkdir()
            frame_paths, height, width = self._write_frames(video_path, frame_dir)
            image_source, image = load_image(str(frame_paths[0]))
            boxes, confidences, _labels = predict(
                model=grounding_model,
                image=image,
                caption=phrase + ".",
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device,
            )
            if boxes.numel() == 0:
                raise RuntimeError(f"Grounding DINO found no object for phrase: {phrase!r}")
            boxes = boxes * torch.tensor([width, height, width, height])
            input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
            image_predictor.set_image(image_source)
            masks, _scores, _logits = image_predictor.predict(
                point_coords=None, point_labels=None, box=input_boxes, multimask_output=False
            )
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            state = video_predictor.init_state(video_path=str(frame_dir), offload_video_to_cpu=True)
            for object_id, box in enumerate(input_boxes, start=1):
                video_predictor.add_new_points_or_box(state, frame_idx=0, obj_id=object_id, box=box)
            segments: dict[int, np.ndarray] = {}
            for frame_idx, object_ids, mask_logits in video_predictor.propagate_in_video(state):
                frame_mask = np.zeros((height, width), dtype=np.bool_)
                for object_index, _object_id in enumerate(object_ids):
                    frame_mask |= (mask_logits[object_index] > 0).cpu().numpy().squeeze().astype(bool)
                segments[int(frame_idx)] = frame_mask
            if len(segments) != len(frame_paths):
                raise RuntimeError(f"SAM2 propagated {len(segments)} of {len(frame_paths)} frames")
            pixel_mask = np.stack([segments[index] for index in range(len(frame_paths))])
        metadata = {
            "backend": "grounded_sam2",
            "release": self.config.get("release", "v1.0"),
            "commit": self.config.get("commit", "dd4c514"),
            "sam_checkpoint": self.config.get("sam_checkpoint", "sam2.1_hiera_large.pt"),
            "groundingdino_checkpoint": self.config.get("groundingdino_checkpoint", "groundingdino_swint_ogc.pth"),
            "phrase": phrase,
            "boxes": int(len(input_boxes)),
            "box_confidence_max": round(float(confidences.max().item()), 4),
            "pixel_edit_fraction": round(float(pixel_mask.mean()), 4),
        }
        return pixel_mask, metadata

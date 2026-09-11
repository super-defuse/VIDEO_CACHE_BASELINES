from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from core.protocol import ModelAdapter
from core.types import Generation, Request
from .teacache import TeaCacheController


class Wan21Adapter(ModelAdapter):
    model_name = "Wan2.1"

    def __init__(self, config):
        super().__init__(config)
        self._delegate = None
        self._native_dense_c0_dir: Path | None = None
        self._native_dense_c0_runtime = None
        self._wan21_native_runtime = None

    def enable_native_dense_c0(self, video_dir: Path) -> None:
        """Use the resident generic Wan pipeline with native Dense C0."""
        if self._delegate is not None:
            raise RuntimeError("native Dense C0 must be configured before loading Wan")
        self._native_dense_c0_dir = Path(video_dir).resolve()

    def set_native_dense_c0_output_dir(self, video_dir: Path) -> None:
        if self._native_dense_c0_dir is None:
            raise RuntimeError("native Dense C0 is not enabled")
        self._native_dense_c0_dir = Path(video_dir).resolve()

    def _load_delegate(self):
        if self._delegate is None:
            from methods.chai.adapters.wan_t2v import WanT2VAdapter
            config = dict(self.config)
            if self._native_dense_c0_dir is not None:
                config["native_dense_c0"] = True
            self._delegate = WanT2VAdapter(config)
        return self._delegate

    def prepare(self):
        self._load_delegate().prepare()

    def reset_hooks(self):
        delegate = self._load_delegate()
        delegate.state.set_cached_tokens(None, False)

    def generate(self, request: Request, *, fast=False, cached_latent=None, capture=False, method="baseline"):
        if method == "teacache":
            return self._generate_teacache(request)
        if method == "nirvana_vid":
            return self._generate_nirvana(request, cached_latent=cached_latent, capture=capture)
        if method in {"chorus", "chorus_no_tgaa", "chorus_mbase_only",
                      "chorus_tgaa_joint_gamma_1_6", "chorus_tgaa_joint_gamma_2_1",
                      "chorus_tgaa_joint_gamma_2_0"}:
            return self._generate_chorus(request, cached_latent=cached_latent, capture=capture, method=method)
        if method == "chorus2":
            return self._generate_chorus2(request, cached_latent=cached_latent, capture=capture)
        if method == "carnator_boundary_b5":
            return self._generate_carnator(request, cached_latent=cached_latent, capture=capture)
        delegate = self._load_delegate()
        result, dit_forward_s = self._timed_delegate_generate(
            delegate, request, fast=fast, cached_latent=cached_latent, capture=capture
        )
        extra = {**result.extra, "dit_forward_s": dit_forward_s}
        return Generation(result.video, result.latent, result.latent_step, result.extra.get("executed_steps"), extra=extra)

    @staticmethod
    def _timed_delegate_generate(delegate: Any, request: Request, *, fast: bool,
                                 cached_latent: Any, capture: bool) -> tuple[Any, float]:
        """Generate through the delegate while timing CUDA-synchronized DiT forwards.

        The dense Wan2.1 pipeline does not expose per-forward timing, but the
        replay experiment contract requires ``dit_forward_s`` on every dense
        run. Wrapping ``model.forward`` counts each conditional/unconditional
        pass, matching the hand-rolled loops of the other Wan2.1 methods.
        """
        import time

        delegate._load()
        pipeline = delegate.pipeline
        assert pipeline is not None
        torch = delegate.torch
        original = pipeline.model.forward
        elapsed = 0.0

        def timed_forward(*args: Any, **kwargs: Any) -> Any:
            nonlocal elapsed
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                elapsed += time.perf_counter() - started

        pipeline.model.forward = timed_forward
        try:
            result = delegate.generate(
                request.prompt, request.seed, fast=fast,
                cached_latent=cached_latent, capture=capture,
            )
        finally:
            pipeline.model.forward = original
        return result, elapsed

    def _wan21_runtime(self):
        """The carnator-shaped runtime around the resident Wan2.1 pipeline."""

        from carnator.adapters import wan21_native

        if self._wan21_native_runtime is None:
            self._wan21_native_runtime = wan21_native.build_wan21_runtime(
                self._load_delegate(), self.config
            )
        return self._wan21_native_runtime

    def _request_cache_root(self) -> Path:
        """Directory the replay runner assigned for this request's cache."""

        root = getattr(self, "request_cache_dir", None)
        if root is None:
            raise RuntimeError("Wan2.1 cache methods require an assigned request cache directory")
        return Path(root)

    @staticmethod
    def _ensure_model_on_device(pipeline: Any) -> None:
        if getattr(pipeline, "init_on_cpu", False):
            pipeline.model.to(pipeline.device)

    def _generate_native_dense_c0(self, request: Request) -> Generation | None:
        if self._native_dense_c0_dir is None:
            return None
        from carnator.adapters import wan_ti2v_5b as native

        delegate = self._load_delegate()
        delegate._load()
        pipeline = delegate.pipeline
        assert pipeline is not None
        if self._native_dense_c0_runtime is None:
            profiler = native.CudaModuleProfiler()
            profiler.attach(pipeline.model)
            self._native_dense_c0_runtime = native.NativeRuntime(
                torch=delegate.torch,
                wan=native.bootstrap_wan(),
                pipe=pipeline,
                cfg=pipeline.config,
                device=pipeline.device,
                profiler=profiler,
            )
        args = native.build_parser().parse_args([
            "--no-env-setup", "--model-dir", str(self.config["model_path"]),
            "--device", str(self.config["device"]), "--height", str(self.config["height"]),
            "--width", str(self.config["width"]), "--frames", str(self.config["frames"]),
            "--fps", str(self.config.get("fps", 24)), "--steps", str(self.config["baseline_steps"]),
            "--seed", str(request.seed), "--scheduler", str(self.config["solver"]),
            "--flow-shift", str(self.config["shift"]), "--guidance-scale", str(self.config["guidance_scale"]),
            "--outer-attn-backend", "dense", "--ffn-backend", "fp", "--disable-progress",
        ])
        video_path = self._native_dense_c0_dir / f"{request.request_index:04d}_{request.uuid}.mp4"
        cache_dir = self._native_dense_c0_dir / ".dense-c0" / request.uuid
        if video_path.exists():
            raise FileExistsError(f"native Dense C0 output already exists: {video_path}")
        import shutil

        try:
            stage = native.run_native_t2v_stage(
                self._native_dense_c0_runtime,
                args,
                label="dense_c0",
                prompt=request.prompt,
                out_video=video_path,
                cache_dir=cache_dir,
            )
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)
            try:
                cache_dir.parent.rmdir()
            except OSError:
                pass
        if not video_path.is_file():
            raise RuntimeError(f"native Dense C0 did not write {video_path}")
        timing = dict(stage.get("timing", {}))
        dit_forward_s = timing.get("dit_forward_seconds")
        if dit_forward_s is None:
            raise RuntimeError("native Dense C0 did not report DiT forward time")
        return Generation(
            video=None,
            video_path=video_path,
            executed_steps=int(stage["completed_steps"]),
            extra={
                "executed_steps": int(stage["completed_steps"]),
                "restart_step": 0,
                "source_trajectory": False,
                "cache_steps": [10],
            "dit_forward_s": float(dit_forward_s),
            "dense_c0": {"version": "wan21-dense-c0-v1", "steps": int(args.steps),
                             "executor": "native.run_native_t2v_stage",
                             "dtype": str(pipeline.param_dtype)},
                "native_dense_c0_timing": timing,
            },
        )

    def _generate_chorus(self, request: Request, *, cached_latent: Any, capture: bool,
                         method: str = "chorus") -> Generation:
        """Run Chorus full reuse, SRD, and dense refinement on native Wan."""
        delegate = self._load_delegate()
        delegate._load()
        pipeline = delegate.pipeline
        assert pipeline is not None
        if self.config["solver"] != "unipc":
            raise ValueError("Chorus Wan adapter currently requires solver=unipc")

        import math
        import random
        import time
        from contextlib import contextmanager

        import torch
        from methods.chorus.llm import extract_prompt_diff
        from methods.chorus.mask import build_edit_mask
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        self._ensure_model_on_device(pipeline)

        chorus_cfg = dict(self.config.get("chorus", {}))
        if method == "chorus_tgaa_joint_gamma_1_6":
            probe_cfg = chorus_cfg.get("joint_gamma_1_6", {})
            chorus_cfg["gamma_k"] = float(probe_cfg.get("gamma_k", 1.30))
            chorus_cfg["gamma_o"] = float(probe_cfg.get("gamma_o", 1.30))
        elif method == "chorus_tgaa_joint_gamma_2_1":
            probe_cfg = chorus_cfg.get("joint_gamma_2_1", {})
            chorus_cfg["gamma_k"] = float(probe_cfg.get("gamma_k", 1.6))
            chorus_cfg["gamma_o"] = float(probe_cfg.get("gamma_o", 1.6))
        elif method == "chorus_tgaa_joint_gamma_2_0":
            probe_cfg = chorus_cfg.get("joint_gamma_2_0", {})
            chorus_cfg["gamma_k"] = float(probe_cfg.get("gamma_k", 2.0))
            chorus_cfg["gamma_o"] = float(probe_cfg.get("gamma_o", 2.0))
        stages = getattr(self, "chorus_stage_boundaries", None) or chorus_cfg.get("stages", {})
        steps = int(self.config["baseline_steps"])
        k1 = int(stages.get("k1", 10))
        k2 = int(stages.get("k2", 40))
        if not 0 <= k1 < k2 <= steps:
            raise ValueError(f"Invalid Chorus boundaries: K1={k1}, K2={k2}, N={steps}")
        is_source = cached_latent is None
        if not is_source and (not isinstance(cached_latent, dict) or
                              cached_latent.get("format") != "chorus-trajectory-v2" or
                              "trajectory_paths" not in cached_latent):
            raise ValueError("Chorus cache is not a per-step trajectory; clear the old chorus namespace")
        if is_source and not capture:
            raise RuntimeError("Chorus cache miss must be handled before generation")

        frame_num = int(self.config["frames"])
        size = (int(self.config["width"]), int(self.config["height"]))
        target_shape = (
            pipeline.vae.model.z_dim,
            (frame_num - 1) // pipeline.vae_stride[0] + 1,
            size[1] // pipeline.vae_stride[1],
            size[0] // pipeline.vae_stride[2],
        )
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) /
            (pipeline.patch_size[1] * pipeline.patch_size[2]) * target_shape[1] / pipeline.sp_size
        ) * pipeline.sp_size
        seed = request.seed if request.seed >= 0 else random.randint(0, 2**63 - 1)
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        if not pipeline.t5_cpu:
            pipeline.text_encoder.model.to(pipeline.device)
            context = pipeline.text_encoder([request.prompt], pipeline.device)
            context_null = pipeline.text_encoder([pipeline.sample_neg_prompt], pipeline.device)
        else:
            context = [item.to(pipeline.device) for item in pipeline.text_encoder([request.prompt], torch.device("cpu"))]
            context_null = [item.to(pipeline.device) for item in pipeline.text_encoder([pipeline.sample_neg_prompt], torch.device("cpu"))]

        if is_source:
            initial = torch.randn(*target_shape, dtype=torch.float32, device=pipeline.device, generator=generator)
            trajectories: list[Any] | None = [initial.detach().cpu()] if capture else None
            latents = [initial]
            start = 0
        else:
            trajectories = None
            source_trajectory = [Path(item) for item in cached_latent["trajectory_paths"]]
            if len(source_trajectory) != steps + 1 or any(not path.is_file() for path in source_trajectory):
                raise ValueError(f"Chorus source trajectory is incomplete; expected {steps + 1} per-step files")
            # Resume from the completed state before scheduler step K1, as in
            # Carnator's resume_start_step convention.
            latents = [self._load_chorus_state(source_trajectory[k1], pipeline.device)]
            start = k1

        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=pipeline.num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        scheduler.set_timesteps(steps, device=pipeline.device, shift=float(self.config["shift"]))
        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}
        latent_edit_mask = None
        mask_metadata: dict[str, Any] | None = None
        llm_diff: dict[str, Any] | None = None
        tgaa_patches: list[tuple[Any, Any]] = []
        srd_patches: list[tuple[Any, Any]] = []
        srd_state: dict[str, Any] = {
            "active": False,
            "tgaa_active": False,
            "step": 0,
            "total_steps": steps,
            "tgaa_calls": 0,
        }
        llm_s = sam_s = 0.0
        dit_forward_s = 0.0
        tgaa_enabled = method in {"chorus", "chorus_mbase_only", "chorus_tgaa_joint_gamma_1_6",
                                  "chorus_tgaa_joint_gamma_2_1", "chorus_tgaa_joint_gamma_2_0"}
        if not is_source:
            source_prompt = request.cache_prompt or request.prompt
            llm_started = time.perf_counter()
            prompt_diff_override = chorus_cfg.get("prompt_diff_override")
            if prompt_diff_override is not None:
                if not isinstance(prompt_diff_override, dict):
                    raise ValueError("Chorus prompt_diff_override must be a mapping")
                llm_diff = dict(prompt_diff_override)
                llm_diff.setdefault("backend", "manual_override")
                llm_diff.setdefault("latency_s", 0.0)
            else:
                llm_diff = extract_prompt_diff(source_prompt, request.prompt)
            llm_s = time.perf_counter() - llm_started
            # Keep the original Chorus entity selection unchanged. The
            # no-TGAA ablation prefers removed tokens so LLM-added scene nouns
            # cannot change its edit region in addition to removing TGAA.
            source_entities = (
                (llm_diff.get("removed_tokens") or llm_diff.get("source_entities") or [])
                if method != "chorus" else
                (llm_diff.get("source_entities") or llm_diff.get("removed_tokens") or [])
            )
            target_entities = llm_diff.get("target_entities") or []
            unique_entities = list(dict.fromkeys(str(item).strip() for item in source_entities if str(item).strip()))
            phrase = " ".join(unique_entities[:4]).strip()
            if not phrase:
                raise RuntimeError("LLM prompt diff did not return a source object for Grounded-SAM-2")
            source_video = getattr(self, "chorus_source_video_path", None)
            if not source_video:
                raise RuntimeError("Chorus source video is unavailable for Grounded-SAM-2")
            sam_started = time.perf_counter()
            latent_edit_mask, mask_metadata = build_edit_mask(
                Path(source_video), phrase,
                latent_shape=(target_shape[1], target_shape[2], target_shape[3]),
                patch_size=(int(pipeline.patch_size[1]), int(pipeline.patch_size[2])),
                config={**chorus_cfg.get("grounded_sam2", {}),
                        "mask_dilation_edit": chorus_cfg.get("mask_dilation_edit", 8),
                        "mask_dilation_see": chorus_cfg.get("mask_dilation_see", 14)},
            )
            sam_s = time.perf_counter() - sam_started
            update_mask_name = "Mbase" if method == "chorus_mbase_only" else "Medit"
            latent_edit_mask = self._select_chorus_update_masks(
                latent_edit_mask, update_mask_name
            )
            mask_metadata = {**mask_metadata, "update_mask": update_mask_name}
            if tgaa_enabled:
                diff_indices = self._differential_token_indices(
                    pipeline.text_encoder, request.prompt, target_entities
                )
                if not diff_indices:
                    raise RuntimeError(
                        "TGAA could not map the LLM differential tokens to target prompt tokens"
                    )
                tgaa_patches = self._install_native_tgaa(
                    pipeline.model, chorus_cfg, diff_indices, srd_state
                )
            srd_patches = self._install_native_srd(pipeline.model, latent_edit_mask, srd_state)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(pipeline.model, "no_sync", noop_no_sync)
        try:
            with torch.amp.autocast("cuda", dtype=pipeline.param_dtype), torch.no_grad(), no_sync():
                for index, timestep_value in enumerate(scheduler.timesteps[start:], start=start):
                    stage2_active = not is_source and k1 <= index < k2
                    if not is_source and index == k2:
                        # SRD only supplies partial model outputs in Stage 2. Do
                        # not carry those regional derivatives into dense Stage
                        # 3; retain the global sigma index for the same schedule.
                        self._reset_chorus_scheduler_history(scheduler)
                        srd_state["scheduler_history_reset"] = True
                    srd_state["active"] = stage2_active
                    srd_state["tgaa_active"] = tgaa_enabled and stage2_active
                    srd_state["step"] = index
                    timestep = torch.stack([timestep_value])
                    torch.cuda.synchronize()
                    model_started = time.perf_counter()
                    cond = pipeline.model(latents, t=timestep, **arg_c)[0]
                    uncond = pipeline.model(latents, t=timestep, **arg_null)[0]
                    torch.cuda.synchronize()
                    dit_forward_s += time.perf_counter() - model_started
                    noise_pred = uncond + float(self.config["guidance_scale"]) * (cond - uncond)
                    sample = scheduler.step(
                        noise_pred.unsqueeze(0), timestep_value, latents[0].unsqueeze(0),
                        return_dict=False, generator=generator,
                    )[0].squeeze(0)
                    # Stage 2 is the only stage that reuses source latent
                    # features. Stage 3 is deliberately full computation so
                    # it can repair discontinuities introduced by SRD.
                    if stage2_active:
                        source_state = self._load_chorus_state(
                            source_trajectory[index + 1], sample.device, sample.dtype
                        )
                        edit = latent_edit_mask["latent_edit"].to(device=sample.device, dtype=sample.dtype)
                        sample = self._fuse_chorus_latent(sample, source_state, edit)
                    latents = [sample]
                    if trajectories is not None:
                        trajectories.append(sample.detach().cpu())
                video = pipeline.vae.decode(latents)[0]
        finally:
            self._restore_native_srd(srd_patches)
            self._restore_native_tgaa(tgaa_patches)

        extra: dict[str, Any] = {
            "chorus_stage": "source_capture" if is_source else "target_reuse",
            "chorus_stages": {"steps": steps, "k1": k1, "k2": k2,
                              "stage1": [0, k1], "stage2": [k1, k2], "stage3": [k2, steps]},
            "executed_steps": steps if is_source else steps - k1,
            "dit_forward_s": dit_forward_s,
        }
        if is_source:
            payload = None
            if trajectories is not None:
                payload = {"trajectory": trajectories, "steps": steps, "k1": k1, "k2": k2,
                           "target_shape": list(target_shape), "seed": seed}
            return Generation(video=video, latent=payload, latent_step=k1, executed_steps=steps, extra=extra)
        preprocess_s = llm_s + sam_s
        extra.update({"chorus_llm": llm_diff, "chorus_mask": mask_metadata,
                      "chorus_preprocess_s": preprocess_s,
                      "chorus_llm_s": llm_s, "chorus_sam_s": sam_s,
                      "excluded_latency_s": preprocess_s,
                      "chorus_tgaa": {
                          "enabled": tgaa_enabled,
                          "gamma_k": chorus_cfg.get("gamma_k", 1.5),
                          "gamma_o": chorus_cfg.get("gamma_o", 1.5),
                          "schedule": "linear_step_decay_approximation",
                          "paper_exact_schedule": "not specified",
                          "scope": "stage2",
                          "blocks": "all",
                          "calls": int(srd_state.get("tgaa_calls", 0)),
                      },
                      "chorus_scheduler": {
                          "stage3_history_reset": bool(srd_state.get("scheduler_history_reset", False)),
                          "stage3_history_reset_step": k2,
                          "stage3_history_reset_semantics": "clear UniPC derivative history while preserving the global timestep index",
                      },
                      "chorus_cache": {
                          "source_trajectory_steps": steps,
                          "format": "per_step_manifest",
                          "stage2_latent_fusion": True,
                          "latent_fusion_update_mask": mask_metadata.get("update_mask", "Medit"),
                          "latent_fusion_from_step": k1,
                          "latent_fusion_through_step": k2 - 1,
                          "latent_fusion_semantics": "source latent fused only during Stage 2; Stage 3 disables latent reuse",
                          "srd": {
                              "enabled": True,
                              "update_mask": mask_metadata.get("update_mask", "Medit"),
                              "edit_fraction": mask_metadata.get("edit_fraction") if mask_metadata else None,
                              "see_fraction": mask_metadata.get("see_fraction") if mask_metadata else None,
                              "self_attention": f"{mask_metadata.get('update_mask', 'Medit')} queries / Msee keys-values",
                              "cross_attention": f"{mask_metadata.get('update_mask', 'Medit')} queries",
                              "ffn": f"{mask_metadata.get('update_mask', 'Medit')} tokens",
                          },
                      },
                  })
        return Generation(video=video, latent=None, latent_step=k1,
                          executed_steps=steps - k1, extra=extra)

    @staticmethod
    def _select_chorus_update_masks(mask_bundle: dict[str, Any], update_mask: str) -> dict[str, Any]:
        """Select the region updated by Chorus while retaining Msee context."""
        if update_mask not in {"Mbase", "Medit"}:
            raise ValueError(f"unsupported Chorus update mask: {update_mask}")
        token_key = "token_base" if update_mask == "Mbase" else "token_edit"
        latent_key = "latent_base" if update_mask == "Mbase" else "latent_edit"
        missing = [key for key in (token_key, latent_key, "token_see") if key not in mask_bundle]
        if missing:
            raise KeyError(f"Chorus mask bundle is missing {missing} for {update_mask}")
        selected = dict(mask_bundle)
        selected["token_edit"] = mask_bundle[token_key]
        selected["latent_edit"] = mask_bundle[latent_key]
        selected["update_mask"] = update_mask
        return selected

    @staticmethod
    def _fuse_chorus_latent(sample: Any, source_state: Any, edit_mask: Any) -> Any:
        """Keep target values in the active update region and source elsewhere.

        This mirrors Carnator's ``torch.where(latent_medit, latent,
        source_latent)`` while accepting the unbatched Wan latent layout
        ``[C, T, H, W]`` used by this adapter.
        """
        import torch

        if sample.ndim != 4 or source_state.ndim != 4:
            raise ValueError(
                "Chorus latent fusion expects [C,T,H,W] tensors, got "
                f"sample={tuple(sample.shape)}, source={tuple(source_state.shape)}"
            )
        if sample.shape != source_state.shape:
            raise ValueError(
                "Chorus source/target latent shapes differ: "
                f"sample={tuple(sample.shape)}, source={tuple(source_state.shape)}"
            )
        if edit_mask.ndim == 3:
            edit_mask = edit_mask.unsqueeze(0)
        if edit_mask.ndim != 4 or edit_mask.shape[1:] != sample.shape[1:]:
            raise ValueError(
                "Chorus latent mask is not aligned with the latent: "
                f"mask={tuple(edit_mask.shape)}, latent={tuple(sample.shape)}"
            )
        edit_mask = edit_mask.to(device=sample.device, dtype=torch.bool)
        return torch.where(edit_mask, sample, source_state.to(device=sample.device, dtype=sample.dtype))

    @staticmethod
    def _reset_chorus_scheduler_history(scheduler: Any) -> None:
        """Restart UniPC multistep warmup without changing the global step."""
        if hasattr(scheduler, "model_outputs"):
            scheduler.model_outputs = [None] * len(scheduler.model_outputs)
        if hasattr(scheduler, "timestep_list"):
            scheduler.timestep_list = [None] * len(scheduler.timestep_list)
        if hasattr(scheduler, "lower_order_nums"):
            scheduler.lower_order_nums = 0
        if hasattr(scheduler, "last_sample"):
            scheduler.last_sample = None
        if hasattr(scheduler, "this_order"):
            scheduler.this_order = 1

    @staticmethod
    def _load_chorus_state(path: Path, device: Any, dtype: Any | None = None) -> Any:
        import torch

        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        return state.to(device=device, dtype=dtype or state.dtype)

    @staticmethod
    def _differential_token_indices(text_encoder: Any, prompt: str, entities: list[Any]) -> list[int]:
        """Map target entities to actual T5 token positions for TGAA."""
        tokenizer = getattr(getattr(text_encoder, "tokenizer", None), "tokenizer", None)
        if tokenizer is None:
            return []
        try:
            encoded = tokenizer(prompt, add_special_tokens=True, truncation=True,
                                max_length=int(text_encoder.text_len), return_tensors="pt")
            ids = encoded.input_ids[0].tolist()
            pieces = [str(tokenizer.decode([token])).strip().lower() for token in ids]
        except Exception:
            return []
        terms = [str(item).strip().lower() for item in entities if str(item).strip()]
        return [index for index, piece in enumerate(pieces)
                if piece and any(term in piece or piece in term for term in terms)]

    def _install_native_tgaa(
        self,
        model: Any,
        chorus_cfg: dict[str, Any],
        differential_tokens: list[int],
        state: dict[str, Any],
    ) -> list[tuple[Any, Any]]:
        from wan.modules.attention import flash_attention

        patches: list[tuple[Any, Any]] = []
        configured_blocks = chorus_cfg.get("tgaa_blocks", "all")
        if configured_blocks in (None, "all"):
            blocks = set(range(len(model.blocks)))
        else:
            blocks = set(int(item) for item in configured_blocks)
        gamma_k = float(chorus_cfg.get("gamma_k", 1.5))
        gamma_o = float(chorus_cfg.get("gamma_o", 1.5))
        token_indices = list(differential_tokens)
        if not token_indices:
            raise ValueError("TGAA requires at least one mapped differential token")
        for index, block in enumerate(model.blocks):
            if index not in blocks:
                continue
            attention = block.cross_attn
            original = attention.forward

            def patched_cross(x, context_value, context_lens, *, _attention=attention, _original=original):
                if not state.get("tgaa_active", False):
                    return _original(x, context_value, context_lens)
                batch, heads, head_dim = x.size(0), _attention.num_heads, _attention.head_dim
                query = _attention.norm_q(_attention.q(x)).view(batch, -1, heads, head_dim)
                # ``context_value`` is shared by every transformer block.
                # TGAA must change this block's attention key only; an
                # in-place scale on a view would compound into later blocks.
                key = _attention.norm_k(_attention.k(context_value)).view(
                    batch, -1, heads, head_dim
                ).clone()
                value = _attention.v(context_value).view(batch, -1, heads, head_dim)
                # The paper leaves F_k(t,m) and F_o(t,m) unspecified. Keep
                # the existing step-only linear decay as an explicit
                # approximation rather than inventing a matching-score law.
                progress = min(1.0, max(0.0, float(state.get("step", 0)) /
                                       max(1, int(state.get("total_steps", 1)) - 1)))
                scale_k = 1.0 + (gamma_k - 1.0) * (1.0 - progress)
                scale_o = 1.0 + (gamma_o - 1.0) * (1.0 - progress)
                valid = [item for item in token_indices if item < key.shape[1]]
                state["tgaa_calls"] = int(state.get("tgaa_calls", 0)) + 1
                if valid:
                    key[:, valid] *= scale_k
                output = flash_attention(query, key, value, k_lens=context_lens)
                return _attention.o(output.flatten(2)) * scale_o

            attention.forward = patched_cross
            patches.append((attention, original))
        return patches

    def _install_native_srd(self, model: Any, masks: dict[str, Any], state: dict[str, Any]) -> list[tuple[Any, Any]]:
        """Patch Wan blocks to compute active update queries with Msee context."""
        import torch

        patches: list[tuple[Any, Any]] = []
        token_edit = masks["token_edit"].bool().cpu()
        token_see = masks["token_see"].bool().cpu()

        def indices_for(length: int, device: Any) -> tuple[Any, Any]:
            edit = torch.zeros(length, dtype=torch.bool)
            see = torch.zeros(length, dtype=torch.bool)
            count = min(length, token_edit.numel())
            edit[:count] = token_edit[:count]
            see[:count] = token_see[:count]
            see |= edit
            edit_idx = edit.nonzero(as_tuple=False).flatten().to(device=device)
            see_idx = see.nonzero(as_tuple=False).flatten().to(device=device)
            if edit_idx.numel() == 0 or see_idx.numel() == 0:
                raise RuntimeError("SRD mask has no active tokens for the Wan sequence")
            return edit_idx, see_idx

        def get_indices(length: int, device: Any) -> tuple[Any, Any]:
            cached = state.get("indices")
            if cached is None or cached[0].numel() > length or cached[0].device != device:
                cached = indices_for(length, device)
                state["indices"] = cached
            return cached

        for block in model.blocks:
            original = block.forward

            def patched_block(x, e, seq_lens, grid_sizes, freqs, context, context_lens,
                              *, _block=block, _original=original):
                if not state.get("active", False):
                    return _original(x, e, seq_lens, grid_sizes, freqs, context, context_lens)
                edit_idx, see_idx = get_indices(x.shape[1], x.device)
                return self._native_srd_block(
                    _block, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
                    edit_idx, see_idx,
                )

            block.forward = patched_block
            patches.append((block, original))

        # Keep Wan's native head dense. The sparse block path leaves non-edit
        # hidden states unchanged, but the head must still produce a full
        # model output so the scheduler does not record zeros outside Medit.
        return patches

    @staticmethod
    def _native_srd_block(
        block: Any,
        x: Any,
        e: Any,
        seq_lens: Any,
        grid_sizes: Any,
        freqs: Any,
        context: Any,
        context_lens: Any,
        edit_idx: Any,
        see_idx: Any,
    ) -> Any:
        import torch

        from wan.modules.attention import flash_attention
        from wan.modules.model import rope_apply

        # Wan2.1 chunks (block.modulation + e) on dim 1 into [B,1,C] pieces
        # that broadcast over every token.
        with torch.amp.autocast("cuda", dtype=torch.float32):
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                block.modulation + e
            ).chunk(6, dim=1)
        attn = block.self_attn
        batch, heads, head_dim = x.shape[0], attn.num_heads, attn.head_dim
        norm = (block.norm1(x).float() * (1 + scale_msa) + shift_msa).type_as(x)
        query = attn.norm_q(attn.q(norm)).view(batch, -1, heads, head_dim)
        key = attn.norm_k(attn.k(norm)).view(batch, -1, heads, head_dim)
        value = attn.v(norm).view(batch, -1, heads, head_dim)
        query = rope_apply(query, grid_sizes, freqs).index_select(1, edit_idx)
        key = rope_apply(key, grid_sizes, freqs).index_select(1, see_idx)
        value = value.index_select(1, see_idx)
        q_lens = torch.full((batch,), edit_idx.numel(), dtype=torch.int32, device=x.device)
        k_lens = torch.full((batch,), see_idx.numel(), dtype=torch.int32, device=x.device)
        y = flash_attention(query, key, value, q_lens=q_lens, k_lens=k_lens,
                            window_size=attn.window_size)
        x_edit = x.index_select(1, edit_idx) + attn.o(y.flatten(2)) * gate_msa
        x_edit = x_edit + block.cross_attn(block.norm3(x_edit), context, context_lens)
        ffn_input = block.norm2(x_edit).float() * (1 + c_scale_msa) + c_shift_msa
        y = block.ffn(ffn_input)
        x_edit = x_edit + y * c_gate_msa
        output = x.clone()
        output[:, edit_idx] = x_edit.to(dtype=output.dtype)
        return output

    @staticmethod
    def _restore_native_srd(patches: list[tuple[Any, Any]]) -> None:
        for module, original in reversed(patches):
            module.forward = original

    @staticmethod
    def _restore_native_tgaa(patches: list[tuple[Any, Any]]) -> None:
        for module, original in reversed(patches):
            module.forward = original

    def _generate_teacache(self, request: Request) -> Generation:
        delegate = self._load_delegate()
        delegate._load()
        config = self.config["teacache"]
        steps = int(self.config["baseline_steps"])
        controller = TeaCacheController(
            steps=steps,
            threshold=float(config["relative_l1_threshold"]),
            max_skip_steps=int(config["max_skip_steps"]),
        )
        restore = controller.patch(delegate.pipeline.model, branches=2)
        try:
            video = delegate.pipeline.t2v(
                input_prompt=request.prompt,
                size=(int(self.config["width"]), int(self.config["height"])),
                frame_num=int(self.config["frames"]),
                shift=float(self.config["shift"]),
                sample_solver=self.config["solver"],
                sampling_steps=steps,
                guide_scale=float(self.config["guidance_scale"]),
                seed=request.seed,
                offload_model=False,
            )
        finally:
            restore()
        return Generation(
            video=video,
            executed_steps=len(controller.executed_steps),
            extra={"executed_steps": len(controller.executed_steps), "skipped_steps": steps - len(controller.executed_steps),
                   "dit_forward_s": controller.forward_seconds},
        )

    def _generate_nirvana(self, request: Request, *, cached_latent: Any, capture: bool) -> Generation:
        """Resume the native Wan schedule from a cached diffusion latent."""
        delegate = self._load_delegate()
        delegate._load()
        pipeline = delegate.pipeline
        assert pipeline is not None
        if cached_latent is None and not capture:
            raise RuntimeError("NIRVANA-VID cache miss must be handled before generation")
        if self.config["solver"] != "unipc":
            raise ValueError("NIRVANA-VID Wan adapter currently requires solver=unipc")

        import math
        import random
        from contextlib import contextmanager

        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        torch = delegate.torch
        self._ensure_model_on_device(pipeline)
        steps = int(self.config["baseline_steps"])
        reuse_start = int(getattr(self, "nirvana_reuse_start_step", self.config["nirvana_vid"]["reuse_start_step"]))
        if not 0 < reuse_start < steps:
            raise ValueError("nirvana_vid.reuse_start_step must be within the sampling schedule")
        cache_steps = tuple(sorted({int(step) for step in self.config["nirvana_vid"].get("cache_steps", [reuse_start])}))
        if not cache_steps or any(not 0 < step < steps for step in cache_steps):
            raise ValueError("nirvana_vid.cache_steps must be non-empty steps within the sampling schedule")
        start = reuse_start if cached_latent is not None else 0
        frame_num = int(self.config["frames"])
        size = (int(self.config["width"]), int(self.config["height"]))
        target_shape = (
            pipeline.vae.model.z_dim,
            (frame_num - 1) // pipeline.vae_stride[0] + 1,
            size[1] // pipeline.vae_stride[1],
            size[0] // pipeline.vae_stride[2],
        )
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) /
            (pipeline.patch_size[1] * pipeline.patch_size[2]) * target_shape[1] / pipeline.sp_size
        ) * pipeline.sp_size
        seed = request.seed if request.seed >= 0 else random.randint(0, 2**63 - 1)
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        if not pipeline.t5_cpu:
            pipeline.text_encoder.model.to(pipeline.device)
            context = pipeline.text_encoder([request.prompt], pipeline.device)
            context_null = pipeline.text_encoder([pipeline.sample_neg_prompt], pipeline.device)
        else:
            context = [item.to(pipeline.device) for item in pipeline.text_encoder([request.prompt], torch.device("cpu"))]
            context_null = [item.to(pipeline.device) for item in pipeline.text_encoder([pipeline.sample_neg_prompt], torch.device("cpu"))]
        latents = [cached_latent.to(pipeline.device)] if cached_latent is not None else [
            torch.randn(*target_shape, dtype=torch.float32, device=pipeline.device, generator=generator)
        ]
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=pipeline.num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        scheduler.set_timesteps(steps, device=pipeline.device, shift=float(self.config["shift"]))
        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}
        captured: dict[int, Any] = {}
        dit_forward_s = 0.0

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(pipeline.model, "no_sync", noop_no_sync)
        with torch.amp.autocast("cuda", dtype=pipeline.param_dtype), torch.no_grad(), no_sync():
            for index, timestep_value in enumerate(scheduler.timesteps[start:], start=start):
                timestep = torch.stack([timestep_value])
                torch.cuda.synchronize()
                model_started = time.perf_counter()
                cond = pipeline.model(latents, t=timestep, **arg_c)[0]
                uncond = pipeline.model(latents, t=timestep, **arg_null)[0]
                torch.cuda.synchronize()
                dit_forward_s += time.perf_counter() - model_started
                noise_pred = uncond + float(self.config["guidance_scale"]) * (cond - uncond)
                sample = scheduler.step(
                    noise_pred.unsqueeze(0), timestep_value, latents[0].unsqueeze(0), return_dict=False, generator=generator
                )[0]
                latents = [sample.squeeze(0)]
                completed_step = index + 1
                if capture and completed_step in cache_steps:
                    captured[completed_step] = latents[0].detach().cpu()
            video = pipeline.vae.decode(latents)[0]
        return Generation(
            video=video,
            latent=captured if capture else None,
            latent_step=max(cache_steps),
            executed_steps=steps - start,
            extra={"executed_steps": steps - start, "restart_step": start, "source_trajectory": cached_latent is not None,
                   "cache_steps": list(cache_steps), "dit_forward_s": dit_forward_s,
                   "dense_c0": ({"version": "wan21-dense-c0-v1", "steps": steps} if cached_latent is None else None)},
        )

    def _generate_chorus2(self, request: Request, *, cached_latent: Any, capture: bool) -> Generation:
        """Chorus2 on Wan2.1: capture source routing masks, replay them sparsely.

        The source runs the full dense 50-step schedule while capturing
        per-layer block-attention routing; the target re-runs its own complete
        schedule with every block served by SpargeAttn from completed step
        ``sparse_start_step + 1``. No source latent is reused.
        """
        from carnator.adapters import wan21_native
        from carnator.adapters import wan_ti2v_5b as native
        from carnator.outer_sparse import (
            OuterSparseAttentionHooks,
            OuterSparseCaptureHooks,
            OuterSparseExecutor,
            OuterSparseState,
            SourceBlockMaskCache,
            cfg_branches,
        )

        runtime = self._wan21_runtime()
        steps = int(self.config["baseline_steps"])
        chorus2_cfg = dict(self.config.get("chorus2", {}))
        sparse_start = int(chorus2_cfg.get("sparse_start_step", 10))
        if not 0 <= sparse_start < steps:
            raise ValueError(f"chorus2.sparse_start_step must be in [0, {steps})")
        layer_refresh_size = int(chorus2_cfg.get("layer_refresh_size", 8))
        outer_backend = str(chorus2_cfg.get("outer_backend", "sparge-sage2"))
        layers = list(range(len(runtime.pipe.model.blocks)))
        branch_count = cfg_branches(float(self.config["guidance_scale"]))
        is_source = cached_latent is None
        if is_source and not capture:
            raise RuntimeError("Chorus2 cache miss must be handled before generation")
        if not is_source and (not isinstance(cached_latent, dict)
                              or cached_latent.get("format") != "wan21-chorus2-source/v1"):
            raise ValueError("Chorus2 cache payload is not a Wan2.1 source mask directory")

        cache_root = self._request_cache_root()
        args = wan21_native.build_wan21_args(self.config, seed=request.seed, reuse_start_step=0)
        execution_dtype = native.outer_sparse_execution_dtype(args)
        if is_source:
            mask_dir = cache_root / "chorus2_source" / "block_masks"
            stage_dir = cache_root / "chorus2_source"
            out_video = stage_dir / f"{request.uuid}.source.mp4"
            cache = SourceBlockMaskCache(
                mask_dir,
                native.outer_mask_config(args),
                source_run_id=request.uuid,
                cfg_branch_count=branch_count,
                create=True,
                pack_format=SourceBlockMaskCache.CHORUS2_PACKED_FORMAT,
            )
            state = OuterSparseState(cfg_branch_count=branch_count, start_completed_step=sparse_start + 1)
            with OuterSparseCaptureHooks(
                runtime.pipe.model, runtime.wan["model_module"], state, cache, layers,
                execution_dtype=execution_dtype,
            ):
                stage = native.run_native_t2v_stage(
                    runtime, args, label="chorus2_source", prompt=request.prompt,
                    out_video=out_video, cache_dir=stage_dir / "dense_cache",
                    outer_sparse_state=state,
                )
            missing = [
                (step, branch, layer)
                for step in range(sparse_start + 1, steps + 1)
                for branch in (("cond", "uncond") if branch_count == 2 else ("cond",))
                for layer in layers
                if (step, branch, layer) not in cache.entries
            ]
            if missing:
                raise RuntimeError(f"Chorus2 source mask capture is incomplete: {len(missing)} missing entries")
            payload = {
                "format": "wan21-chorus2-source/v1",
                "mask_dir": str(mask_dir),
                "steps": steps,
                "sparse_start_step": sparse_start,
                "seed": int(request.seed),
            }
            return Generation(
                video=None,
                video_path=out_video,
                latent=payload,
                latent_step=steps,
                executed_steps=steps,
                extra={
                    "chorus2_stage": "source_capture",
                    "executed_steps": steps,
                    "dit_forward_s": float(stage["timing"]["dit_forward_seconds"]),
                    "chorus2": {"layers": layers, "sparse_start_step": sparse_start,
                                "captured_entries": len(cache.entries)},
                },
            )

        mask_dir = Path(str(cached_latent["mask_dir"]))
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Chorus2 source mask cache is missing: {mask_dir}")
        target_dir = cache_root / "chorus2_target"
        out_video = target_dir / f"{request.uuid}.mp4"
        cache = SourceBlockMaskCache(mask_dir, native.outer_mask_config(args))
        cache.expect_cfg_branches(branch_count)
        state = OuterSparseState(cfg_branch_count=branch_count, start_completed_step=sparse_start + 1)
        executor = OuterSparseExecutor(
            cache, outer_backend, execution_dtype=execution_dtype,
            layer_refresh_size=layer_refresh_size,
        )
        # Decode the captured masks to host memory only. Wan2.1's VAE stride
        # gives 109,120 tokens for a 1280x704x121 input, so a single
        # [heads, q_blocks, k_blocks] bool mask is 16.6 MiB and the full ~2400
        # entry set is ~40 GiB of host RAM; keeping them resident on the GPU
        # pushed the target stage past 80 GiB. Bit-packing shrinks the on-disk
        # masks 8x but not this decoded host copy. SourceBlockMaskCache.get()
        # copies each mask to the query device per call, so replay stays
        # functional.
        cache_prepare = cache.prepare()
        with OuterSparseAttentionHooks(
            runtime.pipe.model, runtime.wan["model_module"], state, executor, layers,
        ):
            stage = native.run_native_t2v_stage(
                runtime, args, label="chorus2_target", prompt=request.prompt,
                out_video=out_video, cache_dir=target_dir / "dense_cache",
                outer_sparse_state=state,
            )
        stage["cache_prepare"] = cache_prepare
        outer = stage.get("outer_sparse", {})
        expected_runs = len(layers) * (steps - sparse_start) * branch_count
        if outer.get("runs") != expected_runs or outer.get("dense_fallback_count") != 0:
            raise RuntimeError(
                f"Chorus2 target sparse coverage is wrong: runs={outer.get('runs')} "
                f"expected={expected_runs}, fallbacks={outer.get('dense_fallback_count')}"
            )
        if stage.get("completed_steps") != steps:
            raise RuntimeError(f"Chorus2 target completed {stage.get('completed_steps')} steps, expected {steps}")
        return Generation(
            video=None,
            video_path=out_video,
            latent=None,
            latent_step=None,
            executed_steps=steps,
            extra={
                "chorus2_stage": "target_reuse",
                "executed_steps": steps,
                "dit_forward_s": float(stage["timing"]["dit_forward_seconds"]),
                "cache_prepare_s": float((cache_prepare or {}).get("elapsed_seconds", 0.0)),
                "chorus2": {
                    "layers": layers,
                    "sparse_start_step": sparse_start,
                    "backend": outer_backend,
                    "layer_refresh_size": layer_refresh_size,
                    "runs": int(outer.get("runs", 0)),
                    "dense_fallback_count": int(outer.get("dense_fallback_count", 0)),
                },
            },
        )

    def _generate_carnator(self, request: Request, *, cached_latent: Any, capture: bool) -> Generation:
        """Carnator B5 on Wan2.1, algorithm-identical to the Wan2.2 native path.

        Source: dense 50-step run with the multi-noun Early Signature probe at
        completed step 5, the full latent trajectory, fusion-mask template,
        outer routing masks from completed step 11, and the source video.
        Target: per-noun mask selection (the source-only noun of a one-noun
        substitution), a dense 5-step target probe kept for timing parity with
        the Wan2.2 path, resume from the source step-9 latent, soft B5 boundary
        latent injection, region sparse blocks (10-20) with the middle
        attention mask, outer sparse attention (0-9, 21-29), and w8a8 FFN.
        """
        from contextlib import ExitStack

        from carnator.adapters import wan21_native
        from carnator.adapters import wan_ti2v_5b as native
        from carnator.early_signature import materialize_per_noun_inference_mask_bundle
        from carnator.masks import load_mask_bundle
        from carnator.outer_sparse import (
            OuterSparseAttentionHooks,
            OuterSparseCaptureHooks,
            OuterSparseExecutor,
            OuterSparseState,
            SourceBlockMaskCache,
            cfg_branches,
        )
        from carnator.quantized_ffn import QuantizedFFNHooks

        runtime = self._wan21_runtime()
        # The canonical Carnator target ran sparge-sage2 + w8a8 and the soft B5
        # boundary blend; every other constant equals the shared parser defaults.
        carnator_config = self.config.get("carnator_b5", {})
        ffn_backend = str(carnator_config.get("ffn_backend", "w8a8"))
        outer_backend = str(carnator_config.get("outer_backend", "sparge-sage2"))
        injection_mode = str(carnator_config.get("latent_injection_mode", "boundary-blend"))
        alpha_function = str(carnator_config.get("boundary_alpha_function", "smoothstep"))
        # The parser defaults are dense/fp/hard, so every Carnator constant that
        # differs has to be passed explicitly or the target silently degrades.
        args = wan21_native.build_wan21_args(
            self.config, seed=request.seed,
            extra_flags=[
                "--ffn-backend", ffn_backend,
                "--outer-attn-backend", outer_backend,
                "--latent-injection-mode", injection_mode,
                "--boundary-alpha-function", alpha_function,
            ],
        )
        steps = int(args.steps)
        branch_count = cfg_branches(float(args.guidance_scale))
        execution_dtype = native.outer_sparse_execution_dtype(args)
        is_source = cached_latent is None
        if is_source and not capture:
            raise RuntimeError("Carnator cache miss must be handled before generation")
        if not is_source and (not isinstance(cached_latent, dict)
                              or cached_latent.get("format") != "wan21-carnator-source/v1"):
            raise ValueError("Carnator cache payload is not a Wan2.1 source cache directory")

        cache_root = self._request_cache_root()
        if is_source:
            source_dir = cache_root / "carnator_source"
            out_video = source_dir / f"{request.uuid}.source.mp4"
            nouns = wan21_native.extract_request_nouns(request)
            outer_mask_dir = source_dir / "outer_block_masks"
            source_cache = SourceBlockMaskCache(
                outer_mask_dir,
                native.outer_mask_config(args),
                source_run_id=request.uuid,
                cfg_branch_count=branch_count,
                create=True,
                pack_format=SourceBlockMaskCache.PACKED_FORMAT,
            )
            outer_state = OuterSparseState(
                cfg_branch_count=branch_count, start_completed_step=args.reuse_start_step + 1,
            )
            with OuterSparseCaptureHooks(
                runtime.pipe.model, runtime.wan["model_module"], outer_state, source_cache,
                native.block_mask_capture_layers(args), execution_dtype=execution_dtype,
            ):
                stage = wan21_native.run_wan21_carnator_source(
                    runtime, args, prompt=request.prompt, nouns=nouns,
                    cache_dir=source_dir, out_video=out_video, outer_sparse_state=outer_state,
                )
            payload = {
                "format": "wan21-carnator-source/v1",
                "cache_dir": str(source_dir),
                "steps": steps,
                "reuse_start_step": int(args.reuse_start_step),
                "seed": int(request.seed),
            }
            return Generation(
                video=None,
                video_path=out_video,
                latent=payload,
                latent_step=steps,
                executed_steps=steps,
                extra={
                    "carnator_stage": "source_capture",
                    "executed_steps": steps,
                    "dit_forward_s": float(stage["timing"]["dit_forward_seconds"]),
                    "carnator": {"nouns": list(nouns), "source_video": str(out_video)},
                },
            )

        source_dir = Path(str(cached_latent["cache_dir"]))
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Carnator source cache is missing: {source_dir}")
        signature = {"signature_path": str(source_dir / "online_color_signature.json")}
        target_dir = cache_root / "carnator_target"
        out_video = target_dir / f"{request.uuid}.mp4"
        target_nouns = wan21_native.extract_request_nouns(request)
        selection = materialize_per_noun_inference_mask_bundle(
            signature,
            target_nouns=target_nouns,
            source_fusion_mask_dir=source_dir / "fusion_masks",
            output_dir=target_dir / "inference_masks",
        )
        mask_bundle = load_mask_bundle(Path(selection["mask_bundle_dir"]), None)
        source_manifest = native.read_json(source_dir / "manifest.json")
        source_trajectory = {
            int(item["step_index"]): Path(item["path"])
            for item in source_manifest.get("trajectory", [])
            if item.get("kind") == "post_scheduler"
        }
        resume_latent_path = source_trajectory[args.reuse_start_step - 1]
        # Timing-parity probe: the same 5-step dense prefix the Wan2.2 path
        # runs before reuse.  Its result is audit metadata; the replay decision
        # and the mask selection above are already fixed.
        probe = wan21_native.run_wan21_target_probe(
            runtime, args, prompt=request.prompt, nouns=target_nouns,
            output_dir=target_dir / "target_probe",
            source_signature=signature,
            probe_seed=int(self.config.get("seed", 20260218)),
            selection=selection,
        )

        block_mask_cache = SourceBlockMaskCache(source_dir / "outer_block_masks", native.outer_mask_config(args))
        block_mask_cache.expect_cfg_branches(branch_count)
        sparse_state = native.NativeSparseState(
            model_module=runtime.wan["model_module"],
            mask_bundle=mask_bundle,
            layers=native.parse_int_ranges(args.sparse_layers),
            attention_mask_cache=block_mask_cache if args.middle_attention_mask else None,
            attention_mask_execution_dtype=execution_dtype if args.middle_attention_mask else None,
            cfg_branch_count=branch_count,
        )
        outer_state = OuterSparseState(
            cfg_branch_count=branch_count, start_completed_step=args.reuse_start_step + 1,
        )
        outer_executor = OuterSparseExecutor(
            block_mask_cache, outer_backend, execution_dtype=execution_dtype,
        )
        cache_prepare = block_mask_cache.prepare(device=runtime.device)
        with ExitStack() as stack:
            ffn_state = None
            if args.ffn_backend == "w8a8":
                ffn_hooks = QuantizedFFNHooks(runtime.pipe.model)
                ffn_state = stack.enter_context(ffn_hooks)
                ffn_hooks.attach_profiler(runtime.profiler)
            stack.enter_context(wan21_native.Wan21SparseBlockHooks(runtime.pipe.model, sparse_state))
            stack.enter_context(OuterSparseAttentionHooks(
                runtime.pipe.model, runtime.wan["model_module"], outer_state, outer_executor,
                native.parse_int_ranges(args.outer_sparse_layers),
            ))
            stage = native.run_native_t2v_stage(
                runtime, args, label="carnator_target", prompt=request.prompt,
                out_video=out_video, cache_dir=target_dir / "dense_cache",
                resume_latent_path=resume_latent_path,
                resume_start_step=args.reuse_start_step,
                source_trajectory=source_trajectory,
                mask_bundle=mask_bundle,
                sparse_state=sparse_state,
                outer_sparse_state=outer_state,
                ffn_state=ffn_state,
            )
        stage["cache_prepare"] = cache_prepare
        injection = stage.get("masked_injection", {})
        if injection.get("mode") != args.latent_injection_mode or injection.get("calls") != steps - args.reuse_start_step:
            raise RuntimeError(f"Carnator target latent injection is incomplete: {injection}")
        sparse_summary = stage.get("sparse_blocks", {})
        if sparse_summary.get("dense_fallback_count") != 0:
            raise RuntimeError("Carnator target region sparse blocks fell back to dense")
        probe_steps = int(probe["completed_steps"])
        probe_dit_forward_s = float(probe["dit_forward_seconds"])
        reuse_steps = steps - args.reuse_start_step
        return Generation(
            video=None,
            video_path=out_video,
            latent=None,
            latent_step=None,
            executed_steps=probe_steps + reuse_steps,
            extra={
                "carnator_stage": "target_reuse",
                "executed_steps": probe_steps + reuse_steps,
                "dit_forward_s": float(stage["timing"]["dit_forward_seconds"]) + probe_dit_forward_s,
                "cache_prepare_s": float((cache_prepare or {}).get("elapsed_seconds", 0.0)),
                "probe_dit_forward_s": probe_dit_forward_s,
                "probe_steps": probe_steps,
                "probe_comparison": probe["comparison"],
                "carnator": {
                    "inference_mask_selection": {k: v for k, v in selection.items() if k != "mask_bundle_dir"},
                    "target_nouns": list(target_nouns),
                    "target_probe": {"completed_steps": probe_steps, "dit_forward_s": probe_dit_forward_s},
                    "masked_injection": injection,
                    "sparse_blocks": {"calls": sparse_summary.get("sparse_call_count"),
                                      "layers": sparse_summary.get("layers")},
                    "outer_sparse": {"runs": stage.get("outer_sparse", {}).get("runs"),
                                     "backend": outer_backend},
                    "ffn_backend": args.ffn_backend,
                },
            },
        )

    # Pointer payloads reference an on-disk source cache directory instead of
    # carrying tensors; they serialize as plain JSON.
    POINTER_FORMATS = ("wan21-chorus2-source/v1", "wan21-carnator-source/v1")

    def save_latent(self, latent, path: Path):
        if isinstance(latent, dict) and latent.get("format") in self.POINTER_FORMATS:
            import json

            path.write_text(json.dumps(latent, indent=2) + "\n", encoding="utf-8")
            return
        if isinstance(latent, dict) and "trajectory" in latent and "steps" in latent:
            import json
            import shutil

            delegate = self._load_delegate()
            delegate._load()
            step_dir = path.with_suffix(".steps")
            if step_dir.exists():
                shutil.rmtree(step_dir)
            step_dir.mkdir(parents=True, exist_ok=True)
            trajectory = list(latent["trajectory"])
            files: list[str] = ["latents_initial.pt"]
            delegate.torch.save(trajectory[0], step_dir / files[0])
            for index, state in enumerate(trajectory[1:]):
                name = f"latents_step_{index:03d}.pt"
                delegate.torch.save(state, step_dir / name)
                files.append(name)
            manifest = {
                "format": "chorus-trajectory-v2",
                "steps": int(latent["steps"]),
                "k1": int(latent["k1"]),
                "k2": int(latent["k2"]),
                "target_shape": list(latent["target_shape"]),
                "seed": int(latent["seed"]),
                "trajectory_files": files,
            }
            path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            return
        self._load_delegate().save_latent(latent, path)

    def load_latent(self, path: Path):
        import json

        path = Path(path)
        if path.is_dir():
            manifest_path = path / "manifest.json"
            try:
                native_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid native Chorus source cache: {manifest_path}") from exc
            steps = int(native_manifest.get("steps", -1))
            trajectory = {
                int(item["step_index"]): Path(item["path"])
                for item in native_manifest.get("trajectory", [])
                if item.get("kind") == "post_scheduler"
            }
            initial = path / "latents_initial.pt"
            if steps < 1 or sorted(trajectory) != list(range(steps)) or not initial.is_file():
                raise ValueError(f"native Chorus source trajectory is incomplete: {path}")
            paths = [initial, *(trajectory[index] for index in range(steps))]
            if any(not item.is_file() for item in paths):
                raise ValueError(f"native Chorus source trajectory is missing files: {path}")
            return {
                "format": "chorus-trajectory-v2",
                "steps": steps,
                "k1": 5,
                "k2": 30,
                "target_shape": list(native_manifest.get("latent_shape", [])),
                "seed": int(native_manifest.get("seed", -1)),
                "trajectory_paths": paths,
                "source_format": "native-carnator-source-cache-v1",
            }
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return self._load_delegate().load_latent(path)
        if manifest.get("format") in self.POINTER_FORMATS:
            return manifest
        if manifest.get("format") != "chorus-trajectory-v2":
            return self._load_delegate().load_latent(path)
        step_dir = path.with_suffix(".steps")
        files = list(manifest.get("trajectory_files", []))
        paths = [step_dir / name for name in files]
        if len(paths) != int(manifest.get("steps", -1)) + 1 or any(not item.is_file() for item in paths):
            raise ValueError(f"Chorus trajectory sidecar is incomplete for {path}")
        return {
            **manifest,
            "trajectory_paths": paths,
        }

    def save_video(self, video, path: Path):
        delegate = self._load_delegate()
        delegate._load()
        # Wan2.1 utils ships cache_video only (no save_video).
        from wan.utils.utils import cache_video

        # Wan returns [C, T, H, W]; its writer expects a batch dimension.
        cache_video(video[None], str(path), fps=24, nrow=1, normalize=True, value_range=(-1, 1))

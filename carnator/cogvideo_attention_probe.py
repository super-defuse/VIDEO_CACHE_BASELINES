"""CogVideoX attention probing for Carnator fusion mask."""

from __future__ import annotations

import torch
from typing import Any, Dict, Tuple, Optional


class CogVideoAttentionProbe:
    """Capture self and cross attention scores from CogVideoX joint attention."""

    def __init__(
        self,
        transformer: Any,
        layer_index: int = 15,
        text_tokens: Optional[int] = None,
        target_step: Optional[int] = None,
    ):
        """
        Args:
            transformer: CogVideoX transformer model
            layer_index: Which layer to probe (default 15, middle layer)
            text_tokens: Number of text tokens (will be detected if None)
            target_step: Only capture at this denoising step (None = capture first)
        """
        self.transformer = transformer
        self.layer_index = int(layer_index)
        self.text_tokens = text_tokens
        self.target_step = target_step
        self.current_step = 0
        self.captured: Dict[str, Any] = {}
        self.handle = None

    def __enter__(self):
        """Install hook to capture attention scores."""
        target_block = self.transformer.transformer_blocks[self.layer_index]
        self.handle = target_block.attn1.register_forward_hook(self._capture_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False

    def _capture_hook(self, module, input, output):
        """Hook function to capture hidden states during forward pass."""
        # Input tuple: (hidden_states, encoder_hidden_states, ...)
        # For CogVideoX joint attention, hidden_states is the concatenated [text, video] sequence

        # Increment step counter
        self.current_step += 1

        # Skip if we already captured or if not at target step
        if 'hidden_states' in self.captured:
            return

        if self.target_step is not None and self.current_step != self.target_step:
            return

        if len(input) > 0:
            # Capture the hidden states (concatenated text + video tokens)
            hidden_states = input[0]
            self.captured['hidden_states'] = hidden_states.detach().cpu()

        if len(input) > 1 and input[1] is not None:
            # Capture encoder hidden states (if separate)
            encoder_hidden_states = input[1]
            self.captured['encoder_hidden_states'] = encoder_hidden_states.detach().cpu()

        # Store reference for later extraction
        self.captured['module'] = module
        if output is not None:
            self.captured['output'] = output

    def extract_scores_from_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Manually compute attention scores to extract self and cross components.

        Args:
            hidden_states: Video latent states [B, T_video, D]
            encoder_hidden_states: Text embeddings [B, T_text, D]
            rotary_emb: Rotary position embeddings

        Returns:
            self_scores: [B, H, T_video, T_video] - Video→Video attention
            cross_scores: [B, H, T_video, T_text] - Video→Text attention
        """
        if self.layer_index not in range(len(self.transformer.transformer_blocks)):
            raise ValueError(f"Layer {self.layer_index} out of range")

        attn = self.transformer.transformer_blocks[self.layer_index].attn1

        # Get text token count
        text_tokens = encoder_hidden_states.shape[1]
        if self.text_tokens is None:
            self.text_tokens = text_tokens

        # Concatenate text and video tokens
        concat_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        batch, total_tokens, _ = concat_states.shape

        # Compute Q, K, V
        query = attn.to_q(concat_states)
        key = attn.to_k(concat_states)

        # Reshape to multi-head format
        head_dim = query.shape[-1] // attn.heads
        query = query.view(batch, total_tokens, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch, total_tokens, attn.heads, head_dim).transpose(1, 2)

        # Apply normalization if present
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply rotary embeddings to video tokens only
        if rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query[:, :, text_tokens:] = apply_rotary_emb(query[:, :, text_tokens:], rotary_emb)
            key[:, :, text_tokens:] = apply_rotary_emb(key[:, :, text_tokens:], rotary_emb)

        # Compute attention scores: [B, H, T_total, T_total]
        scale = head_dim ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        # Extract video query scores (rows corresponding to video tokens)
        video_query_scores = scores[:, :, text_tokens:, :]  # [B, H, T_video, T_total]

        # Split into self (V→V) and cross (V→T) attention
        self_scores = video_query_scores[:, :, :, text_tokens:]  # [B, H, T_video, T_video]
        cross_scores = video_query_scores[:, :, :, :text_tokens]  # [B, H, T_video, T_text]

        return self_scores, cross_scores

    def extract_scores_from_captured(self, text_tokens: int, rotary_emb: Any = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract attention scores from captured hidden states during forward pass.

        Args:
            text_tokens: Number of text tokens
            rotary_emb: Rotary position embeddings

        Returns:
            self_scores: [B, H, T_video, T_video] - Video→Video attention
            cross_scores: [B, H, T_video, T_text] - Video→Text attention
        """
        if 'hidden_states' not in self.captured:
            raise RuntimeError("No hidden states captured. Make sure the probe hook was triggered.")

        hidden_states = self.captured['hidden_states']

        # For CogVideoX joint attention, hidden_states is concatenated [text, video] sequence
        # Shape should be [B, T_total, D] where T_total = T_text + T_video

        # Use empty encoder_hidden_states since CogVideoX uses joint attention
        encoder_hidden_states = hidden_states[:, :text_tokens, :]  # Text tokens
        video_states = hidden_states[:, text_tokens:, :]  # Video tokens

        return self.extract_scores_from_forward(
            hidden_states=video_states,
            encoder_hidden_states=encoder_hidden_states,
            rotary_emb=rotary_emb,
        )

    def get_scores(self) -> Dict[str, torch.Tensor]:
        """Return captured scores."""
        return self.captured


def compute_self_contrast_score(
    self_scores: torch.Tensor,
    token_grid_shape: Tuple[int, int, int],
    noun_seed_indices: Optional[torch.Tensor] = None,
    background_seed_indices: Optional[torch.Tensor] = None,
    top_p: float = 0.02,
    background_p: float = 0.02,
) -> torch.Tensor:
    """
    Compute self-contrast score from self-attention scores.

    Args:
        self_scores: [B, H, T_video, T_video] - Self-attention scores
        token_grid_shape: (T, H, W) - 3D token grid dimensions
        noun_seed_indices: Indices of noun seed tokens (if None, computed from cross-attention)
        background_seed_indices: Indices of background seed tokens
        top_p: Fraction of tokens to use as noun seeds
        background_p: Fraction of tokens to use as background seeds

    Returns:
        self_contrast: [T, H, W] - 3D self-contrast score
    """
    # Average over attention heads
    self_scores_avg = self_scores.mean(dim=1)  # [B, T_video, T_video]

    batch, num_tokens, _ = self_scores_avg.shape

    # If seed indices not provided, use top-p selection
    if noun_seed_indices is None or background_seed_indices is None:
        # Use mean attention to all other tokens as a proxy
        token_importance = self_scores_avg[0].sum(dim=0)  # [T_video]

        # Select top-p tokens as noun seeds
        k_noun = max(1, int(num_tokens * top_p))
        noun_seed_indices = torch.topk(token_importance, k_noun).indices

        # Select bottom-p tokens as background seeds
        k_bg = max(1, int(num_tokens * background_p))
        background_seed_indices = torch.topk(token_importance, k_bg, largest=False).indices

    # Compute attention from each token to noun seeds and background seeds
    noun_attention = self_scores_avg[:, :, noun_seed_indices].max(dim=-1)[0]  # [B, T_video]
    bg_attention = self_scores_avg[:, :, background_seed_indices].max(dim=-1)[0]  # [B, T_video]

    # Self-contrast = noun_attention - background_attention
    self_contrast_flat = noun_attention - bg_attention  # [B, T_video]

    # Reshape to 3D token grid
    T, H, W = token_grid_shape
    if self_contrast_flat.shape[1] != T * H * W:
        raise ValueError(f"Token count mismatch: expected {T*H*W}, got {self_contrast_flat.shape[1]}")

    self_contrast_3d = self_contrast_flat.reshape(batch, T, H, W)

    return self_contrast_3d[0]  # Return first batch element [T, H, W]


def compute_cross_attention_score(
    cross_scores: torch.Tensor,
    token_grid_shape: Tuple[int, int, int],
    noun_token_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute cross-attention score (video→text attention).

    Args:
        cross_scores: [B, H, T_video, T_text] - Cross-attention scores
        token_grid_shape: (T, H, W) - 3D token grid dimensions
        noun_token_indices: Text token indices corresponding to the noun

    Returns:
        cross_score: [T, H, W] - 3D cross-attention score
    """
    # Average over attention heads
    cross_scores_avg = cross_scores.mean(dim=1)  # [B, T_video, T_text]

    if noun_token_indices is not None:
        # Use specified noun tokens
        noun_cross = cross_scores_avg[:, :, noun_token_indices].max(dim=-1)[0]  # [B, T_video]
    else:
        # Use maximum cross-attention over all text tokens
        noun_cross = cross_scores_avg.max(dim=-1)[0]  # [B, T_video]

    # Reshape to 3D token grid
    T, H, W = token_grid_shape
    batch = noun_cross.shape[0]

    if noun_cross.shape[1] != T * H * W:
        raise ValueError(f"Token count mismatch: expected {T*H*W}, got {noun_cross.shape[1]}")

    cross_score_3d = noun_cross.reshape(batch, T, H, W)

    return cross_score_3d[0]  # Return first batch element [T, H, W]


def infer_token_grid_shape(latent_shape: Tuple[int, ...]) -> Tuple[int, int, int]:
    """
    Infer 3D token grid shape from latent shape.

    Args:
        latent_shape: Shape of latent tensor, e.g. [B, C, T, H, W]

    Returns:
        (T, H, W): Token grid dimensions
    """
    if len(latent_shape) == 5:
        # [B, C, T, H, W] format
        _, _, t, h, w = latent_shape
    elif len(latent_shape) == 4:
        # [B, T, H, W] format
        _, t, h, w = latent_shape
    else:
        raise ValueError(f"Unexpected latent shape: {latent_shape}")

    # CogVideoX token grid is same as latent spatial dimensions
    # (already in token space after VAE encoding and patchifying)
    return (t, h, w)

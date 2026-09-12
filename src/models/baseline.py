"""Baseline action-conditioned next-frame model, composed of reusable modules."""

from __future__ import annotations

import dataclasses
import math
from typing import NamedTuple, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(num_channels: int, preferred: int = 16) -> int:
    """Largest divisor of ``preferred`` that also divides ``num_channels``.

    ``nn.GroupNorm`` requires ``num_channels % groups == 0``, which the greatest
    common divisor guarantees for any width. Note this is not the largest valid
    divisor below ``preferred``: 24 channels give 8 groups rather than 12, because
    the count is always kept a divisor of ``preferred`` too.
    """

    return math.gcd(num_channels, preferred)


def _conv_stage(in_channels: int, out_channels: int, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.ReLU(inplace=True),
    )


def _upsample_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.ReLU(inplace=True),
    )


class EncodedFrames(NamedTuple):
    """Per-frame embeddings plus the feature maps of each encoder stage."""

    embedding: torch.Tensor
    stage_features: list[torch.Tensor]


class FrameEncoder(nn.Module):
    """Encode frames to embeddings, exposing stage features for decoder skips."""

    def __init__(
        self,
        image_channels: int = 3,
        embedding_dim: int = 128,
        grid_size: int = 8,
        projected_dim: int | None = None,
        hidden_channels: Sequence[int] = (32, 64),
    ) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.embedding_dim = embedding_dim
        self.grid_size = grid_size
        self.projected_dim = (
            projected_dim if projected_dim is not None else embedding_dim
        )
        self.stage_channels = (*hidden_channels, embedding_dim)

        self.stages = nn.ModuleList(
            _conv_stage(in_channels, out_channels)
            for in_channels, out_channels in zip(
                (image_channels, *self.stage_channels[:-1]), self.stage_channels
            )
        )
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.projection = nn.Linear(
            embedding_dim * grid_size * grid_size,
            self.projected_dim,
        )

    def forward(self, image: torch.Tensor) -> EncodedFrames:
        features = image
        stage_features: list[torch.Tensor] = []
        for stage in self.stages:
            features = stage(features)
            stage_features.append(features)

        pooled = self.pool(features).flatten(start_dim=1)
        return EncodedFrames(self.projection(pooled), stage_features)

    @property
    def output_dim(self) -> int:
        return self.projected_dim


class TemporalEncoder(nn.Module):
    """Summarise a sequence of frame embeddings into a single latent."""

    def __init__(self, input_dim: int, latent_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.GRU(
            input_size=input_dim,
            hidden_size=latent_dim,
            batch_first=True,
        )

    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        _, hidden_state = self.network(frame_embeddings)
        return hidden_state[-1]


class ConditionEncoder(nn.Module):
    """Embed the ego state and the action applied at the last observed frame."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((state, action), dim=-1))


class FrameDecoder(nn.Module):
    """Decode a latent into a residual image bounded to [-1, 1].

    ``skip_channels`` lists encoder stage widths from finest to coarsest spatial
    resolution. Skips are aligned to the highest-resolution blocks, so the finest
    skip feeds the final block, letting texture reach the output without passing
    through the latent bottleneck. The output convolution is zero-initialised, so
    an untrained decoder emits a zero residual.
    """

    def __init__(
        self,
        latent_dim: int,
        image_channels: int = 3,
        initial_size: int = 8,
        base_channels: int = 128,
        min_channels: int = 16,
        max_output_size: int = 256,
        skip_channels: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self.initial_size = initial_size
        self.base_channels = base_channels
        self.input_projection = nn.Linear(
            latent_dim,
            base_channels * initial_size * initial_size,
        )

        num_stages = max(
            1, math.ceil(math.log2(max(max_output_size / initial_size, 1)))
        )
        widths = [base_channels]
        for _ in range(num_stages):
            widths.append(max(min_channels, widths[-1] // 2))

        self.skip_index_per_block = self._align_skips(num_stages, len(skip_channels))
        self.up_blocks = nn.ModuleList(
            _upsample_block(
                in_channels + self._skip_width(block_index, skip_channels),
                out_channels,
            )
            for block_index, (in_channels, out_channels) in enumerate(
                zip(widths[:-1], widths[1:])
            )
        )

        residual_conv = nn.Conv2d(widths[-1], image_channels, kernel_size=3, padding=1)
        nn.init.zeros_(residual_conv.weight)
        nn.init.zeros_(residual_conv.bias)
        self.to_residual = nn.Sequential(residual_conv, nn.Tanh())

    @staticmethod
    def _align_skips(num_stages: int, num_skips: int) -> list[int | None]:
        alignment: list[int | None] = [None] * num_stages
        for skip_index in range(num_skips):
            block_index = num_stages - 1 - skip_index
            if block_index < 0:
                break
            alignment[block_index] = skip_index
        return alignment

    def _skip_width(self, block_index: int, skip_channels: Sequence[int]) -> int:
        skip_index = self.skip_index_per_block[block_index]
        return 0 if skip_index is None else skip_channels[skip_index]

    def forward(
        self,
        latent: torch.Tensor,
        output_size: tuple[int, int],
        skip_features: Sequence[torch.Tensor] = (),
    ) -> torch.Tensor:
        decoded = self.input_projection(latent).reshape(
            latent.shape[0], self.base_channels, self.initial_size, self.initial_size
        )

        target_h, target_w = output_size
        for block_index, block in enumerate(self.up_blocks):
            current_h, current_w = decoded.shape[-2:]
            if current_h < target_h or current_w < target_w:
                decoded = F.interpolate(
                    decoded, scale_factor=2, mode="bilinear", align_corners=False
                )

            skip_index = self.skip_index_per_block[block_index]
            if skip_index is not None and skip_index < len(skip_features):
                decoded = torch.cat(
                    (
                        decoded,
                        F.interpolate(
                            skip_features[skip_index],
                            size=decoded.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ),
                    ),
                    dim=1,
                )

            decoded = block(decoded)

        decoded = F.interpolate(
            decoded, size=output_size, mode="bilinear", align_corners=False
        )
        return self.to_residual(decoded)


@dataclasses.dataclass(frozen=True)
class BaselineModelConfig:
    """Architecture hyperparameters of :class:`BaselineWorldModel`."""

    image_channels: int = 3
    latent_dim: int = 128
    frame_grid_size: int = 8
    encoder_hidden_channels: tuple[int, ...] = (32, 64)
    decoder_initial_size: int = 8
    decoder_base_channels: int = 128
    decoder_min_channels: int = 16
    max_output_size: int = 256


class BaselineWorldModel(nn.Module):
    """Action-conditioned next-frame predictor.

    The decoder predicts a residual that is added to the last observed frame, so a
    zero residual reproduces that frame and the model starts from the copy-last-frame
    baseline. Predictions are returned unclamped; clamp to [0, 1] before rendering or
    computing image metrics.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: BaselineModelConfig = BaselineModelConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.frame_encoder = FrameEncoder(
            image_channels=config.image_channels,
            embedding_dim=config.latent_dim,
            grid_size=config.frame_grid_size,
            projected_dim=config.latent_dim,
            hidden_channels=config.encoder_hidden_channels,
        )
        self.temporal_encoder = TemporalEncoder(
            input_dim=self.frame_encoder.output_dim,
            latent_dim=config.latent_dim,
        )
        self.condition_encoder = ConditionEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=config.latent_dim,
        )
        self.decoder = FrameDecoder(
            latent_dim=config.latent_dim * 2,
            image_channels=config.image_channels,
            initial_size=config.decoder_initial_size,
            base_channels=config.decoder_base_channels,
            min_channels=config.decoder_min_channels,
            max_output_size=config.max_output_size,
            skip_channels=self.frame_encoder.stage_channels,
        )

    @property
    def image_channels(self) -> int:
        return self.config.image_channels

    def _validate_inputs(
        self,
        frame_history: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> None:
        if frame_history.dim() != 5:
            raise ValueError(
                "frame_history must have shape (batch, sequence, channels, height, "
                f"width); got {tuple(frame_history.shape)}"
            )
        batch_size, _, channels, _, _ = frame_history.shape
        if channels != self.image_channels:
            raise ValueError(
                f"Expected {self.image_channels} image channels, got {channels}"
            )
        if state.shape[0] != batch_size or action.shape[0] != batch_size:
            raise ValueError(
                "Batch size mismatch: frame_history has batch "
                f"{batch_size}, state has {state.shape[0]}, action has "
                f"{action.shape[0]}"
            )
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state_dim={self.state_dim}, got state of shape "
                f"{tuple(state.shape)}"
            )
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action_dim={self.action_dim}, got action of shape "
                f"{tuple(action.shape)}"
            )

    def forward(
        self,
        frame_history: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(frame_history, state, action)
        batch_size, sequence_length, channels, height, width = frame_history.shape

        encoded = self.frame_encoder(
            frame_history.reshape(batch_size * sequence_length, channels, height, width)
        )
        history_latent = self.temporal_encoder(
            encoded.embedding.reshape(batch_size, sequence_length, -1)
        )
        condition_latent = self.condition_encoder(state, action)
        dynamics_latent = torch.cat((history_latent, condition_latent), dim=-1)

        last_frame_skips = [
            features.reshape(batch_size, sequence_length, *features.shape[1:])[:, -1]
            for features in encoded.stage_features
        ]
        residual = self.decoder(
            dynamics_latent,
            output_size=(height, width),
            skip_features=last_frame_skips,
        )
        return frame_history[:, -1] + residual


def build_baseline_model(
    state_dim: int,
    action_dim: int,
    config: BaselineModelConfig | None = None,
) -> BaselineWorldModel:
    return BaselineWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config or BaselineModelConfig(),
    )

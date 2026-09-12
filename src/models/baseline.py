from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _safe_num_groups(num_channels: int, preferred: int = 16) -> int:

    for groups in (preferred, 8, 4, 2, 1):
        if groups <= num_channels and num_channels % groups == 0:
            return groups
    return 1


def _conv_stage(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
        nn.GroupNorm(_safe_num_groups(out_channels), out_channels),
        nn.ReLU(inplace=True),
    )


class FrameEncoder(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        embedding_dim: int = 128,
        grid_size: int = 8,
        projected_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.embedding_dim = embedding_dim
        self.grid_size = grid_size
        self.projected_dim = (
            projected_dim if projected_dim is not None else embedding_dim
        )

        stage_channels = [32, 64, embedding_dim]
        self.stages = nn.ModuleList(
            _conv_stage(in_channels, out_channels)
            for in_channels, out_channels in zip(
                [image_channels] + stage_channels[:-1], stage_channels
            )
        )
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.projection = nn.Linear(
            embedding_dim * grid_size * grid_size,
            self.projected_dim,
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return the frame embedding and the per-stage feature maps."""

        features = image
        stage_features: list[torch.Tensor] = []
        for stage in self.stages:
            features = stage(features)
            stage_features.append(features)

        pooled = self.pool(features).flatten(start_dim=1)
        return self.projection(pooled), stage_features

    @property
    def output_dim(self) -> int:
        return self.projected_dim

    @property
    def stage_channels(self) -> list[int]:
        return [32, 64, self.embedding_dim]


class TemporalEncoder(nn.Module):
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
    """Decode a latent into a bounded residual image.

    ``skip_channels`` lists encoder stage widths from finest to coarsest spatial
    resolution; each is fused into the matching upsampling block so that texture
    reaches the output without passing through the latent bottleneck.
    """

    def __init__(
        self,
        latent_dim: int,
        image_channels: int = 3,
        initial_size: int = 8,
        base_channels: int = 128,
        max_output_size: int = 256,
        skip_channels: list[int] | None = None,
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

        channels = [base_channels]
        for _ in range(num_stages):
            channels.append(max(16, channels[-1] // 2))

        self.skip_channels = list(skip_channels or [])
        fused_channels = [0] * num_stages
        for skip_index, skip_width in enumerate(self.skip_channels):
            block_index = num_stages - 1 - skip_index
            if block_index < 0:
                break
            fused_channels[block_index] = skip_width

        self.up_blocks = nn.ModuleList()
        for block_index, (in_channels, out_channels) in enumerate(
            zip(channels[:-1], channels[1:])
        ):
            self.up_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels + fused_channels[block_index],
                        out_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.GroupNorm(_safe_num_groups(out_channels), out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        residual_conv = nn.Conv2d(
            channels[-1], image_channels, kernel_size=3, padding=1
        )
        nn.init.zeros_(residual_conv.weight)
        nn.init.zeros_(residual_conv.bias)
        self.to_residual = nn.Sequential(residual_conv, nn.Tanh())

    def _skip_for_block(
        self,
        block_index: int,
        skip_features: list[torch.Tensor],
    ) -> torch.Tensor | None:
        skip_index = len(self.up_blocks) - 1 - block_index
        if 0 <= skip_index < len(skip_features):
            return skip_features[skip_index]
        return None

    def forward(
        self,
        latent: torch.Tensor,
        output_size: tuple[int, int],
        skip_features: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size = latent.shape[0]
        decoded = self.input_projection(latent)
        decoded = decoded.reshape(
            batch_size, self.base_channels, self.initial_size, self.initial_size
        )

        skip_features = list(skip_features or [])
        target_h, target_w = output_size
        for block_index, block in enumerate(self.up_blocks):
            current_h, current_w = decoded.shape[-2:]
            if current_h < target_h or current_w < target_w:
                decoded = F.interpolate(
                    decoded, scale_factor=2, mode="bilinear", align_corners=False
                )

            skip = self._skip_for_block(block_index, skip_features)
            if skip is not None:
                resized_skip = F.interpolate(
                    skip,
                    size=decoded.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                decoded = torch.cat((decoded, resized_skip), dim=1)

            decoded = block(decoded)

        decoded = F.interpolate(
            decoded, size=output_size, mode="bilinear", align_corners=False
        )
        return self.to_residual(decoded)


class BaselineWorldModel(nn.Module):
    """Action-conditioned next-frame predictor.

    The decoder predicts a residual that is added to the last observed frame, so
    a zero output reproduces that frame. Predictions are returned unclamped;
    clamp to [0, 1] before rendering or computing image metrics.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        image_channels: int = 3,
        latent_dim: int = 128,
        frame_grid_size: int = 8,
        max_output_size: int = 256,
    ) -> None:
        super().__init__()

        self.image_channels = image_channels
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.frame_encoder = FrameEncoder(
            image_channels=image_channels,
            embedding_dim=latent_dim,
            grid_size=frame_grid_size,
            projected_dim=latent_dim,
        )
        self.temporal_encoder = TemporalEncoder(
            input_dim=self.frame_encoder.output_dim,
            latent_dim=latent_dim,
        )
        self.condition_encoder = ConditionEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
        )
        self.decoder = FrameDecoder(
            latent_dim=latent_dim * 2,
            image_channels=image_channels,
            max_output_size=max_output_size,
            skip_channels=self.frame_encoder.stage_channels,
        )

    def forward(
        self,
        frame_history: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        if frame_history.dim() != 5:
            raise ValueError(
                "frame_history must have shape (batch, sequence, channels, height, "
                f"width); got {tuple(frame_history.shape)}"
            )
        batch_size, sequence_length, channels, height, width = frame_history.shape
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

        flattened_history = frame_history.reshape(
            batch_size * sequence_length,
            channels,
            height,
            width,
        )
        encoded_frames, stage_features = self.frame_encoder(flattened_history)
        encoded_frames = encoded_frames.reshape(batch_size, sequence_length, -1)
        history_latent = self.temporal_encoder(encoded_frames)

        condition_latent = self.condition_encoder(state, action)
        dynamics_latent = torch.cat((history_latent, condition_latent), dim=-1)

        skip_features = [
            features.reshape(batch_size, sequence_length, *features.shape[1:])[:, -1]
            for features in stage_features
        ]
        residual = self.decoder(
            dynamics_latent,
            output_size=(height, width),
            skip_features=skip_features,
        )
        return frame_history[:, -1] + residual


def build_baseline_model(
    state_dim: int,
    action_dim: int,
    image_channels: int = 3,
    latent_dim: int = 128,
    frame_grid_size: int = 8,
    max_output_size: int = 256,
) -> BaselineWorldModel:
    return BaselineWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        image_channels=image_channels,
        latent_dim=latent_dim,
        frame_grid_size=frame_grid_size,
        max_output_size=max_output_size,
    )

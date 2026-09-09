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

        self.network = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(_safe_num_groups(32), 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(_safe_num_groups(64), 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embedding_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(_safe_num_groups(embedding_dim), embedding_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
        )
        self.projection = nn.Linear(
            embedding_dim * grid_size * grid_size,
            self.projected_dim,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.network(image)
        flattened = features.flatten(start_dim=1)
        return self.projection(flattened)

    @property
    def output_dim(self) -> int:
        return self.projected_dim


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
    def __init__(
        self,
        latent_dim: int,
        image_channels: int = 3,
        initial_size: int = 8,
        base_channels: int = 128,
        max_output_size: int = 512,
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

        self.up_blocks = nn.ModuleList()
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            self.up_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(_safe_num_groups(out_channels), out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels[-1], image_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        latent: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        batch_size = latent.shape[0]
        decoded = self.input_projection(latent)
        decoded = decoded.reshape(
            batch_size, self.base_channels, self.initial_size, self.initial_size
        )

        target_h, target_w = output_size
        for block in self.up_blocks:
            current_h, current_w = decoded.shape[-2:]
            if current_h < target_h or current_w < target_w:
                decoded = F.interpolate(
                    decoded, scale_factor=2, mode="bilinear", align_corners=False
                )
            decoded = block(decoded)

        decoded = F.interpolate(
            decoded, size=output_size, mode="bilinear", align_corners=False
        )
        return self.to_rgb(decoded)


class BaselineWorldModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        image_channels: int = 3,
        latent_dim: int = 128,
        frame_grid_size: int = 8,
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
        encoded_frames = self.frame_encoder(flattened_history)
        encoded_frames = encoded_frames.reshape(batch_size, sequence_length, -1)
        history_latent = self.temporal_encoder(encoded_frames)

        condition_latent = self.condition_encoder(state, action)
        dynamics_latent = torch.cat((history_latent, condition_latent), dim=-1)

        return self.decoder(dynamics_latent, output_size=(height, width))


def build_baseline_model(
    state_dim: int,
    action_dim: int,
    image_channels: int = 3,
    latent_dim: int = 128,
    frame_grid_size: int = 8,
) -> BaselineWorldModel:
    return BaselineWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        image_channels=image_channels,
        latent_dim=latent_dim,
        frame_grid_size=frame_grid_size,
    )

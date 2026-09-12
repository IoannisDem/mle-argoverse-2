"""Train the baseline action-conditioned next-frame model."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import TypeVar
import dataclasses

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_creation.loader import (
    EpisodeFrameWindowDataset_V1,
    EpisodeFrameWindowDataset_V2,
    Transformations,
    compute_state_standardizer,
    load_episode_path_sequences,
    load_episode_sequences,
)
from models.baseline import build_baseline_model
from train.train_model import TrainDataset, TrainingConfig, train_model


EpisodeT = TypeVar("EpisodeT")


def split_episodes(
    episodes: list[EpisodeT],
    validation_fraction: float,
    seed: int,
) -> tuple[list[EpisodeT], list[EpisodeT]]:
    """Split complete episodes, keeping all windows from an episode together."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(episodes) < 2:
        raise ValueError(
            "At least two episodes are required for a train/validation split"
        )

    shuffled = list(episodes)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, round(len(shuffled) * validation_fraction))
    validation_size = min(validation_size, len(shuffled) - 1)
    return shuffled[validation_size:], shuffled[:validation_size]


@dataclasses.dataclass
class DataLoaderConfig:
    batch_size: int
    num_workers: int


def build_dataloaders_v1(
    data_dir: Path,
    window_size: int,
    stride: int,
    validation_fraction: float,
    seed: int,
    data_loader_config: DataLoaderConfig,
) -> tuple[DataLoader, DataLoader, int, int]:
    episodes = load_episode_sequences(data_dir)
    train_episodes, validation_episodes = split_episodes(
        episodes,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    transformations = Transformations()

    train_dataset = EpisodeFrameWindowDataset_V1(
        train_episodes,
        transformations=transformations,
        window_size=window_size,
        stride=stride,
    )
    validation_dataset = EpisodeFrameWindowDataset_V1(
        validation_episodes,
        transformations=transformations,
        window_size=window_size,
        stride=stride,
    )
    if not len(train_dataset) or not len(validation_dataset):
        raise ValueError("Train and validation datasets must both contain datapoints")

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_loader_config.batch_size,
        shuffle=True,
        num_workers=data_loader_config.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=data_loader_config.batch_size,
        num_workers=data_loader_config.num_workers,
        shuffle=False,
    )
    sample = train_dataset[0]
    state_dim = sample["state"].shape[-1]
    action_dim = sample["action"].shape[-1]
    return train_loader, validation_loader, state_dim, action_dim


def build_dataloaders_v2(
    data_dir: Path,
    window_size: int,
    stride: int,
    validation_fraction: float,
    seed: int,
    data_loader_config: DataLoaderConfig,
) -> tuple[DataLoader, DataLoader, int, int]:
    episodes = load_episode_path_sequences(data_dir)
    train_episodes, validation_episodes = split_episodes(
        episodes,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    transformations = Transformations(
        state_transform=compute_state_standardizer(train_episodes),
    )

    train_dataset = EpisodeFrameWindowDataset_V2(
        train_episodes,
        transformations=transformations,
        window_size=window_size,
        stride=stride,
    )
    validation_dataset = EpisodeFrameWindowDataset_V2(
        validation_episodes,
        transformations=transformations,
        window_size=window_size,
        stride=stride,
    )
    if not len(train_dataset) or not len(validation_dataset):
        raise ValueError("Train and validation datasets must both contain datapoints")

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_loader_config.batch_size,
        shuffle=True,
        num_workers=data_loader_config.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=data_loader_config.batch_size,
        shuffle=False,
        num_workers=data_loader_config.num_workers,
    )
    sample = train_dataset[0]
    state_dim = sample["state"].shape[-1]
    action_dim = sample["action"].shape[-1]
    return train_loader, validation_loader, state_dim, action_dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/baseline"),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    torch.manual_seed(args.seed)

    data_loader_config = DataLoaderConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader, validation_loader, state_dim, action_dim = build_dataloaders_v2(
        data_dir=args.data_dir,
        window_size=args.window_size,
        stride=args.stride,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        data_loader_config=data_loader_config,
    )
    model = build_baseline_model(state_dim=state_dim, action_dim=action_dim)
    history = train_model(
        model=model,
        dataset=TrainDataset(train_loader, validation_loader),
        criterion=nn.L1Loss(),
        config=TrainingConfig(
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir,
        ),
    )
    for metrics in history:
        print(
            f"epoch={metrics.epoch:03d} "
            f"train_loss={metrics.train_loss:.6f} "
            f"val_loss={metrics.val_loss:.6f} "
            f"persistence_l1={metrics.val_persistence_l1:.6f} "
            f"skill={metrics.val_skill:.3f} "
            f"psnr={metrics.val_psnr:.2f}dB "
            f"ssim={metrics.val_ssim:.4f}"
        )


if __name__ == "__main__":
    main()

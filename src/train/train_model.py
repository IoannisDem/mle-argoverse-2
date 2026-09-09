from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

import wandb


logger = logging.getLogger(__name__)

Batch = dict[str, torch.Tensor]


@dataclasses.dataclass
class TrainingConfig:
    num_epochs: int = 50
    learning_rate: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path | None = None
    early_stopping_patience: int | None = 5
    log_every_n_epochs: int = 1
    use_wandb: bool = False


@dataclasses.dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float


@dataclasses.dataclass
class TrainDataset:
    train_loader: DataLoader
    val_loader: DataLoader


@dataclasses.dataclass
class TrainState:
    criterion: Callable
    device: str
    optimizer: torch.optim.Optimizer | None = None


def train_model(
    model: nn.Module,
    dataset: TrainDataset,
    criterion: Callable,
    config: TrainingConfig,
) -> list[EpochMetrics]:
    model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    logger.info(
        "Starting training for %d epochs on %s with learning rate %.2e",
        config.num_epochs,
        config.device,
        config.learning_rate,
    )

    train_state = TrainState(criterion, config.device, optimizer)
    eval_state = TrainState(criterion, config.device, optimizer=None)

    history: list[EpochMetrics] = []

    for epoch in range(1, config.num_epochs + 1):
        train_loss = _run_epoch(model, train_state, dataset.train_loader)
        val_loss = _run_epoch(model, eval_state, dataset.val_loader)

        metrics = EpochMetrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
        history.append(metrics)

        if (
            epoch % config.log_every_n_epochs == 0
            or epoch == 1
            or epoch == config.num_epochs
        ):
            logger.info(
                "Epoch %d/%d - train_loss=%.6f, val_loss=%.6f",
                epoch,
                config.num_epochs,
                train_loss,
                val_loss,
            )

        if config.use_wandb:
            wandb.log(dataclasses.asdict(metrics), step=epoch)

    logger.info("Training complete")
    return history


def _run_epoch(model: nn.Module, state: TrainState, dataloader: DataLoader) -> float:
    is_training = state.optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        batch = _move_batch_to_device(batch, state.device)
        if is_training:
            batch_loss = training_step(model, state, batch)
        else:
            batch_loss = validation_step(model, state, batch)

        total_loss += batch_loss
        num_batches += 1

    return total_loss / num_batches


def training_step(model: nn.Module, state: TrainState, batch: Batch) -> float:
    state.optimizer.zero_grad()
    predicted_frame = _forward_pass(model, batch)
    loss = _compute_loss(predicted_frame, batch, state.criterion)
    loss.backward()
    state.optimizer.step()

    return loss.item()


@torch.no_grad()
def validation_step(model: nn.Module, state: TrainState, batch: Batch) -> float:
    predicted_frame = _forward_pass(model, batch)
    loss = _compute_loss(predicted_frame, batch, state.criterion)
    return loss.item()


def _move_batch_to_device(batch: Batch, device: str) -> Batch:
    return {key: value.to(device) for key, value in batch.items()}


def _forward_pass(model: nn.Module, batch: Batch) -> torch.Tensor:
    return model(
        frame_history=batch["frame_history"],
        state=batch["state"],
        action=batch["action"],
    )


def _compute_loss(
    predicted_frame: torch.Tensor, batch: Batch, criterion: Callable
) -> torch.Tensor:
    return criterion(predicted_frame, batch["frame_next"])

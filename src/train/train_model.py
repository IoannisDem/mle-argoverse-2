from __future__ import annotations

import dataclasses
import json
import logging
import math
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

import wandb

from train.metrics import FrameMetricAccumulator, FrameMetrics


logger = logging.getLogger(__name__)

Batch = dict[str, torch.Tensor]


@dataclasses.dataclass
class TrainingConfig:
    num_epochs: int = 50
    learning_rate: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path | None = None
    early_stopping_patience: int | None = 5
    log_every_n_epochs: int = 1
    use_wandb: bool = False
    wandb_project: str = "mle-argoverse-2"
    warmup_steps: int = 200
    min_learning_rate_ratio: float = 0.01


@dataclasses.dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float | None = None
    val_persistence_l1: float | None = None
    val_skill: float | None = None
    val_psnr: float | None = None
    val_ssim: float | None = None


@dataclasses.dataclass
class TrainDataset:
    train_loader: DataLoader
    val_loader: DataLoader


@dataclasses.dataclass
class TrainState:
    criterion: Callable
    device: str
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay, stepped once per optimizer step.

    Warmup matters here because the residual head is zero-initialised, so the first
    steps train only that head while every upstream gradient is still zero.
    """

    total_steps = max(1, config.num_epochs * steps_per_epoch)
    warmup_steps = min(config.warmup_steps, total_steps)
    floor = config.min_learning_rate_ratio

    def learning_rate_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return floor + (1 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)


def train_model(
    model: nn.Module,
    dataset: TrainDataset,
    criterion: Callable,
    config: TrainingConfig,
) -> list[EpochMetrics]:
    model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = build_scheduler(
        optimizer, config, steps_per_epoch=len(dataset.train_loader)
    )

    logger.info(
        "Starting training for %d epochs on %s with peak learning rate %.2e "
        "(%d warmup steps, then cosine decay)",
        config.num_epochs,
        config.device,
        config.learning_rate,
        min(config.warmup_steps, config.num_epochs * len(dataset.train_loader)),
    )

    if config.use_wandb:
        wandb.init(project=config.wandb_project, config=dataclasses.asdict(config))

    train_state = TrainState(criterion, config.device, optimizer, scheduler)
    eval_state = TrainState(criterion, config.device, optimizer=None)

    history: list[EpochMetrics] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    if config.checkpoint_dir is not None:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.num_epochs + 1):
        train_loss, _ = _run_epoch(model, train_state, dataset.train_loader)
        accumulator = FrameMetricAccumulator()
        val_loss, frame_metrics = _run_epoch(
            model, eval_state, dataset.val_loader, accumulator
        )

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=scheduler.get_last_lr()[0],
            **_frame_metric_fields(frame_metrics),
        )
        history.append(metrics)

        if (
            epoch % config.log_every_n_epochs == 0
            or epoch == 1
            or epoch == config.num_epochs
        ):
            logger.info(
                "Epoch %d/%d - train_loss=%.6f, val_loss=%.6f%s, lr=%.2e",
                epoch,
                config.num_epochs,
                train_loss,
                val_loss,
                _format_frame_metrics(frame_metrics),
                metrics.learning_rate,
            )

        if config.use_wandb:
            wandb.log(dataclasses.asdict(metrics), step=epoch)

        if config.checkpoint_dir is not None:
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                metrics=metrics,
                history=history,
                config=config,
                filename="checkpoint_last.pt",
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if config.checkpoint_dir is not None:
                _save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    metrics=metrics,
                    history=history,
                    config=config,
                    filename="checkpoint_best.pt",
                )
        else:
            epochs_without_improvement += 1

        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            logger.info(
                "Early stopping after %d epochs without validation improvement",
                epochs_without_improvement,
            )
            break

    if config.checkpoint_dir is not None:
        with (config.checkpoint_dir / "loss_history.json").open("w") as file:
            json.dump([dataclasses.asdict(item) for item in history], file, indent=2)

    logger.info("Training complete after %d epochs", len(history))
    return history


def _frame_metric_fields(metrics: FrameMetrics | None) -> dict[str, float]:
    if metrics is None:
        return {}
    return {
        "val_persistence_l1": metrics.persistence_l1,
        "val_skill": metrics.skill,
        "val_psnr": metrics.psnr,
        "val_ssim": metrics.ssim,
    }


def _format_frame_metrics(metrics: FrameMetrics | None) -> str:
    if metrics is None:
        return ""
    return (
        f", persistence_l1={metrics.persistence_l1:.6f}"
        f", skill={metrics.skill:.3f}"
        f", psnr={metrics.psnr:.2f}dB"
        f", ssim={metrics.ssim:.4f}"
    )


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: EpochMetrics,
    history: list[EpochMetrics],
    config: TrainingConfig,
    filename: str,
) -> None:
    """Save model state, optimizer state, and training progress."""

    checkpoint = {
        "epoch": metrics.epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": metrics.train_loss,
        "val_loss": metrics.val_loss,
        "history": [dataclasses.asdict(item) for item in history],
        "config": dataclasses.asdict(config),
    }
    torch.save(checkpoint, config.checkpoint_dir / filename)


def _run_epoch(
    model: nn.Module,
    state: TrainState,
    dataloader: DataLoader,
    accumulator: FrameMetricAccumulator | None = None,
) -> tuple[float, FrameMetrics | None]:
    is_training = state.optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        batch = _move_batch_to_device(batch, state.device)
        if is_training:
            batch_loss = training_step(model, state, batch)
        else:
            batch_loss, predicted_frame = validation_step(model, state, batch)
            if accumulator is not None:
                accumulator.update(predicted_frame, batch)

        total_loss += batch_loss
        num_batches += 1

    frame_metrics = accumulator.compute() if accumulator is not None else None
    return total_loss / num_batches, frame_metrics


def training_step(model: nn.Module, state: TrainState, batch: Batch) -> float:
    state.optimizer.zero_grad()
    predicted_frame = _forward_pass(model, batch)
    loss = _compute_loss(predicted_frame, batch, state.criterion)
    loss.backward()
    state.optimizer.step()
    if state.scheduler is not None:
        state.scheduler.step()

    return loss.item()


@torch.no_grad()
def validation_step(
    model: nn.Module, state: TrainState, batch: Batch
) -> tuple[float, torch.Tensor]:
    predicted_frame = _forward_pass(model, batch)
    loss = _compute_loss(predicted_frame, batch, state.criterion)
    return loss.item(), predicted_frame


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

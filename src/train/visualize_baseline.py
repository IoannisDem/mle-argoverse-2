"""Visualize baseline next-frame predictions from a saved checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from models.baseline import build_baseline_model
from train.train_baseline import DataLoaderConfig, build_dataloaders_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("baseline_predictions.png"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect input, target, and predicted frames from a DataLoader."""

    input_frames = []
    target_frames = []
    predicted_frames = []
    collected = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            frame_history = batch["frame_history"].to(device)
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            prediction = model(
                frame_history=frame_history,
                state=state,
                action=action,
            )

            take = min(num_samples - collected, prediction.shape[0])
            input_frames.append(frame_history[:take, -1].cpu())
            target_frames.append(batch["frame_next"][:take].cpu())
            predicted_frames.append(prediction[:take].cpu())
            collected += take
            if collected >= num_samples:
                break

    return (
        torch.cat(input_frames),
        torch.cat(target_frames),
        torch.cat(predicted_frames),
    )


def make_prediction_grid(
    input_frames: torch.Tensor,
    target_frames: torch.Tensor,
    predicted_frames: torch.Tensor,
    output_path: Path,
) -> None:
    """Save a grid comparing the last input, target, and prediction."""

    num_samples = input_frames.shape[0]
    figure, axes = plt.subplots(
        num_samples,
        3,
        figsize=(12, 4 * num_samples),
        squeeze=False,
    )
    column_titles = ("Last input frame", "Ground-truth next frame", "Predicted frame")

    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title)

    for row in range(num_samples):
        for column, frame in enumerate(
            (
                input_frames[row],
                target_frames[row],
                predicted_frames[row],
            )
        ):
            image = frame.permute(1, 2, 0).numpy().clip(0.0, 1.0)
            axes[row, column].imshow(image)
            axes[row, column].axis("off")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    train_loader, validation_loader, state_dim, action_dim = build_dataloaders_v2(
        data_dir=args.data_dir,
        window_size=args.window_size,
        stride=args.stride,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        data_loader_config=DataLoaderConfig(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        ),
    )
    model = build_baseline_model(state_dim=state_dim, action_dim=action_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    loader = train_loader if args.split == "train" else validation_loader
    input_frames, target_frames, predicted_frames = collect_predictions(
        model=model,
        loader=loader,
        device=str(device),
        num_samples=args.num_samples,
    )
    make_prediction_grid(
        input_frames=input_frames,
        target_frames=target_frames,
        predicted_frames=predicted_frames,
        output_path=args.output,
    )
    print(f"Saved prediction grid to {args.output}")


if __name__ == "__main__":
    main()

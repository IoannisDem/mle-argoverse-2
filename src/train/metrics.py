"""Frame-prediction metrics, scored against the copy-last-frame baseline."""

from __future__ import annotations

import dataclasses

import torch
from torch.nn import functional as F


def _gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates = coordinates - (window_size - 1) / 2
    profile = torch.exp(-(coordinates**2) / (2 * sigma**2))
    profile = profile / profile.sum()
    window = torch.outer(profile, profile)
    return window.expand(channels, 1, window_size, window_size).contiguous()


def structural_similarity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Mean SSIM over a batch of images shaped (batch, channels, height, width)."""

    channels = prediction.shape[1]
    window = _gaussian_window(
        window_size=window_size,
        sigma=sigma,
        channels=channels,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    padding = window_size // 2

    def filter_2d(image: torch.Tensor) -> torch.Tensor:
        return F.conv2d(image, window, padding=padding, groups=channels)

    mean_prediction = filter_2d(prediction)
    mean_target = filter_2d(target)
    variance_prediction = filter_2d(prediction * prediction) - mean_prediction**2
    variance_target = filter_2d(target * target) - mean_target**2
    covariance = filter_2d(prediction * target) - mean_prediction * mean_target

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mean_prediction * mean_target + c1) * (2 * covariance + c2)
    denominator = (mean_prediction**2 + mean_target**2 + c1) * (
        variance_prediction + variance_target + c2
    )
    return (numerator / denominator).mean()


def peak_signal_noise_ratio(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    mean_squared_error = F.mse_loss(prediction, target)
    return 10 * torch.log10(data_range**2 / mean_squared_error.clamp_min(1e-12))


@dataclasses.dataclass
class FrameMetrics:
    l1: float
    persistence_l1: float
    psnr: float
    ssim: float

    @property
    def skill(self) -> float:
        """Error relative to copying the last frame; below 1.0 beats that baseline."""

        return self.l1 / self.persistence_l1


class FrameMetricAccumulator:
    """Accumulate sample-weighted frame metrics over a validation pass."""

    def __init__(self) -> None:
        self._totals = dict.fromkeys(("l1", "persistence_l1", "psnr", "ssim"), 0.0)
        self._num_samples = 0

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, batch: dict[str, torch.Tensor]) -> None:
        target = batch["frame_next"]
        last_frame = batch["frame_history"][:, -1]
        clamped = prediction.clamp(0.0, 1.0)
        batch_size = target.shape[0]

        values = {
            "l1": F.l1_loss(clamped, target),
            "persistence_l1": F.l1_loss(last_frame, target),
            "psnr": peak_signal_noise_ratio(clamped, target),
            "ssim": structural_similarity(clamped, target),
        }
        for name, value in values.items():
            self._totals[name] += float(value) * batch_size
        self._num_samples += batch_size

    def compute(self) -> FrameMetrics | None:
        if not self._num_samples:
            return None
        averages = {
            name: total / self._num_samples for name, total in self._totals.items()
        }
        return FrameMetrics(**averages)

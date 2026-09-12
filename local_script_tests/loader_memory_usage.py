import os
import time
import threading
import dataclasses
from collections.abc import Callable
import psutil
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.train.train_baseline import (
    DataLoaderConfig,
    build_dataloaders_v1,
    build_dataloaders_v2,
)


@dataclasses.dataclass
class LoaderBuilderConfig:
    data_dir: str
    window_size: int
    stride: int
    validation_fraction: float
    seed: int
    data_loader_config: DataLoaderConfig


class MemoryMonitor:

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.peak_rss = 0
        self._stop_event = threading.Event()
        self._thread = None

    def _current_total_rss(self) -> int:
        total = 0
        try:
            total += self.process.memory_info().rss
        except psutil.NoSuchProcess:
            return self.peak_rss
        for child in self.process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total

    def _poll(self):
        while not self._stop_event.is_set():
            self.peak_rss = max(self.peak_rss, self._current_total_rss())
            time.sleep(self.interval)

    def __enter__(self):
        self._stop_event.clear()
        self.peak_rss = self._current_total_rss()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop_event.set()
        self._thread.join()
        self.peak_rss = max(self.peak_rss, self._current_total_rss())


def run_one(
    builder: Callable,
    num_workers: int,
    base_kwargs: dict,
    batch_size: int,
    poll_interval: float = 0.05,
) -> tuple[float, float]:
    data_loader_config = DataLoaderConfig(
        batch_size=batch_size,
        num_workers=num_workers,
    )
    config = LoaderBuilderConfig(
        data_loader_config=data_loader_config,
        **base_kwargs,
    )

    with MemoryMonitor(interval=poll_interval) as monitor:
        train_loader, _, _, _ = builder(
            config.data_dir,
            config.window_size,
            config.stride,
            config.validation_fraction,
            config.seed,
            config.data_loader_config,
        )
        start = time.perf_counter()
        for _ in train_loader:
            pass
        iteration_time = time.perf_counter() - start

    peak_mb = monitor.peak_rss / 1024**2
    return iteration_time, peak_mb


def main() -> None:
    base_kwargs = dict(
        data_dir="data/raw/traffic_0.15_accident_0_steps_1000",
        window_size=3,
        stride=1,
        validation_fraction=0.2,
        seed=42,
    )

    batch_size = 8
    worker_counts = list(range(0, 9, 2))
    for builder_name, builder in (
        # ("V1", build_dataloaders_v1),
        ("V2", build_dataloaders_v2),
    ):
        times = []
        peaks_mb = []
        for num_workers in worker_counts:
            print(f"Running {builder_name}, num_workers={num_workers} ...")
            iteration_time, peak_mb = run_one(
                builder,
                num_workers,
                base_kwargs,
                batch_size,
            )
            print(
                f"  iteration_time={iteration_time:.3f}s "
                f"peak_rss={peak_mb:.1f} MB"
            )
            times.append(iteration_time)
            peaks_mb.append(peak_mb)

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color1 = "tab:blue"
        ax1.set_xlabel("num_workers")
        ax1.set_ylabel("Iteration time (s)", color=color1)
        line1 = ax1.plot(
            worker_counts,
            times,
            marker="o",
            color=color1,
            label="Iteration time",
        )
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_xticks(worker_counts)

        ax2 = ax1.twinx()
        color2 = "tab:red"
        ax2.set_ylabel("Peak RSS across processes (MB)", color=color2)
        line2 = ax2.plot(
            worker_counts,
            peaks_mb,
            marker="s",
            color=color2,
            label="Peak memory",
        )
        ax2.tick_params(axis="y", labelcolor=color2)

        lines = line1 + line2
        ax1.legend(lines, [line.get_label() for line in lines], loc="upper center")
        fig.suptitle(
            f"{builder_name} DataLoader Benchmark — batch size = {batch_size}"
        )
        fig.tight_layout()

        out_path = f"dataloader_profile_{builder_name.lower()}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
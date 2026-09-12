import dataclasses
from pathlib import Path
from typing import Callable, TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset

import json
from tqdm import tqdm


@dataclasses.dataclass
class FrameSpec:
    image_array: np.array
    current_state: np.array
    next_action: np.array


@dataclasses.dataclass
class EpisodeSequence:
    episode_name: str
    frame_sequence: list[FrameSpec]


@dataclasses.dataclass
class FrameDatapoint:
    frames: list[FrameSpec]
    label: FrameSpec


def load_episode_sequences(data_dir: str | Path) -> list[EpisodeSequence]:
    """Load saved episode arrays from a directory of ``episode_*`` folders."""

    data_dir = Path(data_dir)
    episode_dirs = sorted(path for path in data_dir.glob("episode_*") if path.is_dir())
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found in {data_dir}")

    episodes: list[EpisodeSequence] = []
    for episode_dir in tqdm(episode_dirs[:15]):
        images = np.load(episode_dir / "images.npy", allow_pickle=False)
        states = np.load(episode_dir / "states.npy", allow_pickle=False)
        actions = np.load(episode_dir / "actions.npy", allow_pickle=False)

        if not (len(images) == len(states) == len(actions)):
            raise ValueError(
                f"Mismatched lengths in {episode_dir}: "
                f"images={len(images)}, states={len(states)}, actions={len(actions)}"
            )

        frame_sequence = [
            FrameSpec(
                image_array=images[index],
                current_state=states[index],
                next_action=actions[index],
            )
            for index in range(len(images))
        ]
        episodes.append(
            EpisodeSequence(
                episode_name=episode_dir.name,
                frame_sequence=frame_sequence,
            )
        )

    return episodes


class ModelInput(TypedDict):
    frame_history: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    frame_next: torch.Tensor


def get_datapoints(
    episode: EpisodeSequence,
    window_size: int = 3,
    stride: int = 1,
) -> list[FrameDatapoint]:

    datapoints: list[FrameDatapoint] = []
    frames = episode.frame_sequence

    for start in range(0, len(frames) - window_size, stride):
        datapoints.append(
            FrameDatapoint(
                frames=frames[start : start + window_size],
                label=frames[start + window_size],
            )
        )

    return datapoints


def default_frame_history_transform(frame_history: np.ndarray) -> torch.Tensor:
    frame_history = np.asarray(frame_history)
    if frame_history.ndim == 5 and frame_history.shape[-1] == 1:
        frame_history = frame_history[..., 0]
    if frame_history.ndim != 4:
        raise ValueError(
            "Expected frame history with shape [time, height, width, channels] "
            f"or [time, height, width, channels, 1], got {frame_history.shape}"
        )
    return torch.from_numpy(frame_history).permute(0, 3, 1, 2).float() / 255.0


def default_frame_transform(image_array: np.ndarray) -> torch.Tensor:
    image_array = np.asarray(image_array)
    if image_array.ndim == 4 and image_array.shape[-1] == 1:
        image_array = image_array[..., 0]
    if image_array.ndim != 3:
        raise ValueError(
            "Expected frame with shape [height, width, channels] or "
            f"[height, width, channels, 1], got {image_array.shape}"
        )
    return torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0


def default_state_transform(state: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(state)).float()


def default_action_transform(action: np.ndarray) -> torch.Tensor:
    # The recorded policy output is unbounded, but MetaDrive clips to [-1, 1] in
    # BaseVehicle before applying it, so only the clipped value drove the sim.
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    return torch.from_numpy(action).float()


@dataclasses.dataclass
class StateStandardizer:
    """Standardize state vectors with per-dimension training statistics."""

    mean: np.ndarray
    scale: np.ndarray

    def __call__(self, state: np.ndarray) -> torch.Tensor:
        standardized = (np.asarray(state, dtype=np.float32) - self.mean) / self.scale
        return torch.from_numpy(standardized).float()


def compute_state_standardizer(
    episodes: list["EpisodePathSequence"],
    epsilon: float = 1e-6,
) -> StateStandardizer:
    """Fit state statistics on the given episodes only.

    Dimensions with no variance are mapped to exactly zero rather than amplified.
    """

    states = np.concatenate(
        [np.load(episode.state_path, allow_pickle=False) for episode in episodes]
    ).astype(np.float32)
    standard_deviation = states.std(axis=0)
    return StateStandardizer(
        mean=states.mean(axis=0),
        scale=np.where(standard_deviation > epsilon, standard_deviation, 1.0),
    )


@dataclasses.dataclass
class Transformations:
    frame_history_transform: Callable[[np.ndarray], torch.Tensor] = (
        default_frame_history_transform
    )
    frame_transform: Callable[[np.ndarray], torch.Tensor] = default_frame_transform
    state_transform: Callable[[np.ndarray], torch.Tensor] = default_state_transform
    action_transform: Callable[[np.ndarray], torch.Tensor] = default_action_transform


class EpisodeFrameWindowDataset_V1(Dataset):
    def __init__(
        self,
        episodes: list[EpisodeSequence],
        transformations: Transformations,
        window_size: int = 3,
        stride: int = 1,
    ):
        self._transformations = transformations
        self._datapoints = [
            datapoint
            for episode in episodes
            for datapoint in get_datapoints(
                episode,
                window_size=window_size,
                stride=stride,
            )
        ]

    def __len__(self) -> int:
        return len(self._datapoints)

    def __getitem__(self, index: int) -> ModelInput:
        datapoint = self._datapoints[index]

        window = datapoint.frames
        target_frame = datapoint.label

        frame_history = np.stack(
            [frame.image_array for frame in window],
            axis=0,
        )

        frame_history_t = self._transformations.frame_history_transform(frame_history)

        last_frame = window[-1]
        current_state = last_frame.current_state
        next_action = last_frame.next_action

        state_t = self._transformations.state_transform(current_state)
        action_t = self._transformations.action_transform(next_action)
        target_t = self._transformations.frame_transform(target_frame.image_array)

        return {
            "frame_history": frame_history_t,
            "state": state_t,
            "action": action_t,
            "frame_next": target_t,
        }


@dataclasses.dataclass
class EpisodePathSequence:
    episode_name: str
    frame_path: Path
    state_path: Path
    action_path: Path
    num_steps: int


def load_episode_path_sequences(
    data_dir: str | Path,
) -> list[EpisodePathSequence]:

    data_dir = Path(data_dir)
    episode_dirs = sorted(path for path in data_dir.glob("episode_*") if path.is_dir())
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found in {data_dir}")

    episodes: list[EpisodePathSequence] = []
    for episode_dir in episode_dirs[:15]:
        frame_path = episode_dir / "images.npy"
        state_path = episode_dir / "states.npy"
        action_path = episode_dir / "actions.npy"
        meta_path = episode_dir / "meta.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
        num_steps = meta["num_steps"]
        episodes.append(
            EpisodePathSequence(
                episode_name=episode_dir.name,
                frame_path=frame_path,
                state_path=state_path,
                action_path=action_path,
                num_steps=num_steps,
            )
        )
    return episodes


@dataclasses.dataclass
class Datapoint:
    episode_index: int
    start_frame_index: int
    end_frame_index: int


class EpisodeFrameWindowDataset_V2(Dataset):
    def __init__(
        self,
        episodes: list[EpisodePathSequence],
        transformations: Transformations,
        window_size: int = 3,
        stride: int = 1,
    ):
        self._transformations = transformations
        self._episodes = episodes
        self._window_size = window_size
        self._stride = stride
        self._datapoint_mapping = self._build_datapoint_mapping()

    def _build_datapoint_mapping(self) -> list[Datapoint]:
        datapoints: list[Datapoint] = []
        for episode_index, episode in enumerate(self._episodes):
            for step in range(
                0, episode.num_steps - self._window_size, self._stride
            ):
                datapoints.append(
                    Datapoint(
                        episode_index=episode_index,
                        start_frame_index=step,
                        end_frame_index=step + self._window_size,
                    )
                )
        return datapoints

    def __len__(self) -> int:
        return len(self._datapoint_mapping)

    def __getitem__(self, index: int) -> ModelInput:

        datapoint = self._datapoint_mapping[index]
        episode = self._episodes[datapoint.episode_index]
        frame = np.load(episode.frame_path, mmap_mode="r", allow_pickle=False)
        state = np.load(episode.state_path, mmap_mode="r", allow_pickle=False)
        action = np.load(episode.action_path, mmap_mode="r", allow_pickle=False)

        frame_history = frame[datapoint.start_frame_index : datapoint.end_frame_index]
        target_frame = frame[datapoint.end_frame_index]
        frame_history_t = self._transformations.frame_history_transform(frame_history)
        state_t = self._transformations.state_transform(
            state[datapoint.end_frame_index - 1]
        )
        action_t = self._transformations.action_transform(
            action[datapoint.end_frame_index - 1]
        )
        target_t = self._transformations.frame_transform(target_frame)

        return {
            "frame_history": frame_history_t,
            "state": state_t,
            "action": action_t,
            "frame_next": target_t,
        }

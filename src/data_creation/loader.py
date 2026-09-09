import dataclasses
from pathlib import Path
from typing import Callable, TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset


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
    for episode_dir in episode_dirs:
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
    return torch.from_numpy(frame_history).permute(0, 3, 1, 2).float() / 255.0


def default_frame_transform(image_array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image_array)).permute(2, 0, 1).float() / 255.0


def default_state_transform(state: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(state)).float()


def default_action_transform(action: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(action)).float()


@dataclasses.dataclass
class Transformations:
    frame_history_transform: Callable[[np.ndarray], torch.Tensor] = (
        default_frame_history_transform
    )
    frame_transform: Callable[[np.ndarray], torch.Tensor] = default_frame_transform
    state_transform: Callable[[np.ndarray], torch.Tensor] = default_state_transform
    action_transform: Callable[[np.ndarray], torch.Tensor] = default_action_transform


class EpisodeFrameWindowDataset(Dataset):
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

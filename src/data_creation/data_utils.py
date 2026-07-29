import json
from pathlib import Path
import dataclasses
from typing import Any
from metadrive.envs.metadrive_env import MetaDriveEnv
import numpy as np

from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.component.sensors.rgb_camera import RGBCamera
from enum import Enum
from utils import to_numpy_image
import logging

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("src/data_creation/config.yaml")
OUTPUT_DIR = Path("data/raw")


class SensorType(Enum):
    RGB_CAMERA = "rgb_camera"


SENSOR_REGISTRY = {
    SensorType.RGB_CAMERA: RGBCamera,
}


class ExpertPolicyType(Enum):
    EXPERT_POLICY = "expert_policy"


POLICY_REGISTRY = {
    ExpertPolicyType.EXPERT_POLICY: ExpertPolicy,
}


@dataclasses.dataclass
class EnvConfig:
    image_observation: bool = True
    image_on_cuda: bool = False
    vehicle_config: dict = dataclasses.field(default_factory=dict)
    sensors: dict = dataclasses.field(default_factory=dict)
    agent_policy: type = ExpertPolicy
    num_scenarios: int = 100
    start_seed: int = 0
    traffic_density: float = 0.15
    accident_prob: float = 0.3
    log_level: int = 50


@dataclasses.dataclass
class DataConfig:
    output_path: str
    env_config: EnvConfig


def _create_env_config(config: dict[str, Any]) -> EnvConfig:
    config = config.copy()

    config["sensors"] = {
        name: (
            SENSOR_REGISTRY[SensorType(sensor["type"])],
            sensor["width"],
            sensor["height"],
        )
        for name, sensor in config.get("sensors", {}).items()
    }

    if "agent_policy" in config:
        config["agent_policy"] = POLICY_REGISTRY[
            ExpertPolicyType(config["agent_policy"])
        ]

    return EnvConfig(**config)


def parse_meta_env_config(config: dict[str, Any]) -> DataConfig:
    return DataConfig(
        output_path=config["output_path"],
        env_config=_create_env_config(config["meta_env"]),
    )


def define_env(config: DataConfig) -> MetaDriveEnv:
    return MetaDriveEnv(dataclasses.asdict(config))


@dataclasses.dataclass
class EpisodeResult:
    images: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    info: dict[str, Any] | None


def collect_episode(
    env: MetaDriveEnv, seed: int, max_iterations: int = 1000
) -> EpisodeResult:
    obs, _ = env.reset(seed=seed)
    images, states, actions = [], [], []
    done = False
    info = {}
    logger.info(f"This is iteratio number {seed}")
    step_count = 0

    while not done and step_count <= max_iterations:
        action = env.engine.get_policy(env.agent.id).act(env.agent.id)

        images.append(to_numpy_image(obs["image"]))
        states.append(np.asarray(obs["state"], dtype=np.float32).copy())
        actions.append(np.asarray(action, dtype=np.float32).copy())

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if not step_count % 100:
            logger.info(f"Iteration: {seed}, Step: {step_count}")
        step_count += 1

    return EpisodeResult(
        np.stack(images, axis=0),
        np.stack(states, axis=0),
        np.stack(actions, axis=0),
        info,
    )


def save_episode(
    episode_idx: int,
    seed: int,
    episode_result: EpisodeResult,
    output_path: Path,
) -> dict[str, Any]:
    ep_dir = output_path / f"episode_{episode_idx:03d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    np.save(ep_dir / "images.npy", episode_result.images)
    np.save(ep_dir / "states.npy", episode_result.states)
    np.save(ep_dir / "actions.npy", episode_result.actions)

    relevant_info = {
        key: value
        for key, value in episode_result.info.items()
        if key not in ["action", "raw_action"]
    }

    meta = {
        "episode_idx": episode_idx,
        "seed": seed,
        "num_steps": int(episode_result.images.shape[0]),
        **relevant_info,
    }
    with open(ep_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta

import yaml
import numpy as np
import torch
from pathlib import Path
import json


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_yaml(file_path: str | Path) -> dict:
    file_path = Path(file_path)

    with file_path.open("r") as file:
        return yaml.safe_load(file)


def read_json(file_path: str | Path) -> dict:
    file_path = Path(file_path)

    with file_path.open("r") as file:
        return json.load(file)


def _to_uint8(img: np.ndarray) -> np.ndarray:

    if img.dtype == np.uint8:
        return img

    if img.max() <= 1.5:
        img = np.clip(img, 0, 1) * 255

    return img.astype(np.uint8)


def to_numpy_image(img: torch.Tensor | np.ndarray) -> np.ndarray:

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    else:
        img = np.asarray(img)

    return _to_uint8(img)

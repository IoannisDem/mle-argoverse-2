from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import glob

import utils


def render_video(
    arrays: Sequence[np.ndarray | str | Path],
    names: Sequence[str],
    output_path: str | Path,
    fps: int = 10,
):


    if len(arrays) != len(names):
        raise ValueError("arrays and names must have the same length")

    # Load first array to determine video size
    first = arrays[0]
    if isinstance(first, (str, Path)):
        first = np.load(first)

    H, W = first.shape[1:3]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    for array, name in zip(arrays, names):

        if isinstance(array, (str, Path)):
            array = np.load(array)

        # Expect shape (B, H, W, 3, stack)
        if array.ndim != 5:
            raise ValueError(f"Expected 5D array, got {array.shape}")

        # Extract current frame (0th in stack)
        frames = array[..., 0]  # -> (B, H, W, 3)

        for i, frame in enumerate(frames):

            frame = frame.copy()

            # Convert to uint8 if necessary
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (255 * frame).astype(np.uint8)
                else:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)

            # RGB -> BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Top-left: sequence name
            cv2.putText(
                frame,
                name,
                (10, 30),
                font,
                0.3,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Top-right: frame index
            text = f"{i + 1}/{len(frames)}"
            (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)

            cv2.putText(
                frame,
                text,
                (W - tw - 10, 30),
                font,
                0.3,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)

    writer.release()



def main():
    current_path = Path.cwd()
    data_path = current_path / "data" / "raw" / "traffic_0.15_accident_0_steps_1000"
    images_data_paths = glob.glob(str(data_path / "**" / "images.npy"), recursive=True)

    images_data_paths = images_data_paths
    data_paths = {path.split("/")[-2]: path for path in images_data_paths}
    episodes_images = {key:utils.to_numpy_image( np.load(value)) for key, value in data_paths.items()}

    names = list(episodes_images.keys())
    images = list(episodes_images.values())

    render_video(
        images,
        names,
        "sample.mp4",
        fps = 10,
    )

if __name__ == "__main__":
    main()
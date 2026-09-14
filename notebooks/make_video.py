import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


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

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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

        if array.ndim == 5 and array.shape[-1] == 1:
            frames = array[..., 0]
        elif array.ndim == 4:
            frames = array
        else:
            raise ValueError(
                "Expected images with shape (frames, height, width, 3) "
                f"or (frames, height, width, 3, 1), got {array.shape}"
            )

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
    parser = argparse.ArgumentParser(description="Render one episode as a video.")
    parser.add_argument(
        "--episode",
        type=Path,
        required=True,
        help="Episode directory containing images.npy, or a path to images.npy.",
    )
    parser.add_argument("--output", type=Path, default=Path("sample.mp4"))
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    images_path = (
        args.episode if args.episode.suffix == ".npy"
        else args.episode / "images.npy"
    )
    images = np.load(images_path, allow_pickle=False)
    name = args.episode.parent.name if args.episode.suffix == ".npy" else args.episode.name

    render_video(
        (images,),
        (name,),
        args.output,
        fps=args.fps,
    )
    print(f"Saved {name} video to {args.output}")

if __name__ == "__main__":
    main()
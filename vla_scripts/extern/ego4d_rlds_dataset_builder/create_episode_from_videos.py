import argparse
import json
import os
from multiprocessing import Pool
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for simple video->episode conversion.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw videos into Ego4D-compatible episode .npy files (one video = one episode).\n"
            "Each episode contains a list of steps with keys: image, wrist_image, state, action, language_instruction.\n"
            "No temporal slicing; optional frame sampling by fps or frame-interval."
        )
    )

    # IO
    parser.add_argument(
        "--source_videos_dir",
        type=str,
        required=True,
        help="Directory containing input videos (e.g., .mp4, .avi).",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help=(
            "Directory to save episode_*.npy files. To work with the existing TFDS builder,"
            " point this to vla-scripts/extern/ego4d_rlds_dataset_builder/ego4d/data/train"
        ),
    )
    parser.add_argument(
        "--captions_json",
        type=str,
        default=None,
        help=(
            "Optional path to a JSON mapping {video_basename: caption}.\n"
            "If omitted, caption is set to empty string or video filename."
        ),
    )

    # Sampling & preprocessing
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "Target FPS to sample from each video. If not set, use all frames."
            " Mutually exclusive with --frame_interval."
        ),
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=None,
        help=(
            "Keep 1 frame every N frames. If not set and --fps not provided, use all frames."
        ),
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Output image size as 'H W' (default: 256 256).",
    )

    # Exec
    parser.add_argument(
        "--processes",
        type=int,
        default=8,
        help="Number of worker processes to decode videos in parallel.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify each saved episode by reloading it.",
    )

    args = parser.parse_args()
    if args.fps is not None and args.frame_interval is not None:
        raise ValueError("--fps and --frame_interval are mutually exclusive. Choose one.")
    return args


def center_crop_and_resize(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Center-crop the longer side and resize to target size.

    Args:
        image: Input HxWxC uint8 RGB image.
        target_size: (H, W) output size.

    Returns:
        Resized HxWxC uint8 RGB image.
    """
    height, width, _ = image.shape
    if height < width:
        crop_size = height
        start_x = (width - crop_size) // 2
        start_y = 0
    else:
        crop_size = width
        start_x = 0
        start_y = (height - crop_size) // 2

    cropped = image[start_y:start_y + crop_size, start_x:start_x + crop_size, :]
    pil_img = Image.fromarray(cropped)
    resampling_enum = getattr(Image, 'Resampling', None)
    if resampling_enum is not None:
        resample_method = getattr(resampling_enum, 'BILINEAR')
    else:
        resample_method = getattr(Image, 'BILINEAR', 2)
    resized = pil_img.resize((target_size[1], target_size[0]), resample=resample_method)
    return np.asarray(resized, dtype=np.uint8)


def _iter_video_frames(
    video_path: str,
    target_fps: Optional[float],
    frame_interval: Optional[int],
) -> List[np.ndarray]:
    """Decode frames from a video with optional sampling.

    Args:
        video_path: Path to the input video.
        target_fps: If provided, resample frames to approximate this fps.
        frame_interval: If provided, keep 1 in every N frames.

    Returns:
        List of BGR frames (uint8) as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    keep_every = 1
    if frame_interval is not None and frame_interval > 1:
        keep_every = int(frame_interval)
    elif target_fps is not None and orig_fps > 0:
        ratio = max(orig_fps / max(target_fps, 1e-6), 1.0)
        keep_every = int(round(ratio))
        keep_every = max(1, keep_every)

    frames: List[np.ndarray] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % keep_every == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def _derive_caption(
    captions_map: Optional[Dict[str, str]],
    video_basename: str,
) -> str:
    """Get caption for a video from a map or default to empty/filename.

    Args:
        captions_map: Optional dict mapping basename (without ext) -> caption.
        video_basename: Basename without extension.

    Returns:
        Caption string.
    """
    if captions_map is None:
        return ""
    return captions_map.get(video_basename, "")


def process_single_video(
    video_path: str,
    target_dir: str,
    target_size: Tuple[int, int],
    target_fps: Optional[float],
    frame_interval: Optional[int],
    captions_map: Optional[Dict[str, str]],
    verify: bool,
) -> Optional[str]:
    """Convert a single video to one episode_*.npy file.

    The episode schema matches ego4d_dataset_builder.py expectations.

    Args:
        video_path: Path to the video file.
        target_dir: Directory to save .npy episodes.
        target_size: (H, W) for images.
        target_fps: Optional FPS sampling.
        frame_interval: Optional keep-1-in-N frames.
        captions_map: Optional mapping from video stem to caption.
        verify: If True, reload after saving.

    Returns:
        Saved path or None if failed.
    """
    # Video id and episode id
    stem = os.path.splitext(os.path.basename(video_path))[0]
    caption = _derive_caption(captions_map, stem)

    frames_bgr = _iter_video_frames(video_path, target_fps, frame_interval)
    if len(frames_bgr) == 0:
        return None

    episode: List[Dict[str, Any]] = []
    for bgr in frames_bgr:
        rgb = bgr[:, :, ::-1]
        rgb = center_crop_and_resize(rgb, target_size)
        episode.append(
            {
                "image": np.asarray(rgb, dtype=np.uint8),
                "wrist_image": np.asarray(np.zeros([1, 1, 1]), dtype=np.uint8),
                "state": np.asarray(np.zeros(7), dtype=np.float32),
                "action": np.asarray(np.zeros(7), dtype=np.float32),
                "language_instruction": caption,
            }
        )

    os.makedirs(target_dir, exist_ok=True)
    save_path = os.path.join(target_dir, f"episode_{stem}.npy")
    np.save(save_path, np.array(episode, dtype=object), allow_pickle=True)

    if verify:
        try:
            _ = np.load(save_path, allow_pickle=True)
        except Exception:
            return None
    return save_path


def main() -> None:
    args = parse_arguments()
    os.makedirs(args.target_dir, exist_ok=True)

    captions_map: Optional[Dict[str, str]] = None
    if args.captions_json is not None and os.path.isfile(args.captions_json):
        with open(args.captions_json, "r") as f:
            # Expect {"video_basename": "caption", ...}
            captions_map = json.load(f)

    # Gather video files
    supported_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_paths: List[str] = []
    for fname in sorted(os.listdir(args.source_videos_dir)):
        fpath = os.path.join(args.source_videos_dir, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext in supported_exts:
            video_paths.append(fpath)

    print(f"Found {len(video_paths)} videos. Converting to episodes...")

    worker = partial(
        process_single_video,
        target_dir=args.target_dir,
        target_size=tuple(args.target_size),
        target_fps=args.fps,
        frame_interval=args.frame_interval,
        captions_map=captions_map,
        verify=args.verify,
    )

    if args.processes <= 1:
        results = []
        for p in tqdm(video_paths, desc="Processing videos"):
            results.append(worker(p))
    else:
        with Pool(processes=args.processes) as pool:
            results = list(
                tqdm(pool.imap_unordered(worker, video_paths), total=len(video_paths), desc="Processing videos")
            )

    num_ok = sum(1 for r in results if r is not None)
    print(f"Done. Saved {num_ok}/{len(video_paths)} episodes to: {args.target_dir}")


if __name__ == "__main__":
    main()



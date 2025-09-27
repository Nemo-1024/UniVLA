"""Self-contained test for ego4d_dataset_builder.Builder using fake data.

This test avoids real TF-Hub downloads and Apache Beam by:
- Monkeypatching `tensorflow_hub` import with a stub (so the module can import).
- Overriding `ego4d_dataset_builder._get_embed` to a cheap deterministic function.
- Replacing `tfds.core.lazy_imports.apache_beam` with a minimal shim so that
  `beam.Create(...) | beam.Map(...)` executes locally without Beam installed.

Run:
  python test_ego4d_builder_fake.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _install_tfhub_stub() -> None:
    """Install a minimal stub for `tensorflow_hub` to allow module import.

    The builder module imports `tensorflow_hub as hub` at top-level. We only need
    the import to succeed; actual embedding is overridden later via `_get_embed`.
    """

    if "tensorflow_hub" in sys.modules:
        return
    import types

    hub_stub = types.SimpleNamespace(load=lambda url: None)
    sys.modules["tensorflow_hub"] = hub_stub


class _BeamShim:
    """Minimal shim to emulate `apache_beam` used in the builder.

    Supports the expression: `beam.Create(iterable) | beam.Map(fn)` and returns
    an iterator of mapped results.
    """

    @staticmethod
    def Create(iterable: Iterable[Any]) -> List[Any]:
        return list(iterable)

    class _Map:
        def __init__(self, fn):
            self.fn = fn

        # Right-hand or operator so that: left | beam.Map(fn) works when left
        # does not implement __or__.
        def __ror__(self, other: Iterable[Any]):  # type: ignore[override]
            return map(self.fn, other)

    @staticmethod
    def Map(fn):
        return _BeamShim._Map(fn)


def _load_builder_module(module_path: Path):
    """Dynamically load the builder module from file path.

    Args:
        module_path: Absolute path to `ego4d_dataset_builder.py`.

    Returns:
        The loaded python module object.
    """

    spec = importlib.util.spec_from_file_location("ego4d_dataset_builder", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load ego4d_dataset_builder module spec.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ego4d_dataset_builder"] = module
    spec.loader.exec_module(module)
    return module


def _override_embed_fn(module) -> None:
    """Override the module-level `_get_embed` with a deterministic TF function.

    The function returns a callable that maps list[str] -> tf.Tensor[batch, 512].
    """

    def fake_get_embed():
        def _fn(texts: List[str]) -> tf.Tensor:
            batch_size = len(texts)
            # Deterministic but non-trivial content based on text length
            lengths = np.array([len(t or "") for t in texts], dtype=np.float32).reshape(-1, 1)
            base = np.zeros((batch_size, 512), dtype=np.float32)
            base[:, 0:1] = lengths
            return tf.convert_to_tensor(base)

        return _fn

    module._get_embed = fake_get_embed  # type: ignore[attr-defined]


def _override_beam_lazy_import() -> None:
    """Patch TFDS lazy import for apache_beam to use the local shim."""

    tfds.core.lazy_imports.apache_beam = _BeamShim  # type: ignore[attr-defined]


def _make_fake_episode(num_steps: int) -> List[Dict[str, Any]]:
    """Create a single fake episode with the required keys and shapes.

    Args:
        num_steps: Number of steps in the episode.

    Returns:
        A list of dicts representing steps.
    """

    steps: List[Dict[str, Any]] = []
    for i in range(num_steps):
        steps.append(
            {
                "image": np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8),
                "wrist_image": np.zeros((1, 1, 1), dtype=np.uint8),
                "state": np.zeros((7,), dtype=np.float32),
                "action": np.zeros((7,), dtype=np.float32),
                "language_instruction": f"do something {i}",
            }
        )
    return steps


def _write_fake_dataset(root_dir: Path, num_episodes: int, num_steps: int) -> None:
    """Write fake episodes to `root_dir/data/train`.

    Args:
        root_dir: The directory that will contain the `data/` folder.
        num_episodes: Number of episodes to write.
        num_steps: Number of steps per episode.
    """

    train_dir = root_dir / "data" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    for e in range(num_episodes):
        ep = _make_fake_episode(num_steps)
        np.save(train_dir / f"episode_{e:04d}.npy", ep)


def _validate_sample(sample: Dict[str, Any], expected_steps: int) -> None:
    """Validate a parsed sample from the builder.

    Raises AssertionError on mismatch.
    """

    steps = sample["steps"]
    assert isinstance(steps, list), "steps should be a list of dicts"
    assert len(steps) == expected_steps, f"expected {expected_steps} steps, got {len(steps)}"

    first = steps[0]
    assert "observation" in first, "missing 'observation' in step"
    obs = first["observation"]

    assert obs["image"].shape == (256, 256, 3) and obs["image"].dtype == np.uint8
    assert obs["wrist_image"].shape == (1, 1, 1) and obs["wrist_image"].dtype == np.uint8
    assert obs["state"].shape == (7,) and obs["state"].dtype == np.float32

    assert first["action"].shape == (7,) and first["action"].dtype == np.float32

    for k in ["discount", "reward", "is_first", "is_last", "is_terminal", "language_instruction", "language_embedding"]:
        assert k in first, f"missing '{k}' in step"

    assert isinstance(first["language_instruction"], str)

    lang_emb = first["language_embedding"]
    if hasattr(lang_emb, "numpy"):
        lang_emb = lang_emb.numpy()
    lang_emb = np.asarray(lang_emb)
    assert lang_emb.shape == (512,), "embedding size should be 512"


def main() -> None:
    parser = argparse.ArgumentParser(description="Test ego4d_dataset_builder with fake data")
    parser.add_argument("--module_path", type=str, default=str(Path(__file__).with_name("ego4d_dataset_builder.py")), help="Path to ego4d_dataset_builder.py")
    parser.add_argument("--num_episodes", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=3)
    args = parser.parse_args()

    module_path = Path(args.module_path).resolve()
    module_dir = module_path.parent

    # Prepare environment
    _install_tfhub_stub()
    module = _load_builder_module(module_path)
    _override_embed_fn(module)
    _override_beam_lazy_import()

    # Create fake data in a temp directory under the module dir so relative paths work
    tmp_root = Path(tempfile.mkdtemp(prefix="ego4d_fake_", dir=str(module_dir)))
    try:
        _write_fake_dataset(tmp_root, args.num_episodes, args.num_steps)

        # Change CWD so that builder's relative glob paths resolve
        prev_cwd = Path.cwd()
        os.chdir(tmp_root)
        try:
            builder = module.Builder()  # type: ignore[attr-defined]

            # Sanity-check DatasetInfo
            info = builder._info()
            assert "steps" in info.features, "Expected 'steps' in features"

            # Generate examples (Beam shim executes locally)
            train_gen = builder._split_generators(None)["train"]
            items: List[Tuple[str, Dict[str, Any]]] = list(train_gen)
            assert len(items) == args.num_episodes, (
                f"Expected {args.num_episodes} episodes, got {len(items)}"
            )

            # Validate content
            for key, sample in items:
                assert key.endswith(".npy"), f"Key should be episode file path, got {key}"
                _validate_sample(sample, args.num_steps)

            print(
                json.dumps(
                    {
                        "status": "ok",
                        "num_episodes": len(items),
                        "num_steps": args.num_steps,
                        "tmp_root": str(tmp_root),
                    }
                )
            )
        finally:
            os.chdir(prev_cwd)
    finally:
        # Cleanup temporary data
        # Comment the next two lines to inspect generated files after the run.
        import shutil

        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()



from typing import Iterator, Tuple, Any

import glob
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
try:
    import tensorflow_hub as hub  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency for real builds
    hub = None  # type: ignore[assignment]

# Module-level lazy loader for TFHub model to avoid multi-processing serialization issues.
_EMBED = None

def _get_embed():
    """Return the local TF-Hub SavedModel embedding function (lazy-loaded)."""
    global _EMBED
    if _EMBED is None:
        if hub is None:
            raise RuntimeError("tensorflow_hub is required but not installed.")
        _EMBED = hub.load("/data/home/jlchen/code/UniVLA/weights")
    return _EMBED


def _parse_example(episode_path: str) -> Tuple[str, Any]:
    print(episode_path)
    # Load raw data (list of dicts with required keys)
    data = np.load(episode_path, allow_pickle=True)

    # Assemble episode; demos set reward to 1 at the end
    episode = []
    for i, step in enumerate(data):
        emb_batch = _get_embed()([step['language_instruction']])
        # Support TF Tensor or NumPy outputs
        try:
            language_embedding = emb_batch[0].numpy()  # TensorFlow tensor path
        except AttributeError:
            language_embedding = np.asarray(emb_batch[0], dtype=np.float32)  # NumPy path

        episode.append({
            'observation': {
                'image': step['image'],
                'wrist_image': step['wrist_image'],
                'state': step['state'],
            },
            'action': step['action'],
            'discount': 1.0,
            'reward': float(i == (len(data) - 1)),
            'is_first': i == 0,
            'is_last': i == (len(data) - 1),
            'is_terminal': i == (len(data) - 1),
            'language_instruction': step['language_instruction'],
            'language_embedding': language_embedding,
        })

    sample = {
        'steps': episode,
        'episode_metadata': {
            'file_path': episode_path
        }
    }

    return episode_path, sample


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Ego4D RLDS dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    # Avoid heavy initialization in __init__; lazy-load in workers instead.

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(1, 1, 1),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc=(
                                'Robot state, consists of [7x robot joint angles, '
                                '2x gripper position, 1x door opening angle].'
                            ),
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc=(
                            'Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].'
                        ),
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.',
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.',
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.',
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.',
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc=(
                            'True on last step of the episode if it is a terminal step, True for demos.'
                        ),
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.',
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc=(
                            'Kona language embedding. See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                        ),
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/train/episode_*.npy'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path: str) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        episode_paths = glob.glob(path)

        # Optional: disable Beam for environments where Beam workers cannot spawn (e.g., restricted GRPC/ports).
        if os.environ.get("EGO4D_TFDS_DISABLE_BEAM") == "1":
            for sample in episode_paths:
                yield _parse_example(sample)
            return

        # For large datasets use Beam to parallelize data parsing.
        beam = tfds.core.lazy_imports.apache_beam
        return (
            beam.Create(episode_paths)
            | beam.Map(_parse_example)
        )


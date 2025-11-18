"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
# Must set EGL-related env vars BEFORE any mujoco / robosuite / OpenGL imports
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# Optional but often helps in headless: pick GPU 0 and use surfaceless EGL
os.environ.setdefault("EGL_DEVICE_ID", "0")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # 禁用并行化
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from collections import deque
import draccus
import numpy as np
import tqdm

import wandb
import traceback

"""Ensure project and LIBERO package roots are on sys.path before imports that depend on them."""
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
LIBERO_TOP = FILE_DIR  # contains top-level package name `libero`

# Prepend absolute paths for reliability (works in notebook and direct script runs)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LIBERO_TOP))

from .libero.libero import benchmark
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

home_path = "/mnt/mnt/public/jlchen"
@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    vlm_path: str = home_path + "/code/UniVLA/vla_scripts/vla_log/1027_122446+nsvq_bridgensvq_bridge/checkpoints/checkpoint-4000"
    lam_path: str = home_path + "/code/UniVLA/latent_action_model/logs/nsvq_ddp/version_0/checkpoints/epoch=1.ckpt"
    flow_path:  str = home_path + "/code/UniVLA/vla_scripts/world_vla_log/1027_221512+checkpoint-4000+libero_object_no_noops+lr-0.0001+freeze_vlm/checkpoints/step-029000/flow.pt"     # Pretrained checkpoint path
    vision_model_id: str = home_path + "/weights/vjepa2-vitl-fpc64-256"
    dataset_statistics_path: str = home_path + "/code/UniVLA/vla_scripts/world_vla_log/1015_220104+checkpoint-4000+libero_object_no_noops+lr-0.0001+freeze_vlm/dataset_statistics.json"
    center_crop: bool = False                         # Center crop? (if trained w/ random crop image aug)
    image_resolution: int = 256
    window_size: int = 10
    guidance_scale: float = 1.0
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_object"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "univla-libero"        # Name of W&B project to log to (use default!)

    seed: int = 7                                    # Random Seed (for reproducibility)
    save_video: bool = True                          # Whether to save a replay video of the episode
    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    # resize_size = get_image_resize_size(cfg)
    resize_size = cfg.image_resolution

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, resolution=cfg.image_resolution)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 240  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 300  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 320  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 550  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 420  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")

            action_queue = deque(maxlen=cfg.window_size)
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)
                    # print("img: ", img.shape)
                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                    print("setp: ", t, "state: ", observation["state"].min(),observation["state"].max())
                    assert observation["state"].shape == (8,), f"Observation state shape: {observation['state'].shape} not 8"
                    if len(action_queue) == 0:
                        # Query model to get action
                        actions = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                        )
                        action_queue.extend(actions)
                        print("actions length: ", len(action_queue))
                    action = action_queue.popleft()
                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)
                    
                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"Caught exception: {e}\n{tb}")
                    log_file.write(f"Caught exception: {e}\n{tb}\n")
                    log_file.flush()
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            if cfg.save_video:
                save_rollout_video(
                    replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
                )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )
        break
    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
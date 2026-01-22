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

home_path = "/mnt/project_rlinf/jlchen"
@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    model_id: str = "/mnt/project_rlinf/jlchen/code/UniVLA/vla_scripts/world_vla_log/0113_161659+weights+libero_object_no_noops+lr-0.0001+frzVis+frzLLM+unfrzLast4+frzEmb--finetune_libero_emb_gt/checkpoints/step-016000"
    # ↑ 完整模型 checkpoint 目录，包含 VLM、LAM extra 权重（latent_vla_extra.pt）和 Flow 权重（flow.pt）
    lam_path: str = home_path + "/code/UniVLA/latent_action_model/logs/dino_base_ae_bigwin/version_0/checkpoints/epoch=21.ckpt"
    lam_yaml_path: str = home_path + "/code/UniVLA/latent_action_model/logs/dino_base_ae_bigwin/version_0/dino_base_ae.yaml"  # LAM 配置文件
    # dataset_statistics_path 将自动从 model_id 推导：向上两级目录 + dataset_statistics.json
    center_crop: bool = False                         # Center crop? (if trained w/ random crop image aug)
    image_resolution: int = 256
    window_size: int = 10
    num_actions_to_use: int = 5
    num_inference_steps: int = 20
    guidance_scale: float = 1.0  # 1.0 = 关闭CFG（仅使用条件分支）；>1.0 = 启用CFG引导
    use_history_frame: bool = True                  # Whether to use history frame (should match training config)
    use_simple_model: bool = False
    future_prediction: bool = False  # 是否使用h_t1_pred（未来特征）；False=使用h_t（当前特征）
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
    
    # 自动推导 dataset_statistics_path：从 model_id 向上两级目录 + dataset_statistics.json
    # 例如：.../checkpoints/step-028000 -> .../dataset_statistics.json
    cfg.dataset_statistics_path = os.path.join(
        os.path.dirname(os.path.dirname(cfg.model_id)),
        "dataset_statistics.json"
    )
    print(f"[*] Auto-derived dataset_statistics_path: {cfg.dataset_statistics_path}")

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Get Hugging Face processor
    processor = get_processor(cfg)
    
    # [OpenVLA] Check that the processor contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in processor.norm_stats and f"{cfg.unnorm_key}_no_noops" in processor.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        elif cfg.unnorm_key not in processor.norm_stats and "libero_all_merged" in processor.norm_stats:
            cfg.unnorm_key = "libero_all_merged"
        assert cfg.unnorm_key in processor.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in processor `norm_stats`!"

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
            observation_history = deque(maxlen=cfg.window_size + 1)  # 维护历史帧，最长存储 window_size+1 个观测
            
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
                    # IMPORTANT: State structure must match training data!
                    # Training uses: [EEF_state(6), padding(1), gripper_state(1)]
                    # where EEF_state = [eef_pos(3), axisangle(3)]
                    # and gripper_state = single scalar (average of two gripper joints)
                    eef_state = np.concatenate([
                        obs["robot0_eef_pos"],              # 3D position
                        quat2axisangle(obs["robot0_eef_quat"])  # 3D axis-angle orientation
                    ])  # 6 dimensions total
                    
                    # gripper_state: average of two gripper joints (to match training data)
                    # 训练数据中使用的是单个gripper状态值
                    gripper_state = obs["robot0_gripper_qpos"][-1]  # scalar -> 1D array
                    
                    # Construct state matching training format: [eef_state(6), 0(1), gripper(1)]
                    observation = {
                        "full_image": img,
                        "state": np.concatenate([
                            eef_state,              # 6 dimensions
                            np.array([0.0]),        # 1 padding dimension (matches training)
                            np.array([gripper_state])  # 1 gripper dimension
                        ]),  # Total: 8 dimensions
                    }
                    
                    # Debug: print state range
                    if t == cfg.num_steps_wait:  # only print first valid timestep
                        print(f"[DEBUG] State构成: eef_pos={obs['robot0_eef_pos']}, "
                              f"axisangle={quat2axisangle(obs['robot0_eef_quat'])}, "
                              f"gripper={gripper_state}")
                    # print("step: ", t, "state min/max: ", observation["state"].min(), observation["state"].max())
                    assert observation["state"].shape == (8,), f"Observation state shape: {observation['state'].shape} != 8"
                    if len(action_queue) == 0:
                        # 选择与当前观测相隔 window_size 个时间步的历史帧；
                        # 若尚未累积足够历史，则一直使用初始观测
                        if len(observation_history) == 0:
                            prev_observation = observation
                        else:
                            hist_idx = max(0, len(observation_history) - cfg.window_size)
                            prev_observation = observation_history[hist_idx]

                        # Query model to get action
                        # 首次推理时启用诊断模式，打印中间特征统计信息
                        is_first_inference = (total_episodes == 0 and t == cfg.num_steps_wait and len(action_queue) == 0)
                        actions = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                            prev_obs=prev_observation,  # 传递历史帧
                            debug=is_first_inference,  # 首次推理时打印诊断信息
                        )
                        # print("actions: ", actions)
                        action_queue.extend(actions[:cfg.num_actions_to_use]) 
                        # print("actions length: ", len(action_queue))
                    
                    # 更新历史帧（为下一次推理准备）
                    observation_history.append(observation)
                    action = action_queue.popleft()
                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)
                    
                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    t += 1
                    
                    # Check if episode is done (success, failure, or timeout)
                    if done:
                        # Only count as success if info indicates success
                        if info.get('success', False):
                            task_successes += 1
                            total_successes += 1
                        break

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
        # break
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
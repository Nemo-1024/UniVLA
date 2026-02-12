"""
LatentWorldVLA inference wrapper for simpler_env Bridge evaluation.

This module provides a policy interface compatible with simpler_env's expected API,
wrapping LatentWorldVLA for real2sim Bridge dataset evaluation.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, List
from collections import deque

import numpy as np
import torch
from PIL import Image
from transforms3d.euler import euler2axangle

# Add project root to path
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[4]  # Navigate up to UniVLA root
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robot.openvla_utils import get_vla, get_processor, get_vla_action
from experiments.robot.robot_utils import load_training_yaml


@dataclass
class SimplerEnvConfig:
    """
    Configuration for LatentWorldVLA in simpler_env evaluation.
    
    Minimal configuration that mirrors the fields needed by get_vla/get_processor.
    Most parameters are automatically loaded from the training yaml.
    """
    # Required: checkpoint path
    model_id: str
    
    # Dataset key for action denormalization
    unnorm_key: str = "bridge_oxe"
    
    # Auto-derived from checkpoint path
    dataset_statistics_path: str = ""
    
    # Inference parameters (can override yaml defaults)
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    
    # These will be loaded from yaml
    use_history_frame: Optional[bool] = None
    window_size: Optional[int] = None
    image_resolution: Optional[int] = None


class LatentWorldVLABridgeInference:
    """
    Policy wrapper for LatentWorldVLA to work with simpler_env Bridge evaluation.
    
    This class matches the interface expected by simpler_env evaluation scripts,
    providing:
    - reset(task_description): Reset policy for new episode
    - step(image, task_description): Generate action from observation
    
    The wrapper handles:
    - Loading LatentWorldVLA model and processor from checkpoint
    - Converting numpy images to model input format
    - Action chunking (temporal ensembling)
    - Converting model output to Bridge action format
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        unnorm_key: str = "bridge_oxe",
        device: str = "cuda:0",
        # Optional inference overrides
        action_horizon: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ):
        """
        Initialize LatentWorldVLA inference wrapper.
        
        Args:
            checkpoint_path: Path to model checkpoint directory
            unnorm_key: Dataset key for action denormalization
            device: Device to run model on
            action_horizon: Override for action chunking window (None = use window_size from yaml)
            num_inference_steps: Override for Flow Matching steps (None = use yaml default)
            guidance_scale: Override for CFG guidance (None = use yaml default)
        
        Note:
            Model parameters (use_history_frame, image_resolution, window_size, etc.)
            are automatically loaded from the training yaml in the experiment directory.
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Create config object
        cfg = SimplerEnvConfig(
            model_id=checkpoint_path,
            unnorm_key=unnorm_key,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        # Auto-derive dataset_statistics_path from checkpoint path
        cfg.dataset_statistics_path = os.path.join(
            os.path.dirname(os.path.dirname(checkpoint_path)),
            "dataset_statistics.json"
        )
        print(f"[LatentWorldVLABridgeInference] Dataset statistics: {cfg.dataset_statistics_path}")
        
        # Load training yaml (same approach as run_libero_eval.py)
        # load_training_yaml will also populate missing eval fields on cfg.
        training_cfg = load_training_yaml(cfg)
        print("[*] Eval parameters from yaml:")
        print(f"  - use_history_frame = {cfg.use_history_frame}")
        print(f"  - window_size = {cfg.window_size}")
        print(f"  - image_resolution = {cfg.image_resolution}")
        
        # Store config and parameters
        self.cfg = cfg
        self.device = torch.device(device) if isinstance(device, str) else device
        self.unnorm_key = unnorm_key
        self.use_history_frame = cfg.use_history_frame
        self.window_size = cfg.window_size
        self.image_resolution = cfg.image_resolution
        
        # Action horizon: use override if provided, otherwise use window_size
        self.action_horizon = action_horizon if action_horizon is not None else self.window_size
        
        # Store inference parameters for get_vla_action
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        # Load model and processor (reusing openvla_utils functions)
        self.model = get_vla(cfg, training_cfg)
        self.processor = get_processor(cfg)
        
        print(f"[LatentWorldVLABridgeInference] Model loaded on {self.device}")
        
        # Action queue for temporal ensembling
        self.action_queue = deque(maxlen=self.action_horizon)
        self.observation_history = deque(maxlen=self.window_size + 1)
        
        # Task state
        self.task_description = None
        
        # Action scale (fixed for Bridge)
        self.action_scale = 1.0
        
        print(f"[LatentWorldVLABridgeInference] Initialization complete")
        print(f"  - Action horizon: {self.action_horizon}")
        print(f"  - Use history frame: {self.use_history_frame}")
        print(f"  - Image resolution: {self.image_resolution}")
        
    def reset(self, task_description: Union[str, List[str]]) -> None:
        """
        Reset policy for new episode.
        
        Args:
            task_description: Task instruction (str or list of str for batch)
        """
        # Handle both single string and list input
        if isinstance(task_description, list):
            self.task_description = task_description[0]
        else:
            self.task_description = task_description
        
        # Clear action queue and history
        self.action_queue.clear()
        self.observation_history.clear()
        
        print(f"[LatentWorldVLABridgeInference] Reset for task: {self.task_description}")
    
    def step(
        self, 
        image: np.ndarray, 
        task_description: Optional[Union[str, List[str]]] = None,
        *args, 
        **kwargs
    ) -> Tuple[dict, dict]:
        """
        Generate action from observation.
        
        Args:
            image: RGB image as np.ndarray of shape (H, W, 3), uint8
            task_description: Optional task description (if different from reset)
            
        Returns:
            raw_action: dict with model outputs (for logging/debugging)
            action: dict with processed action for environment:
                - 'world_vector': np.ndarray of shape (3,), xyz translation
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), always 0
        """
        # Update task description if provided
        if task_description is not None:
            if isinstance(task_description, list):
                task_description = task_description[0]
            if task_description != self.task_description:
                self.reset(task_description)
        
        # Convert image to numpy if it's a tensor
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Squeeze batch dimensions if present (ManiSkill returns shape like (1, 1, H, W, 3) or (1, H, W, 3))
        while image.ndim > 3:
            image = image.squeeze(0)
        
        # Ensure image is uint8 and shape (H, W, 3)
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Create dummy state (8D) - simpler_env doesn't provide proprio
        dummy_state = np.zeros(8, dtype=np.float32)
        
        # Create dummy wrist image (same as main image if not available)
        wrist_image = image.copy()
        
        # Prepare observation dict (matching LIBERO format expected by get_vla_action)
        observation = {
            "full_image": image,
            "wrist_image": wrist_image,
            "state": dummy_state,
        }
        
        # Store in observation history
        self.observation_history.append(observation)
        
        # Query model for new actions if queue is empty
        if len(self.action_queue) == 0:
            # Select history frame (matching LIBERO logic in run_libero_eval.py)
            if self.use_history_frame and len(self.observation_history) > 0:
                hist_idx = max(0, len(self.observation_history) - self.window_size)
                prev_observation = self.observation_history[hist_idx]
            else:
                prev_observation = observation
            
            # Get action using openvla_utils.get_vla_action
            actions = get_vla_action(
                vla=self.model,
                processor=self.processor,
                obs=observation,
                task_label=self.task_description,
                unnorm_key=self.unnorm_key,
                guidance_scale=self.guidance_scale,
                use_history_frame=self.use_history_frame,
                prev_obs=prev_observation if self.use_history_frame else None,
                num_inference_steps=self.num_inference_steps,
                debug=False,
                return_intermediates=False,
            )
            
            # Fill action queue
            self.action_queue.extend(actions[:self.action_horizon])
        
        # Pop next action from queue
        action_7d = self.action_queue.popleft()
        
        # Convert to Bridge action format
        raw_action, action = self._format_action(action_7d)
        
        return raw_action, action
    
    def _format_action(self, action_7d: np.ndarray) -> Tuple[dict, dict]:
        """
        Convert 7D action to Bridge format.
        
        Args:
            action_7d: Action array of shape (7,) containing
                       [x, y, z, rx, ry, rz, gripper]
                       
        Returns:
            raw_action: dict with separate components (for logging)
            action: dict with Bridge-formatted action
        """
        # Split action
        world_vector = action_7d[:3]
        rotation_delta = action_7d[3:6]
        gripper_raw = action_7d[6:7]
        
        # Raw action dict
        raw_action = {
            "world_vector": world_vector,
            "rotation_delta": rotation_delta,
            "open_gripper": gripper_raw,  # range [0, 1]; 1 = open; 0 = close
        }
        
        # Process action for environment
        action = {}
        
        # Translation (apply scale)
        action["world_vector"] = world_vector * self.action_scale
        
        # Rotation: euler angles -> axis-angle
        roll, pitch, yaw = rotation_delta.astype(np.float64)
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale
        
        # Gripper: binarize to -1 (close) or +1 (open)
        # Bridge convention: +1 = open, -1 = close
        action["gripper"] = 2.0 * (gripper_raw > 0.5) - 1.0
        
        # Termination flag (always 0)
        action["terminate_episode"] = np.array([0.0])
        
        return raw_action, action

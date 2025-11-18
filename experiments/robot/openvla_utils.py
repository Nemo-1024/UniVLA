"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor, LatentVLAProcessor

# LatentWorldVLA loading (mirrors vla_scripts/finetune_libero.py)
from prismatic.models.vlas.latent_world_vla import LatentWorldVLA, LatentWorldVLAConfig
from safetensors.torch import load_file as load_safetensors

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint (new self-loading API)."""



    # Build config for self-loading
    model_cfg = LatentWorldVLAConfig(
        model_id=cfg.vlm_path,
        lam_ckpt_path=cfg.lam_ckpt_path,
        lam_yaml_path=cfg.lam_yaml_path,
        freeze_vlm=True,
    )

    # Instantiate model and load flow weights
    vla = LatentWorldVLA(model_cfg=model_cfg)
    print(f"[*] Loaded flow weights from: {cfg.flow_path}")
    state_dict = torch.load(cfg.flow_path)
    # print(state_dict.keys())
    vla.flow.load_state_dict(state_dict,strict=True)

    for p in vla.parameters():
        p.requires_grad = False
    # Load dataset stats used during finetuning (for action un-normalization).
    if os.path.isfile(cfg.dataset_statistics_path):
        with open(cfg.dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        raise ValueError(f"Dataset statistics file {cfg.dataset_statistics_path} load failed.")
    # print("[*] Instantiating Pretrained VLA model")
    vla = vla.to(DEVICE,dtype=torch.float32).eval()

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    base_processor = AutoProcessor.from_pretrained(cfg.vlm_path, trust_remote_code=True)
    processor = LatentVLAProcessor.from_internvl_processor(base_processor, dataset_statistics_path=cfg.dataset_statistics_path)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


from PIL import Image
import numpy as np
import torch

# 在 get_vla_action 内部使用的 buffer 需要是函数属性，首次调用时初始化
def get_vla_action(vla, processor, obs, task_label, unnorm_key,
                   center_crop=False, guidance_scale=1.0, window_size=10,):
    """
    Generates an action with the VLA policy, with internal temporal smoothing.
    
    Each model call outputs a [window_size, 7] action chunk.
    The temporal smoothing averages the first frame of each recent chunk
    with exponential decay weighting to produce a smooth action.
    """
    # # 初始化函数内部 buffer（首次调用）
    # if not hasattr(get_vla_action, "_buffer"):
    #     get_vla_action._buffer = np.zeros((window_size, window_size, 7), dtype=np.float32)  # buffer 保存 window_size 个 chunk，每个 chunk 是 (window_size, 7)
    #     get_vla_action._mask = np.zeros(window_size, dtype=np.bool_)           # 标记有效 chunk
    #     get_vla_action._weights = np.exp(-decay_factor * np.arange(window_size))[:, None]  # 时间衰减

    # 1️⃣ 处理观测
    image = Image.fromarray(obs["full_image"]).convert("RGB")
    proprio = torch.from_numpy(obs["state"])
    
    messages = [
        {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a robot controller. Based on visual input and instructions, always output exactly 4 latent action tokens chosen from <ACT_0> ... <ACT_15>."}
                    ],
                },
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"What action should the robot take to {task_label.lower()}?"},
        ]},
    ]
    
    feats = processor.build_vla_features(messages=messages, proprio=proprio, unnorm_key=unnorm_key)
    
    # 确保数据类型与模型权重一致
    dtype = next(vla.vlm.parameters()).dtype
    
    # 将特征移到设备上并转换数据类型（input_ids 必须保持为 Long 类型）
    feats["pixel_values"] = feats["pixel_values"].to(DEVICE, dtype=dtype)
    feats["input_ids"] = feats["input_ids"].to(DEVICE)  # 保持为 Long 类型，不转换 dtype
    feats["image_4_jepa"] = feats["image_4_jepa"].to(DEVICE, dtype=dtype)
    feats["proprio"] = feats["proprio"].to(DEVICE, dtype=dtype)
    
    # 2️⃣ 获取模型预测动作 chunk
    with torch.inference_mode():
        actions_norm = vla.predict_action(
            pixel_values=feats["pixel_values"],
            input_ids=feats["input_ids"],
            image_4_jepa=feats["image_4_jepa"],
            proprio=feats["proprio"],
            guidance_scale=guidance_scale,
            window_size=window_size,
        )
    
    # 转为 numpy，[window_size, 7]
    # actions_chunk = actions_norm.detach().cpu().numpy().reshape(window_size, 7)
    
    # # 3️⃣ 更新 buffer（滑动 + 插入最新 chunk）
    # get_vla_action._buffer[1:] = get_vla_action._buffer[:-1]
    # get_vla_action._mask[1:] = get_vla_action._mask[:-1]
    # get_vla_action._buffer[0] = actions_chunk
    # get_vla_action._mask[0] = True
    
    # # 4️⃣ 时间加权融合（只融合每个 chunk 的第0帧）
    # first_frames = get_vla_action._buffer[:, 0, :]  # shape [window_size, 7]
    # weighted_actions = first_frames * get_vla_action._mask[:, None] * get_vla_action._weights
    # smoothed_action_norm = np.sum(weighted_actions, axis=0) / (np.sum(get_vla_action._mask[:, None] * get_vla_action._weights, axis=0) + 1e-8)
    
    # 5️⃣ 后处理（反归一化）
    action = processor.postprocess_actions(actions_norm.reshape(window_size, 7), unnorm_key=unnorm_key)
    
    # 转换为 numpy 数组以便后续的就地修改
    action = action.detach().cpu().numpy()
    
    return [action[i] for i in range(action.shape[0])]



def get_vla_latent_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False, hist_action=''):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    
    if len(hist_action) > 0:
        prompt = f"In: What action should the robot take to {task_label.lower()}? History action {hist_action}\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(vla.device, dtype=torch.bfloat16)

    # Get latent action.
    action = vla.predict_latent_action(**inputs, unnorm_key=unnorm_key, do_sample=True, temperature=0.75, top_p = 0.9)

    return action
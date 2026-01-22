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
from prismatic.models.vlas.latent_world_vla import LatentWorldVLA, LatentWorldVLAConfig, SimpleLatentWorldVLA
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
    """
    加载 LatentWorldVLA 模型用于评估。
    
    所有加载逻辑都在 LatentWorldVLA.from_config 内部完成。
    此函数仅负责：
    1. 构造配置
    2. 调用 from_config（自动加载 VLM, LAM, Flow 权重）
    3. 移到设备并设置为评估模式
    
    注意：
    - Flow 权重会自动从 cfg.model_id/flow.pt 加载（如果存在）
    - 支持两种场景：
      * LatentVLAModel checkpoint（无 flow.pt）→ 用于开始训练
      * LatentWorldVLA checkpoint（有 flow.pt）→ 用于推理/继续训练
    - 数据集统计信息（dataset_statistics）由 LatentVLAProcessor 单独加载
    """
    # 构造模型配置
    model_cfg = LatentWorldVLAConfig(
        model_id=cfg.model_id,
        lam_ckpt_path=cfg.lam_path,
        lam_yaml_path=cfg.lam_yaml_path,
        cfg_guidance_scale=cfg.guidance_scale,
        num_inference_steps=cfg.num_inference_steps,
        supervise_quantized=True,
        future_prediction=cfg.future_prediction,  # 🔧 控制是否使用未来特征预测
    )
    
    # 使用 from_config 加载模型（内部完成所有权重加载，包括可选的 Flow）
    print("[*] Loading LatentWorldVLA model...")
    if cfg.use_simple_model:
        vla, processor = SimpleLatentWorldVLA.from_config(model_cfg)
    else:
        vla, processor = LatentWorldVLA.from_config(model_cfg)
    # vla, processor = SimpleLatentWorldVLA.from_config(model_cfg)
    # 移到设备并设置为评估模式
    vla = vla.to(DEVICE, dtype=torch.float32).eval()
    
    # 冻结所有参数（推理模式）
    for p in vla.parameters():
        p.requires_grad = False
    
    print(f"[*] Model loaded successfully on {DEVICE}")
    
    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    base_processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
    processor = LatentVLAProcessor.from_vlm_processor(base_processor, dataset_statistics_path=cfg.dataset_statistics_path)
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

def get_vla_action(vla, processor, obs, task_label, unnorm_key,
                   center_crop=False, guidance_scale=1.0,
                   use_history_frame=False, prev_obs=None, num_inference_steps=20,
                   debug=False):
    """
    使用 LatentWorldVLA 生成动作序列。
    
    Args:
        vla: LatentWorldVLA 模型实例
        processor: LatentVLAProcessor 实例
        obs: 观测字典，包含 "full_image" 和 "state"
        task_label: 任务描述文本
        unnorm_key: 反归一化使用的数据集键
        center_crop: 是否中心裁剪（已废弃，保留接口兼容）
        guidance_scale: CFG 引导强度
        use_history_frame: 是否使用历史帧（需与训练配置一致）
        prev_obs: 前一帧观测字典（包含 "full_image"），仅当 use_history_frame=True 时使用
        
    Returns:
        actions: List[np.ndarray]，长度为 window_size 的动作列表
    """
    # 1. 处理观测
    image = Image.fromarray(obs["full_image"]).convert("RGB")
    proprio = torch.from_numpy(obs["state"])
    
    # 2. 构造输入消息（根据 use_history_frame 决定是否添加历史帧）
    if use_history_frame:
        # 训练时使用两帧：video[0]（历史帧）和 video[1]（当前帧）
        if prev_obs is not None:
            prev_image = Image.fromarray(prev_obs["full_image"]).convert("RGB")
        else:
            # 第一次推理时，重复当前帧作为历史帧
            prev_image = image
        
        user_content = [
            {"type": "image", "image": prev_image},
            {"type": "image", "image": image},
            {"type": "text", "text": f"What action should the robot take to {task_label.lower()}?"},
        ]
    else:
        # 训练时只使用一帧：video[1]（当前帧）
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": f"What action should the robot take to {task_label.lower()}?"},
        ]
    
    # 构造 assistant 的回复，包含 <ACT_PH> 占位符
    # 占位符数量等于 LAM 的 num_queries
    num_queries = vla.lam.num_queries
    placeholder_str = "".join(["<ACT_PH>" for _ in range(num_queries)])
    
    messages = [
        {
            "role": "user", 
            "content": user_content
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": placeholder_str}
            ],
        },
    ]
    
    # 3. 构造模型输入特征
    feats = processor.build_vla_features(messages=messages, proprio=proprio, unnorm_key=unnorm_key,observation=image)
    
    # 🔍 诊断：打印推理时的输入形状
    if debug:
        print(f"[DEBUG INFER] use_history_frame={use_history_frame}")
        print(f"[DEBUG INFER] num_images_in_messages={sum(1 for c in user_content if c.get('type')=='image')}")
        print(f"[DEBUG INFER] input_ids shape: {feats['input_ids'].shape}")
    
    # 4. 确保数据类型与设备一致
    dtype = next(vla.vlm.parameters()).dtype
    feats["pixel_values"] = feats["pixel_values"].to(DEVICE, dtype=dtype)
    feats["input_ids"] = feats["input_ids"].to(DEVICE)
    feats["lam_image"] = feats["lam_image"].to(DEVICE, dtype=dtype)
    feats["proprio"] = feats["proprio"].to(DEVICE, dtype=dtype)
    # 处理 image_grid_thw（如果存在）
    if "image_grid_thw" in feats and feats["image_grid_thw"] is not None:
        feats["image_grid_thw"] = feats["image_grid_thw"].to(DEVICE)
    
    # 5. 准备 lam_videos：从 [B, C, H, W] 扩展为 [B, 1, C, H, W]
    # lam_image 是当前帧，添加时间维度后成为 lam_videos
    lam_videos = feats["lam_image"].unsqueeze(1)  # [B, 1, C, H, W]
    
    # 6. 调用推理管线
    with torch.inference_mode():
        actions_norm = vla.predict_action(
            pixel_values=feats["pixel_values"],
            input_ids=feats["input_ids"],
            lam_videos=lam_videos,
            proprio=feats["proprio"],
            image_grid_thw=feats.get("image_grid_thw", None),
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            debug=debug,  # 传递诊断开关
        )
    
    # 7. 后处理：反归一化
    window_size = actions_norm.shape[1]
    actions = processor.postprocess_actions(
        actions_norm.reshape(window_size, 7),
        unnorm_key=unnorm_key,
    )
    
    # 🔍 调试：反归一化后动作范围
    # print(f"actions (unnorm): range=[{actions.min().item():.3f}, {actions.max().item():.3f}], first={actions[0, :3].tolist()}")
    
    # 8. 转换为 numpy 并返回列表格式
    actions = actions.detach().cpu().numpy()
    return [actions[i] for i in range(actions.shape[0])]



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
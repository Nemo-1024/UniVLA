"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoProcessor

# LatentWorldVLA loading (mirrors vla_scripts/finetune_libero.py)
from prismatic.models.vlas.latent_world_vla import LatentWorldVLA, LatentWorldVLAConfig, SimpleLatentWorldVLA
from prismatic.vla.latent_vla_processor import LatentVLAProcessor
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


def get_vla(cfg, training_cfg=None):
    """
    加载 LatentWorldVLA 模型用于评估。
    
    Args:
        cfg: 评估配置（GenerateConfig）
        training_cfg: 已加载的训练yaml配置字典（由调用方传入）
    
    工作流程：
    1. 用yaml中的model_cfg参数构造LatentWorldVLAConfig
    2. 用评估配置中的推理参数覆盖
    3. 调用 from_config 加载模型
    """
    import dataclasses
    from prismatic.models.vlas.flowmatching_expert import ConditionalFlowMatchingConfig
    
    if training_cfg is None:
        raise ValueError("training_cfg is required. Please load yaml first using load_training_yaml().")
    
    # 1. 准备模型配置参数字典
    model_cfg_kwargs = {'model_id': cfg.model_id}  # model_id始终使用命令行指定的
    
    # 2. 从yaml的model_cfg中提取参数
    yaml_model_cfg = training_cfg.get('model_cfg', {})
    config_fields = {f.name for f in dataclasses.fields(LatentWorldVLAConfig)}
    
    print("[*] Model parameters from yaml:")
    for key, value in yaml_model_cfg.items():
        if key in config_fields and key != 'model_id':
            if key == 'flow_cfg' and isinstance(value, dict):
                flow_config_fields = {f.name for f in dataclasses.fields(ConditionalFlowMatchingConfig)}
                flow_cfg_kwargs = {k: v for k, v in value.items() if k in flow_config_fields}
                model_cfg_kwargs['flow_cfg'] = ConditionalFlowMatchingConfig(**flow_cfg_kwargs)
                print(f"  - flow_cfg: {len(flow_cfg_kwargs)} parameters")
            else:
                model_cfg_kwargs[key] = value
                print(f"  - {key} = {value}")
    
    # 3. 特殊处理：顶层参数映射
    if 'use_simple_model' in training_cfg:
        model_cfg_kwargs['use_simple_action_head'] = training_cfg['use_simple_model']
        print(f"  - use_simple_action_head = {training_cfg['use_simple_model']}")
    
    # 4. 评估参数覆盖
    if hasattr(cfg, 'guidance_scale') and cfg.guidance_scale is not None:
        model_cfg_kwargs['cfg_guidance_scale'] = cfg.guidance_scale
        print(f"[*] Eval override: guidance_scale = {cfg.guidance_scale}")
    if hasattr(cfg, 'num_inference_steps') and cfg.num_inference_steps is not None:
        model_cfg_kwargs['num_inference_steps'] = cfg.num_inference_steps
        print(f"[*] Eval override: num_inference_steps = {cfg.num_inference_steps}")
    
    # 5. 构造模型配置并加载模型
    model_cfg = LatentWorldVLAConfig(**model_cfg_kwargs)
    
    print(f"[*] Loading model ({len(model_cfg_kwargs)} parameters)...")
    if model_cfg.use_simple_action_head:
        print("[*] Using SimpleLatentWorldVLA")
        vla, processor = SimpleLatentWorldVLA.from_config(model_cfg)
    else:
        print("[*] Using LatentWorldVLA (flow matching)")
        vla, processor = LatentWorldVLA.from_config(model_cfg)
    
    vla = vla.to(DEVICE, dtype=torch.float32).eval()
    for p in vla.parameters():
        p.requires_grad = False
    
    print(f"[*] Model loaded on {DEVICE}")
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
                   debug=False, return_intermediates=False, image_resolution=256, num_queries=8, num_flow_queries=8):
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
        return_intermediates: 是否返回中间特征（h_t, h_t1_pred）
        image_resolution: 图像分辨率
    Returns:
        如果 return_intermediates=False:
            actions: List[np.ndarray]，长度为 window_size 的动作列表
        如果 return_intermediates=True:
            (actions, intermediates): 其中 intermediates 包含 h_t, h_t1_pred, vision_tokens_hw
    """
    def _resize_pil(img: Image.Image, resolution):
        """Resize PIL image to `resolution` (int or (w,h))."""
        if resolution is None:
            return img
        if isinstance(resolution, int):
            size = (resolution, resolution)
        elif isinstance(resolution, (tuple, list)) and len(resolution) == 2:
            # PIL expects (width, height)
            size = (int(resolution[0]), int(resolution[1]))
        else:
            raise ValueError(f"Unsupported image_resolution={resolution!r}; expected int or (w,h).")
        if img.size == size:
            return img
        resampling = getattr(Image, "Resampling", Image)
        return img.resize(size, resample=getattr(resampling, "BICUBIC"))

    # 1. 处理观测（先 resize 到 image_resolution）
    image = Image.fromarray(obs["full_image"]).convert("RGB")
    image = _resize_pil(image, image_resolution)
    proprio = torch.from_numpy(obs["state"])
    wrist_image = Image.fromarray(obs["wrist_image"]).convert("RGB")
    wrist_image = _resize_pil(wrist_image, image_resolution)
    # 2. 构造输入消息（根据 use_history_frame 决定是否添加历史帧）
    if use_history_frame:
        # 训练时使用两帧：video[0]（历史帧）和 video[1]（当前帧）
        if prev_obs is not None:
            prev_image = Image.fromarray(prev_obs["full_image"]).convert("RGB")
            prev_image = _resize_pil(prev_image, image_resolution)
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
    # 占位符数量 = latent queries + flow queries
    total_queries = int(num_queries) + int(num_flow_queries)
    placeholder_str = "".join(["<ACT_PH>" for _ in range(total_queries)])
    
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
    feats = processor.build_vla_features(messages=messages, proprio=proprio, unnorm_key=unnorm_key, observation=image, wrist_image=wrist_image)
    
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
    feats["wrist_image"] = feats["wrist_image"].to(DEVICE, dtype=dtype)
    feats["proprio"] = feats["proprio"].to(DEVICE, dtype=dtype)
    # 处理 attention_mask（如果存在）
    if "attention_mask" in feats and feats["attention_mask"] is not None:
        feats["attention_mask"] = feats["attention_mask"].to(DEVICE)
    # 处理 image_grid_thw（如果存在）
    if "image_grid_thw" in feats and feats["image_grid_thw"] is not None:
        feats["image_grid_thw"] = feats["image_grid_thw"].to(DEVICE)
    
    # 5. 准备 lam_videos：从 [B, C, H, W] 扩展为 [B, 1, C, H, W]
    # lam_image 是当前帧，添加时间维度后成为 lam_videos
    lam_videos = feats["lam_image"].unsqueeze(1)  # [B, 1, C, H, W]
    wrist_videos = feats["wrist_image"].unsqueeze(1)  # [B, 1, C, H, W]
    # 6. 调用推理管线
    with torch.inference_mode():
        result = vla.predict_action(
            pixel_values=feats["pixel_values"],
            input_ids=feats["input_ids"],
            attention_mask=feats.get("attention_mask", None),  # 传递 attention_mask
            lam_videos=lam_videos,
            wrist_videos=wrist_videos,
            proprio=feats["proprio"],
            image_grid_thw=feats.get("image_grid_thw", None),
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            debug=debug,  # 传递诊断开关
            return_intermediates=return_intermediates,
        )
    
    # 处理返回值
    if return_intermediates:
        actions_norm, intermediates = result
    else:
        actions_norm = result
        intermediates = None
    
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
    actions_list = [actions[i] for i in range(actions.shape[0])]
    
    if not return_intermediates:
        return actions_list
    else:
        return actions_list, intermediates



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

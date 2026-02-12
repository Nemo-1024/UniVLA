"""Utils for evaluating robot policies in various environments."""

import os
import random
import time
from pathlib import Path
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt
try:
    import cv2
except ImportError:
    cv2 = None
    print("[WARNING] cv2 not available, grid overlay will use PIL instead")

from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
    get_vla_latent_action,
)

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


def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg, training_cfg=None):
    """Load model for evaluation.
    
    Args:
        cfg: 评估配置
        training_cfg: 训练yaml配置字典（已加载）
    """
    if cfg.model_family == "openvla":
        model = get_vla(cfg, training_cfg)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    print(f"Loaded model: {type(model)}")
    return model


def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    if cfg.model_family == "openvla":
        resize_size = 256
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


def get_action(cfg, model, obs, task_label, processor=None, prev_obs=None, debug=False, return_intermediates=False):
    """Queries the model to get an action."""

    action = get_vla_action(
        model,
        processor,
        obs,
        task_label,
        cfg.unnorm_key,
        center_crop=cfg.center_crop,
        guidance_scale=cfg.guidance_scale,
        use_history_frame=cfg.use_history_frame,
        prev_obs=prev_obs,
        num_inference_steps=cfg.num_inference_steps,
        debug=debug,  # 传递诊断开关
        return_intermediates=return_intermediates,
        image_resolution = cfg.image_resolution,
    )

    return action


def get_latent_action(cfg, model, obs, task_label, processor=None, hist_action=''):
    """Queries the model to get an action."""
    latent_action = get_vla_latent_action(
        model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop, hist_action=hist_action,
    )

    return latent_action


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action

def load_training_yaml(cfg) -> dict:
    """
    加载训练yaml配置（统一入口）。
    
    返回完整的yaml配置字典，同时将评估所需的参数设置到cfg中。
    
    Returns:
        training_cfg: 完整的yaml配置字典，供后续模型加载使用
    """
    import yaml
    
    checkpoint_path = Path(cfg.model_id).resolve()
    experiment_dir = checkpoint_path.parent.parent
    yaml_files = list(experiment_dir.glob("*.yaml"))
    
    if not yaml_files:
        raise FileNotFoundError(f"No training yaml found in {experiment_dir}.")
    
    yaml_path = yaml_files[0]
    print(f"[*] Loading training config from: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        training_cfg = yaml.safe_load(f)
        
        # Populate missing eval fields from training config when available.
    for key in ("image_resolution", "use_history_frame", "window_size"):
        if getattr(cfg, key, None) is None and key in training_cfg:
            setattr(cfg, key, training_cfg[key])
    
    return training_cfg


def compute_cross_similarity(
    h_t1_pred,
    anchor_feat,
    vision_tokens_hw,
    target_image,
    vmin=0.4,
    vmax=1.0,
    alpha=0.5,
    cmap='jet',
):
    """
    计算 h_t1_pred 与锚点特征的相似度热力图，并叠加到目标图像上。
    
    Args:
        h_t1_pred: [K, D] 预测的未来特征 tokens（已在 CPU 上）
        anchor_feat: [D,] 锚点特征向量（已归一化）
        vision_tokens_hw: (H, W) 特征网格尺寸
        target_image: [H_img, W_img, 3] 目标图像（uint8, RGB）
        vmin, vmax: colormap 范围
        alpha: 叠加透明度
        cmap: colormap 名称
        
    Returns:
        overlay_image: [H_img, W_img, 3] 叠加热力图的图像（uint8, RGB）
    """
    H, W = vision_tokens_hw
    K, D = h_t1_pred.shape
    
    # 1. Reshape 为网格特征 [D, H, W]
    h_t1_grid = h_t1_pred.T.reshape(D, H, W)  # [K, D] -> [D, K] -> [D, H, W]
    
    # 2. L2 归一化
    h_t1_norm = F.normalize(h_t1_grid, p=2, dim=0)  # [D, H, W]
    
    # 3. 与锚点做点积得到相似度图 [H, W]
    anchor_feat = anchor_feat.view(D, 1, 1)  # [D,] -> [D, 1, 1]
    sim_map = (h_t1_norm * anchor_feat).sum(dim=0)  # [H, W]
    
    # 4. 上采样到图像尺寸
    img_h, img_w = target_image.shape[:2]
    sim_map_resized = F.interpolate(
        sim_map.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
        size=(img_h, img_w),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()  # [H_img, W_img]
    
    # 5. 应用 colormap
    colormap = plt.get_cmap(cmap)
    # 归一化到 [0, 1]
    sim_norm = np.clip((sim_map_resized - vmin) / (vmax - vmin), 0, 1)
    heatmap_colored = colormap(sim_norm)  # [H_img, W_img, 4] (RGBA)
    heatmap_rgb = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    
    # 6. 叠加到目标图像
    overlay_image = (
        (1 - alpha) * target_image + alpha * heatmap_rgb
    ).astype(np.uint8)
    
    return overlay_image


def extract_anchor_feature(h_t, src_row, src_col, vision_tokens_hw):
    """
    从初始帧特征 h_t 中提取锚点 patch 的特征向量（已归一化）。
    
    Args:
        h_t: [K, D] 初始帧特征 tokens（已在 CPU 上）
        src_row, src_col: 锚点 patch 的行列索引
        vision_tokens_hw: (H, W) 特征网格尺寸
        
    Returns:
        anchor_feat: [D,] 归一化的锚点特征向量
    """
    H, W = vision_tokens_hw
    K, D = h_t.shape
    
    # Reshape 为网格特征 [D, H, W]
    h_t_grid = h_t.T.reshape(D, H, W)
    
    # L2 归一化
    h_t_norm = F.normalize(h_t_grid, p=2, dim=0)
    
    # 提取锚点特征
    anchor_feat = h_t_norm[:, src_row, src_col]  # [D,]
    
    return anchor_feat


def save_video(frames, path, fps=30):
    """
    保存帧序列为视频文件。
    
    Args:
        frames: List[np.ndarray] 图像帧列表，每帧为 [H, W, 3] uint8 数组
        path: str 输出视频路径
        fps: int 帧率
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    video_writer = imageio.get_writer(path, fps=fps)
    for img in frames:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved video at path {path}")


def draw_feature_grid_overlay(image, grid_h=16, grid_w=16, line_color=(128, 128, 128), line_width=1, highlight_patch=None):
    """
    在图像上绘制特征网格，方便选择锚点 patch 坐标。
    
    Args:
        image: [H, W, 3] uint8 图像数组
        grid_h: int 网格行数（默认 16）
        grid_w: int 网格列数（默认 16）
        line_color: tuple RGB 颜色（默认灰色）
        line_width: int 线条宽度
        highlight_patch: tuple (row, col) 可选，高亮显示某个 patch
        
    Returns:
        overlay_image: [H, W, 3] uint8 带网格的图像
    """
    img_h, img_w = image.shape[:2]
    overlay = image.copy()
    
    # 计算每个 patch 的尺寸
    patch_h = img_h // grid_h
    patch_w = img_w // grid_w
    
    if cv2 is not None:
        # 使用 cv2 绘制（更高效）
        # 绘制网格线
        for i in range(grid_h + 1):
            y = i * patch_h
            cv2.line(overlay, (0, y), (img_w, y), line_color, line_width)
        
        for j in range(grid_w + 1):
            x = j * patch_w
            cv2.line(overlay, (x, 0), (x, img_h), line_color, line_width)
        
        # 高亮显示指定 patch
        if highlight_patch is not None:
            row, col = highlight_patch
            y1 = row * patch_h
            y2 = (row + 1) * patch_h
            x1 = col * patch_w
            x2 = (col + 1) * patch_w
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        # 使用 PIL 绘制（备用方案）
        from PIL import ImageDraw
        pil_image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(pil_image)
        
        # 绘制网格线
        for i in range(grid_h + 1):
            y = i * patch_h
            draw.line([(0, y), (img_w, y)], fill=line_color, width=line_width)
        
        for j in range(grid_w + 1):
            x = j * patch_w
            draw.line([(x, 0), (x, img_h)], fill=line_color, width=line_width)
        
        # 高亮显示指定 patch
        if highlight_patch is not None:
            row, col = highlight_patch
            y1 = row * patch_h
            y2 = (row + 1) * patch_h
            x1 = col * patch_w
            x2 = (col + 1) * patch_w
            draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)
        
        overlay = np.array(pil_image)
    
    return overlay


def save_reference_image(image, path, grid_h=16, grid_w=16, highlight_patch=None):
    """
    保存带特征网格标注的参考图像，用于选择锚点坐标。
    
    Args:
        image: [H, W, 3] uint8 图像数组
        path: str 输出图像路径
        grid_h: int 网格行数（默认 16）
        grid_w: int 网格列数（默认 16）
        highlight_patch: tuple (row, col) 可选，高亮显示某个 patch
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    overlay_image = draw_feature_grid_overlay(image, grid_h, grid_w, highlight_patch=highlight_patch)
    imageio.imwrite(path, overlay_image)
    print(f"Saved reference image with feature grid at path {path}")
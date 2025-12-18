"""
traj_transforms.py

Contains trajectory transforms used in the orca data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory length).
"""

import logging
from typing import Dict

import tensorflow as tf


def chunk_act_obs(
    traj,
    window_size,
    future_action_window_size,
    proprio_threshold_min: float = 0.05,
    proprio_threshold_max: float = 0.8,
):
    """
    仅取窗口首尾两帧的轻量版 chunk：
    - 每个 chunk 由 [t, t+window_size-1] 两个时刻组成（第二个索引会截断到轨迹末尾）。
    - 根据 proprio 首尾差值/夹爪变化过滤静止片段，逻辑参考 `chunk_act_obs_uniform_resample`。
    """
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]

    # Create indices for the first and last elements within the window size
    # 只保留完整窗口：起点范围为 [0, traj_len - window_size]
    max_start = tf.maximum(traj_len - window_size, 0)
    first_indices = tf.range(max_start + 1, dtype=tf.int32)[:, None]  # First index is the current timestep
    last_indices = first_indices + (window_size - 1)

    # Observation chunks: add a historical frame (~30% window back) plus first/last
    hist_offset = tf.cast(tf.cast(window_size, tf.float32) * 0.5, tf.int32)
    obs_hist_indices = first_indices - hist_offset
    obs_chunk_indices = tf.concat([obs_hist_indices, first_indices, last_indices], axis=1)  # [num_chunks, 3]

    # Action chunks: only first/last (no history)
    action_first_indices = first_indices
    action_last_indices = last_indices
    action_chunk_indices = tf.concat([action_first_indices, action_last_indices], axis=1)  # [num_chunks, 2]

    # Ensure indices are bounded (for observations)
    floored_obs_chunk_indices = tf.maximum(tf.minimum(obs_chunk_indices, tf.maximum(traj_len - 1, 0)), 0)

    # goal timestep 处理
    if "timestep" in traj["task"]:
        goal_timestep = traj["task"]["timestep"]
    else:
        goal_timestep = tf.fill([traj_len], tf.maximum(traj_len - 1, 0))

    chunk_goal = tf.gather(goal_timestep, tf.squeeze(first_indices, axis=1))
    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), chunk_goal[:, None])

    # === 根据 proprio 差值过滤静止片段 ===
    # 这里直接在原始时间索引上计算首尾位移，与 `chunk_act_obs_uniform_resample` 一致
    num_chunks = tf.shape(obs_chunk_indices)[0]
    keep_indices = tf.range(num_chunks, dtype=tf.int32)
    last_t = tf.maximum(num_chunks - 1, 0)

    if proprio_threshold_min > 0 and proprio_threshold_max > 0 and "proprio" in traj["observation"]:
        # keep_indices: [N_kept]，此处初始为 [0..T-1]
        kept_chunk_indices = tf.gather(obs_chunk_indices, keep_indices)  # [N_kept, 3]
        floored_kept_indices = tf.maximum(
            tf.minimum(kept_chunk_indices, tf.maximum(traj_len - 1, 0)),
            0,
        )

        # [N_kept, 3, proprio_dim]
        kept_proprio = tf.gather(traj["observation"]["proprio"], floored_kept_indices)

        # 计算窗口首尾的位移 (L2 norm)
        delta = kept_proprio[:, -1, :3] - kept_proprio[:, 0, :3]
        displacement = tf.norm(delta, axis=-1)  # [N_kept]

        # 检查夹爪状态变化 (假设最后一维是夹爪状态)
        gripper_delta = tf.abs(kept_proprio[:, -1, -1] - kept_proprio[:, 0, -1])
        gripper_changed = gripper_delta > 0.5

        # 位移必须小于最大值，即使有夹爪动作也要满足这个条件
        is_moving = ((displacement > proprio_threshold_min) | gripper_changed) & (
            displacement < proprio_threshold_max
        )

        # 如果存在移动的片段，则仅保留移动的；否则（全静止）只保留最后一个窗口
        has_moving = tf.reduce_any(is_moving)
        keep_indices = tf.cond(
            has_moving,
            lambda: tf.boolean_mask(keep_indices, is_moving),
            lambda: keep_indices[-1:] if tf.size(keep_indices) > 0 else keep_indices,
        )

    # 若无任何合法窗口，兜底保留最后一个窗口
    keep_indices = tf.cond(
        tf.size(keep_indices) > 0,
        lambda: keep_indices,
        lambda: tf.cond(
            num_chunks > 0,
            lambda: tf.reshape(last_t, [1]),
            lambda: tf.zeros([0], dtype=tf.int32),
        ),
    )

    # 先构造 chunk 结果，再按 keep_indices 构建输出
    chunked_obs = tf.nest.map_structure(
        lambda x: tf.gather(x, floored_obs_chunk_indices), traj["observation"]
    )  # [T, 3, ...]
    chunked_act = tf.gather(traj["action"], floored_action_chunk_indices)  # [T, 2, act_dim]
    chunked_pad = obs_chunk_indices >= 0  # [T, 3]

    new_traj = {}
    new_traj["observation"] = tf.nest.map_structure(
        lambda x: tf.gather(x, keep_indices), chunked_obs
    )  # [N_kept, 3, ...]
    new_traj["observation"]["pad_mask"] = tf.gather(chunked_pad, keep_indices)
    new_traj["action"] = tf.gather(chunked_act, keep_indices)

    # 处理 task 字段（若存在）
    if "task" in traj:
        new_traj["task"] = tf.nest.map_structure(
            lambda x: tf.gather(x, keep_indices)
            if hasattr(x, "shape") and len(x.shape) > 0 and tf.shape(x)[0] == traj_len
            else x,
            traj["task"],
        )

    # 处理其他按时间维对齐的字段
    for key in traj:
        if key not in ["observation", "action", "task"]:
            if hasattr(traj[key], "shape") and len(traj[key].shape) > 0 and tf.shape(traj[key])[0] == traj_len:
                new_traj[key] = tf.gather(traj[key], keep_indices)
            else:
                new_traj[key] = traj[key]

    # 绝对/相对动作的中性处理（保持与原逻辑一致）
    if "absolute_action_mask" not in traj and future_action_window_size > 0:
        logging.warning(
            "future_action_window_size > 0 but no absolute_action_mask was provided. "
            "Assuming all actions are relative for the purpose of making neutral actions."
        )
    absolute_action_mask = traj.get("absolute_action_mask", tf.zeros([traj_len, action_dim], dtype=tf.bool))
    sampled_mask = tf.gather(absolute_action_mask, keep_indices)
    neutral_actions = tf.where(
        sampled_mask[:, None, :],
        new_traj["action"],
        tf.zeros_like(new_traj["action"]),
    )

    reduced_action_chunk_indices = tf.gather(action_chunk_indices, keep_indices)
    reduced_goal = tf.gather(chunk_goal, keep_indices)
    action_past_goal = reduced_action_chunk_indices > reduced_goal[:, None]
    new_traj["action"] = tf.where(action_past_goal[:, :, None], neutral_actions, new_traj["action"])

    return new_traj


def chunk_act_obs_libero(traj: Dict, window_size: int, future_action_window_size: int = 0) -> Dict:
    """
    Chunks actions and observations into the given window_size.

    "observation" keys are given a new axis (at index 1) of size `window_size` containing `window_size - 1`
    observations from the past and the current observation. "action" is given a new axis (at index 1) of size
    `window_size + future_action_window_size` containing `window_size - 1` actions from the past, the current
    action, and `future_action_window_size` actions from the future. "pad_mask" is added to "observation" and
    indicates whether an observation should be considered padding (i.e. if it had come from a timestep
    before the start of the trajectory).
    """
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]
    chunk_indices = tf.broadcast_to(tf.range(-window_size + 1, 1), [traj_len, window_size]) + tf.broadcast_to(
        tf.range(traj_len)[:, None], [traj_len, window_size]
    )
    # print('chunk_indices', chunk_indices)
    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(traj_len)[:, None],
        [traj_len, window_size + future_action_window_size],
    )

    floored_chunk_indices = tf.maximum(chunk_indices, 0)

    if "timestep" in traj["task"]:
        goal_timestep = traj["task"]["timestep"]
    else:
        goal_timestep = tf.fill([traj_len], traj_len - 1)

    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])

    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj["observation"])
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    # indicates whether an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # if no absolute_action_mask was provided, assume all actions are relative
    if "absolute_action_mask" not in traj and future_action_window_size > 0:
        logging.warning(
            "future_action_window_size > 0 but no absolute_action_mask was provided. "
            "Assuming all actions are relative for the purpose of making neutral actions."
        )
    absolute_action_mask = traj.get("absolute_action_mask", tf.zeros([traj_len, action_dim], dtype=tf.bool))
    neutral_actions = tf.where(
        absolute_action_mask[:, None, :],
        traj["action"],  # absolute actions are repeated (already done during chunking)
        tf.zeros_like(traj["action"]),  # relative actions are zeroed
    )

    # actions past the goal timestep become neutral
    action_past_goal = action_chunk_indices > goal_timestep[:, None]
    traj["action"] = tf.where(action_past_goal[:, :, None], neutral_actions, traj["action"])

    return traj


def chunk_act_obs_uniform_resample(
    traj: Dict,
    window_size: int,
    future_action_window_size: int = 0,  # 保持签名兼容，实际不使用
    *,
    fixed_obs_len: int = 5,
    proprio_threshold_min: float = 0.1,
    proprio_threshold_max: float = 0.8,
) -> Dict:
    """
    基于等距重采样的简化版本：
    - 在物理时间跨度 [-window_size+1, 0] 内，等距采样 fixed_obs_len 帧作为 observation。
    - 动作与观测使用相同的时间索引（不使用 future_action_window_size）。
    - 输出时间维固定为 fixed_obs_len，便于跨数据集对齐。
    - proprio_threshold_min 和 proprio_threshold_max: 若 > 0，则计算窗口首尾 proprio 的欧氏距离，若小于该阈值则丢弃该 chunk（静止片段）。
    """
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]

    # 等距偏移：[-window_size+1, 0] -> fixed_obs_len 个点
    obs_offsets = tf.cast(
        tf.round(
            tf.linspace(
                tf.cast(-window_size + 1, tf.float32),
                tf.cast(0, tf.float32),
                fixed_obs_len,
            )
        ),
        tf.int32,
    )  # [fixed_obs_len]

    # 观测与动作使用相同的时间索引（按时间 t 构造 chunk）
    all_t = tf.range(traj_len, dtype=tf.int32)
    chunk_indices = tf.broadcast_to(obs_offsets, [traj_len, fixed_obs_len]) + tf.broadcast_to(
        all_t[:, None], [traj_len, fixed_obs_len]
    )
    floored_chunk_indices = tf.maximum(chunk_indices, 0)

    # goal timestep 处理
    if "timestep" in traj["task"]:
        goal_timestep = traj["task"]["timestep"]
    else:
        goal_timestep = tf.fill([traj_len], traj_len - 1)

    floored_action_chunk_indices = tf.minimum(tf.maximum(chunk_indices, 0), goal_timestep[:, None])

    # === 仅保留从 t >= window_size-1 开始，按 stride 采样的窗口；并强制包含最后一个窗口，减少尾部重复 ===
    stride = tf.maximum(window_size // 2, 1)
    start_t = tf.maximum(window_size - 1, 0)
    last_t = tf.maximum(traj_len - 1, 0)

    # 安全 range：若轨迹过短，令 limit = max(traj_len, start_t)，避免 start > limit 报错
    limit_for_range = tf.maximum(traj_len, start_t)
    base_indices = tf.range(start_t, limit_for_range, delta=stride, dtype=tf.int32)
    # 若最后一个索引未命中，强制补上最后一个窗口（空序列时不访问 base_indices[-1]）
    need_append_last = tf.cond(
        tf.size(base_indices) > 0,
        lambda: tf.not_equal(base_indices[-1], last_t),
        lambda: tf.constant(False),
    )
    keep_indices = tf.cond(
        need_append_last,
        lambda: tf.concat([base_indices, tf.reshape(last_t, [1])], axis=0),
        lambda: base_indices,
    )
    # 过滤掉包含越界索引（负索引，被 0 填充导致重复）的窗口
    # 不改变窗口内等间隔取样（允许窗口内索引重复），仅移除发生越界填充的整段窗口
    window_all_in_range = tf.reduce_all(chunk_indices >= 0, axis=1)  # [traj_len]
    keep_indices = tf.boolean_mask(keep_indices, tf.gather(window_all_in_range, keep_indices))
    
    # === 根据 proprio 差值过滤静止片段 ===
    # 对于 ego4d、agibot 等无有效状态的数据集，跳过该过滤逻辑，直接使用基于 stride 的窗口采样结果。
    if proprio_threshold_min > 0 and proprio_threshold_max > 0 and "proprio" in traj["observation"]:
        # 判断当前轨迹是否来自 ego4d 或 agibot（dataset_name 在 make_dataset_from_rlds 中被写入）
        if "dataset_name" in traj:
            ds_name0 = traj["dataset_name"][0]
            is_ego4d = tf.strings.regex_full_match(ds_name0, ".*ego4d.*")
            is_agibot = tf.strings.regex_full_match(ds_name0, ".*agibot.*")
            is_ego4d = tf.logical_or(is_ego4d, is_agibot)
        else:
            is_ego4d = tf.constant(False)

        def apply_motion_filter():
            nonlocal keep_indices
            # 获取候选窗口的 proprio 数据
            # keep_indices: [N_kept]
            # chunk_indices: [traj_len, fixed_obs_len]
            kept_chunk_indices = tf.gather(chunk_indices, keep_indices)  # [N_kept, fixed_obs_len]
            floored_kept_indices = tf.maximum(kept_chunk_indices, 0)
            
            # [N_kept, fixed_obs_len, proprio_dim]
            kept_proprio = tf.gather(traj["observation"]["proprio"], floored_kept_indices)
            
            # 计算窗口首尾的位移 (L2 norm)
            # kept_proprio[:, -1] 是当前时刻 t，kept_proprio[:, 0] 是 t - window_size + 1
            delta = kept_proprio[:, -1, :3] - kept_proprio[:, 0, :3]
            displacement = tf.norm(delta, axis=-1)  # [N_kept]
            
            # 检查夹爪状态变化 (假设最后一维是夹爪状态)
            # 如果夹爪状态变化超过阈值（例如 0.1），则认为是有效动作
            gripper_delta = tf.abs(kept_proprio[:, -1, -1] - kept_proprio[:, 0, -1])
            gripper_changed = gripper_delta > 0.5

            # 位移必须小于最大值，即使有夹爪动作也要满足这个条件
            is_moving = ((displacement > proprio_threshold_min) | gripper_changed) & (
                displacement < proprio_threshold_max
            )
            
            # 如果存在移动的片段，则仅保留移动的；否则（全静止）只保留最后一个窗口
            # 全静止时只保留一个窗口，避免批次间数据分布不均匀
            has_moving = tf.reduce_any(is_moving)
            return tf.cond(
                has_moving,
                lambda: tf.boolean_mask(keep_indices, is_moving),
                lambda: keep_indices[-1:] if tf.size(keep_indices) > 0 else keep_indices,  # 全静止时只保留最后一个
            )

        def skip_motion_filter():
            # 不基于 proprio 做任何过滤，直接返回原始 keep_indices
            return keep_indices

        keep_indices = tf.cond(is_ego4d, skip_motion_filter, apply_motion_filter)

    # 若无任何合法窗口，兜底保留最后一个窗口（允许使用 padding）
    keep_indices = tf.cond(
        tf.size(keep_indices) > 0,
        lambda: keep_indices,
        lambda: tf.reshape(last_t, [1]),
    )

    # 先构造 chunk 结果，再按 keep_indices（已兜底非空）构建输出
    chunked_obs = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj["observation"])  # [T,fixed_obs_len,...]
    chunked_act = tf.gather(traj["action"], floored_action_chunk_indices)  # [T,fixed_obs_len,act]
    chunked_pad = chunk_indices >= 0  # [T,fixed_obs_len]

    new_traj = {}
    new_traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, keep_indices), chunked_obs)
    new_traj["observation"]["pad_mask"] = tf.gather(chunked_pad, keep_indices)
    new_traj["action"] = tf.gather(chunked_act, keep_indices)
    if "task" in traj:
        new_traj["task"] = tf.nest.map_structure(
            lambda x: tf.gather(x, keep_indices) if hasattr(x, 'shape') and len(x.shape) > 0 and tf.shape(x)[0] == traj_len else x,
            traj["task"],
        )
    for key in traj:
        if key not in ["observation", "action", "task"]:
            if hasattr(traj[key], 'shape') and len(traj[key].shape) > 0 and tf.shape(traj[key])[0] == traj_len:
                new_traj[key] = tf.gather(traj[key], keep_indices)
            else:
                new_traj[key] = traj[key]

    # 绝对/相对动作的中性处理（保持与原逻辑一致）
    if "absolute_action_mask" not in traj and future_action_window_size > 0:
        logging.warning(
            "future_action_window_size > 0 but no absolute_action_mask was provided. "
            "Assuming all actions are relative for the purpose of making neutral actions."
        )
    absolute_action_mask = traj.get("absolute_action_mask", tf.zeros([traj_len, action_dim], dtype=tf.bool))
    sampled_mask = tf.gather(absolute_action_mask, keep_indices)
    neutral_actions = tf.where(
        sampled_mask[:, None, :],
        new_traj["action"],
        tf.zeros_like(new_traj["action"]),
    )

    reduced_chunk_indices = tf.gather(chunk_indices, keep_indices)
    reduced_goal = tf.gather(goal_timestep, keep_indices)
    action_past_goal = reduced_chunk_indices > reduced_goal[:, None]
    new_traj["action"] = tf.where(action_past_goal[:, :, None], neutral_actions, new_traj["action"])

    return new_traj

def chunk_act_obs_half_stride(traj: Dict, window_size: int, future_action_window_size: int = 0) -> Dict:
    """
    使用半窗口步长对轨迹进行分块处理，步进大小为window_size//2
    
    与chunk_act_obs的区别：
    - 步进从1改为window_size//2，减少数据重叠
    - 确定性采样，避免1-3等特定时刻
    - 仍然只取首尾帧，保持内存效率
    
    Args:
        traj: 包含观测和动作的轨迹字典
        window_size: 窗口大小
        future_action_window_size: 未来动作窗口大小
        
    Returns:
        处理后的轨迹字典，chunk数量约为原来的1/(window_size//2)
        
    Examples:
        window_size=10 → stride=5 → 起始位置[0,5,10,15...] → 重叠率50%
        window_size=8  → stride=4 → 起始位置[0,4,8,12...]  → 重叠率50%
    """
    # 调试：打印轨迹字段信息
    if tf.executing_eagerly():  # 只在eager模式下打印
        tf.print("=== DEBUG: Trajectory keys ===")
        for key in traj.keys():
            if hasattr(traj[key], 'shape'):
                tf.print(f"  {key}: shape = {tf.shape(traj[key])}")
            else:
                tf.print(f"  {key}: {type(traj[key])}")
        tf.print("===============================")
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]
    
    # 计算步长：窗口大小的一半，至少为1
    stride = tf.maximum(window_size // 2, 1)
    
    # 生成采样的起始位置 - 确定性采样
    max_start = tf.maximum(traj_len - window_size, 0)
    sample_starts = tf.range(0, traj_len, stride, dtype=tf.int32)
    
    # 过滤掉超出范围的起始位置
    valid_starts = tf.boolean_mask(sample_starts, sample_starts <= max_start)
    
    # 确保至少有一个有效位置
    valid_starts = tf.cond(
        tf.size(valid_starts) > 0,
        lambda: valid_starts,
        lambda: tf.constant([0], dtype=tf.int32)
    )
    
    num_chunks = tf.shape(valid_starts)[0]
    
    # 创建chunk索引 - 仍然只取首尾帧
    first_indices = valid_starts[:, None]
    last_indices = tf.minimum(
        first_indices + (window_size - 1), 
        traj_len - 1
    )
    chunk_indices = tf.concat([first_indices, last_indices], axis=1)
    
    # 创建动作chunk索引  
    action_first_indices = first_indices
    action_last_indices = tf.minimum(
        first_indices + (window_size + future_action_window_size - 1),
        traj_len - 1
    )
    action_chunk_indices = tf.concat([action_first_indices, action_last_indices], axis=1)
    
    # 确保索引有效
    floored_chunk_indices = tf.maximum(tf.minimum(chunk_indices, traj_len - 1), 0)
    
    # 处理goal_timestep
    if "timestep" in traj["task"]:
        goal_timestep = tf.gather(traj["task"]["timestep"], valid_starts)
    else:
        goal_timestep = tf.fill([num_chunks], traj_len - 1)
        
    floored_action_chunk_indices = tf.minimum(
        tf.maximum(action_chunk_indices, 0), 
        goal_timestep[:, None]
    )
    
    # 重构轨迹 - 注意：这里改变了轨迹的长度
    new_traj = {}
    
    # 采样观测 - 只取首尾帧
    new_traj["observation"] = tf.nest.map_structure(
        lambda x: tf.gather(x, floored_chunk_indices), 
        traj["observation"]
    )
    
    # 采样动作
    new_traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)
    
    # 添加pad_mask
    new_traj["observation"]["pad_mask"] = chunk_indices >= 0
    
    # 处理task字段 - 需要重新采样
    if "task" in traj:
        new_traj["task"] = tf.nest.map_structure(
            lambda x: tf.gather(x, valid_starts) if hasattr(x, 'shape') and len(x.shape) > 0 and tf.shape(x)[0] == traj_len else x,
            traj["task"]
        )
    
    # 处理其他轨迹级别的字段
    for key in traj:
        if key not in ["observation", "action", "task"]:
            # 检查是否是轨迹长度相关的字段
            if hasattr(traj[key], 'shape') and len(traj[key].shape) > 0 and tf.shape(traj[key])[0] == traj_len:
                new_traj[key] = tf.gather(traj[key], valid_starts)
            else:
                new_traj[key] = traj[key]
    
    # 处理absolute_action_mask和neutral_actions
    if "absolute_action_mask" not in traj and future_action_window_size > 0:
        logging.warning(
            "future_action_window_size > 0 but no absolute_action_mask was provided. "
            "Assuming all actions are relative for the purpose of making neutral actions."
        )
    
    # 获取原始mask，然后采样
    absolute_action_mask = traj.get("absolute_action_mask", tf.zeros([traj_len, action_dim], dtype=tf.bool))
    sampled_mask = tf.gather(absolute_action_mask, valid_starts)
    
    # 创建neutral actions
    neutral_actions = tf.where(
        sampled_mask[:, None, :],
        new_traj["action"],  # absolute actions are repeated
        tf.zeros_like(new_traj["action"]),  # relative actions are zeroed
    )
    
    # 处理超过goal timestep的动作
    action_past_goal = action_chunk_indices > goal_timestep[:, None]
    new_traj["action"] = tf.where(
        action_past_goal[:, :, None], 
        neutral_actions, 
        new_traj["action"]
    )
    
    return new_traj


def subsample(traj: Dict, subsample_length: int) -> Dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)

    return traj


def add_pad_mask_dict(traj: Dict) -> Dict:
    """
    Adds a dictionary indicating which elements of the observation/task should be treated as padding.
        =>> traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]

    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            # Handles "language_instruction", "image_*", and "depth_*"
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0

            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]["pad_mask_dict"] = pad_mask_dict

    return traj


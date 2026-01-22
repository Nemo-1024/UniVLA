# LIBERO 评估问题诊断报告

## 🔍 问题现象
- 评估时成功率恒为 0%
- 可视化显示机械臂胡乱移动
- 代码可以正常运行，无报错

## 🎯 根本原因

### **核心问题：State/Proprio 构造方式不匹配**

#### **训练时的 State 构造**（来自 `prismatic/vla/datasets/rlds/oxe/configs.py`）
```python
"state_obs_keys": ["EEF_state", None, "gripper_state"]
```

根据 `transforms.py` 中的 `libero_dataset_transform`：
- `EEF_state`: observation["state"][:, :6]  → 6维（位置xyz + 轴角姿态）
- `None`: 填充 1 个零
- `gripper_state`: observation["state"][:, -1:]  → 1维（单个标量）

**总维度：6 + 1 + 1 = 8**

具体内容：
```python
EEF_state = [
    eef_pos[0],      # x位置
    eef_pos[1],      # y位置
    eef_pos[2],      # z位置
    axisangle[0],    # 轴角x分量
    axisangle[1],    # 轴角y分量
    axisangle[2],    # 轴角z分量
]  # 6维

padding = [0.0]  # 1维

gripper_state = [mean(gripper_qpos)]  # 1维，两个夹爪关节的平均值
```

#### **评估时的 State 构造**（原始的 `run_libero_eval.py`）
```python
"state": np.concatenate(
    (obs["robot0_eef_pos"],              # 3维：xyz位置
     quat2axisangle(obs["robot0_eef_quat"]),  # 3维：轴角姿态
     obs["robot0_gripper_qpos"])         # 2维：两个夹爪关节
)
```

**总维度：3 + 3 + 2 = 8**

#### **问题分析**

虽然维度都是 8，但是**内容和结构完全不同**：

| 索引 | 训练时 | 评估时（原始） | 是否匹配 |
|-----|--------|--------------|---------|
| 0-2 | eef_pos (xyz) | eef_pos (xyz) | ✅ 匹配 |
| 3-5 | axisangle | axisangle | ✅ 匹配 |
| 6   | 0.0 (padding) | gripper_qpos[0] | ❌ **不匹配** |
| 7   | gripper_state (scalar) | gripper_qpos[1] | ❌ **不匹配** |

**关键差异**：
1. 训练时第6维是固定的 0（padding），评估时是第一个夹爪关节值
2. 训练时第7维是单个gripper状态值（通常是两个关节的平均），评估时是第二个夹爪关节值

这导致：
- ✅ 前6维（位置+姿态）匹配 
- ❌ 后2维（gripper相关）完全不匹配
- 归一化参数应用错误
- 模型收到的输入分布与训练时完全不同
- 预测的动作完全错误

### 从 dataset_statistics.json 验证

```json
"proprio": {
  "mean": [..., 0.0, -0.030556727200746536],
  "std": [..., 0.0, 0.009197483770549297],
  "q01": [..., 0.0, -0.04015838988125324],
  "q99": [..., 0.0, -0.00910604484379293]
}
```

**第6维（索引6）的统计值全为 0**，证实训练时该维度确实是填充的零！

## ✅ 修复方案

已修改 `experiments/robot/libero/run_libero_eval.py`，使 state 构造与训练时完全一致：

```python
# 构造 EEF 状态（6维）
eef_state = np.concatenate([
    obs["robot0_eef_pos"],              # 3维：xyz位置
    quat2axisangle(obs["robot0_eef_quat"])  # 3维：轴角姿态
])

# 计算 gripper 状态（1维标量）- 使用两个关节的平均值
gripper_state = np.mean(obs["robot0_gripper_qpos"])

# 按照训练时的顺序构造 state: [EEF(6), padding(1), gripper(1)]
observation = {
    "full_image": img,
    "state": np.concatenate([
        eef_state,              # 6维
        np.array([0.0]),        # 1维 padding（匹配训练）
        np.array([gripper_state])  # 1维 gripper
    ]),
}
```

## 🔧 其他已验证的配置

### ✅ use_history_frame 配置一致
- 训练时：`use_history_frame: false`（finetune_libero_emb.yaml）
- 评估时：`use_history_frame: False`（run_libero_eval.py）
- **状态：匹配**

### ✅ Gripper 动作处理正确

训练时流程：
1. 原始 gripper ∈ [-1, 1] → clip 到 [0, 1]
2. Invert: `1 - x`（0→1, 1→0）
3. Mask=False，不归一化，保存为 [0, 1]（1=open, 0=close）

评估时流程：
1. 模型预测 gripper ∈ [0, 1]（1=open, 0=close）
2. Mask=False，不反归一化
3. `normalize_gripper_action`: [0, 1] → [-1, +1] → binarize → {-1, +1}
4. `invert_gripper_action`: {-1, +1} → {+1, -1}
5. 环境期望：-1=open, +1=close

映射验证：
- 模型预测 1（open）→ +1 → -1（环境的open）✅
- 模型预测 0（close）→ -1 → +1（环境的close）✅

## 📋 测试建议

1. **运行修复后的评估脚本**：
```bash
cd /mnt/project_rlinf/jlchen/code/UniVLA
python experiments/robot/libero/run_libero_eval.py
```

2. **观察首个时间步的 DEBUG 输出**：
```
[DEBUG] State构成: eef_pos=[...], axisangle=[...], gripper=...
```
确认 state 各维度的值在合理范围内

3. **对比 state 统计值**：
查看打印的 state min/max 是否与 dataset_statistics.json 中的 q01/q99 范围接近

## 🎓 经验教训

1. **推理时必须严格匹配训练时的数据预处理流程**
   - 不仅维度要匹配，内容和顺序也必须一致
   - 包括 padding 这样看似无用的维度

2. **dataset_statistics.json 是重要的调试线索**
   - 全零的统计值表明该维度在训练时被填充
   - 统计范围可以验证数据处理是否正确

3. **训练/推理代码路径不同时容易出错**
   - 训练使用 RLDS + transform pipeline
   - 推理直接从环境获取观测
   - 需要确保两边的预处理逻辑完全一致

## 📝 待验证

修复后需要验证：
1. 成功率是否提升
2. 机械臂运动是否合理（不再胡乱移动）
3. 各个任务的表现是否符合预期


# LatentVLA 双模式解耦重构计划

## 目标

- **把“token CE (latent action 索引监督)”与“`supervise_quantized` (连续 quantized 回归监督)”拆成两个类**，消除大量 `if self.supervise_quantized` 分支，让 forward/predict 逻辑更直观。
- **保留训练器依赖的行为**：
- `model.supervise_quantized` 属性用于跳过 action accuracy
- `forward()` 返回包含 `loss` 与 `logits` 的 dict（以及 `loss_main/loss_distill/loss_perceptual/loss_kl`）
- `save_pretrained()` 仍能保存 HF 权重，并在需要时额外保存 wrapper 参数（query / projection head）

## 方案（多文件 + 基类 + 两子类）

新增一个子包 `prismatic/vla/latent_vla/`：

- [`prismatic/vla/latent_vla/base.py`](prismatic/vla/latent_vla/base.py)
- `LatentVLABase(nn.Module)`：持有 `vlm/lam/processor` 与通用配置（token id、codebook size、debug 等）
- 通用方法下沉：
    - `train()`：始终保持 LAM eval
    - `_lam_vq_encode()`、`_maybe_debug_check_lam_codes()`
    - `_compute_kl_loss()`、`_sum_losses()`
    - `save_pretrained()` / `load_latent_vla_extra()`（extra 权重保存/加载放在基类；子类可通过钩子提供要保存的额外参数）
    - `state_dict()` / `load_state_dict()` 继续透传到 `self.vlm`（保持现有 checkpoint 行为）
- [`prismatic/vla/latent_vla/token_ce.py`](prismatic/vla/latent_vla/token_ce.py)
- `LatentVLATokenCE(LatentVLABase)`：只负责离散 action token 的 teacher-forcing 与 CE loss
- 从当前类里迁移并“定型”这些逻辑（不再出现 quantized 分支）：
    - `_prepare_vlm_token_inputs()`（替换 `<ACT_PH>` -> `<ACT_i>`）
    - `_vlm_forward()`：直接 `self.vlm(input_ids=..., labels=..., output_hidden_states=True)`
    - `forward()`：
    - LAM `vq_encode` 得到 `indices`/`quantized`
    - 构造 action token ids 替换输入/labels
    - 取 `vlm_out.loss` 为 `loss_main`
    - （可选）`enable_lam_kl_loss` 计算 `loss_kl`
    - 汇总 loss 并返回 dict（`loss_distill/loss_perceptual` 固定为 None）
- `predict_latent_actions()`：只保留当前 token 生成 + argmax 的路径
- [`prismatic/vla/latent_vla/quantized.py`](prismatic/vla/latent_vla/quantized.py)
- `LatentVLAQuantizedRegression(LatentVLABase)`：只负责 query 注入 + quantized 回归监督
- 从当前类里迁移并“定型”这些逻辑：
    - `act_query` 参数、`vlm_to_lam` 投影头、梯度放大 hook（`act_query_lr_scale` / `vlm_to_lam_lr_scale` / `new_token_lr_scale`）
    - `_vlm_forward()`：通过 `inputs_embeds` 在 `<ACT_PH>` 位置注入 query
    - `_project_action_hidden_to_lam()`、`_compute_quantized_loss()`
    - `_compute_decoder_perceptual_loss()`（仅该子类支持；否则无意义）
    - `forward()`：
    - `loss_main=None`（与当前一致）
    - `loss_distill`（quantized 回归）+ 可选 perceptual + 可选 KL
    - 继续保留 `total = total + 0*logits.sum()` 的 DDP safety 逻辑
- `predict_latent_actions()`：返回 `pred_quantized`，可选返回 teacher `lam_latent_idx/lam_quantized`
- [`prismatic/vla/latent_vla/factory.py`](prismatic/vla/latent_vla/factory.py)
- `build_latent_vla_from_config(cfg, overwatch, debug_mode, yaml_path=None, preserve_checkpoint_model_id=True)`：
    - 复用当前 `from_config` 的 VLM/LAM 加载、token 注册、(可选) YAML override、extra 权重加载逻辑
    - **根据 `cfg.supervise_quantized` 返回对应子类实例**
- [`prismatic/vla/latent_vla/__init__.py`](prismatic/vla/latent_vla/__init__.py)
- 导出 `LatentVLATokenCE` / `LatentVLAQuantizedRegression` / `build_latent_vla_from_config`

## 训练入口改动（按你选择的“允许改 API”）

- 更新 [`vla_scripts/train.py`](vla_scripts/train.py)：
- `from prismatic.vla.latent_vla import build_latent_vla_from_config`
- 替换 `LatentVLAModel.from_config(...)` 为 `build_latent_vla_from_config(...)`

## 向后兼容（可选但推荐）

为避免 notebooks/旧脚本断掉，我会让 [`prismatic/vla/latent_vla_model.py`](prismatic/vla/latent_vla_model.py) 保留一个很薄的兼容层：

- 提供同名 `LatentVLAModel.from_config(...)`，内部直接调用 `build_latent_vla_from_config(...)`
- 或者在文件顶层提供 `from_config = build_latent_vla_from_config` 并保留 `LatentVLAModel` 作为别名

## 校验点（重构后必须满足）
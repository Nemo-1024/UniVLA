# 未提交改动检查报告（对比分支：InternVL）

本报告针对当前工作区相对 **origin/InternVL** 的未提交改动进行潜在问题检查。

---

## 1. 配置未从 YAML 传入模型（潜在 Bug）

**位置**: `prismatic/vla/latent_vla_model.py` → `LatentVLAModel.from_config()`

**问题**: `from_config` 在调用 `cls(...)` 构造模型时，**没有**从 `cfg` 传入以下参数，导致 YAML 中若配置了也不会生效，始终使用构造函数默认值：

- `num_queries`（默认 8）
- `qformer_layers`（默认 1）
- `qformer_heads`（默认 8）
- `qformer_ffn_expansion`（默认 4.0）
- `qformer_dropout`（默认 0.0）

**建议**: 在 `from_config` 的 `model = cls(...)` 调用中增加上述参数，例如：

```python
num_queries=getattr(cfg, "num_queries", 8),
qformer_layers=getattr(cfg, "qformer_layers", 1),
qformer_heads=getattr(cfg, "qformer_heads", 8),
qformer_ffn_expansion=getattr(cfg, "qformer_ffn_expansion", 4.0),
qformer_dropout=getattr(cfg, "qformer_dropout", 0.0),
```

---

## 2. train.py 中 `latent_action_num_queries` 硬编码

**位置**: `vla_scripts/train.py` → `get_latent_vla_dataset_and_collator(..., latent_action_num_queries=8, ...)`

**问题**: 数据侧占位符数量写死为 `8`，若将来在配置中修改模型的 `num_queries`，会与数据 pipeline 不一致，导致 placeholder 数量不匹配报错。

**建议**: 改为使用模型侧一致来源，例如：

```python
latent_action_num_queries=latent_vla_model.num_queries,
```

这样与 `LatentVLAModel.num_queries` 始终一致；若后续在 `from_config` 中支持从 cfg 读取 `num_queries`，此处也会自动对齐。

---

## 3. 文件末尾缺少换行符

**位置**: `prismatic/vla/materialize.py`

**问题**: diff 显示 `No newline at end of file`，不符合常见代码规范（如 PEP 8 建议文本文件以换行符结尾）。

**建议**: 在 `materialize.py` 最后一行的 `return train_dataset, val_dataset, collator` 后增加一行换行。

---

## 4. PaddedCollatorForActionPrediction_LIBERO 与 LAM 接口不一致（理论风险，当前未使用）

**位置**: `prismatic/util/data_utils.py` → `PaddedCollatorForActionPrediction_LIBERO.__call__`

**问题**: 该 Collator 内调用了 `self.action_tokenizer.get_latent_action(..., dec_in=..., tgt=...)`，而 `LatentLAMModel.get_latent_action` 只接受 `dec_videos`，不接受 `dec_in`/`tgt`，若把 LAM 当 `action_tokenizer` 用会报 `TypeError`。

**当前用法**: 经确认，**现有代码并未使用** `PaddedCollatorForActionPrediction_LIBERO`：`finetune_libero.py` 使用的是 `PaddedCollatorForLatentWorldVLA_LIBERO`，`get_latent_vla_dataset_and_collator` 也从未传入该类或设置 `action_tokenizer`。因此当前不会触发该错误，仅属未使用代码的理论风险。

**建议**: 若后续仍不打算使用该类，可考虑删除或标注为遗留/备用；若打算使用并传入 LAM，再按 LAM 接口改 collator 或为 LAM 增加兼容参数。

---

## 5. LIBERO 占位符数量与模型一致性

**位置**: `prismatic/util/data_utils.py` → `PaddedCollatorForLatentWorldVLA_LIBERO`；`vla_scripts/finetune_libero.py`

**说明**: LIBERO 使用 `latent_action_num_queries=8` 与 `flow_action_num_queries`（默认 8），即共 8+8=16 个 `<ACT_PH>`。需确保 `LatentWorldVLA` / `LatentVLAModel` 侧对 act 与 flow 的 query 数量与 collator 一致（例如模型里也是 8 act + 8 flow）。当前默认配置下若模型与 collator 均为 8+8，则一致；若将来修改任一侧，需同步检查另一侧。

---

## 6. 已删除文件与引用（已确认无残留）

- `prismatic/vla/latent_vla_model_continuous.py` 已删除；已 grep，无剩余引用。
- `experiments/robot/simpler-bridge/`、`vla_scripts/config/vla_vq.yaml`、`dino_base_bn/ln/ln_2q.yaml` 等已删除；无代码引用。

---

## 7. 其他改动摘要（无问题或仅风格）

- **train.py**: 移除 `tensorflow.python.data.util.structure.NoneTensorSpec` 导入、`supervise_quantized` 改为固定使用 latent 回归、`target_seq_len` 与 `save_interval`/`max_steps` 默认值调整 —— 逻辑一致，无发现问题。
- **latent_vla_model.py**: 从 VQ/离散 token 监督改为连续 latent 回归、新增 `VLMToLAMQFormer`、统一使用 `lam.code_dim` 等 —— 与当前 LAM 的 `code_dim` 一致。
- **data_utils / datasets / materialize**: `load_camera_views`、`flow_placeholder_mask`、`wrist_videos` 等扩展 —— 与现有 forward 和 LIBERO 使用方式一致。
- **materialize.py**: 仅需补上文件末尾换行（见第 3 条）。

---

## 建议修复优先级

1. **高**: 第 1 条（若需要通过 YAML 配置 num_queries / QFormer）。第 4 条当前无实际用法，可忽略或后续清理/兼容。
2. **中**: 第 2 条（避免未来改 num_queries 时出错）；第 3 条（规范）。
3. **低**: 第 5 条（在修改 act/flow query 数量时做一次交叉核对即可）。

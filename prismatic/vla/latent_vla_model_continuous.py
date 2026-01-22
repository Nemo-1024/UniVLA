import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List
from typing import Dict, Optional, Any, Tuple

from latent_action_model.core.lam_model import LatentLAMModel, load_latent_action_model
from prismatic.models import load_vlm_auto


class LatentVLAModel(nn.Module):
    """
    组合模型：封装 VLM 与 LAM

    - 在 forward 中调用 LAM.vq_encode 取得离散 latent（indices）与连续表征（quantized）
    - 将占位符 token (<ACT_PH>) 替换为真实 <ACT_i>，并同步替换 labels
    - 计算主 CE 损失（VLM 内部）与 encoder 蒸馏损失（VLM hidden 对齐 LAM quantized）
    - decoder 感知损失接口预留，默认关闭
    """

    def __init__(
        self,
        vlm: nn.Module,
        lam: LatentLAMModel,
        *,
        action_token_begin_id: int,
        codebook_size: int,
        placeholder_token_id: int,
        enable_lam_encoder_distill: bool = False,
        enable_lam_decoder_perceptual: bool = False,
        lam_encoder_distill_weight: float = 1.0,
        lam_decoder_perceptual_weight: float = 0.0,
        vlm_hidden_dim: int = 1024,
        lam_code_dim: int = 256,
        processor: Optional[Any] = None,
        new_token_ids: Optional[Any] = None,
        new_token_lr_scale: float = 1.0,
        enable_lam_kl_loss: bool = False,
        lam_kl_weight: float = 1.0,
        lam_kl_temperature: float = 0.1,
        debug_mode: bool = False,
    ) -> None:
        super().__init__()
        self.vlm = vlm
        self.lam = lam.eval()
        for p in self.lam.parameters():
            p.requires_grad = False
        self.processor = processor

        self.action_token_begin_id = int(action_token_begin_id)
        self.codebook_size = int(codebook_size)
        self.placeholder_token_id = int(placeholder_token_id)

        self.enable_lam_encoder_distill = bool(enable_lam_encoder_distill)
        self.enable_lam_decoder_perceptual = bool(enable_lam_decoder_perceptual)
        self.lam_encoder_distill_weight = float(lam_encoder_distill_weight)
        self.lam_decoder_perceptual_weight = float(lam_decoder_perceptual_weight)
        self.new_token_lr_scale = float(new_token_lr_scale)
        self.enable_lam_kl_loss = bool(enable_lam_kl_loss)
        self.lam_kl_weight = float(lam_kl_weight)
        self.lam_kl_temperature = float(lam_kl_temperature)
        self.new_token_ids = list(new_token_ids) if new_token_ids is not None else []
        self.debug_mode = bool(debug_mode)
        # 训练时的 latent 数量（Q）；推理沿用同一数目
        nq = getattr(lam, "num_queries", None)
        if nq is None and hasattr(lam, "vq"):
            nq = getattr(lam.vq, "num_queries", None)
        self.num_queries = int(nq) if nq is not None else None

        # 若未启用 encoder 蒸馏，则投影层不参与训练，避免 DDP 报未使用参数
        if self.enable_lam_encoder_distill:
            self.vlm_to_lam = nn.Linear(vlm_hidden_dim, lam_code_dim)
        else:
            self.vlm_to_lam = None

        # 针对新增 token 行做梯度放大，实现“只对新增 embedding 提高有效学习率”
        if self.new_token_lr_scale != 1.0 and len(self.new_token_ids) > 0:
            try:
                emb = self.vlm.get_input_embeddings()
                # 构建放大系数并注册为 buffer，避免 device 不一致
                row_mask = torch.zeros(
                    emb.weight.size(0), device=emb.weight.device, dtype=emb.weight.dtype
                )
                for tid in self.new_token_ids:
                    tid_int = int(tid)
                    if 0 <= tid_int < row_mask.numel():
                        row_mask[tid_int] = 1.0
                scale = self.new_token_lr_scale
                factor = 1.0 + (scale - 1.0) * row_mask  # [vocab]
                self.register_buffer("_new_token_factor", factor, persistent=False)

                def _scale_grad(g):
                    # g: [vocab, dim]
                    if not hasattr(self, "_new_token_factor") or self._new_token_factor is None:
                        return g
                    f = self._new_token_factor
                    if f.device != g.device:
                        f = f.to(device=g.device, dtype=g.dtype)
                    else:
                        f = f.to(dtype=g.dtype)
                    return g * f.unsqueeze(1)

                emb.weight.register_hook(_scale_grad)
            except Exception:
                self._new_token_factor = None
                print("Failed to register new token gradient scaling hook")

    def train(self, mode: bool = True):
        """
        Keep LAM in eval mode even when the wrapper model is toggled to train.
        This avoids dropout/other train-time randomness inside the frozen LAM,
        which would otherwise lead to non-deterministic VQ indices.
        """
        super().train(mode)
        self.lam.eval()
        return self
    @property
    def tokenizer(self):
        return self.processor.tokenizer if self.processor is not None else None

    # 仅保存/加载 VLM 权重，保持与原有管线一致的 checkpoint 行为
    def save_pretrained(self, save_directory, **kwargs):
        return self.vlm.save_pretrained(save_directory, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.vlm.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.vlm.load_state_dict(state_dict, strict=strict)

    def _build_prefix_from_placeholders(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        *,
        strict_placeholder_count: bool,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        """
        截断到每个样本第一个 <ACT_PH> 之前作为 prefix，并对 batch 进行 padding。
        返回 (prefix_input_ids, prefix_attention_mask, placeholder_counts)
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        device = input_ids.device
        placeholder_mask = input_ids == self.placeholder_token_id  # [B, L]
        counts = placeholder_mask.sum(dim=1)  # [B]

        if self.num_queries is not None and strict_placeholder_count:
            mismatch = counts != self.num_queries
            if mismatch.any():
                raise ValueError(
                    f"[LatentVLAModel] placeholder count mismatch: "
                    f"found={counts.tolist()}, expected={int(self.num_queries)}"
                )

        B, L = input_ids.shape
        idx_range = torch.arange(L, device=device)
        first_pos = torch.where(
            placeholder_mask, idx_range.unsqueeze(0).expand_as(input_ids), torch.full_like(input_ids, L)
        ).min(dim=1).values  # [B]

        prefixes: List[torch.Tensor] = []
        prefix_masks: List[torch.Tensor] = []
        for b in range(B):
            cut = int(first_pos[b].item())
            prefixes.append(input_ids[b, :cut])
            prefix_masks.append(attention_mask[b, :cut])

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.vlm.config, "pad_token_id", 0)

        prefix_input_ids = pad_sequence(prefixes, batch_first=True, padding_value=pad_token_id)
        prefix_attention_mask = pad_sequence(prefix_masks, batch_first=True, padding_value=0)
        return prefix_input_ids, prefix_attention_mask, counts

    @classmethod
    def from_config(cls, cfg: Any, overwatch=None, debug_mode: bool = False) -> Tuple["LatentVLAModel", Any]:
        """
        内部加载 VLM 与 LAM，返回 (model, processor)
        占位符 id 可通过 model.placeholder_token_id 访问
        """
        if overwatch is None:
            import logging
            overwatch = logging.getLogger(__name__)

        overwatch.info(f"🔄 加载基础 VLM `{cfg.model_id}`（HF from_pretrained, generic）")
        vlm, processor = load_vlm_auto(cfg.model_id, cfg.hf_cache_dir, dtype=torch.bfloat16)
        tokenizer = processor.tokenizer
        vlm.generation_config.max_new_tokens = int(getattr(cfg, "max_new_tokens", 4))
        vlm.config.loss_type = str(getattr(cfg, "loss_type", "ForCausalLMLoss"))
        vlm.config.use_cache = False

        overwatch.info(f"🔄 加载 LAM（yaml=`{cfg.lam_yaml_path}`）")
        lam = load_latent_action_model(cfg.lam_ckpt_path, cfg.lam_yaml_path)

        # 注册动作 token 与占位符
        special_tokens_dict = {"additional_special_tokens": [f"<ACT_{i}>" for i in range(lam.vq.codebook_size)]}
        tokenizer.add_special_tokens(special_tokens_dict)  # type: ignore[attr-defined]
        placeholder_token = getattr(cfg, "latent_action_placeholder_token", "<ACT_PH>")
        tokenizer.add_special_tokens({"additional_special_tokens": [placeholder_token]})  # type: ignore[attr-defined]


        # 打印当前 tokenizer 词表大小与 VLM embedding 大小，检查是否超出预留长度
        try:
            vocab_size = len(tokenizer)
            emb = vlm.get_input_embeddings()
            num_embeddings = emb.num_embeddings if hasattr(emb, "num_embeddings") else None
            overwatch.info(
                f"[LatentVLAModel] tokenizer vocab_size={vocab_size}, "
                f"vlm.num_embeddings={num_embeddings}"
            )
            if num_embeddings is not None and vocab_size > num_embeddings:
                overwatch.info(
                    f"[LatentVLAModel] WARNING: vocab_size({vocab_size}) > num_embeddings({num_embeddings}); "
                    "请确认 VLM 是否为本地权重并已预留足够 embedding 空位。"
                )
        except Exception:
            pass

        act_tokens = [f"<ACT_{i}>" for i in range(lam.vq.codebook_size)]
        act_ids = tokenizer.convert_tokens_to_ids(act_tokens)
        action_token_begin_id = min(act_ids)
        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

        new_token_ids = act_ids

        model = cls(
            vlm=vlm,
            lam=lam,
            action_token_begin_id=action_token_begin_id,
            codebook_size=lam.vq.codebook_size,
            placeholder_token_id=placeholder_token_id,
            enable_lam_encoder_distill=getattr(cfg, "enable_lam_encoder_distill", False),
            enable_lam_decoder_perceptual=getattr(cfg, "enable_lam_decoder_perceptual", False),
            lam_encoder_distill_weight=getattr(cfg, "lam_encoder_distill_weight", 1.0),
            lam_decoder_perceptual_weight=getattr(cfg, "lam_decoder_perceptual_weight", 0.0),
            enable_lam_kl_loss=getattr(cfg, "enable_lam_kl_loss", False),
            lam_kl_weight=getattr(cfg, "lam_kl_weight", 1.0),
            lam_kl_temperature=getattr(cfg, "lam_kl_temperature", 1.0),
            vlm_hidden_dim=vlm.config.text_config.hidden_size,
            lam_code_dim=lam.vq.code_dim,
            processor=processor,
            new_token_ids=new_token_ids,
            new_token_lr_scale=getattr(cfg, "new_token_lr_scale", 1.0),
            debug_mode=bool(debug_mode),
        )

        # 同步 cfg 中的字段（便于后续使用）
        cfg.action_token_begin_id = action_token_begin_id
        cfg.codebook_size = lam.vq.codebook_size

        return model, processor

    def forward(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        act_placeholder_mask: torch.BoolTensor,
        lam_videos: torch.FloatTensor,
        lam_states: torch.FloatTensor,
        lam_dec_videos: Optional[torch.FloatTensor] = None,
        lam_dataset_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # 1) LAM 编码：获取离散 codes 与连续 teacher 表征
        # LAM 视觉编码 + VQ：关闭 autocast，使用 FP32 以获得稳定的 code/特征
        with torch.no_grad():
            device_type = lam_videos.device.type if hasattr(lam_videos, "device") else "cuda"
            with torch.autocast(device_type=device_type, enabled=False):
                vq_out = self.lam.vq_encode(
                    videos=lam_videos,
                    states=lam_states,
                    dec_videos=lam_dec_videos,
                    predict_future_frame=self.enable_lam_decoder_perceptual,
                    dataset_ids=lam_dataset_ids,
                    return_teacher_probs=self.enable_lam_kl_loss,
                    teacher_temperature=self.lam_kl_temperature,
                )

        # 可选：校验 latent_codes 是否与首次迭代一致（仅用于 debug_repeat_batch）
        if self.debug_mode:
            try:
                codes = vq_out.get("indices", None)
                if codes is not None:
                    codes = codes.detach().clone()
                    if not hasattr(self, "_ref_latent_codes"):
                        self._ref_latent_codes = None  # type: ignore[attr-defined]
                    if self._ref_latent_codes is None:
                        self._ref_latent_codes = codes  # type: ignore[attr-defined]
                    else:
                        ref_codes = self._ref_latent_codes.to(codes.device)  # type: ignore[attr-defined]
                        if ref_codes.shape == codes.shape:
                            if not torch.equal(codes, ref_codes):
                                mismatch = (codes != ref_codes).sum().item()
                                print(f"[LamCodes Debug] mismatch found, count = {mismatch}")
                                print(codes.cpu().numpy())
                                print(ref_codes.cpu().numpy())
                        else:
                            print(
                                "[LamCodes Debug] shape mismatch, ref vs current: "
                                f"{tuple(ref_codes.shape)} vs {tuple(codes.shape)}; skip diff"
                            )
                else:
                    print("[LamCodes Debug] codes is None; skipping diff (vq_training=True?)")
            except Exception as e:
                print(f"[LamCodes Debug] diff check failed: {e}")

        latent_codes = vq_out["indices"].detach().clone()  # [B, Q]
        # print(latent_codes.cpu().numpy())
        # print(lam_states[0].cpu().numpy())
        # vq_encode 在 inference_mode 下返回 inference tensor，克隆以参与反传
        z_teacher = vq_out["quantized"].detach().clone()  # [B, Q, D_lam]

        # 2) codes -> <ACT_i> token id，并替换占位符
        latent_codes = latent_codes.to(input_ids.device)
        z_teacher = z_teacher.to(input_ids.device)
        act_token_ids = self.action_token_begin_id + latent_codes  # [B, Q]

        input_ids_flat = input_ids.view(-1)
        mask_flat = act_placeholder_mask.view(-1)
        act_ids_flat = act_token_ids.view(-1)
        assert mask_flat.sum().item() == act_ids_flat.numel(), "占位符数量与 latent 数量不匹配。mask_flat.sum() = {}, act_ids_flat.numel() = {}".format(mask_flat.sum().item(), act_ids_flat.numel())

        input_ids_flat[mask_flat] = act_ids_flat
        input_ids_replaced = input_ids_flat.view_as(input_ids)

        if labels is not None:
            labels_flat = labels.view(-1)
            labels_flat[mask_flat] = act_ids_flat
            labels_replaced = labels_flat.view_as(labels)
        else:
            labels_replaced = None

        # 3) VLM 前向
        vlm_out = self.vlm(
            input_ids=input_ids_replaced,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels_replaced,
            output_hidden_states=True,
        )
        hidden = vlm_out.hidden_states[-1]  # [B, L, D_vlm]
        logits = vlm_out.logits
        loss_main = getattr(vlm_out, "loss", None)

        # 4) encoder 蒸馏：动作 hidden 对齐 LAM quantized
        loss_distill = None
        if self.enable_lam_encoder_distill:

            h_act = hidden[act_placeholder_mask]  # [B*Q, D_vlm]
            B, Q = latent_codes.shape
            h_act = h_act.view(B, Q, -1)  # [B, Q, D_vlm]
            h_proj = self.vlm_to_lam(h_act)  # [B, Q, D_lam]
            loss_distill = 1- F.cosine_similarity(h_proj, z_teacher, dim=-1).mean()

        # 5) decoder 感知损失（预留，默认关闭）
        loss_perceptual = None
        if self.enable_lam_decoder_perceptual and self.lam_decoder_perceptual_weight > 0:
            # 可根据需要填充 recon/tgt 感知损失
            pass

        # 5) KL(p_teacher || q_student)：teacher 来自 LAM 的软分布，student 为 VLM 动作 logits
        loss_kl = None
        if self.enable_lam_kl_loss:
            teacher_probs = vq_out.get("vq_probs", None)
            if teacher_probs is not None:
                # student: 仅抽取动作 token 段 logits
                act_token_range = torch.arange(
                    self.codebook_size, device=logits.device
                ) + self.action_token_begin_id
                student_logits_all = logits.index_select(dim=-1, index=act_token_range)

                # 对齐因果 LM 的 “logits 预测下一个 token”，需左移一位
                mask_student = act_placeholder_mask[:, 1:]
                student_logits = student_logits_all[:, :-1, :]

                # 若出现极端情况（首位就是占位符导致数量不匹配），回退到不移位
                num_targets = teacher_probs.numel() // teacher_probs.shape[-1]
                if mask_student.sum().item() != num_targets:
                    mask_student = act_placeholder_mask
                    student_logits = student_logits_all

                student_log_probs = F.log_softmax(student_logits, dim=-1)
                # 使用展平后的 mask 与 logits 对齐，避免 2D mask 直接索引 3D 张量带来的歧义
                mask_flat = mask_student.reshape(-1)
                student_log_probs_flat = student_log_probs.reshape(-1, student_log_probs.shape[-1])
                student_log_probs_flat = student_log_probs_flat[mask_flat]

                teacher_probs = teacher_probs.to(device=student_log_probs_flat.device)
                teacher_probs_flat = teacher_probs.view(-1, teacher_probs.shape[-1])
                teacher_probs_flat = teacher_probs_flat.to(student_log_probs_flat.dtype)
                if student_log_probs_flat.shape[0] != teacher_probs_flat.shape[0]:
                    raise ValueError(
                        f"[LatentVLAModel] KL shape mismatch: student={student_log_probs_flat.shape} "
                        f"teacher={teacher_probs_flat.shape}, mask_sum={mask_flat.sum().item()}, "
                        f"teacher_targets={teacher_probs_flat.shape[0]}"
                    )
                loss_kl = F.kl_div(
                    student_log_probs_flat,
                    teacher_probs_flat,
                    reduction="mean",
                )

        # 6) 总损失
        total_loss = torch.tensor(0.0, device=input_ids.device, dtype=hidden.dtype)
        if loss_main is not None:
            total_loss = total_loss + loss_main
        if loss_distill is not None:
            total_loss = total_loss + self.lam_encoder_distill_weight * loss_distill
        if loss_perceptual is not None:
            total_loss = total_loss + self.lam_decoder_perceptual_weight * loss_perceptual
        if loss_kl is not None:
            total_loss = total_loss +self.lam_kl_weight * loss_kl

        return {
            "loss": total_loss,
            "loss_main": loss_main,
            "loss_distill": loss_distill,
            "loss_perceptual": loss_perceptual,
            "loss_kl": loss_kl,
            "logits": logits,
        }

    @torch.inference_mode()
    def predict_latent_actions(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        lam_videos: torch.FloatTensor,
        lam_states: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        lam_dec_videos: Optional[torch.FloatTensor] = None,
        lam_dataset_ids: Optional[torch.Tensor] = None,
        act_placeholder_mask: Optional[torch.BoolTensor] = None,
        strict_placeholder_count: bool = True,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        推理：同时返回
          - lam_latent_idx: LAM vq_encode 的 teacher indices，形状 [B, Q]
          - vlm_latent_idx: VLM generate 在动作词表上的 argmax indices，形状 [B, Q]
        Q 由训练时配置 self.num_queries 决定；若缺失则尝试从 LAM 输出推断。
        """
        if self.num_queries is None:
            raise ValueError("[LatentVLAModel] num_queries is None; ensure LAM提供了 num_queries")
        Q = int(self.num_queries)

        # 1) LAM 编码（teacher）
        device_type = lam_videos.device.type if hasattr(lam_videos, "device") else "cuda"
        with torch.no_grad():
            with torch.autocast(device_type=device_type, enabled=False):
                vq_out = self.lam.vq_encode(
                    videos=lam_videos,
                    states=lam_states,
                    dec_videos=lam_dec_videos,
                    predict_future_frame=False,
                    dataset_ids=lam_dataset_ids,
                    return_teacher_probs=False,
                )

        lam_latent_idx = vq_out["indices"].detach()
        if lam_latent_idx.shape[1] != Q:
            raise ValueError(
                f"[LatentVLAModel] LAM indices length mismatch: got {lam_latent_idx.shape[1]}, expected {Q}"
            )
        lam_latent_idx = lam_latent_idx.to(device=input_ids.device)

        # 2) 构造生成前缀（去掉占位符）
        placeholder_mask = (
            act_placeholder_mask.to(dtype=torch.bool, device=input_ids.device)
            if act_placeholder_mask is not None
            else input_ids == self.placeholder_token_id
        )
        if placeholder_mask.any():
            prefix_input_ids, prefix_attention_mask, _ = self._build_prefix_from_placeholders(
                input_ids=input_ids,
                attention_mask=attention_mask,
                strict_placeholder_count=strict_placeholder_count,
            )
        else:
            prefix_input_ids = input_ids
            prefix_attention_mask = attention_mask

        # 3) VLM 生成 Q 个动作 token，并在动作词表上取 argmax
        gen_kwargs = dict(generate_kwargs or {})
        gen_kwargs.setdefault("min_new_tokens", Q)
        gen_kwargs.setdefault("max_new_tokens", Q)
        gen_kwargs.setdefault("do_sample", False)
        gen_kwargs.setdefault("return_dict_in_generate", True)
        gen_kwargs.setdefault("output_scores", True)

        gen_out = self.vlm.generate(
            input_ids=prefix_input_ids,
            attention_mask=prefix_attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **gen_kwargs,
        )
        scores = gen_out.scores
        if scores is None or len(scores) < Q:
            raise ValueError(f"[LatentVLAModel] generate returned insufficient scores, len={0 if scores is None else len(scores)}")

        act_token_range = torch.arange(self.codebook_size, device=scores[0].device) + self.action_token_begin_id
        step_indices = []
        for s in scores[:Q]:
            act_logits = s.index_select(dim=-1, index=act_token_range)  # [B, K]
            step_indices.append(torch.argmax(act_logits, dim=-1))  # [B]
        vlm_latent_idx = torch.stack(step_indices, dim=0).transpose(0, 1).contiguous()  # [B, Q]

        return {
            "lam_latent_idx": lam_latent_idx,
            "vlm_latent_idx": vlm_latent_idx.to(device=input_ids.device),
        }
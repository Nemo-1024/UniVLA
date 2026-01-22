import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Dict, List, Optional, Any
from .vq import VQ, NSVQ, EMAVQ, AEQuantizer
from .vjepa_encoder import build_vision_encoder, DINOv3Encoder

import torch
import torch.nn as nn
import math
import yaml
from .utils.lam_encoder import LAMEncoder
from .utils.lam_decoder import LAMDecoder, LAMDecoder_v2, StatePredictor
from .utils.modules import PatchEmbed





class LatentLAMModel(nn.Module):
    """
    LAM主模型：自动实例化Encoder/Decoder/NSVQ，QFormer实现稀疏离散化。
    现在包含物理接地状态差解码器。
    """
    def __init__(
        self,
        dim: int=1024,
        num_heads: int = 16,
        ffn_expansion_factor: int = 2,
        enc_layers: int = 6,
        codebook_size: int = 16,
        code_dim: int = 256,
        num_frames: int = 5,
        num_queries: int = 1,
        ar_prediction: bool = False,
        vq_kwargs: Optional[Dict[str, Any]] = None,
        dec_layers: int = 6,
        dropout: float = 0.1,
        # 新增：状态差预测器参数
        enable_state_delta_prediction: bool = True,
        vq_type: str = "nsvq",
        disable_vq: bool = False,
        norm_latents: bool = False,
        norm_latents_type: str = "l2",
        vision_model_id: str = "facebook/vjepa2-vitl-fpc64-256",
        enc_add_state: bool = False,
        enc_modal_mask: bool = False,
        latent_layer_to_use: Any = 23,
        multi_input: bool = False,
        dataset_vocab_size: int = 16,
        **kwargs
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 集成视觉编码器：负责将 videos 编码为 [B, T, N, D] 特征
        # 依据 latent_layer_to_use 的长度控制 DINO 中 BatchNorm 的个数：
        # - 若为单层（int），只需一个 BN
        # - 若为多层（list/tuple），为每个层位分配独立 BN
        if isinstance(latent_layer_to_use, (list, tuple)):
            num_latent_layers = len(latent_layer_to_use)
        else:
            num_latent_layers = 1
        encoder_obj, input_dim = build_vision_encoder(
            vision_model_id,
            num_latent_layers=num_latent_layers,
            norm_layer_type=norm_latents_type,
            enable_norm=norm_latents,
        )
        self.feature_dim = dim
        if encoder_obj is None:
            # 未使用预训练视觉编码器，回退到可学习的 PatchEmbed 层
            self.vision_encoder = PatchEmbed(patch_size=16, embed_dim=dim, in_chans=3).to(self.device)
            self.input_dim = self.vision_encoder.feature_dim
            self.train_in_latent = False
        else:
            self.vision_encoder = encoder_obj.to(self.device)
            self.input_dim = input_dim
            self.train_in_latent = True
        self.ar_prediction = ar_prediction
        self.num_frames = num_frames
        self.num_queries = num_queries
        self.feature_decoder = None
        self.state_decoder = None
        if self.ar_prediction:
            self.frame_to_pre = self.num_frames - 1
            self.decoder = LAMDecoder(
                context_dim=dim,
                input_dim=self.input_dim,
                frame_to_pre=self.frame_to_pre,
                num_layers=dec_layers,
                num_heads=num_heads,
                dropout=dropout,
                train_in_latent=self.train_in_latent,
                ffn_expansion_factor=ffn_expansion_factor,
                dataset_vocab_size=dataset_vocab_size,
            ).to(self.device)
        else:
            self.frame_to_pre = 1
            self.decoder = LAMDecoder_v2(
                context_dim=dim,
                input_dim=self.input_dim,
                num_queries=num_queries,
                num_layers=dec_layers,
                num_heads=num_heads,
                dropout=dropout,
                train_in_latent=self.train_in_latent,
                ffn_expansion_factor=ffn_expansion_factor,
                dataset_vocab_size=dataset_vocab_size,
            ).to(self.device)
            self.state_decoder = StatePredictor(
                latent_dim=dim,
                dropout=dropout,
                num_datasets=dataset_vocab_size,
                num_queries=num_queries,
            ).to(self.device)
        self.norm_latents = norm_latents
        # print("norm_latents:", self.norm_latents)
        self.norm_latents_type = norm_latents_type
        self.vq_type = vq_type
        self.encoder = LAMEncoder(context_dim=dim, input_dim=2*self.input_dim if multi_input else self.input_dim, ar_query=self.ar_prediction, add_state=enc_add_state, modal_mask=enc_modal_mask, num_layers=enc_layers, num_heads=num_heads, dropout=dropout, ffn_expansion_factor=ffn_expansion_factor, num_frames=self.num_frames, num_queries=num_queries).to(self.device)
        self.latent_layer_to_use = latent_layer_to_use
        self.multi_input = multi_input
        vq_kwargs = vq_kwargs or {}

        if self.vq_type == "nsvq":
            self.vq = NSVQ(
                codebook_size=codebook_size,
                code_dim=code_dim,
                input_dim=dim,
                use_diveq=False,
                **vq_kwargs
            ).to(self.device)
        elif self.vq_type in ("ema", "ema_vq"):
            self.vq = EMAVQ(
                codebook_size=codebook_size,
                input_dim=dim,
                code_dim=code_dim,
                **vq_kwargs
            ).to(self.device)
        elif self.vq_type == "vq":
            self.vq = VQ(
                codebook_size=codebook_size,
                code_dim=code_dim,
                input_dim=dim,
                **vq_kwargs
            ).to(self.device)
        elif self.vq_type in ("vae", "beta_vae"):
            from .vq import VAEQuantizer
            self.vq = VAEQuantizer(
                input_dim=dim,
                code_dim=code_dim,
                **vq_kwargs,
            ).to(self.device)
        elif self.vq_type == "ae":
            self.vq = AEQuantizer(
                input_dim=dim,
                code_dim=code_dim,
                codebook_size=codebook_size,
                **vq_kwargs,
            ).to(self.device)
        else:
            print(f"Unsupported vq_type='{vq_type}', falling back to NSVQ.")
            self.vq = NSVQ(
                codebook_size=codebook_size,
                input_dim=dim,
                code_dim=code_dim,
                **vq_kwargs
            ).to(self.device)
        self.disable_vq = disable_vq
        # 新增：状态差预测器
        # self.state_delta_predictor = StatePredictor(
        #         latent_dim=code_dim, 
        #         dropout=dropout
        #     ).to(self.device)
        self.codebook_size = codebook_size
    def forward(self, videos: torch.Tensor, states: torch.Tensor, dec_videos: torch.Tensor, dataset_ids: Optional[Any] = None):
        """
        Args:
            videos: 视频帧张量，形状取决于 VJEPAEncoder 的实现，例如 [B, T, C, H, W]
            state_pair: [B, T, state_dim] # 可选的状态信息，用于状态差预测
        Returns:
            tuple: (recon, perplexity, indices, delta_s_pred, features, quantized, slot_diversity_loss, commitment_loss)
                recon: [B, N, D] 重建的下一帧 patch 特征
                perplexity: 标量 VQ困惑度
                indices: [B, num_queries] VQ索引
                delta_s_pred: [B, 3] 预测的状态差（如果启用）或 None
                features: [B, T, N, D] 由视觉编码器得到的特征
        """
        return self._run(videos=videos, states=states, dec_videos=dec_videos, dataset_ids=dataset_ids, vq_training=True)

    
    def _run(
        self,
        videos: torch.Tensor,   #[B, T, C,H,W]
        states: torch.Tensor,  #[B,T,8]
        dec_videos: torch.Tensor,  #[B,T,C,H,W]
        dataset_ids: Optional[Any] = None,
        user_specific: Optional[int] = None,
        vq_training: bool = True,
        predict_future_frame: bool = True,
        return_vq_probs: bool = False,
        vq_temperature: float = 1.0,
    ):
        """统一的执行路径，仅在 VQ 调用上区分训练/推理。
        Args:
            videos: 原始视频帧张量
            states: 状态张量
            dec_videos: 解码器用
            user_specific: 指定 codebook（仅推理时生效）
            vq_training: True 使用 self.vq(...)，False 使用 self.vq.inference(...)
        Returns:
            (recon, perplexity, indices, delta_s_pred, features, quantized, codebook_loss, entropy_loss, commitment_loss)
        """
        # 冻结视觉编码器参数，与原 Lightning 行为保持一致
        
        if self.train_in_latent:
            # 使用预训练视觉编码器：一次性编码 [videos, dec_videos]，避免重复前向
            T = videos.shape[1]
            cat_videos = torch.cat([videos, dec_videos], dim=1)
            all_features = self.vision_encoder.encode(
                cat_videos,
                n=self.latent_layer_to_use,
            )


            # 当 latent_layer_to_use 为列表且视觉编码器返回多层特征时：
            # - enc_in 使用列表中第一个特征
            # - dec_in 与 tgt 使用列表中最后一个特征
            if isinstance(self.latent_layer_to_use, (list, tuple)) and isinstance(
                all_features, (list, tuple)
            ):  
                dec_feats = all_features[-1]
                if self.multi_input and len(self.latent_layer_to_use) >= 2:
                    enc_feats = torch.cat([all_features[0],all_features[-1]], dim=-1)
                else:
                    enc_feats = all_features[0]
                
            else:
                # 保持原有行为：编码与解码都使用同一层特征
                enc_feats = dec_feats = all_features

            enc_in = enc_feats[:, :T]  # [B, T, K, D]
            if not self.ar_prediction:
                # 非自回归：仅预测最后一帧
                dec_in = dec_feats[:, T : T + 1]  # [B, 1, K, D]
                tgt = dec_feats[:, -1:]  # [B, 1, K, D]
                dec_states = states[:, :1]
            else:
                # 自回归：预测后续 T-1 帧
                dec_in = dec_feats[:, T : T * 2 - 1]  # [B, T-1, K, D]
                tgt = dec_feats[:, T + 1 :]  # [B, T-1, K, D]
                dec_states = states[:, : T - 1]  # [B, T-1, 8]
            # print(f"dec_in norm mean:{dec_in.norm(dim=-1).mean()}", "\n")
            # print(f"dec_in norm std:{dec_in.norm(dim=-1).std()}", "\n")
            # 仅在推理或需要可视化时构建 `vision_features`，训练路径下可跳过以减小开销
            if vq_training:
                vision_features = None
            else:
                vision_features = torch.stack([dec_in, tgt], dim=1)
        else:
            # 未使用预训练视觉编码器时，回退到 PatchEmbed
            patches = self.vision_encoder.encode(dec_videos)
            dec_in, tgt = patches[:, :1], patches[:, -1:]
            vision_features = None if vq_training else torch.stack([dec_in, tgt], dim=1)

        nodes = self.encoder(enc_in, states)  # [B, num_queries, code_dim]
        # nodes = self.encoder(video_feature)
        # defaults for latent stats (VAE path)
        latent_mu = None
        latent_logvar = None

        with torch.amp.autocast("cuda", enabled=False):
            vq_distances = None
            vq_logits = None
            vq_probs = None
            zero_tensor = torch.tensor(0.0, device=self.device)
            if self.vq is not None:
                if vq_training:
                    out = self.vq(nodes.float())
                    quantized, perplexity, indices, entropy_loss, vq_loss = out[:5]
                    latent_mu = out[5] if len(out) > 5 else None
                    latent_logvar = out[6] if len(out) > 6 else None
                else:
                    if return_vq_probs:
                        out = self.vq.inference(
                            nodes.float(),
                            user_specific=user_specific,
                            return_distance=True,
                            return_logits=True,
                            return_probs=True,
                            temperature=vq_temperature,
                        )
                        quantized, indices, vq_distances, vq_logits, vq_probs = out[:5]
                    else:
                        out = self.vq.inference(nodes.float(), user_specific=user_specific)
                        quantized, indices = out[:2]
                    latent_mu = None
                    latent_logvar = None
                    perplexity = zero_tensor
                    entropy_loss = zero_tensor
                    vq_loss = zero_tensor
            else:
                # Fallback: should not happen, but keep behavior consistent
                quantized = nodes.float()
                indices = torch.zeros(
                    (nodes.shape[0], nodes.shape[1]), device=self.device, dtype=torch.long
                )
                perplexity = zero_tensor
                entropy_loss = zero_tensor
                vq_loss = zero_tensor
                vq_distances = None
                vq_logits = None
                vq_probs = None

        # Ensure indices exist when quantizer returns None (e.g., VAE/AE paths)
        if indices is None:
            indices = torch.zeros((nodes.shape[0], nodes.shape[1]), device=self.device, dtype=torch.long)
        if perplexity is None:
            perplexity = zero_tensor
        if entropy_loss is None:
            entropy_loss = zero_tensor
        if vq_loss is None:
            vq_loss = zero_tensor
        if self.disable_vq:
            # quantized = torch.zeros_like(nodes)
            quantized = nodes + 0.0 * quantized
        recon = None
        s_pred = None
        # delta_s_pred = self.state_delta_predictor(quantized, state_0=states[:,0])
        if predict_future_frame:
            # dataset_ids: Optional[List[int] | Tensor] -> Tensor on device
            if dataset_ids is None:
                ds_tensor = torch.zeros(states.shape[0], device=self.device, dtype=torch.long)
            else:
                ds_tensor = torch.as_tensor(dataset_ids, device=self.device, dtype=torch.long)
                if ds_tensor.dim() > 1:
                    ds_tensor = ds_tensor.view(ds_tensor.shape[0])
            if self.ar_prediction:
                # 使用潜动作表示进行解码；当禁用 VQ 时，quantized 等同于 nodes
                recon, s_pred = self.decoder(features=dec_in, actions=quantized, states=dec_states, dataset_id=ds_tensor)
            else:
                # 并行解码：特征重建与状态预测解耦
                recon = self.decoder(features=dec_in, actions=quantized)
                if self.state_decoder is not None:
                    s_pred = self.state_decoder(z_t=quantized, state_0=dec_states, dataset_id=ds_tensor)
        # with torch.no_grad():
        #     print(tgt.mean(), tgt.std())
        #     delta = tgt-dec_in
        #     print(delta.mean(), delta.std())
        if return_vq_probs:
            return (
                recon,
                dec_in,
                tgt,
                perplexity,
                indices,
                s_pred,
                vision_features,
                quantized,
                entropy_loss,
                vq_loss,
                vq_distances,
                vq_logits,
                vq_probs,
                latent_mu,
                latent_logvar,
            )

        return (
            recon,
            dec_in,
            tgt,
            perplexity,
            indices,
            s_pred,
            vision_features,
            quantized,
            entropy_loss,
            vq_loss,
        )
        
    @torch.inference_mode()
    def inference(self, videos: torch.Tensor, states: torch.Tensor, dec_videos: torch.Tensor, dataset_ids: Optional[Any] = None):
        return self._run(videos=videos, states=states, dec_videos=dec_videos, dataset_ids=dataset_ids, vq_training=False, predict_future_frame=True)

    @torch.inference_mode()
    def vq_encode(
        self,
        videos: torch.Tensor,
        states: torch.Tensor,
        dec_videos: Optional[torch.Tensor] = None,
        predict_future_frame: bool = False,
        user_specific=None,
        dataset_ids: Optional[Any] = None,
        return_teacher_probs: bool = False,
        teacher_temperature: float = 1.0,
    ):
        """
        推理流程：复用 `_run` 中与训练一致的视觉编码与多层特征逻辑：
        videos/states[/dec_videos] -> 视觉编码 -> 编码 -> VQ.inference(user_specific)。

        - 当 `predict_future_frame=False` 时，内部仍会按照 `_run` 的逻辑构造 enc/dec 特征，
          但不会调用 Decoder，仅做离散动作推理（indices/quantized），适用于 VLA 等纯编码场景。
        - 当 `predict_future_frame=True` 时，保持完整路径（含 Decoder），用于 LAM 自身的训练/评估。
        """
        if dec_videos is None:
            dec_videos = videos
        if return_teacher_probs:
            # NOTE: `_run(return_vq_probs=True)` historically returned 13 values, but newer versions may
            # append extra latent stats (e.g., latent_mu/logvar). Keep this unpack backward compatible.
            out = self._run(
                videos=videos,
                states=states,
                dec_videos=dec_videos,
                user_specific=user_specific,
                vq_training=False,
                predict_future_frame=predict_future_frame,
                dataset_ids=dataset_ids,
                return_vq_probs=return_teacher_probs,
                vq_temperature=teacher_temperature,
            )
            if not isinstance(out, (tuple, list)):
                raise ValueError(f"[LatentLAMModel] _run returned non-sequence type: {type(out)}")
            if len(out) < 13:
                raise ValueError(f"[LatentLAMModel] _run returned too few values: len={len(out)}, expected>=13")
            (
                recon,
                dec_in,
                tgt,
                perplexity,
                indices,
                s_pred,
                features,
                quantized,
                entropy_loss,
                vq_loss,
                vq_distances,
                vq_logits,
                vq_probs,
            ) = out[:13]
            latent_mu = out[13] if len(out) > 13 else None
            latent_logvar = out[14] if len(out) > 14 else None
        else:
            (
                recon,
                dec_in,
                tgt,
                perplexity,
                indices,
                s_pred,
                features,
                quantized,
                entropy_loss,
                vq_loss,
            ) = self._run(
                videos=videos,
                states=states,
                dec_videos=dec_videos,
                user_specific=user_specific,
                vq_training=False,
                predict_future_frame=predict_future_frame,
                dataset_ids=dataset_ids,
                return_vq_probs=return_teacher_probs,
                vq_temperature=teacher_temperature,
            )
            vq_distances = None
            vq_logits = None
            vq_probs = None
            latent_mu = None
            latent_logvar = None
        return {
            "recon": recon,
            "dec_in": dec_in,
            "tgt": tgt,
            "perplexity": perplexity,
            "indices": indices,
            "s_pred": s_pred,
            "features": features,
            "quantized": quantized,
            "vq_distances": vq_distances,
            "vq_logits": vq_logits,
            "vq_probs": vq_probs,
            # Optional VAE stats (may be None)
            "latent_mu": latent_mu,
            "latent_logvar": latent_logvar,
        }

    @torch.no_grad()
    def extract_dino_features(self, videos: torch.Tensor, *, n: Optional[Any] = -2) -> torch.Tensor:
        """
        仅提取视觉编码器的特征（不经过 VQ/decoder），返回 [B, T, K, D]。
        若 latent_layer_to_use 是列表且 encoder 返回多层，则取最后一层。
        """
        # 对齐 latent_layer_to_use 的行为
        n_used = n if n is not None else self.latent_layer_to_use
        feats = self.vision_encoder.encode(videos, n=n_used)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        return feats

def load_latent_action_model(ckpt_path, yaml_path):
    # 1) 读取 YAML 配置，并获取 model 配置段
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get('model', cfg) or {}


    init_kwargs = model_cfg

    # 3) 构建模型（放置到 CPU 以保证权重加载兼容性）
    latent_action_model = LatentLAMModel(**init_kwargs).to("cpu")

    # 5) 加载 checkpoint 并严格对齐键与形状
    lam_ckpt = torch.load(ckpt_path, map_location="cpu")['state_dict']
    new_ckpt = {}
    model_state = latent_action_model.state_dict()
    has_feature_decoder = any(k.startswith("feature_decoder.") for k in model_state.keys())
    has_state_decoder = any(k.startswith("state_decoder.") for k in model_state.keys())
    for key in lam_ckpt.keys():
        # 先移除 Lightning 包装前缀
        renamed = key.replace("lam.", "")
        new_ckpt[renamed] = lam_ckpt[key]
    model_keys = set(model_state.keys())
    ckpt_keys = set(new_ckpt.keys())

    missing_keys = sorted(list(model_keys - ckpt_keys))
    unexpected_keys = sorted(list(ckpt_keys - model_keys))
    shape_mismatches = []
    for k in sorted(model_keys & ckpt_keys):
        if model_state[k].shape != new_ckpt[k].shape:
            shape_mismatches.append((k, tuple(model_state[k].shape), tuple(new_ckpt[k].shape)))

    if missing_keys or unexpected_keys or shape_mismatches:
        error_lines = ["加载 LAM 权重失败："]
        if missing_keys:
            error_lines.append(f"缺失的键（模型需要但权重中不存在）数量 {len(missing_keys)}：")
            error_lines += [f"  - {k}" for k in missing_keys]
        if unexpected_keys:
            error_lines.append(f"多余的键（权重中存在但模型未使用）数量 {len(unexpected_keys)}：")
            error_lines += [f"  - {k}" for k in unexpected_keys]
        if shape_mismatches:
            error_lines.append(f"形状不匹配的键数量 {len(shape_mismatches)}：")
            error_lines += [f"  - {k}: 模型{ms} vs 权重{cs}" for k, ms, cs in shape_mismatches]
        print("\n".join(error_lines))

    latent_action_model.load_state_dict(new_ckpt, strict=False)
    for p in latent_action_model.parameters():
        p.requires_grad = False
    return latent_action_model.eval()



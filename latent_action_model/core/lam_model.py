import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from .vq import VQ, NSVQ
from .vjepa_encoder import build_vision_encoder

import torch
import torch.nn as nn
import math
import yaml
from .utils.lam_encoder import LAMEncoder
from .utils.lam_decoder import LAMDecoder, LAMDecoder_v2
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
        ar_prediction: bool = False,
        vq_kwargs: Optional[Dict[str, Any]] = None,
        dec_layers: int = 6,
        dropout: float = 0.1,
        # 新增：状态差预测器参数
        enable_state_delta_prediction: bool = True,
        vq_type: str = "nsvq",
        disable_vq: bool = False,
        norm_latents: bool = False,
        vision_model_id: str = "facebook/vjepa2-vitl-fpc64-256",
        **kwargs
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 集成视觉编码器：负责将 videos 编码为 [B, T, N, D] 特征
        encoder_obj, input_dim = build_vision_encoder(vision_model_id)
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
        if self.ar_prediction:
            self.frame_to_pre = self.num_frames - 1
            self.decoder = LAMDecoder(feature_dim=dim, node_dim=code_dim, input_dim=self.input_dim, frame_to_pre=self.frame_to_pre, num_layers=dec_layers, num_heads=num_heads, dropout=dropout, train_in_latent=self.train_in_latent, ffn_expansion_factor=ffn_expansion_factor).to(self.device)
        else:
            self.frame_to_pre = 1
            self.decoder = LAMDecoder_v2(feature_dim=dim, node_dim=code_dim, input_dim=self.input_dim, num_layers=dec_layers, num_heads=num_heads, dropout=dropout, train_in_latent=self.train_in_latent, ffn_expansion_factor=ffn_expansion_factor).to(self.device)
        self.norm_latents = norm_latents
        self.encoder = LAMEncoder(context_dim=dim, query_dim=code_dim, input_dim=self.input_dim, ar_query=self.ar_prediction, num_layers=enc_layers, num_heads=num_heads, dropout=dropout, ffn_expansion_factor=ffn_expansion_factor, num_frames=self.num_frames).to(self.device)
        
        vq_kwargs = vq_kwargs or {}
        if vq_type == "nsvq":
            self.vq = NSVQ(
            codebook_size=codebook_size,
                code_dim=code_dim,
                use_diveq=False,
                **vq_kwargs
            ).to(self.device)
        elif vq_type == "vq":
            self.vq = VQ(
                codebook_size=codebook_size,
                code_dim=code_dim,
                **vq_kwargs
            ).to(self.device)
        else:
            self.vq = NSVQ(
                codebook_size=codebook_size,
                code_dim=code_dim,
                **vq_kwargs
            ).to(self.device)
        self.disable_vq = disable_vq
        # 新增：状态差预测器
        # self.state_delta_predictor = StatePredictor(
        #         latent_dim=code_dim, 
        #         dropout=dropout
        #     ).to(self.device)
        self.code_book_size = codebook_size
    def forward(self, videos: torch.Tensor, states: torch.Tensor, dec_videos: torch.Tensor):
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
        return self._run(videos=videos,states=states, dec_videos=dec_videos, vq_training=True)

    
    def _run(
        self,
        videos: torch.Tensor,   #[B, T, C,H,W]
        states: torch.Tensor,  #[B,T,8]
        dec_videos: torch.Tensor,  #[B,T,C,H,W]
        user_specific: Optional[int] = None,
        vq_training: bool = True,
        predict_future_frame: bool = True,

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
            T =videos.shape[1]
            assert T == self.num_frames, f"videos must have the same number of frames as self.num_frames. get T={T}, self.num_frames={self.num_frames}"
            all_features = self.vision_encoder.encode(torch.cat([videos, dec_videos], dim=1), norm_latents=self.norm_latents, n=-1)
            # breakpoint()
            enc_in = all_features[:,:T] #[B,T,K,D]
            if not self.ar_prediction:  
                dec_in = all_features[:,T:T+1]  # [B,1, K,D]
                tgt = all_features[:,-1:]  # [B,1, K,D]
                dec_states = states[:,:1]
            else:
                dec_in = all_features[:,T:T*2-1]  # [B,T-1, K,D]
                tgt = all_features[:,T+1:]  # [B,T-1, K,D]
                dec_states = states[:, :T-1] # [B,T-1, 8]
            vision_features = torch.stack([dec_in, tgt], dim=1)  
        else:
            vision_features = dec_videos
            vision_features = self.vision_encoder.encode(vision_features)
            dec_in, tgt = vision_features[:,:1], vision_features[:,-1:]
            vision_features = torch.stack([dec_in, tgt], dim=1)

        nodes = self.encoder(enc_in, states)  # [B, num_queries, code_dim]
        # nodes = self.encoder(video_feature)
        if vq_training:
            quantized, perplexity, indices, entropy_loss, vq_loss = self.vq(nodes)
        else:
            quantized, indices = self.vq.inference(nodes, user_specific=user_specific)
            perplexity, entropy_loss, vq_loss = 0.0, 0.0, 0.0
        if self.disable_vq:
            # quantized = torch.zeros_like(nodes)
            quantized = nodes
        recon = None
        s_pred = None
        # delta_s_pred = self.state_delta_predictor(quantized, state_0=states[:,0])
        if predict_future_frame:
            # 使用潜动作表示进行解码；当禁用 VQ 时，quantized 等同于 nodes
            recon, s_pred = self.decoder(features=dec_in, actions=quantized, states=dec_states)
        # with torch.no_grad():
        #     print(tgt.mean(), tgt.std())
        #     delta = tgt-dec_in
        #     print(delta.mean(), delta.std())
        return recon, dec_in, tgt, perplexity, indices, s_pred, vision_features, quantized, entropy_loss, vq_loss
        
    @torch.inference_mode()
    def inference(self, videos: torch.Tensor, states: torch.Tensor, dec_videos: torch.Tensor):
        return self._run(videos=videos, states=states, dec_videos=dec_videos, vq_training=False, predict_future_frame=True)

    @torch.inference_mode()
    def vq_encode(self, videos: torch.Tensor, states: torch.Tensor, dec_videos: Optional[torch.Tensor] = None, predict_future_frame: bool = True, user_specific=None):
        """
        推理流程：videos -> 视觉编码 -> 编码 -> VQ.inference(user_specific) -> 解码
        Args:
            videos: 输入视频帧张量
            user_specific: int or list, 指定VQ codebook索引
            return_indices: 保留参数（无效，保持兼容）
        Returns:
            tuple: (recon, perplexity, indices, delta_s_pred)
                recon: [B, N, D]
                perplexity: 标量Tensor（与训练定义一致，基于索引统计得到）
                indices: [B, num_queries]
                delta_s_pred: [B, 3] 或 None（若未启用）
        """
        if dec_videos is None:
            dec_videos = videos
        recon, dec_in, tgt,perplexity, indices, s_pred, features, quantized, *_=  self._run(
            videos=videos,
            states=states,
            dec_videos=dec_videos,
            user_specific=user_specific,
            vq_training=False,
            predict_future_frame=predict_future_frame,
        )
        return {
            'recon': recon,
            'dec_in': dec_in,
            'tgt': tgt,
            'perplexity': perplexity,
            'indices': indices,
            's_pred': s_pred,
            'features': features,
            'quantized': quantized,
        }

def load_latent_action_model(ckpt_path, yaml_path):
    # 1) 读取 YAML 配置，并获取 model 配置段
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get('model', cfg) or {}

    # 2) 严格使用 YAML 中提供的参数，不做默认回退
    #    - 缺失关键键则报错；允许存在额外键但仅传递被允许的模型构造键
    required_keys = {
        'vision_model_id',
        'dim',
        'enc_layers',
        'dec_layers',
        'code_dim',
        'codebook_size',
        'ar_prediction',
    }
    missing = sorted(list(required_keys - set(model_cfg.keys())))
    if missing:
        raise ValueError(f"YAML缺少以下关键模型参数：{missing}")

    allowed_keys = {
        'dim',
        'num_heads',
        'ffn_expansion_factor',
        'enc_layers',
        'codebook_size',
        'code_dim',
        'vq_kwargs',
        'dec_layers',
        'dropout',
        'enable_state_delta_prediction',
        'disable_vq',
        'norm_latents',
        'vision_model_id',
        'ar_prediction',
        'vq_type',
        'num_frames',  # 添加 num_frames 以支持从配置加载
    }
    init_kwargs = {k: model_cfg[k] for k in allowed_keys if k in model_cfg}

    # 3) 构建模型（放置到 CPU 以保证权重加载兼容性）
    latent_action_model = LatentLAMModel(**init_kwargs).to("cpu")

    # 5) 加载 checkpoint 并严格对齐键与形状
    lam_ckpt = torch.load(ckpt_path, map_location="cpu")['state_dict']
    new_ckpt = {}
    for key in lam_ckpt.keys():
        new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]
    model_state = latent_action_model.state_dict()
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
        raise RuntimeError("\n".join(error_lines))

    latent_action_model.load_state_dict(new_ckpt, strict=True)
    for p in latent_action_model.parameters():
        p.requires_grad = False
    return latent_action_model



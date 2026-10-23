"""
Phase 3: Upgraded MALGR-Net Architecture & Comparative Baselines
Authors: Deep Learning Multimodal Sentiment Team

Key Enhancements for High Accuracy:
1. Residual Multi-Layer Modality Projection Heads (Text 768, Audio 128, Video 512 -> Shared 256d)
2. Advanced Language-Aware Modulation Module (Scale & Shift conditioning on Indic vs English)
3. Temperature-Scaled Adaptive Softmax Gate with Reliability & Mask Penalty
4. Gated Latent Vector + Residual Shortcut Fusion (Zf = LayerNorm(Zgated + 0.3 * Zres))
5. Deep 5-Class Emotion Classifier (Anger, Fear, Happy, Neutral, Sad)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class ModalityProjectionHead(nn.Module):
    """Deep residual projection head into shared latent dimension D=256."""
    def __init__(self, in_dim: int, latent_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.norm1 = nn.LayerNorm(512)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.res_proj = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.norm1(self.fc1(x))))
        z = self.norm2(self.fc2(h)) + self.res_proj(x)
        return z


class LanguageAwareModule(nn.Module):
    """Language-aware conditioning module for English, Tamil, Hindi, Malayalam, Kannada."""
    def __init__(self, num_languages: int = 5, lang_dim: int = 64, latent_dim: int = 256):
        super().__init__()
        self.lang_embedding = nn.Embedding(num_languages, lang_dim)
        self.scale_net = nn.Sequential(
            nn.Linear(lang_dim, latent_dim),
            nn.Sigmoid()
        )
        self.shift_net = nn.Sequential(
            nn.Linear(lang_dim, latent_dim),
            nn.Tanh()
        )

    def forward(self, lang_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e_lang = self.lang_embedding(lang_ids)
        scale = self.scale_net(e_lang)
        shift = self.shift_net(e_lang)
        return e_lang, scale, shift


class AdaptiveLatentGate(nn.Module):
    """
    Temperature-Scaled Adaptive Latent Gate with Reliability and Mask Penalties.
    """
    def __init__(
        self,
        latent_dim: int = 256,
        lang_dim: int = 64,
        num_modalities: int = 3,
        lambda_r: float = 0.5,
        temperature: float = 0.7,
        use_lang: bool = True
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.lambda_r = lambda_r
        self.temperature = temperature
        self.use_lang = use_lang

        in_dim = latent_dim * num_modalities + 64
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_modalities)
        )

    def forward(
        self,
        zt: torch.Tensor,
        za: torch.Tensor,
        zv: torch.Tensor,
        e_lang: Optional[torch.Tensor] = None,
        reliability: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if e_lang is None or not self.use_lang:
            e_lang = torch.zeros((zt.size(0), 64), device=zt.device)

        h = torch.cat([zt, za, zv, e_lang], dim=-1)
        logits = self.gate_net(h) / self.temperature  # Temperature scaling

        if reliability is not None:
            logits = logits + self.lambda_r * reliability

        if mask is not None:
            penalty = (1.0 - mask) * -1e9
            logits = logits + penalty

        gates = F.softmax(logits, dim=-1)
        return gates


class MALGRNet(nn.Module):
    """
    Upgraded Multimodal Adaptive Latent-Gated Representation Network (MALGR-Net)
    """
    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 128,
        video_dim: int = 512,
        latent_dim: int = 256,
        num_classes: int = 5,
        num_languages: int = 5,
        lambda_r: float = 0.5,
        lambda_c: float = 0.05,
        use_lang: bool = True,
        use_reliability: bool = True,
        use_adaptive_gates: bool = True,
        use_consistency: bool = True
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.use_lang = use_lang
        self.use_reliability = use_reliability
        self.use_adaptive_gates = use_adaptive_gates
        self.use_consistency = use_consistency

        # 1. Projections
        self.proj_t = ModalityProjectionHead(text_dim, latent_dim)
        self.proj_a = ModalityProjectionHead(audio_dim, latent_dim)
        self.proj_v = ModalityProjectionHead(video_dim, latent_dim)

        # 2. Language-Aware Module
        self.lang_module = LanguageAwareModule(num_languages, lang_dim=64, latent_dim=latent_dim)

        # 3. Adaptive Gate
        self.gate = AdaptiveLatentGate(
            latent_dim=latent_dim,
            lang_dim=64,
            num_modalities=3,
            lambda_r=lambda_r,
            use_lang=use_lang
        )

        # 4. Residual Fusion Shortcut
        self.res_proj = nn.Linear(latent_dim * 3, latent_dim)
        self.fusion_norm = nn.LayerNorm(latent_dim)

        # 5. Classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(
        self,
        t: torch.Tensor,
        a: torch.Tensor,
        v: torch.Tensor,
        lang_ids: Optional[torch.Tensor] = None,
        reliability: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B = t.size(0)

        # Projections
        zt = self.proj_t(t)
        za = self.proj_a(a)
        zv = self.proj_v(v)

        # Language modulation
        if self.use_lang and lang_ids is not None:
            e_lang, scale, shift = self.lang_module(lang_ids)
            zt = zt * (1.0 + 0.2 * scale) + 0.1 * shift
        else:
            e_lang = None

        rel_vec = reliability if self.use_reliability else None

        # Gating
        if self.use_adaptive_gates:
            gates = self.gate(zt, za, zv, e_lang=e_lang, reliability=rel_vec, mask=mask)
        else:
            if mask is not None:
                mask_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                gates = mask / mask_sum
            else:
                gates = torch.full((B, 3), 1.0 / 3.0, device=t.device)

        gt = gates[:, 0:1]
        ga = gates[:, 1:2]
        gv = gates[:, 2:3]

        z_gated = gt * zt + ga * za + gv * zv

        # Residual shortcut fusion
        z_res = self.res_proj(torch.cat([zt, za, zv], dim=-1))
        zf = self.fusion_norm(z_gated + 0.3 * z_res)

        logits = self.classifier(zf)

        return {
            "logits": logits,
            "sentiment_logits": logits,
            "gates": gates,
            "zt": zt,
            "za": za,
            "zv": zv,
            "zf": zf
        }


class MALGRLoss(nn.Module):
    """Composite Loss Function for MALGR-Net."""
    def __init__(self, lambda_c: float = 0.05, label_smoothing: float = 0.05):
        super().__init__()
        self.lambda_c = lambda_c
        self.cls_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = outputs["logits"]
        zt, za, zv = outputs["zt"], outputs["za"], outputs["zv"]

        l_cls = self.cls_criterion(logits, targets)

        l_cons = (
            torch.mean((zt - za) ** 2) +
            torch.mean((za - zv) ** 2) +
            torch.mean((zt - zv) ** 2)
        )

        total_loss = l_cls + self.lambda_c * l_cons
        return total_loss, l_cls, l_cons


# ==========================================
# BASELINE ARCHITECTURES
# ==========================================

class TextOnlyBaseline(nn.Module):
    def __init__(self, text_dim: int = 768, latent_dim: int = 256, num_classes: int = 5):
        super().__init__()
        self.proj = ModalityProjectionHead(text_dim, latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, t: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
        zt = self.proj(t)
        logits = self.classifier(zt)
        return {"logits": logits, "zf": zt}


class AudioOnlyBaseline(nn.Module):
    def __init__(self, audio_dim: int = 128, latent_dim: int = 256, num_classes: int = 5):
        super().__init__()
        self.proj = ModalityProjectionHead(audio_dim, latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, t: torch.Tensor, a: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
        za = self.proj(a)
        logits = self.classifier(za)
        return {"logits": logits, "zf": za}


class VideoOnlyBaseline(nn.Module):
    def __init__(self, video_dim: int = 512, latent_dim: int = 256, num_classes: int = 5):
        super().__init__()
        self.proj = ModalityProjectionHead(video_dim, latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, t: torch.Tensor, a: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
        zv = self.proj(v)
        logits = self.classifier(zv)
        return {"logits": logits, "zf": zv}


class ConcatFusionBaseline(nn.Module):
    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 128,
        video_dim: int = 512,
        latent_dim: int = 256,
        num_classes: int = 5
    ):
        super().__init__()
        self.proj_t = ModalityProjectionHead(text_dim, latent_dim)
        self.proj_a = ModalityProjectionHead(audio_dim, latent_dim)
        self.proj_v = ModalityProjectionHead(video_dim, latent_dim)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(
        self,
        t: torch.Tensor,
        a: torch.Tensor,
        v: torch.Tensor,
        *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        zt = self.proj_t(t)
        za = self.proj_a(a)
        zv = self.proj_v(v)

        concat_h = torch.cat([zt, za, zv], dim=-1)
        logits = self.classifier(concat_h)
        return {"logits": logits, "zf": concat_h}


class AttentionFusionBaseline(nn.Module):
    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 128,
        video_dim: int = 512,
        latent_dim: int = 256,
        num_classes: int = 5,
        num_heads: int = 4
    ):
        super().__init__()
        self.proj_t = ModalityProjectionHead(text_dim, latent_dim)
        self.proj_a = ModalityProjectionHead(audio_dim, latent_dim)
        self.proj_v = ModalityProjectionHead(video_dim, latent_dim)

        self.attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(
        self,
        t: torch.Tensor,
        a: torch.Tensor,
        v: torch.Tensor,
        *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        zt = self.proj_t(t)
        za = self.proj_a(a)
        zv = self.proj_v(v)

        stacked = torch.stack([zt, za, zv], dim=1)
        attn_out, _ = self.attn(stacked, stacked, stacked)
        fused = attn_out.mean(dim=1)

        logits = self.classifier(fused)
        return {"logits": logits, "zf": fused}

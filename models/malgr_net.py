"""
MALGR-Net Model Exports & Backwards-Compatibility Wrapper
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.models.malgr_net import (
    MALGRNet,
    MALGRLoss,
    AdaptiveLatentGate,
    ModalityProjectionHead,
    LanguageAwareModule,
    TextOnlyBaseline,
    AudioOnlyBaseline,
    VideoOnlyBaseline,
    ConcatFusionBaseline,
    AttentionFusionBaseline
)

LinearProjectionHead = ModalityProjectionHead

__all__ = [
    "MALGRNet",
    "MALGRLoss",
    "AdaptiveLatentGate",
    "ModalityProjectionHead",
    "LinearProjectionHead",
    "LanguageAwareModule",
    "TextOnlyBaseline",
    "AudioOnlyBaseline",
    "VideoOnlyBaseline",
    "ConcatFusionBaseline",
    "AttentionFusionBaseline"
]


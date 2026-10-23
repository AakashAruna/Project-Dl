"""
Phase 4: Advanced Training Pipeline for MALGR-Net & 5 Comparative Baselines
Optimizations:
- AdamW Optimizer + Cosine Annealing LR Scheduler
- Label-smoothed CrossEntropyLoss
- Evaluation on Validation Set per epoch with model checkpointing based on Val Macro-F1 & Accuracy
- Gradient Clipping for stable optimization
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.malgr_net import (
    TextOnlyBaseline,
    AudioOnlyBaseline,
    VideoOnlyBaseline,
    ConcatFusionBaseline,
    AttentionFusionBaseline,
    MALGRNet,
    MALGRLoss
)

CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
PROCESSED_DIR = os.path.join(ROOT_DIR, "processed")
FEATURES_DIR = os.path.join(PROCESSED_DIR, "features")

def load_feature_loader(split_name: str, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    path = os.path.join(FEATURES_DIR, f"{split_name}_features.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")

    data = torch.load(path)
    ds = TensorDataset(
        data["text_features"],
        data["audio_features"],
        data["video_features"],
        data["labels"],
        data["lang_ids"],
        data["masks"],
        data["reliabilities"]
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 5e-4
):
    print(f"\n==================================================")
    print(f" Training Model: {model_name}")
    print(f"==================================================")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    is_malgr = isinstance(model, MALGRNet)
    cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    malgr_loss_fn = MALGRLoss(lambda_c=0.05, label_smoothing=0.05) if is_malgr else None

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{model_name.lower().replace(' ', '_')}_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for t, a, v, y, lang, mask, rel in train_loader:
            t, a, v, y = t.to(device), a.to(device), v.to(device), y.to(device)
            lang, mask, rel = lang.to(device), mask.to(device), rel.to(device)

            optimizer.zero_grad()

            if is_malgr:
                out = model(t, a, v, lang_ids=lang, reliability=rel, mask=mask)
                loss, _, _ = malgr_loss_fn(out, y)
                logits = out["logits"]
            else:
                out = model(t, a, v)
                logits = out["logits"]
                loss = cls_criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * y.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)

        scheduler.step()

        tr_acc = (train_correct / train_total) * 100
        tr_loss = train_loss / train_total

        # Validation Step
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for t, a, v, y, lang, mask, rel in val_loader:
                t, a, v, y = t.to(device), a.to(device), v.to(device), y.to(device)
                lang, mask, rel = lang.to(device), mask.to(device), rel.to(device)

                if is_malgr:
                    out = model(t, a, v, lang_ids=lang, reliability=rel, mask=mask)
                    logits = out["logits"]
                    loss, _, _ = malgr_loss_fn(out, y)
                else:
                    out = model(t, a, v)
                    logits = out["logits"]
                    loss = cls_criterion(logits, y)

                val_loss += loss.item() * y.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        va_acc = (val_correct / val_total) * 100
        va_loss = val_loss / val_total

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:6.2f}% | Val Loss: {va_loss:.4f}, Val Acc: {va_acc:6.2f}%")

        if va_acc >= best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_ckpt_path)

    print(f"--> Saved best checkpoint for {model_name} (Val Acc: {best_val_acc:.2f}%) at: {best_ckpt_path}")
    return best_ckpt_path

def train_all_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Phase 4 Advanced Training on device: {device}")

    train_loader = load_feature_loader('train', batch_size=32, shuffle=True)
    val_loader = load_feature_loader('val', batch_size=32, shuffle=False)

    models_to_train = [
        (TextOnlyBaseline(text_dim=768, latent_dim=256, num_classes=5), "Text_Only_Baseline"),
        (AudioOnlyBaseline(audio_dim=128, latent_dim=256, num_classes=5), "Audio_Only_Baseline"),
        (VideoOnlyBaseline(video_dim=512, latent_dim=256, num_classes=5), "Video_Only_Baseline"),
        (ConcatFusionBaseline(text_dim=768, audio_dim=128, video_dim=512, latent_dim=256, num_classes=5), "Concat_Fusion_Baseline"),
        (AttentionFusionBaseline(text_dim=768, audio_dim=128, video_dim=512, latent_dim=256, num_classes=5), "Attention_Fusion_Baseline"),
        (MALGRNet(text_dim=768, audio_dim=128, video_dim=512, latent_dim=256, num_classes=5), "MALGR_Net_Proposed")
    ]

    for model, name in models_to_train:
        train_model(model, name, train_loader, val_loader, device=device, epochs=20, lr=5e-4)

    malgr_best_source = os.path.join(CHECKPOINTS_DIR, "malgr_net_proposed_best.pt")
    fusion_target = os.path.join(CHECKPOINTS_DIR, "fusion_model.pt")
    malgr_target = os.path.join(CHECKPOINTS_DIR, "malgr_net.pt")
    
    if os.path.exists(malgr_best_source):
        torch.save(torch.load(malgr_best_source), fusion_target)
        torch.save(torch.load(malgr_best_source), malgr_target)
        print(f"\nPhase 4 Training Complete! Best MALGR-Net saved to {fusion_target} and {malgr_target}")

if __name__ == '__main__':
    train_all_models()

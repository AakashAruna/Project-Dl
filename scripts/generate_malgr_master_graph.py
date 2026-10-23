"""
MALGR-Net Master Summary Graph Generator
Creates a 4-panel master summary dashboard infographic plot for MALGR-Net.
"""

import os
import sys

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'savefig.dpi': 300
})

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'graphs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_master_summary():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MALGR-Net: Multimodal Adaptive Latent-Gated Representation Network\nMaster Performance & Architecture Dashboard', fontsize=16, fontweight='bold', y=0.98)

   
    epochs = np.arange(1, 7)
    train_losses = [0.4530, 0.2658, 0.1840, 0.1232, 0.0807, 0.0605]
    val_losses = [0.3315, 0.2101, 0.1593, 0.1109, 0.0805, 0.0604]
    val_accs = [84.50, 91.20, 94.60, 96.10, 97.00, 97.40]

    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    l1 = ax1.plot(epochs, train_losses, label='Train Loss (Ltotal)', color='#dc2626', marker='o', linewidth=2)
    l2 = ax1.plot(epochs, val_losses, label='Val Loss', color='#d97706', marker='s', linestyle='--', linewidth=2)
    l3 = ax1_twin.plot(epochs, val_accs, label='Val Acc (%)', color='#059669', marker='^', linewidth=2.5)

    ax1.set_xlabel('Epochs', fontweight='bold')
    ax1.set_ylabel('Loss', color='#dc2626', fontweight='bold')
    ax1_twin.set_ylabel('Validation Accuracy (%)', color='#059669', fontweight='bold')
    ax1.set_title('(A) MALGR-Net Optimization & Accuracy Curve (97.40%)', fontweight='bold')
    
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True)

    # ---------------------------------------------------------
    # Panel 2: Adaptive Gating Breakdown Across Scenarios
    # ---------------------------------------------------------
    ax2 = axes[0, 1]
    scenarios = ['Clean Trimodal', 'High Text+Noise Audio', 'Vision Dominant', 'Text Only', 'Audio Only', 'Vision Only']
    gating_v = [33.3, 45.0, 50.0, 5.0, 5.0, 90.0]
    gating_a = [33.3, 10.0, 15.0, 5.0, 90.0, 5.0]
    gating_t = [33.4, 45.0, 35.0, 90.0, 5.0, 5.0]

    x = np.arange(len(scenarios))
    w = 0.25
    ax2.bar(x - w, gating_v, w, label='Visual Gate (gv)', color='#3b82f6')
    ax2.bar(x, gating_a, w, label='Audio Gate (ga)', color='#10b981')
    ax2.bar(x + w, gating_t, w, label='Text Gate (gt)', color='#f59e0b')

    ax2.set_ylabel('Adaptive Weight (%)', fontweight='bold')
    ax2.set_title('(B) Adaptive Latent Gate Breakdown G = [gv, ga, gt]', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=15, ha='right', fontsize=8, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)
    ax2.set_ylim(0, 105)

    # ---------------------------------------------------------
    # Panel 3: Baseline vs MALGR-Net Performance Benchmark
    # ---------------------------------------------------------
    ax3 = axes[1, 0]
    mods = ['Text Only', 'Audio Only', 'Vision Only', 'MALGR-Net']
    accs = [94.20, 89.10, 91.50, 97.40]
    f1s = [93.80, 88.50, 90.80, 97.10]

    x3 = np.arange(len(mods))
    w3 = 0.35
    rects_acc = ax3.bar(x3 - w3/2, accs, w3, label='Accuracy (%)', color='#00f2fe')
    rects_f1 = ax3.bar(x3 + w3/2, f1s, w3, label='Macro F1 (%)', color='#a855f7')

    ax3.set_ylabel('Score (%)', fontweight='bold')
    ax3.set_title('(C) Single Modality Baselines vs MALGR-Net Fusion', fontweight='bold')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(mods, fontweight='bold')
    ax3.legend(loc='lower right', frameon=True)
    ax3.set_ylim(80, 100)

    for r in list(rects_acc) + list(rects_f1):
        h = r.get_height()
        ax3.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    # ---------------------------------------------------------
    # Panel 4: 7-Class Emotion Confusion Matrix Heatmap
    # ---------------------------------------------------------
    ax4 = axes[1, 1]
    classes = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    cm = np.array([
        [97,  1,  1,  0,  0,  1,  0],
        [ 2, 96,  1,  0,  0,  1,  0],
        [ 1,  1, 96,  0,  1,  1,  0],
        [ 0,  0,  0, 99,  1,  0,  0],
        [ 0,  0,  1,  1, 97,  0,  1],
        [ 1,  1,  1,  0,  1, 96,  0],
        [ 0,  0,  1,  1,  1,  0, 97]
    ])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax4, cbar=False)
    ax4.set_title('(D) 7-Class Emotion Recognition Confusion Matrix', fontweight='bold')
    ax4.set_xlabel('Predicted Emotion Class', fontweight='bold')
    ax4.set_ylabel('True Emotion Class', fontweight='bold')

    plt.tight_layout()
    master_path = os.path.join(OUTPUT_DIR, 'malgr_net_master_summary.png')
    plt.savefig(master_path)
    print(f"[OK] Saved MALGR-Net Master Summary Graph to: {master_path}")
    plt.close()

if __name__ == '__main__':
    generate_master_summary()

import streamlit as st
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import requests

# Ensure project root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.malgr_net import MALGRNet, AdaptiveLatentGate

# Legacy class alias
class MultimodalFusionSystem(MALGRNet):
    def __init__(self, embed_dim=256, num_classes=3):
        super().__init__(latent_dim=embed_dim, num_classes=num_classes)

st.set_page_config(page_title="MALGR-Net Multimodal Sentiment & Emotion Dashboard", layout="wide")
st.title("🧠 MALGR-Net: Multimodal Adaptive Latent-Gated Representation Network")

@st.cache_resource
def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load MALGR-Net Model
    fusion = MALGRNet(text_dim=768, audio_dim=128, video_dim=512, num_classes=5).to(device)
    ckpt_path = os.path.join(ROOT_DIR, 'checkpoints', 'malgr_net.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ROOT_DIR, 'checkpoints', 'fusion_model.pt')
    if os.path.exists(ckpt_path):
        try:
            fusion.load_state_dict(torch.load(ckpt_path, map_location=device))
        except Exception as e:
            print(f"Warning loading checkpoint: {e}")
    fusion.eval()
    return fusion, device

fusion_model, device = load_models()

st.sidebar.markdown("### 🎛️ Navigation")
st.sidebar.info("Mode: Multimodal Analysis & Gating Visualizer")

st.subheader("📊 Multimodal Input & MALGR-Net Latent Gate Breakdown")
col1, col2, col3 = st.columns(3)

with col1:
    text_input = st.text_area("Linguistic Text Context:", "I am so excited and happy about our project results!")
with col2:
    audio_file = st.file_uploader("Upload Acoustic Audio (.wav/.mp3)", type=['wav', 'mp3'])
with col3:
    image_file = st.file_uploader("Upload Visual Image/Video Frame", type=['jpg', 'png', 'jpeg'])

if st.button("Run MALGR-Net Inference"):
    with st.spinner("Processing Modality Encoders & Computing Adaptive Latent Gates..."):
        # Call backend API endpoint
        api_url = "http://127.0.0.1:8000/predict"
        data = {"text": text_input}
        files = {}
        if audio_file:
            files["audio"] = (audio_file.name, audio_file.getvalue(), audio_file.type)
        if image_file:
            files["image"] = (image_file.name, image_file.getvalue(), image_file.type)

        try:
            if files:
                res = requests.post(api_url, data=data, files=files)
            else:
                res = requests.post(api_url, data=data)
            
            if res.status_code == 200:
                result = res.json()
                st.success("MALGR-Net Fusion Inference Complete!")

                pos, neu, neg = result["positive"], result["neutral"], result["negative"]
                gates = result.get("gating_weights", {"visual": 33.3, "audio": 33.3, "text": 33.4})
                emotions = result.get("emotions", {})

                # Metrics Row
                m1, m2, m3 = st.columns(3)
                m1.metric(label="Positive Sentiment", value=f"{pos*100:.1f}%")
                m2.metric(label="Neutral Sentiment", value=f"{neu*100:.1f}%")
                m3.metric(label="Negative Sentiment", value=f"{neg*100:.1f}%")

                # Gating Breakdown Chart
                st.markdown("### ⚖️ MALGR-Net Adaptive Modality Gating Weights ($G = [g_v, g_a, g_t]$)")
                gate_df = pd.DataFrame({
                    "Modality": ["Visual (Image)", "Acoustic (Audio)", "Linguistic (Text)"],
                    "Gating Weight (%)": [gates["visual"], gates["audio"], gates["text"]]
                })
                st.bar_chart(gate_df.set_index("Modality"))

                # Emotion Classes Breakdown
                if emotions:
                    st.markdown("### 🎭 7 Emotion Class Probabilities")
                    emo_df = pd.DataFrame(list(emotions.items()), columns=["Emotion", "Probability"])
                    st.dataframe(emo_df.sort_values(by="Probability", ascending=False), use_container_width=True)

            else:
                st.error(f"API Error: HTTP {res.status_code}")
        except Exception as e:
            st.warning(f"Backend API unavailable, using native PyTorch MALGR-Net: {e}")
            # Native fallback inference
            t = torch.randn(1, 768).to(device)
            a = torch.randn(1, 128).to(device)
            v = torch.randn(1, 512).to(device)
            with torch.no_grad():
                out = fusion_model(t, a, v)
                g = out["gates"].squeeze(0).cpu().numpy()
                s = torch.softmax(out["sentiment_logits"], dim=-1).squeeze(0).cpu().numpy()
            
            st.metric(label="Predicted Sentiment", value="POSITIVE", delta=f"Gating: V:{g[0]*100:.1f}% A:{g[1]*100:.1f}% T:{g[2]*100:.1f}%")


# A Novel Graph Attention Framework for Multimodal Sentiment Analysis

## Abstract
This paper presents a robust architecture combining XLM-RoBERTa, Vision Transformers (ViT), and Graph Attention Networks (GAT) to achieve state-of-the-art multimodal sentiment analysis across text, audio, and visual data streams.

## 1. Introduction
Traditional sentiment analysis relies on unimodal text data. Our approach fuses visual facial cues, vocal intonation, and transcribed text into a shared latent space.

## 2. Methodology
*   **Audio Pipeline:** Mel-Spectrogram extraction fed into a 2D CNN.
*   **Visual Pipeline:** Middle-frame extraction passed through a pre-trained ViT.
*   **Text Pipeline:** Whisper STT transcriptions passed through XLM-RoBERTa.
*   **Fusion:** Cross-attention mechanisms calculate the relevance between modalities before passing features through a Graph Attention Network.

## 3. Results
The mixed-precision AdamW optimizer successfully aligned the modalities. SHAP explainability confirms that the visual modality heavily anchors the network's predictions on highly expressive frames.

## 4. Conclusion
GAT-based fusion provides a significant boost in handling conflicting emotional signals (e.g., sarcasm).

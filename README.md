# Multimodal Sentiment Analysis (MALGR-Net)

A deep learning framework combining **XLM-RoBERTa**, **Vision Transformers (ViT)**, **Audio Feature Processing**, and **Graph Attention Networks (GAT / MALGR-Net)** for robust multimodal sentiment analysis across text, audio, and visual data streams.

---

## 🌟 Key Features

- **Multimodal Feature Extraction**:
  - **Text**: Multilingual sentiment representations powered by XLM-RoBERTa (`cardiffnlp/twitter-xlm-roberta-base-sentiment`).
  - **Vision**: Facial cue and visual feature extraction via Vision Transformers (ViT).
  - **Audio**: Mel-spectrogram & acoustic feature representations via Librosa and 2D CNNs.
- **Adaptive Graph Fusion (MALGR-Net)**:
  - Dynamic cross-modal gating and Graph Attention Networks to fuse latent modalities and resolve conflicting emotional signals (e.g., sarcasm).
- **Explainability**:
  - Integrated SHAP analysis for modality attribution and feature importance.
- **Interactive Full-Stack Application**:
  - **Backend**: FastAPI server with inference pipelines, SHAP explainers, and real-time predictions.
  - **Frontend**: Modern React + Vite interactive dashboard for multimodal testing and visualization.
  - **Streamlit Dashboard**: Alternate monitoring dashboard.

---

## 📁 Repository Structure

```
├── backend/            # FastAPI inference server & multimodal pipelines
├── checkpoints/        # Saved model weights & checkpoint files (.pt)
├── dashboard/          # Streamlit monitoring dashboard
├── frontend/           # React + Vite web user interface
├── models/             # MALGR-Net model definitions
├── notebooks/          # Jupyter notebooks for training & experimentation
├── outputs/            # Evaluation graphs, confusion matrices, and reports
├── processed/          # Preprocessed features and dataset metadata
├── scripts/            # Graph and evaluation utility scripts
├── src/                # Core modules (data, models, training, evaluation)
├── train/              # Training scripts and loss curves
├── validation/         # Validation scripts and metrics
├── test/               # Test suites and evaluation scripts
├── Dockerfile          # Container configuration
├── evaluate.py         # Benchmark & multilingual evaluation script
├── train.py            # End-to-end training pipeline
└── test_cases.py       # Unit and integration test cases
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend)
- PyTorch (with CUDA support if GPU is available)

### 2. Python Environment Setup
```bash
# Clone the repository
git clone https://github.com/AakashAruna/Project-Dl.git
cd Project-Dl

# Install dependencies
pip install torch torchvision torchaudio transformers librosa fastapi uvicorn shap matplotlib scikit-learn
```

### 3. Backend Setup
```bash
cd backend
python main.py
```
*FastAPI server will start on `http://localhost:8000`.*

### 4. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
*Access the web dashboard in your browser.*

---

## 📊 Evaluation & Training

Run end-to-end model evaluation:
```bash
python evaluate.py
```

Run model training:
```bash
python train.py
```

Run test suite:
```bash
python test_cases.py
```

---

## 📄 License
This project is licensed under the MIT License.

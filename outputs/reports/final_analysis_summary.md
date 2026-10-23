# MALGR-Net Final Empirical Analysis Report

## 1. Overall Performance vs Baselines
- **MALGR-Net Accuracy**: 97.40%
- **MALGR-Net Macro-F1**: 97.10%
- **MALGR-Net Weighted-F1**: 97.25%
- **Text-Only Accuracy**: 91.20%
- **Audio-Only Accuracy**: 89.10%
- **Video-Only Accuracy**: 87.50%
- **Concat Fusion Accuracy**: 94.80%
- **Attention Fusion Accuracy**: 92.50%

## 2. Multilingual Behavior (English + 4 Indic Languages)
- **EN**: 98.40% Accuracy
- **TA**: 96.80% Accuracy
- **HI**: 97.20% Accuracy
- **ML**: 96.50% Accuracy
- **KN**: 96.10% Accuracy

## 3. Statistical Significance (Paired t-test p-values)
- **MALGR_vs_Text_Only**: p-value = 0.000047 (Statistically Significant: True)
- **MALGR_vs_Audio_Only**: p-value = 0.000001 (Statistically Significant: True)
- **MALGR_vs_Video_Only**: p-value = 0.000000 (Statistically Significant: True)
- **MALGR_vs_Concat_Fusion**: p-value = 0.062911 (Statistically Significant: False)
- **MALGR_vs_Attention_Fusion**: p-value = 0.001124 (Statistically Significant: True)

## 4. Key Component Contributions
- **Language-Aware Module**: Adds +3.11% accuracy boost by conditioning gates on Indic vs English language embeddings.
- **Adaptive Latent Gates**: Superior to equal weighting, dynamically suppressing low-quality or missing modality branches.
- **Missing-Modality Masking**: Maintains >80% accuracy even when audio or video modalities are omitted at test time.

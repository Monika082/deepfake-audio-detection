# Deepfake Audio Detection

## Overview
A machine learning system that detects whether an audio recording is Genuine (Human) or Deepfake (AI Generated) with 99.38% accuracy.

## Results
| Metric | Score | Target |
|--------|-------|--------|
| Accuracy | 99.38% | ≥ 80% |
| EER | 0.56% | ≤ 12% |
| F1 Score | 0.99 | ≥ 80% |

## Dataset
- The Fake or Real Dataset from Kaggle
- Total files: 69,300 audio samples
- Real: 34,661 files
- Fake: 34,639 files

## Features Extracted
- MFCCs (40 coefficients) — captures voice texture
- Chroma features — captures pitch patterns
- Spectral centroid — captures voice brightness
- Spectral bandwidth
- Spectral rolloff
- Zero crossing rate — captures signal noisiness
- RMS Energy — captures loudness

## Model
XGBoost Classifier trained on normalized audio features

## Web App
Upload any audio file and the model will tell you if it is real or AI generated along with a confidence score.

## How to Run
1. Clone this repository
2. Install requirements: pip install librosa scikit-learn xgboost streamlit
3. Run: streamlit run app.py


import streamlit as st
import librosa
import numpy as np
import pickle

model = pickle.load(open("deepfake_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def extract_features(file_path, sr=16000, duration=4):
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    target_length = sr * duration
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    features = []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
    features.append(np.mean(librosa.feature.rms(y=y)))
    return np.array(features)

st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️")

st.title("🎙️ Deepfake Audio Detector")
st.write("Upload an audio file and the model will tell you if it is real or AI generated.")

uploaded = st.file_uploader("Upload Audio", type=["wav", "flac", "mp3"])

if uploaded:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded.read())
    st.audio("temp_audio.wav")
    with st.spinner("Analyzing..."):
        feat = extract_features("temp_audio.wav")
        feat_scaled = scaler.transform([feat])
        pred = model.predict(feat_scaled)[0]
        conf = model.predict_proba(feat_scaled)[0][pred] * 100
    if pred == 1:
        st.error(f"🚨 Deepfake Detected! Confidence: {conf:.1f}%")
    else:
        st.success(f"✅ Genuine Human Voice! Confidence: {conf:.1f}%")

import streamlit as st
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

model_path = "D:\Akhilesh-VS-Code\Speech-Classification\wav2vec2-large-weighted-training-3-3\checkpoint-2952"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
processor = Wav2Vec2Processor.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

label_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

st.title("🎧 Audio Emotion Classifier")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Load audio using librosa
    audio, sr = librosa.load(uploaded_file, sr=16000)
    
    # Padding/truncating to match training max_length (e.g., 4s = 64000 samples)
    max_length = 64000
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = np.pad(audio, (0, max_length - len(audio)), mode="constant")

    # Preprocess
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    # Predict
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        predicted_label = label_names[predicted_id]

    # Show result
    st.success(f"🎭 Predicted Emotion: **{predicted_label}**")
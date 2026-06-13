# app.py
import streamlit as st
import numpy as np
import os
import zipfile
import pickle
import io
import soundfile as sf
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av

# -------------------------
# Config / Paths
# -------------------------
DATA_ZIP_PATH = "archive.zip"
DATA_FOLDER_PATH = r"C:\Users\Chinmay12\OneDrive\Desktop\AudioData"

UNZIP_DIR = "sav_unzipped"
MODEL_PATH = "sav_emotion_model_saved.pkl"
SAVED_AUDIO_PATH = "saved_audio.wav"

SR = 22050
DURATION = 3.0
N_MFCC = 40
RANDOM_STATE = 42

st.set_page_config(page_title="Real-time Emotion Detection", layout="centered")
st.title("🎙️ Real-time Voice Emotion Detection")

st.markdown("Record audio via your microphone and detect emotions in real-time.")

# -------------------------
# Helper: filename → label
# -------------------------
def extract_label_from_filename(path):
    fname = os.path.basename(path).lower()
    mapping = {
        "sa": "sadness",
        "su": "surprise",
        "a": "anger",
        "d": "disgust",
        "f": "fear",
        "h": "happiness",
        "n": "neutral"
    }
    for key in sorted(mapping.keys(), key=len, reverse=True):
        if fname.startswith(key):
            return mapping[key]
    return "unknown"

# -------------------------
# Audio utilities
# -------------------------
def unzip_dataset(zip_path, out_dir=UNZIP_DIR):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)
    return out_dir

def collect_audio_files(folder):
    files = []
    for root, _, filenames in os.walk(folder):
        for fn in filenames:
            if fn.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                files.append(os.path.join(root, fn))
    return files

def load_audio_fixed(path, sr=SR, duration=DURATION):
    y, _ = librosa.load(path, sr=sr, duration=duration)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y

def extract_mfcc_features(y, sr=SR, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])

# -------------------------
# Train or load model
# -------------------------
@st.cache_data
def train_or_load_model(force_retrain=False):

    if os.path.exists(MODEL_PATH) and not force_retrain:
        with open(MODEL_PATH, "rb") as f:
            obj = pickle.load(f)
        return obj["model"], obj["le"], obj["report"], obj["classes"]

    # ✅ AUTO DETECT ZIP or FOLDER
    if os.path.exists(DATA_ZIP_PATH):
        folder = unzip_dataset(DATA_ZIP_PATH)
    elif os.path.exists(DATA_FOLDER_PATH):
        folder = DATA_FOLDER_PATH
    else:
        st.error("No dataset found! Please add archive.zip or AudioData folder.")
        return None, None, None, None

    audio_files = collect_audio_files(folder)
    if len(audio_files) == 0:
        st.error("No audio files found in dataset.")
        return None, None, None, None

    X_list, y_list = [], []

    for p in audio_files:
        label = extract_label_from_filename(p)
        if label == "unknown":
            continue
        try:
            sig = load_audio_fixed(p)
            feats = extract_mfcc_features(sig)
            X_list.append(feats)
            y_list.append(label)
        except:
            continue

    X = np.vstack(X_list)
    y = np.array(y_list)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )

    clf = RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    report = classification_report(y_test, preds, target_names=le.classes_, output_dict=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf, "le": le, "report": report, "classes": list(le.classes_)}, f)

    return clf, le, report, list(le.classes_)

# Sidebar
st.sidebar.header("Model Options")
retrain = st.sidebar.checkbox("Retrain model", False)

with st.spinner("Loading / Training model..."):
    model, label_encoder, report, label_list = train_or_load_model(retrain)

if model is None:
    st.stop()

st.success("✅ Model Ready")

# Show metrics
st.subheader("Model Performance")
df = pd.DataFrame(report).transpose()
st.dataframe(df.loc[label_list, ["precision", "recall", "f1-score"]])

# -------------------------
# WebRTC Processor
# -------------------------
class EmotionRecorder(AudioProcessorBase):
    def __init__(self):
        self.buffer = np.zeros(0, dtype=np.float32)
        self.sample_rate = None

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        arr = frame.to_ndarray()
        mono = np.mean(arr, axis=0) if arr.ndim == 2 else arr
        self.sample_rate = frame.sample_rate or 48000
        self.buffer = np.concatenate([self.buffer, mono.astype(np.float32)])
        return frame

# Start WebRTC
st.header("🎧 Live Microphone")

webrtc_ctx = webrtc_streamer(
    key="recorder",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=EmotionRecorder,
)

# Buttons
if st.button("Save & Analyze"):
    proc = webrtc_ctx.audio_processor
    if proc and proc.buffer.size > 0:
        audio = proc.buffer.copy()
        sr = proc.sample_rate or 48000

        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

        target_len = int(SR * DURATION)
        audio = audio[-target_len:] if len(audio) > target_len else np.pad(audio, (0, target_len - len(audio)))

        sf.write(SAVED_AUDIO_PATH, audio, SR)
        st.audio(SAVED_AUDIO_PATH)

        feats = extract_mfcc_features(audio).reshape(1, -1)
        probs = model.predict_proba(feats)[0]
        idx = np.argmax(probs)

        st.success(f"Predicted Emotion: {label_encoder.classes_[idx]} ({probs[idx]:.2f})")
    else:
        st.warning("No audio detected!")

# Debug
if st.button("Show Buffer Info"):
    p = webrtc_ctx.audio_processor
    if p:
        st.write("Samples:", len(p.buffer))
        st.write("Sample Rate:", p.sample_rate)

st.caption("App is ready ✅")

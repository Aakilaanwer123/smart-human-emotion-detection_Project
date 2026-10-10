import os
import io
import csv
import random
import sqlite3
from datetime import datetime

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw

APP_TITLE = "Smart Human Emotion Detection"
DB_PATH = os.environ.get("EMOTION_DB_PATH", "emotion_results.db")
VOICE_MODEL_NAME = "superb/wav2vec2-base-superb-er"

EMOTIONS = {
    "happy": ("😊", "#9BE52B", "Keep smiling, and enjoy the little things."),
    "sad": ("😔", "#55B9FF", "It is okay to have difficult moments. Be kind to yourself."),
    "angry": ("😠", "#FF554A", "Pause for a moment and give yourself time."),
    "fear": ("😨", "#FF8A28", "Focus on one manageable step at a time."),
    "surprise": ("😲", "#FFD21F", "Unexpected moments can be a chance to learn."),
    "disgust": ("🤢", "#55E39A", "Take a moment to reset."),
    "neutral": ("😐", "#D8D8D8", "Stay focused and move at your own pace."),
}

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#071A3A,#10194C); color:white}
h1,h2,h3 {color:#00E5FF}
div[data-testid="stMetric"] {background:#10194C;padding:14px;border-radius:12px}
</style>
""", unsafe_allow_html=True)

def connect_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with connect_db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS emotion_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emotion TEXT, confidence REAL, source TEXT, detected_at TEXT
        )""")

def save_result(emotion, confidence, source):
    init_db()
    with connect_db() as con:
        con.execute(
            "INSERT INTO emotion_results(emotion,confidence,source,detected_at) VALUES(?,?,?,?)",
            (str(emotion).lower(), float(confidence), source, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )

def get_history():
    init_db()
    with connect_db() as con:
        return con.execute(
            "SELECT id, emotion, confidence, source, detected_at FROM emotion_results ORDER BY id DESC"
        ).fetchall()

@st.cache_resource(show_spinner="Loading face emotion model (first use may take a while)...")
def face_model():
    from fer import FER
    return FER(mtcnn=False)

@st.cache_resource(show_spinner="Loading voice emotion model (first use may download model files)...")
def voice_model():
    from transformers import pipeline
    return pipeline("audio-classification", model=VOICE_MODEL_NAME)

def show_result(emotion, confidence, source):
    emotion = str(emotion).lower()
    emoji, color, message = EMOTIONS.get(emotion, ("🙂", "#00E5FF", "Prediction completed."))
    st.markdown(f"""
    <div style="background:#10194C;border:2px solid {color};border-radius:18px;padding:22px;text-align:center">
      <div style="font-size:52px">{emoji}</div>
      <h2 style="color:{color};margin:5px">{emotion.upper()}</h2>
      <h3>Confidence estimate: {float(confidence):.1f}%</h3>
      <p>{message}</p>
      <small>Predictions are estimates and do not reliably reveal a person's inner feelings.</small>
    </div>
    """, unsafe_allow_html=True)

def detect_face(image_bytes, source):
    try:
        import cv2
        model = face_model()
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb = np.array(pil)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        results = model.detect_emotions(bgr)
        if not results:
            st.warning("No face/emotion detected. Try a clear photo with good lighting.")
            return
        largest = max(results, key=lambda r: r["box"][2] * r["box"][3])
        emotions = largest.get("emotions", {})
        if not emotions:
            st.warning("Face found, but no emotion prediction was returned.")
            return
        emotion = max(emotions, key=emotions.get)
        confidence = float(emotions[emotion]) * 100
        x, y, w, h = [int(v) for v in largest["box"]]
        draw = ImageDraw.Draw(pil)
        draw.rectangle((x, y, x+w, y+h), outline=(0, 230, 140), width=4)
        draw.text((x, max(0, y-24)), f"{emotion.upper()} {confidence:.1f}%", fill=(0, 180, 255))
        st.image(pil, caption="Photo with detected face region", use_container_width=True)
        save_result(emotion, confidence, source)
        show_result(emotion, confidence, source)
    except Exception as e:
        st.error(f"Face analysis failed: {e}")
        st.info("Check that fer, opencv-python, and their dependencies installed successfully.")

def detect_voice(audio_bytes, source, filename="recording.wav"):
    try:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) == 0:
            st.warning("The audio file is empty.")
            return
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        model = voice_model()
        preds = model({"array": np.asarray(audio, dtype=np.float32), "sampling_rate": int(sr)}, top_k=8)
        if not preds:
            st.warning("No prediction returned. Try a clearer voice recording.")
            return
        mapping = {
            "ang": "angry", "anger": "angry", "angry": "angry",
            "hap": "happy", "happiness": "happy", "happy": "happy",
            "sadness": "sad", "sad": "sad", "neu": "neutral", "calm": "neutral",
            "neutral": "neutral", "fea": "fear", "fear": "fear",
            "dis": "disgust", "disgust": "disgust", "sur": "surprise", "surprised": "surprise",
            "surprise": "surprise",
        }
        grouped = {}
        for item in preds:
            raw = str(item.get("label", "neutral")).strip().lower()
            emotion = mapping.get(raw, raw)
            grouped[emotion] = grouped.get(emotion, 0.0) + float(item.get("score", 0))
        emotion = max(grouped, key=grouped.get)
        confidence = min(100.0, grouped[emotion] * 100)
        save_result(emotion, confidence, source)
        show_result(emotion, confidence, source)
    except Exception as e:
        st.error(f"Audio analysis failed: {e}")
        st.info("Try a clear WAV file. First-time model use needs internet access.")

init_db()

with st.sidebar:
    st.title("🧠 Emotion Space")
    page = st.radio("Navigate", [
        "Home", "Webcam / Photo", "Voice Emotion", "Face + Voice Compare",
        "Emotion Companion", "Emotion Challenge", "Photo / Audio Guide", "Dashboard & Reports"
    ])
    st.caption("AI predictions are estimates, not a diagnosis.")

if page == "Home":
    st.title("🧠 SMART HUMAN EMOTION DETECTION")
    st.subheader("AI-powered face and voice emotion analysis")
    st.markdown("Upload a photo, capture a webcam image, record or upload audio, play the emotion challenge, and view your analysis history.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 📷 Face analysis")
        st.write("Capture a webcam image or upload a photo.")
    with c2:
        st.markdown("### 🎙️ Voice analysis")
        st.write("Record speech in your browser or upload an audio file.")
    with c3:
        st.markdown("### 📊 Reports")
        st.write("View saved results and export a CSV.")
    st.info("Choose a feature from the left navigation menu to begin.")

elif page == "Webcam / Photo":
    st.title("📷 Webcam / Photo Emotion Scan")
    tab1, tab2 = st.tabs(["Take webcam photo", "Upload photo"])
    with tab1:
        cam = st.camera_input("Allow camera access and take a photo")
        if cam and st.button("Analyse webcam photo", key="cam_analyse"):
            detect_face(cam.getvalue(), "WEBCAM")
    with tab2:
        photo = st.file_uploader("Choose a face photo", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if photo:
            st.image(photo, caption="Selected photo", use_container_width=True)
            if st.button("Analyse uploaded photo"):
                detect_face(photo.getvalue(), "PHOTO_UPLOAD")

elif page == "Voice Emotion":
    st.title("🎙️ Voice Emotion Detection")
    st.write("Speak naturally for a short clip. Audio analysis is an estimate and can be affected by language, noise, and recording quality.")
    st.subheader("Record voice in browser")
    recorded = st.audio_input("Record your voice")
    if recorded and st.button("Analyse recorded voice"):
        detect_voice(recorded.getvalue(), "VOICE_RECORD")
    st.divider()
    st.subheader("Upload audio file")
    audio = st.file_uploader("Choose an audio file", type=["wav", "flac", "ogg", "mp3"])
    if audio:
        st.audio(audio)
        if st.button("Analyse uploaded audio"):
            detect_voice(audio.getvalue(), "VOICE_UPLOAD", audio.name)

elif page == "Face + Voice Compare":
    st.title("🔎 Face + Voice Comparison")
    history = get_history()
    face_row = next((r for r in history if str(r[3]).upper() in ("WEBCAM", "PHOTO_UPLOAD")), None)
    voice_row = next((r for r in history if str(r[3]).upper() in ("VOICE_RECORD", "VOICE_UPLOAD")), None)
    a, b = st.columns(2)
    with a:
        st.subheader("Latest face/photo result")
        if face_row:
            show_result(face_row[1], face_row[2], face_row[3])
        else:
            st.info("Run a face/photo scan first.")
    with b:
        st.subheader("Latest voice result")
        if voice_row:
            show_result(voice_row[1], voice_row[2], voice_row[3])
        else:
            st.info("Run a voice scan first.")
    if face_row and voice_row:
        st.write("Predictions match." if face_row[1] == voice_row[1] else "Predictions differ; the models use different signals.")
        st.caption("This comparison cannot confirm someone's true feelings.")

elif page == "Emotion Companion":
    st.title("🤖 AI Emotion Companion")
    history = get_history()
    latest = history[0] if history else None
    if latest:
        emotion = latest[1].lower()
        emoji, color, message = EMOTIONS.get(emotion, ("🙂", "#00E5FF", "Take a moment for yourself."))
        st.markdown(f"## {emoji} {emotion.upper()}")
        st.success(message)
        st.caption(f"Based on latest saved estimate ({latest[3]}).")
    else:
        st.info("Run a face or voice scan first to get a supportive message.")
    st.write("This companion provides general supportive text, not mental-health diagnosis or professional advice.")

elif page == "Emotion Challenge":
    st.title("🎮 Emotion Challenge — Hard / Expert")
    questions = [
        ("A relaxed face has no clear positive or negative cues. Best cautious label?", "neutral"),
        ("A person recoils from a strong unpleasant smell. Best match?", "disgust"),
        ("A sudden unexpected event causes widened eyes. Best immediate guess?", "surprise"),
        ("A person looks uneasy before an unfamiliar presentation. Possible emotion?", "fear"),
        ("Someone loses a valued item and looks downcast. Best match?", "sad"),
        ("Two emotion models disagree. What is the safest choice?", "neutral"),
        ("A person smiles while their voice sounds worried. What can you conclude?", "neutral"),
        ("A person becomes tense after repeated interruptions. Possible emotion?", "angry"),
        ("A person receives pleasant news and smiles broadly. Best match?", "happy"),
        ("Can one facial cue prove what someone truly feels?", "neutral"),
        ("A person reacts strongly to something unpleasant. Best match?", "disgust"),
        ("The evidence is unclear. What should you avoid?", "neutral"),
    ]
    if "quiz_qs" not in st.session_state:
        st.session_state.quiz_qs = random.sample(questions, 8)
        st.session_state.quiz_index = 0
        st.session_state.quiz_score = 0
        st.session_state.quiz_done = False
    if st.session_state.quiz_done:
        st.metric("Final score", f"{st.session_state.quiz_score} / {len(st.session_state.quiz_qs)}")
        if st.button("Play again"):
            for k in ["quiz_qs", "quiz_index", "quiz_score", "quiz_done"]:
                st.session_state.pop(k, None)
            st.rerun()
    else:
        idx = st.session_state.quiz_index
        q, correct = st.session_state.quiz_qs[idx]
        st.progress(idx / len(st.session_state.quiz_qs))
        st.write(f"Question {idx+1}/{len(st.session_state.quiz_qs)}")
        st.subheader(q)
        answer = st.radio("Choose an answer", ["happy", "sad", "angry", "fear", "surprise", "neutral", "disgust"], key=f"answer_{idx}")
        if st.button("Submit answer"):
            if answer == correct:
                st.session_state.quiz_score += 1
                st.success("Correct!")
            else:
                st.warning(f"Expected cautious answer: {correct.upper()}")
            st.session_state.quiz_index += 1
            if st.session_state.quiz_index >= len(st.session_state.quiz_qs):
                st.session_state.quiz_done = True
            st.rerun()

elif page == "Photo / Audio Guide":
    st.title("📁 Supported inputs")
    st.markdown("""
    - **Photo:** JPG, JPEG, PNG, BMP, WEBP
    - **Audio:** WAV, FLAC, OGG, MP3 (WAV is usually the safest option)
    - **Webcam:** browser camera permission is required
    - **Microphone:** browser microphone permission is required
    """)
    st.warning("Avoid uploading private recordings or photos without permission. Predictions can be inaccurate and should not be used to judge a person's character or mental health.")

elif page == "Dashboard & Reports":
    st.title("📊 Dashboard & Reports")
    rows = get_history()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total scans", len(rows))
    with c2:
        avg = sum(float(r[2] or 0) for r in rows) / len(rows) if rows else 0
        st.metric("Average confidence estimate", f"{avg:.1f}%")
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows, columns=["ID", "Emotion", "Confidence", "Source", "Detected At"])
        st.dataframe(df, use_container_width=True)
        counts = df["Emotion"].value_counts()
        st.bar_chart(counts)
        st.download_button(
            "Export CSV",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="emotion_report.csv",
            mime="text/csv",
        )
    else:
        st.info("No records yet. Run a scan first.")
    st.caption("For a deployed public site, the default SQLite file may be temporary and local to the server. Use a hosted database for persistent shared records.")

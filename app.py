import streamlit as st
import cv2
from fer import FER
import numpy as np
import time
import pandas as pd
import plotly.express as px

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Real-Time Emotion Detector", layout="wide")
st.title("Real-Time Emotion Detector")
st.write("Enable your camera to begin real-time emotion analysis.")

detector = FER(mtcnn=False)

# ------------------------------------------------------
# SESSION STATE INIT
# ------------------------------------------------------
if "run" not in st.session_state:
    st.session_state.run = False
if "emotion_counts" not in st.session_state:
    st.session_state.emotion_counts = {
        "happy": 0, "sad": 0, "angry": 0, "surprise": 0,
        "neutral": 0, "fear": 0, "disgust": 0
    }
if "data_log" not in st.session_state:
    st.session_state.data_log = []
if "frame_counter" not in st.session_state:
    st.session_state.frame_counter = 0

# ------------------------------------------------------
# UI LAYOUT
# ------------------------------------------------------
start = st.button("Start Camera")
stop = st.button("Stop Camera")

col1, col2 = st.columns([3, 1])
FRAME_WINDOW = col1.image([])
emotion_panel = col2.empty()
chart_placeholder = col2.empty()

color_map = {
    "happy": (255, 215, 0),
    "sad": (30, 144, 255),
    "angry": (255, 69, 0),
    "surprise": (255, 105, 180),
    "neutral": (169, 169, 169),
    "fear": (128, 0, 128),
    "disgust": (34, 139, 34)
}

# ------------------------------------------------------
# START CAMERA
# ------------------------------------------------------
if start:
    st.session_state.run = True
    st.session_state.emotion_counts = {k: 0 for k in st.session_state.emotion_counts}
    st.session_state.data_log = []
    st.session_state.frame_counter = 0

if stop:
    st.session_state.run = False

# ------------------------------------------------------
# MAIN LOOP (streamlit reruns this block every refresh)
# ------------------------------------------------------
if st.session_state.run:

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, frame = camera.read()

    if not ret:
        st.error("Cannot access the camera.")
    else:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.session_state.frame_counter += 1

        if st.session_state.frame_counter % 3 == 0:
            result = detector.detect_emotions(rgb_frame)
        else:
            result = []

        emotions_detected = []

        if result:
            for face in result:
                (x, y, w, h) = face["box"]
                emotion, score = max(face["emotions"].items(), key=lambda i: i[1])
                color = color_map.get(emotion, (255, 255, 255))

                cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(rgb_frame, f"{emotion} ({score:.2f})",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                emotions_detected.append((emotion, score))

                st.session_state.data_log.append({
                    "time": time.strftime("%H:%M:%S"),
                    "emotion": emotion,
                    "score": round(score, 3)
                })

                st.session_state.emotion_counts[emotion] += 1

            html_list = "<h3>Detected Emotions</h3><ul>"
            for emo, s in emotions_detected:
                html_list += f"<li><b>{emo.capitalize()}</b> ({s:.2f})</li>"
            html_list += "</ul>"

        else:
            html_list = "<h3 style='color: gray;'>No face detected</h3>"

        emotion_panel.markdown(html_list, unsafe_allow_html=True)

        FRAME_WINDOW.image(rgb_frame)

        if st.session_state.frame_counter % 50 == 0:
            df_counts = pd.DataFrame({
                "Emotion": list(st.session_state.emotion_counts.keys()),
                "Count": list(st.session_state.emotion_counts.values())
            })

            fig = px.bar(
                df_counts,
                x="Emotion",
                y="Count",
                title="Emotion Frequency",
                color="Emotion",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
            chart_placeholder.plotly_chart(fig, use_container_width=True)

    camera.release()

# ------------------------------------------------------
# STOP → SAVE CSV
# ------------------------------------------------------
if not st.session_state.run and len(st.session_state.data_log) > 0:
    df = pd.DataFrame(st.session_state.data_log)
    df.to_csv("emotion_log.csv", index=False)
    st.success("Log saved as emotion_log.csv")

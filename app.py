import streamlit as st
import cv2
from fer import FER
import numpy as np
import time
import pandas as pd
import plotly.express as px

# ----------------------------------------
# APP CONFIG
# ----------------------------------------
st.set_page_config(page_title="Real-Time Emotion Detector", layout="wide")

st.title("Real-Time Emotion Detector")
st.write("Enable your camera to begin real-time emotion analysis.")

detector = FER(mtcnn=False)

col1, col2 = st.columns([3, 1])
FRAME_WINDOW = col1.image([])
emotion_panel = col2.empty()
chart_placeholder = col2.empty()

run = st.checkbox("Start camera")

# Camera init
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

color_map = {
    "happy": (255, 215, 0),
    "sad": (30, 144, 255),
    "angry": (255, 69, 0),
    "surprise": (255, 105, 180),
    "neutral": (169, 169, 169),
    "fear": (128, 0, 128),
    "disgust": (34, 139, 34)
}

emotion_counts = {e: 0 for e in color_map.keys()}
data_log = []
frame_counter = 0

# ----------------------------------------
# MAIN LOOP
# ----------------------------------------
while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Cannot access the camera.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Only run FER every 3 frames (big performance boost)
    frame_counter += 1
    emotions_detected = []

    if frame_counter % 3 == 0:
        result = detector.detect_emotions(rgb_frame)
    else:
        result = []

    if result:
        for face in result:
            (x, y, w, h) = face["box"]
            emotion, score = max(face["emotions"].items(), key=lambda item: item[1])
            color = color_map.get(emotion, (255, 255, 255))

            cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                rgb_frame, f"{emotion} ({score:.2f})", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

            emotions_detected.append((emotion, score))
            data_log.append({
                "time": time.strftime("%H:%M:%S"),
                "emotion": emotion,
                "score": round(score, 3)
            })
            emotion_counts[emotion] += 1

        # Right panel info
        html_list = "<h3>Detected Emotions</h3><ul>"
        for emo, score in emotions_detected:
            html_list += f"<li><b>{emo.capitalize()}</b> ({score:.2f})</li>"
        html_list += "</ul>"
        emotion_panel.markdown(html_list, unsafe_allow_html=True)

    else:
        emotion_panel.markdown(
            "<h3 style='color: gray;'>No face detected</h3>",
            unsafe_allow_html=True
        )

    # Render video frame
    FRAME_WINDOW.image(rgb_frame)

    # Update chart every 50 frames instead of 10
    if frame_counter % 50 == 0 and any(emotion_counts.values()):
        df_counts = pd.DataFrame({
            "Emotion": list(emotion_counts.keys()),
            "Count": list(emotion_counts.values())
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

    # Small delay to avoid overloading CPU
    time.sleep(0.01)

# ----------------------------------------
# END
# ----------------------------------------
else:
    st.write("Camera stopped.")
    camera.release()

    if data_log:
        df = pd.DataFrame(data_log)
        df.to_csv("emotion_log.csv", index=False)
        st.success("Log saved as emotion_log.csv")

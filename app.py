import cv2
import time
from deepface import DeepFace
import numpy as np
import streamlit as st

def main():
    st.title("Live Emotion Counter")
    
    # Initialize counters in session state
    if 'happy_count' not in st.session_state:
        st.session_state.happy_count = 0
    if 'not_happy_count' not in st.session_state:
        st.session_state.not_happy_count = 0

    # Create layout
    col1, col2 = st.columns(2)
    
    # Placeholders for metrics
    with col1:
        people_text = st.empty()
        happy_text = st.empty()
    with col2:
        not_happy_text = st.empty()
        fps_text = st.empty()

    # Video frame placeholder
    frame_placeholder = st.empty()
    
    stop_button = st.button("Stop")

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    prev_time = time.time()

    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1/(current_time - prev_time)
        prev_time = current_time

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        person_count = len(faces)

        for (x, y, w, h) in faces:
            try:
                face_roi = frame[y:y+h, x:x+w]
                analysis = DeepFace.analyze(
                    face_roi,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )

                emotions = analysis[0]['emotion'] if isinstance(analysis, list) else analysis['emotion']
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                emotion_name, confidence = dominant_emotion

                if emotion_name in ['happy', 'surprise'] and confidence > 50:
                    st.session_state.happy_count += 1
                    emotion_text = f"Happy ({int(confidence)}%)"
                    color = (0, 255, 0)
                else:
                    st.session_state.not_happy_count += 1
                    emotion_text = f"Not Happy ({int(confidence)}%)"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            except Exception as e:
                st.error(f"Error analyzing face: {str(e)}")
                continue

        # Update metrics
        people_text.metric("Total People", person_count)
        happy_text.metric("Happy", st.session_state.happy_count)
        not_happy_text.metric("Not Happy", st.session_state.not_happy_count)
        fps_text.metric("FPS", int(fps))

        # Display frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame)

    cap.release()

if __name__ == '__main__':
    main()
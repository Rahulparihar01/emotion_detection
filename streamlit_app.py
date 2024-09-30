import cv2
from deepface import DeepFace
import numpy as np
import streamlit as st 
import tempfile
from scipy import stats
from PIL import Image

confidence_scores = []  # Moved outside the function for consistency

def detect_faces(frame, face_cascade):
    """Detects faces in a frame with optimizations."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(60, 60)
    )
    return faces

model = "res10_300x300_ssd_iter_140000.caffemodel"
file_txt = "deploy.prototxt.txt"

def detect_faces_dnn(frame):
    """Detect faces using a DNN model."""
    net = cv2.dnn.readNetFromCaffe(file_txt, model)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))
    
    return faces

def detect_emotion(frame):
    """Detect emotion using DeepFace and calculates confidence metrics."""
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        confidence_score = analysis[0]['emotion'][emotion]
        confidence_scores.append(confidence_score)

        mean_confidence = np.mean(confidence_scores)
        n = len(confidence_scores)
        std_error = np.std(confidence_scores, ddof=1) / np.sqrt(n) if n > 1 else 0
        z = 1.96
        confidence_interval = (mean_confidence - z * std_error, mean_confidence + z * std_error)

        return emotion, confidence_score, confidence_interval
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return None, None, None

def draw_corner_box(frame, x, y, w, h, color=(0, 0, 225), thickness=2, corner_length=20):
    cv2.line(frame, (x, y), (x + corner_length, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + corner_length), color, thickness)
    cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, thickness)
    cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, thickness)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error(f"Error: Could not open video file {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = 3  # Process every 3rd frame for faster performance
    frame_count = 0
    frame_display = st.empty()  # Placeholder for displaying frames in Streamlit

    while True:
        ret, frame = cap.read()
        if not ret:
            st.info("End of video file.")
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        faces = detect_faces_dnn(frame)
        if len(faces) > 0:
            emotion, confidence_score, confidence_interval = detect_emotion(frame)

            if emotion and confidence_score > 85.0:
                for (x, y, w, h) in faces:
                    draw_corner_box(frame, x, y, w, h)
                cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the frame (BGR to RGB) for Streamlit display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display.image(rgb_frame, channels="RGB")

        frame_count += 1
    
    cap.release()

def main():
    st.title("Emotion Detection from Video")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())

        if st.button('Process Video'):
            process_video(tfile.name)

if __name__ == "__main__":
    main()

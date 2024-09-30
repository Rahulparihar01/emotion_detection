import cv2
from deepface import DeepFace
import numpy as np
import streamlit as st 
import  tempfile
from scipy import stats

def detect_faces(frame, face_cascade):
    """Detects faces in a frame with optimizations."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with better parameters
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # Try increasing or decreasing this value to fine-tune detection
        minNeighbors=6,    # Higher value reduces false positives
        minSize=(60, 60)   # Skip faces smaller than 60x60 pixels
    )
    
    return faces
model="res10_300x300_ssd_iter_140000.caffemodel"
file_txt="deploy.prototxt.txt"
def detect_faces_dnn(frame):
    """Detect faces using a DNN model."""
    net = cv2.dnn.readNetFromCaffe(file_txt,model)
    
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


# Initialize a list to store confidence scores
confidence_scores = []

def detect_emotion(frame):
    """Detects emotion using DeepFace and calculates confidence metrics."""
    try:
        # Analyze the frame to detect emotion
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        
        # Get the confidence score for the detected emotion
        confidence_score = analysis[0]['emotion'][emotion]  # Assuming this returns a probability

        # Append the confidence score to the list
        confidence_scores.append(confidence_score)

        # Calculate mean of the confidence scores
        mean_confidence = np.mean(confidence_scores)
        
        # Calculate standard error for the confidence scores
        n = len(confidence_scores)
        std_error = np.std(confidence_scores, ddof=1) / np.sqrt(n) if n > 1 else 0

        # Define z-score for 95% confidence level
        z = 1.96

        # Calculate the confidence interval
        confidence_interval = (mean_confidence - z * std_error, mean_confidence + z * std_error)
        
        print(f"Emotion: {emotion}, Confidence Score: {confidence_score}")
        print(f"Mean Confidence Score: {mean_confidence}")
        print(f"Confidence Interval: {confidence_interval}")

        # Check if the margin of error is below a threshold (e.g., 0.05)
        margin_of_error = z * std_error
        if margin_of_error < 0.05:
            print("Prediction is reliable (margin of error < 0.05).")
        else:
            print("Prediction may be unreliable (margin of error >= 0.05).")
        
        return emotion, confidence_score, confidence_interval

    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return None, None, None


def draw_corner_box(frame, x, y, w, h, color=(0, 0, 225), thickness=2, corner_length=20):
    # Top-left corner
    cv2.line(frame, (x, y), (x + corner_length, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + corner_length), color, thickness)
    
    # Top-right corner
    cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, thickness)
    
    # Bottom-left corner
    cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, thickness)
    
    # Bottom-right corner
    cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, thickness)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))    
    frame_skip = 3  # Process every 3rd frame for faster performance
    frame_count = 0
    
    # Create a named window with the ability to resize
    cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video file.")
            break
        
        # Skip frames for performance
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        
        # Detect faces without resizing the frame
        faces = detect_faces_dnn(frame)
        
        if len(faces) > 0:
            emotion, confidence_score, confidence_interval = detect_emotion(frame)
            # emotion, confidence_score, confidence_interval,requirements_met = detect_emotion(frame)
    
    
    # Mark faces and display emotion if detected
            if emotion and confidence_score > 85.0:
                for (x, y, w, h) in faces:
                    draw_corner_box(frame, x, y, w, h, color=(0, 0, 225), thickness=2, corner_length=20)

                cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(frame, f"Confidence Score: {confidence_score:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(frame, f"requirement_met: {requirements_met}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the confidence interval
                # if confidence_interval is not None:
                #     cv2.putText(frame, f"Confidence Interval: [{confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}]", (10, 90),
                #         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        
        # Display the original frame size
        cv2.imshow('Emotion Detection', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()



def main():
    st.title("Emotion Detection from Video")
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        # st.write(tfile.name)
        
        if st.button('Process Video'):
            process_video(tfile.name)

if __name__ == "__main__":
    main()
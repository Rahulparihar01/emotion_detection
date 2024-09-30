import cv2
from deepface import DeepFace
import numpy as np
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


def detect_emotion(frame):
    """Detects emotion using DeepFace."""
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        return emotion
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
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
            emotion = detect_emotion(frame)
            
            # Mark faces and display emotion if detected
            if emotion:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the original frame size
        cv2.imshow('Emotion Detection', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    video_path = "w.mp4"
    process_video(video_path)

if __name__ == "__main__":
    main()

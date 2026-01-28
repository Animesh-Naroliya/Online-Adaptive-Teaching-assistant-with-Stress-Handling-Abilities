import os
import base64
import warnings
import cv2
import numpy as np
from tf_keras.models import load_model 
from tf_keras.preprocessing.image import img_to_array

warnings.filterwarnings("ignore")

# Define file paths relative to this module's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Gets the path to the VideoAnalyzer.py directory
CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
MODEL_PATH = os.path.join(BASE_DIR, 'video-model.h5') 

# Define emotion labels (Ensure this order matches your model's output)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'] 

# --- GLOBAL MODEL INITIALIZATION ---
FACE_CLASSIFIER = None
VIDEO_CLASSIFIER = None

try:
    # Load the pre-trained model and cascade classifier using relative paths
    VIDEO_CLASSIFIER = load_model(MODEL_PATH)
    FACE_CLASSIFIER = cv2.CascadeClassifier(CASCADE_PATH)
    print("Video Emotion model loaded successfully.")
except Exception as e:
    print(f"Error loading video models from {MODEL_PATH} or {CASCADE_PATH}: {e}. Facial ER will be disabled.")
    FACE_CLASSIFIER = None
    VIDEO_CLASSIFIER = None

def analyze_video_frame(base64_frame: str) -> str:
    """
    Analyzes a single Base64-encoded frame to detect the dominant emotion.
    """
    if not FACE_CLASSIFIER or not VIDEO_CLASSIFIER:
        return 'Model Error'

    try:
        # 1. Decode Base64 string into NumPy array (image)
        base64_decoded = base64_frame.split(',')[1]
        img_bytes = base64.b64decode(base64_decoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return 'Neutral'

        # 2. Preprocess
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CLASSIFIER.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return 'Neutral' 
        
        # Process the largest face found
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        roi_gray = gray[y:y+h, x:x+w]
        
        if roi_gray.size == 0:
            return 'Neutral'
            
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # 3. Normalize and Predict
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            
            # Expand dimensions to match model input shape (1, 48, 48, 1)
            # You may need to verify these axis dimensions against your model's summary.
            roi = np.expand_dims(roi, axis=-1) 
            roi = np.expand_dims(roi, axis=0)

            prediction = VIDEO_CLASSIFIER.predict(roi, verbose=0)[0]
            
            # Determine Dominant Emotion
            label_index = prediction.argmax()
            
            if label_index < len(EMOTION_LABELS):
                 return EMOTION_LABELS[label_index].capitalize()
            else:
                 return 'Prediction Error'

        return 'Neutral'

    except Exception as e:
        print(f"Video analysis exception: {e}")
        return 'Analysis Error'
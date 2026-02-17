import os
import base64
import warnings
import cv2
import numpy as np
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import img_to_array
import time 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque, Counter

SESSION_EMOTION_BUFFER = []
SESSION_STRESS_BUFFER = []

warnings.filterwarnings("ignore")

# Define file paths relative to this module's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Gets the path to the VideoAnalyzer.py directory
CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
MODEL_PATH = os.path.join(BASE_DIR, 'video-model.h5') 
LANDMARKER_PATH = os.path.join(BASE_DIR, 'face_landmarker.task')

# Define emotion labels (Ensure this order matches your model's output)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'] 

# --- GLOBAL MODEL INITIALIZATION ---
FACE_CLASSIFIER = None
VIDEO_CLASSIFIER = None
EMOTION_BUFFER = deque(maxlen=60)   # ~4 seconds at 15 FPS
STRESS_BUFFER = deque(maxlen=60)

try:
    # Load the pre-trained model and cascade classifier using relative paths
    VIDEO_CLASSIFIER = load_model(MODEL_PATH)
    FACE_CLASSIFIER = cv2.CascadeClassifier(CASCADE_PATH)
    print("Video Emotion model loaded successfully.")
except Exception as e:
    print(f"Error loading video models from {MODEL_PATH} or {CASCADE_PATH}: {e}. Facial ER will be disabled.")
    FACE_CLASSIFIER = None
    VIDEO_CLASSIFIER = None

def landmarker_callback(result, output_image, timestamp_ms):
    global latest_landmark_result
    latest_landmark_result = result


latest_landmark_result = None

try:
    base_options = python.BaseOptions(
        model_asset_path=LANDMARKER_PATH
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=landmarker_callback,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )

    FACE_LANDMARKER = vision.FaceLandmarker.create_from_options(options)

    print("Face Landmarker loaded successfully.")

except Exception as e:
    print(f"Error loading Face Landmarker: {e}")
    FACE_LANDMARKER = None


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
    
def analyze_video_frame_with_stress(base64_frame: str):
    """
    Returns (emotion, stress_placeholder)
    """
    emotion = analyze_video_frame(base64_frame)

    if not FACE_LANDMARKER:
        return emotion, "Calm"

    try:
        # Decode frame again (we need full RGB image)
        base64_decoded = base64_frame.split(',')[1]
        img_bytes = base64.b64decode(base64_decoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return emotion, "Calm"

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp_ms = int(time.time() * 1000)

        FACE_LANDMARKER.detect_async(mp_image, timestamp_ms)

        results = latest_landmark_result


        if not results.face_landmarks:
            return emotion, "Calm"

        blendshape_dict = {
            bs.category_name: bs.score
            for bs in results.face_blendshapes[0]
        }

        # ---- Stress Feature Extraction ----

        # Brow tension
        brow_score = (
            blendshape_dict.get("browDownLeft", 0) +
            blendshape_dict.get("browDownRight", 0)
        ) - blendshape_dict.get("browInnerUp", 0)

        # Lip tension
        lip_score = (
            blendshape_dict.get("mouthPressLeft", 0) +
            blendshape_dict.get("mouthPressRight", 0)
        ) / 2

        # Blink intensity
        blink_score = (
            blendshape_dict.get("eyeBlinkLeft", 0) +
            blendshape_dict.get("eyeBlinkRight", 0)
        ) / 2

        # Combine into stress index
        stress_index = (
            0.4 * brow_score +
            0.3 * lip_score +
            0.3 * blink_score
        )

        # Clamp between 0 and 1
        stress_index = max(0, min(1, stress_index))

        # ---- Classification ----
        if stress_index < 0.35:
            stress_level = "Calm"
        elif stress_index < 0.65:
            stress_level = "Light Stress"
        else:
            stress_level = "High Stress"

        return emotion, stress_level


    except Exception as e:
        print(f"Stress analysis error: {e}")
        return emotion, "Calm"

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to base64 (since your function expects base64)
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        base64_frame = "data:image/jpeg;base64," + jpg_as_text

        emotion, stress = analyze_video_frame_with_stress(base64_frame)

        SESSION_EMOTION_BUFFER.append(emotion)
        SESSION_STRESS_BUFFER.append(stress)

        # Display results
        cv2.putText(frame, f"Emotion: {emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.putText(frame, f"Stress: {stress}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("Emotion + Stress Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if SESSION_EMOTION_BUFFER:
        final_emotion = Counter(SESSION_EMOTION_BUFFER).most_common(1)[0][0]
        final_stress = Counter(SESSION_STRESS_BUFFER).most_common(1)[0][0]

        print("Dominant Emotion:", final_emotion)
        print("Dominant Stress Level:", final_stress)


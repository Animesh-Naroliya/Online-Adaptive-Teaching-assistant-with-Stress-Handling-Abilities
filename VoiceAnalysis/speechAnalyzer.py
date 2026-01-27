import os
import io
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from transformers import pipeline

try:
    if os.name == 'nt': # Check if system is Windows
        temp_dir = os.environ.get('TEMP') or os.environ.get('TMP')
        if temp_dir and os.path.exists(temp_dir):
            os.environ['TMPDIR'] = temp_dir
except Exception as e:
    print(f"Could not set TMPDIR: {e}")
# --- CONFIGURATION ---
RATE = 16000
SEGMENT_DURATION = 1.0  # seconds

# --- GLOBAL MODEL INITIALIZATION ---
SER_PIPELINE = None
STT_PIPELINE = None
print("Loading Speech Models...")

try:
    SER_PIPELINE = pipeline(
        "audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )
    print("Emotion model loaded successfully.")
    STT_PIPELINE = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en" 
    )
    print("Speech-to-text Model loaded successfully.")
except Exception as e:
    print(f"Error loading speech models: {e}")
    SER_PIPELINE = None
    STT_PIPELINE = None

# --- CORE ANALYSIS FUNCTION ---
def analyze_audio_blob(audio_blob):
    """
    Accepts raw audio data (blob) and returns the dominant emotion and a placeholder transcription.
    Requires FFmpeg to be installed on the system PATH.
    """
    global SER_PIPELINE, STT_PIPELINE, RATE, SEGMENT_DURATION
    
    if not SER_PIPELINE or not STT_PIPELINE:
        return {'transcription': 'Speech models failed to load. Check dependencies or Try Again.', 'emotion': 'Neutral'}

    if not audio_blob:
        return {'transcription': '', 'emotion': 'Neutral'}

    try:
        if audio_blob:

            # 1. CRITICAL DECODING STEP: Convert compressed WebM blob using pydub/ffmpeg
            audio_file = io.BytesIO(audio_blob)
            audio_segment = AudioSegment.from_file(audio_file, format="webm")
            
            if len(audio_segment) < 100: 
                return {'transcription': 'Recording too short or silent. Speak clearly.', 'emotion': 'Neutral'} 
        
            # --- SPEECH-TO-TEXT (STT) ---
            # Prepare audio segment for the STT pipeline
            stt_buffer = io.BytesIO()
            audio_segment.export(stt_buffer, format="wav") # Export segment to WAV buffer
            stt_buffer.seek(0)
                
            # Perform STT on the whole segment
            stt_result = STT_PIPELINE(stt_buffer.read())
            transcribed_text = stt_result['text'].strip() if stt_result and 'text' in stt_result else 'Could not transcribe.'
            
            # --- EMOTION RECOGNITION (SER) ---
            # Convert pydub segment to numpy array and normalize
            y = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
            y = y.astype(np.float32) / (2**15) # Normalize
            sr = audio_segment.frame_rate

            # Resample if necessary
            if sr != RATE:
                y = librosa.resample(y, orig_sr=sr, target_sr=RATE)
                sr = RATE

            total_samples = len(y)
            segment_samples = int(SEGMENT_DURATION * sr)

            emotions_count = {"angry": 0, "happy": 0, "neutral": 0, "sad": 0}
            num_segments = 0

            # 3. Segment and Predict
            for start in range(0, total_samples, segment_samples):
                end = min(start + segment_samples, total_samples)
                segment = y[start:end]

                if len(segment) == 0:
                    continue 

                # Convert float32 NumPy segment back to int16 for WAV buffer
                segment_int16 = (segment * (2**15)).astype(np.int16) 

                buffer = io.BytesIO()
                sf.write(buffer, segment_int16, sr, format='wav')
                buffer.seek(0)
                
                # Predict
                result = SER_PIPELINE(buffer.read())

                raw_label = result[0]['label'].lower()

                # Mapping user's sentiment groups
                if "angry" in raw_label or "disgust" in raw_label:
                    emotions_count["angry"] += 1
                elif "happy" in raw_label or "surprise" in raw_label:
                    emotions_count["happy"] += 1
                elif "sad" in raw_label or "fear" in raw_label:
                    emotions_count["sad"] += 1
                else:
                    emotions_count["neutral"] += 1

                num_segments += 1
            
            # 4. Determine Dominant Emotion and Transcription Placeholder
            dominant_emotion = max(emotions_count, key=emotions_count.get) if num_segments > 0 else 'Neutral'
            return {
                'transcription': transcribed_text,
                'emotion': dominant_emotion.capitalize()
            }
        else:
            return {'transcription': 'No audio data received.', 'emotion': 'Neutral'}
    
    except FileNotFoundError:
        return {'transcription': 'Error: FFmpeg not found on PATH. Audio decoding failed.', 'emotion': 'Alert'}
    except Exception as e:
        print(f"Audio analysis failed during processing: {e}")
        return {'transcription': f'Audio input error: {e}.', 'emotion': 'Neutral'}
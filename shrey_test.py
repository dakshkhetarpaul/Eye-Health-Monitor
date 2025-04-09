import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import time
from datetime import datetime
from firebase_setup import db_ref
import platform
import threading
import sounddevice as sd  # Cross-platform audio playback
import warnings
from collections import deque

# Suppress ALSA warnings on Linux
warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')

def play_sound(alert_type):
    def _generate_beep(freq=440, duration=0.5, volume=0.5):
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = volume * np.sin(2 * np.pi * freq * t)
        return wave

    def _play_beep(freq, duration):
        try:
            sound = _generate_beep(freq, duration)
            sd.play(sound, samplerate=44100, blocking=False)
            print(f"SOUND PLAYED: {alert_type} alert ({freq}Hz, {duration}s)")
        except Exception as e:
            print(f"Sound failed: {e}")

    sounds = {
        "bad": (440, 0.8),
        "good": (880, 0.3),
        "blink": (660, 0.5)
    }

    if alert_type in sounds:
        freq, duration = sounds[alert_type]
        threading.Thread(target=_play_beep, args=(freq, duration), daemon=True).start()

# Initialize face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants
EAR_THRESHOLD = 0.25
MIN_BLINKS = 3
MONITOR_WINDOW = 10
SPOOF_VARIANCE_THRESHOLD = 0.003  # Adjust as needed

# Blink monitoring variables
blink_count = 0
state = None
blink_timestamps = []
last_alert_time = 0
alert_cooldown = 2
ear_history = deque(maxlen=30)  # For spoof detection

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

cap = cv2.VideoCapture(0)
session_id = "blink_monitoring_" + datetime.now().strftime("%Y%m%d_%H%M%S")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    current_state = None
    current_time = time.time()
    ear = None  # Default if no face detected
    
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[42:48]
        right_eye = shape[36:42]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        # Track EAR values for spoof detection
        ear_history.append(ear)

        # Detect blinks (transition from open to closed)
        if ear < EAR_THRESHOLD and state != 1:
            blink_count += 1
            blink_timestamps.append(current_time)
            current_state = 1
        else:
            current_state = 0 if ear >= EAR_THRESHOLD else 1

        cv2.polylines(frame, [left_eye], True, (0,255,0), 1)
        cv2.polylines(frame, [right_eye], True, (0,255,0), 1)

    # Remove old blinks
    blink_timestamps = [t for t in blink_timestamps if current_time - t <= MONITOR_WINDOW]

    # Blink warning
    if (len(blink_timestamps) < MIN_BLINKS and
        current_time - last_alert_time > alert_cooldown and
        current_time > MONITOR_WINDOW):
        play_sound("bad")
        last_alert_time = current_time
        blink_timestamps = []

    # State alerts
    if current_state is None and current_time - last_alert_time > alert_cooldown:
        play_sound("bad")
        last_alert_time = current_time
    elif current_state == 0 and current_time - last_alert_time > alert_cooldown:
        play_sound("good")
        last_alert_time = current_time

    state = current_state

    # Show spoofing warning if EAR is suspiciously static
    if len(ear_history) == ear_history.maxlen:
        ear_variance = np.var(ear_history)
        if ear_variance < SPOOF_VARIANCE_THRESHOLD:
            cv2.putText(frame, "⚠️ Spoofing suspected!", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(f"[WARNING] Low EAR variance ({ear_variance:.6f}) — possible fake input.")

    # Display info
    active_blinks = len(blink_timestamps)
    window_remaining = max(0, MONITOR_WINDOW - (current_time - (blink_timestamps[0] if blink_timestamps else current_time)))

    cv2.putText(frame, f'Blinks: {blink_count}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f'Recent: {active_blinks}/{MIN_BLINKS}', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (0,0,255) if active_blinks < MIN_BLINKS else (0,255,0), 2)
    cv2.putText(frame, f'Window: {int(window_remaining)}s', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)

    cv2.imshow("Blink Monitor", frame)

    # Firebase upload
    if current_time - last_alert_time > 1 and state is not None:
        try:
            db_ref.child(session_id).push({
                "timestamp": datetime.now().isoformat(),
                "state": int(state) if state is not None else -1,
                "blink_count": active_blinks,
                "window_remaining": window_remaining,
                "alert_status": "active" if active_blinks < MIN_BLINKS else "normal"
            })
        except Exception as e:
            print(f"Firebase error: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

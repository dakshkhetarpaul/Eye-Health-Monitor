
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
import sounddevice as sd
import warnings
import random
from picamera2 import Picamera2

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
        "bad": (440, 2.0),
        "good": (880, 0.3),
        "blink": (660, 0.5),
        "stroke": (620, 1.2),
        "drowsy": (300, 1.0)
    }

    if alert_type in sounds:
        freq, duration = sounds[alert_type]
        threading.Thread(target=_play_beep, args=(freq, duration), daemon=True).start()

# Initialize face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Blink monitoring variables
blink_count = 0
state = None
blink_timestamps = []
last_alert_time = 0
monitor_start_time = time.time()
alert_cooldown = 2  # Seconds between alerts

# Constants
EAR_THRESHOLD = 0.25
MAX_BLINKS = 15  # For stroke risk
MIN_BLINKS = 4   # For drowsiness risk
MONITOR_WINDOW = 20  # Seconds
CONSEC_FRAMES_THRESHOLD = 1  # Minimum frames for a valid blink

# State machine variables
eye_state = "open"
frames_below_thresh = 0

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()
session_id = "sample_data4"

while True:
    eyes_detected = False
    frame_rgb = picam2.capture_array()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    current_state = None
    current_time = time.time()

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[42:48]
        right_eye = shape[36:42]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        if ear < EAR_THRESHOLD:
            frames_below_thresh += 1
            if eye_state == "open" and frames_below_thresh >= CONSEC_FRAMES_THRESHOLD:
                eye_state = "closed"
        else:
            if eye_state == "closed":
                blink_count += 1
                blink_timestamps.append(current_time)
            frames_below_thresh = 0
            eye_state = "open"

        current_state = 1 if ear < EAR_THRESHOLD else 0

        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
        eyes_detected = True

    blink_timestamps = [t for t in blink_timestamps if current_time - t <= MONITOR_WINDOW]

    active_blinks = len(blink_timestamps)
    time_since_session_start = current_time - monitor_start_time
    window_remaining = max(0, MONITOR_WINDOW - time_since_session_start)

    # Immediate stroke alert if max blinks reached
    if active_blinks >= MAX_BLINKS and current_time - last_alert_time > alert_cooldown:
        play_sound("stroke")
        last_alert_time = current_time
        alert_status = "alert_stroke"
        blink_timestamps = []
        monitor_start_time = current_time

    # End of session logic
    elif time_since_session_start >= MONITOR_WINDOW:
        if not eyes_detected:
            alert_status = "no_eyes_detected"
            monitor_start_time = current_time
            blink_timestamps = []
        else:
            if active_blinks < MIN_BLINKS and current_time - last_alert_time > alert_cooldown:
                play_sound("drowsy")
                last_alert_time = current_time
                alert_status = "alert_drowsy"
                blink_timestamps = []
                monitor_start_time = current_time
            else:
                blink_timestamps = []
                monitor_start_time = current_time
                alert_status = "normal"

    state = current_state

    cv2.putText(frame, f'Blinks: {blink_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Recent: {active_blinks} (min {MIN_BLINKS} / max {MAX_BLINKS})', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if active_blinks > MAX_BLINKS or active_blinks < MIN_BLINKS else (0, 255, 0), 2)
    cv2.putText(frame, f'Window: {int(window_remaining)}s', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

    cv2.imshow("Blink Monitor", frame)

    if current_time - last_alert_time > 1 and state is not None:
        try:
            now = datetime.now()
            formatted_time = now.strftime('%H%M%S%f')
            formatted_date = now.strftime('%Y%m%d')

            custom_key = f"S{formatted_time}_{formatted_date}"

            db_ref.child(session_id).update({
                custom_key: {
                    "timestamp": datetime.now().isoformat(),
                    "state": int(state) if state is not None else -1,
                    "blink_count": active_blinks,
                    "window_remaining": window_remaining,
                    "alert_status": alert_status
                }
            })
        except Exception as e:
            print(f"Firebase error: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

import cv2
import dlib
import numpy as np
import os
import collections
import time
from scipy.spatial import distance as dist
from imutils import face_utils
import threading
from firebase_setup import db_ref  # Firebase connection

# Load Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define eye aspect ratio function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    return (A + B) / (2.0 * C)

# Smooth EAR using moving average
EAR_HISTORY = collections.deque(maxlen=5)

def moving_average_ear(new_ear):
    EAR_HISTORY.append(new_ear)
    return np.mean(EAR_HISTORY)

# Apply exponential smoothing to stabilize EAR
SMOOTH_FACTOR = 0.3

def smooth_ear(prev_ear, new_ear):
    return (SMOOTH_FACTOR * new_ear) + ((1 - SMOOTH_FACTOR) * prev_ear)

# Firebase function to update blink count & duration
def update_firebase_blink_thread(user_id, blink_count, blink_duration, ear_value):
    timestamp = int(time.time())  # Get current timestamp

    db_ref.child(user_id).update({
        "blinks": blink_count,
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    })

    db_ref.child(user_id).child("ear_values").update({
        str(timestamp): ear_value
    })

    db_ref.child(user_id).child("blink_durations").update({
        str(timestamp): blink_duration
    })

    print(f"✅ Firebase Updated: User={user_id}, Blinks={blink_count}, Duration={blink_duration:.3f}s, EAR={ear_value}")

# Call Firebase update in a new thread
def update_blink_count(user_id, blink_count, blink_duration, avg_EAR):
    update_thread = threading.Thread(target=update_firebase_blink_thread, args=(user_id, blink_count, blink_duration, avg_EAR))
    update_thread.start()

# Landmark indices for eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Constants
EAR_THRESHOLD = 0.25  # Blink threshold
FRAME_RATE = 30  # Adjust based on your camera FPS
MIN_BLINK_FRAMES = int(0.1 * FRAME_RATE)  # 100ms blink
MAX_BLINK_FRAMES = int(0.4 * FRAME_RATE)  # 400ms max blink

# Variables
blink_count = 0
frame_counter = 0
frame_count = 0
blink_start_time = None
blink_durations = []  # Store blink durations

cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

prev_face = None
user_id = "user_123"  # Change for multiple users

# Initialize avg_EAR before entering the loop
avg_EAR = 0.0  

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1

    # Detect face every DETECT_EVERY frames
    if frame_count % 10 == 0 or prev_face is None:
        faces = detector(gray)
        if len(faces) > 0:
            prev_face = faces[0]
    else:
        faces = [prev_face] if prev_face else []

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]

        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = moving_average_ear((left_EAR + right_EAR) / 2.0)

        # Adjust EAR threshold dynamically
        if frame_count == 1:
            EAR_THRESHOLD = avg_EAR * 0.85

        avg_EAR = smooth_ear(avg_EAR, (left_EAR + right_EAR) / 2.0)

        # Draw eye contours
        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

        # Blink detection & duration measurement
        if avg_EAR < EAR_THRESHOLD:
            if frame_counter == 0:
                blink_start_time = time.time()  # Start blink timer
            frame_counter += 1
        else:
            if frame_counter >= MIN_BLINK_FRAMES:  # Valid blink detected
                blink_duration = time.time() - blink_start_time
                if MIN_BLINK_FRAMES / FRAME_RATE <= blink_duration <= MAX_BLINK_FRAMES / FRAME_RATE:
                    blink_durations.append(blink_duration)
                    blink_count += 1
                    update_blink_count(user_id, blink_count, blink_duration, avg_EAR)  # Firebase update

                    print(f"Blink {blink_count}: {blink_duration:.3f} sec")
            frame_counter = 0  # Reset counter

    # Display blink count
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # EAR Progress Bar Visualization (Ensure avg_EAR is defined)
    if avg_EAR is not None and avg_EAR > 0:
        bar_length = int(avg_EAR * 200)
        cv2.rectangle(frame, (10, 50), (10 + bar_length, 70), (0, 255, 0), -1)

    cv2.imshow("Eye Blink Tracker", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print all recorded blink durations
print("Recorded Blink Durations:", blink_durations)

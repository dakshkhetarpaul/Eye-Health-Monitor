import cv2
import dlib
import numpy as np
import os
import collections
from scipy.spatial import distance as dist
from imutils import face_utils

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

# Sound alert function
def play_alert():
    if os.name == 'nt':  # Windows
        import winsound
        winsound.Beep(1000, 300)  # 1000 Hz for 300ms
    else:  # macOS/Linux
        os.system("afplay /System/Library/Sounds/Glass.aiff")  # macOS default sound

# Landmark indices for eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Constants
EAR_THRESHOLD = 0.25  # Default blink threshold
CONSEC_FRAMES = 2  # Minimum consecutive frames for blink
DETECT_EVERY = 10  # Detect face every 10 frames for better performance
blink_count = 0
frame_counter = 0
frame_count = 0

# Start webcam feed
cap = cv2.VideoCapture(0)
prev_face = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1

    # Detect face only every DETECT_EVERY frames
    if frame_count % DETECT_EVERY == 0 or prev_face is None:
        faces = detector(gray)
        if len(faces) > 0:
            prev_face = faces[0]  # Store last detected face
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

        # Dynamic EAR threshold based on initial EAR calibration
        if frame_count == 1:
            EAR_THRESHOLD = avg_EAR * 0.85

        avg_EAR = smooth_ear(avg_EAR, (left_EAR + right_EAR) / 2.0)

        # Draw eye contours
        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

        # Blink detection
        if avg_EAR < EAR_THRESHOLD:
            frame_counter += 1
        else:
            if frame_counter >= CONSEC_FRAMES:
                blink_count += 1
            frame_counter = 0  # Reset counter

    # Display blink count
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # EAR Progress Bar Visualization
    bar_length = int(avg_EAR * 200)  # Scale EAR value for better visibility
    cv2.rectangle(frame, (10, 50), (10 + bar_length, 70), (0, 255, 0), -1)

    # Alert if blink count is too low
    # if blink_count < 3 and frame_count > 300:  # Check after ~10 seconds
    #    play_alert()

    cv2.imshow("Eye Blink Tracker", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import time  # Import time to work with timestamps
import os
import threading 


# Load Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define eye aspect ratio function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    return (A + B) / (2.0 * C)

def play_alert():
    threading.Thread(target=lambda: os.system("afplay /System/Library/Sounds/Glass.aiff")).start()

# Landmark indices for eyes (left, right)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Constants
EAR_THRESHOLD = 0.25  # Blink threshold
CONSEC_FRAMES = 2  # Minimum consecutive frames to count as blink
blink_count = 0
frame_counter = 0
CONSECUTIVE_TIME_THRESHOLD = 5 #sound

# Initialize array to hold eye states (1 for blink, 0 for eyes open)
eye_states = []

# Start webcam feed
cap = cv2.VideoCapture(0)

# Track the time for each frame
start_time = time.time()
last_blink_time = time.time()  # Initialize last blink time sound


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]

        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # Draw eyes on frame
        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

        # Blink detection
        if avg_EAR < EAR_THRESHOLD:
            frame_counter += 1

            if time.time() - last_blink_time >= CONSECUTIVE_TIME_THRESHOLD:#sound
                play_alert()  # Play sound after 5 seconds
                last_blink_time = time.time()  # Reset timer after alert sound
        else:
            if frame_counter >= CONSEC_FRAMES:
                blink_count += 1
            frame_counter = 0

        # Timestamp every 1/10th of a second (100 ms)
        elapsed_time = time.time() - start_time
        if int(elapsed_time * 10) % 1 == 0:  # Check every 1/10th of a second
            state = 1 if avg_EAR < EAR_THRESHOLD else 0
            eye_states.append(state)  # Append 1 or 0 to the array

    # Display blink count
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Eye Blink Tracker", frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# After the loop, print the eye states array
print("Eye States (1 for blink, 0 for eyes open):")
print(eye_states)

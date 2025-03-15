import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import pandas as pd
import winsound  # Windows sound library (for sound alerts)
from datetime import datetime

# Load the facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"d:\Capstone_Stuff\myenv\Scripts\shape_predictor_68_face_landmarks.dat") #CHANGE FILE PATH BASED ON YOUR SYSTEM

# Initialize variables
blink_count = 0
state = None  # Start with no state
last_state_change_time = time.time()

# Define constants
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold for blink detection
STATE_HOLD_TIME = 5  # 5 seconds of same state

# Function to compute Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)  # Try to improve frame rate

# Initialize an empty list to store the data for the spreadsheet
data = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    current_state = None
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[42:48]
        right_eye = shape[36:42]
        
        left_eye_ear = eye_aspect_ratio(left_eye)
        right_eye_ear = eye_aspect_ratio(right_eye)
        ear = (left_eye_ear + right_eye_ear) / 2.0

        if ear < EAR_THRESHOLD:
            blink_count += 1
            current_state = 1  # Blink state
        else:
            current_state = 0  # No blink
        
        # Draw eye contours
        cv2.polylines(frame, [left_eye], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(frame, [right_eye], isClosed=True, color=(0, 255, 0), thickness=2)

    # Check if state has remained the same for 5+ seconds
    if current_state is not None and current_state == state:
        if time.time() - last_state_change_time >= STATE_HOLD_TIME:
            winsound.Beep(1000, 500)  # Play sound
            last_state_change_time = time.time()  # Reset timer
    else:
        last_state_change_time = time.time()
    
    state = current_state
    
    # Display information
    cv2.putText(frame, f'Blinks: {blink_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'State: {state}', (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the collected data to an Excel file
file_path = r'd:\Capstone_Stuff\myenv\Scripts\blink_data.xlsx'
df = pd.DataFrame(data, columns=['Timestamp', 'Blink Count', 'State'])
df.to_excel(file_path, index=False)

# Cleanup
cap.release()
cv2.destroyAllWindows()


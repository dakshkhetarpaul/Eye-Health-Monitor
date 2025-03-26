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
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize variables
blink_count = 0
state = None  # Start with no state
consecutive_time = 0  # Track consecutive time in the same state
last_state_time = time.time()  # Time of the last state change

# Define constants
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold for blink detection
CONSECUTIVE_TIME_THRESHOLD = 5  # 5 seconds of same state

# Function to compute Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    # Calculate the Euclidean distances between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Calculate the Euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # Return the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize an empty list to store the data for the spreadsheet
data = []

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    for face in faces:
        # Get facial landmarks
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates
        left_eye = shape[42:48]
        right_eye = shape[36:42]

        # Calculate the eye aspect ratio for both eyes
        left_eye_ear = eye_aspect_ratio(left_eye)
        right_eye_ear = eye_aspect_ratio(right_eye)

        # Calculate the average EAR
        ear = (left_eye_ear + right_eye_ear) / 2.0

        # Detect blink (if EAR is below threshold, consider it a blink)
        if ear < EAR_THRESHOLD:
            blink_count += 1
            current_state = 1  # State 1: Blink

            # Play sound if in blink state for 5 seconds
            if state == 1 and time.time() - last_state_time >= CONSECUTIVE_TIME_THRESHOLD:
                winsound.Beep(1000, 500)  # Play sound

        else:
            current_state = 0  # State 0: No Blink

        # Track consecutive states
        if current_state == state:
            consecutive_time += 1
        else:
            consecutive_time = 0

        if consecutive_time >= CONSECUTIVE_TIME_THRESHOLD:
            winsound.Beep(1000, 500)  # Play sound

        # Update state and last state time
        state = current_state
        last_state_time = time.time()

        # Display the blink count in the top left corner
        cv2.putText(frame, f'Blinks: {blink_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the current state (0 or 1) in the top right corner
        cv2.putText(frame, f'State: {state}', (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw green highlight around the eyes
        cv2.polylines(frame, [left_eye], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(frame, [right_eye], isClosed=True, color=(0, 255, 0), thickness=2)

        # Save the data (timestamp, blink count, state)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data.append([timestamp, blink_count, state])

    # Calculate blink rate
    elapsed_time = (time.time() - last_state_time) / 60  # Convert to minutes
    if elapsed_time > 0:
        blink_rate = blink_count / elapsed_time
    else:
        blink_rate = 0

    # Check for low blink rate and play a warning sound
    if blink_rate < 20:
        winsound.Beep(1500, 500)  # Higher pitch warning beep
    
    # Show the resulting frame
    cv2.imshow("Frame", frame)

    # Break the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the collected data to an Excel file
file_path = r'C:\Capstone_Stuff\myenv\Scripts\blink_data.xlsx'
df = pd.DataFrame(data, columns=['Timestamp', 'Blink Count', 'State'])
df.to_excel(file_path, index=False)

# Cleanup
cap.release()
cv2.destroyAllWindows()

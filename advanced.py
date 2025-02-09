import cv2
import dlib
import numpy as np
import os
import collections
import time
from scipy.spatial import distance as dist
from imutils import face_utils
import threading
from firebase_setup import db_ref  # Import the Firebase reference

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

# Firebase function to update blink count (in a separate thread)
def update_firebase_blink_thread(user_id, blink_count, ear_value):
    timestamp = int(time.time())  # Get current timestamp

    db_ref.child(user_id).update({
        "blinks": blink_count,
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    })

    db_ref.child(user_id).child("ear_values").update({
        str(timestamp): ear_value
    })

    print(f"✅ Updated Firebase: User={user_id}, Blinks={blink_count}, EAR={ear_value}")


# In your main loop, call the Firebase update in a new thread
def update_blink_count(user_id, blink_count, avg_EAR):
    # Create a thread to handle the Firebase update
    update_thread = threading.Thread(target=update_firebase_blink_thread, args=(user_id, blink_count, avg_EAR))
    update_thread.start()

# Firebase function to get the latest blink count
def get_latest_blink_count(user_id):
    data = db_ref.child(user_id).get()
    if data and "blinks" in data:
        return data["blinks"]
    return 0

# Firebase real-time listener (optional)
def stream_handler(event):
    print(f"🔥 Realtime Update: {event.data}")

# Uncomment this line if you want real-time updates:
# db_ref.listen(stream_handler)

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

# Initialize avg_EAR to a default value (for the first frame)
avg_EAR = 0.0  # You can adjust this default value if needed

# Start webcam feed
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

prev_face = None
user_id = "user_123"  # Change this to match different users

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
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
                update_blink_count(user_id, blink_count, avg_EAR)  # Use threaded update
            frame_counter = 0  # Reset counter

    # Display blink count
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # EAR Progress Bar Visualization
    if avg_EAR > 0:  # Ensure avg_EAR has been calculated
        bar_length = int(avg_EAR * 200)  # Scale EAR value for better visibility
        cv2.rectangle(frame, (10, 50), (10 + bar_length, 70), (0, 255, 0), -1)

    cv2.imshow("Eye Blink Tracker", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

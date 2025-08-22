# Capstone SPR 30 - Eye-Health-Monitor 

Video Demo: https://drive.google.com/file/d/1chfHauSKd59mspsd9X3wivSsWxjJlGgZ/view?usp=sharing

This project presents a conceptual design for a real-time drowsiness and stroke risk detection system based on eye-blink monitoring through computer vision and data analysis. The primary objective is to enhance safety by identifying early signs of eye fatigue, which, if left unaddressed, could lead to conditions such as dry eyes. The system utilizes a standard webcam to continuously capture facial data, employing facial landmark detection to compute the Eye Aspect Ratio (EAR)—a well-established metric for detecting blinks and eye closures.

The conceptual design outlines the interaction between hardware components, image processing pipelines, decision logic, and audio feedback mechanisms. The software component, developed using Python libraries such as OpenCV, dlib, and NumPy, processes each video frame to detect subtle eye movements in real-time. Blinking patterns are tracked over fixed intervals to determine whether the frequency deviates from predefined thresholds. If an anomaly is detected, the system immediately triggers auditory alerts to notify the user or caregiver. To support remote monitoring, the system integrates with Firebase to log alerts and blink metrics, enabling historical data analysis and continuous tracking.

Key constraints such as system latency, frame rate stability, lighting conditions, and hardware limitations are addressed to ensure robustness and scalability. Additionally, the system is designed with portability in mind, enabling future deployment on embedded platforms like the Raspberry Pi.

This project lays a strong foundation for developing a cost-effective, non-invasive safety device suitable for healthcare monitoring, driver fatigue prevention, and assisted living environments. Future work may involve incorporating machine learning models to more accurately classify blink patterns and adaptively respond to diverse user behaviors.

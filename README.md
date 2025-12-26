Driver Drowsiness & Distraction Detection 🚗💤
This project uses Computer Vision and Deep Learning to detect signs of driver fatigue and distraction in real-time. By monitoring facial landmarks, we calculate the Eye Aspect Ratio (EAR) to determine if a driver’s eyes are closed for an unsafe duration.

🛠️ Stage 0: Environment Setup (Critical)
        During the initial setup, we encountered significant issues with Windows OneDrive syncing and broken MediaPipe installations.
        1. The "Neutral Zone" Directory
        2. Virtual Environment (.venv)
           We use a dedicated virtual environment to manage dependencies.
        3. Dependency Installation

🎭 Stage 1: Face Mesh Foundation
We initialize the MediaPipe Face Mesh solution. This model identifies 478 3D landmarks on the face in real-time.

        1.Image Processing: OpenCV captures frames in BGR format, which we convert to RGB for MediaPipe processing.
        2.Performance: We use refine_landmarks=True to get high-accuracy tracking around the eyes and irises.

👁️ Stage 2: Eye Landmark Isolation
In this stage, we filter the 478 points to focus only on the eyelids using predefined indices.

        *The Predefined Map
        The model assigns a fixed ID to every facial feature:
            Left Eye: A set of 16 points (e.g., 362, 374, 386).
            Right Eye: A set of 16 points (e.g., 33, 145, 159).

        Implementation Logic
            Coordinate Transformation: The AI returns "normalized" coordinates (0.0 to 1.0). We multiply these by the frame's width and height to get exact pixel locations.

            Visual Verification:
            Blue Circles: Applied to Left Eye landmarks.
            Red Circles: Applied to Right Eye landmarks.
                
            convertion -> mesh_points = [(int(p.x * img_w), int(p.y * img_h)) for p in face_landmarks.landmark]
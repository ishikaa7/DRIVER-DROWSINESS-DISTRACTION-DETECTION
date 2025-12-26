import cv2
import mediapipe as mp

# 1. Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmark indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    
    # We define the width and height of the camera frame here
    img_h, img_w, _ = frame.shape 
    # --- FIX END ---

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Convert normalized coordinates (0-1) to pixel coordinates
            mesh_points = [(int(point.x * img_w), int(point.y * img_h)) 
                           for point in face_landmarks.landmark]
            
            # Draw Left Eye (Blue)
            for idx in LEFT_EYE:
                cv2.circle(frame, mesh_points[idx], 1, (255, 0, 0), -1)
            
            # Draw Right Eye (Red)
            for idx in RIGHT_EYE:
                cv2.circle(frame, mesh_points[idx], 1, (0, 0, 255), -1)

    cv2.imshow('Stage 2: Eye Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
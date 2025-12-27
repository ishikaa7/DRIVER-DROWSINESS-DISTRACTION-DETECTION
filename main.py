import cv2
import mediapipe as mp
import math

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_ear(eye_points, mesh_points):
    # Vertical distances: using indices 12/4 and 11/5 from our eye lists
    v1 = get_distance(mesh_points[eye_points[12]], mesh_points[eye_points[4]])
    v2 = get_distance(mesh_points[eye_points[11]], mesh_points[eye_points[5]])
    # Horizontal distance: using indices 0/8
    h = get_distance(mesh_points[eye_points[0]], mesh_points[eye_points[8]])
    return (v1 + v2) / (2.0 * h)

# 1. Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye & Iris landmark indices
L_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
R_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
IRIS = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477]

# Logic Variables
COUNTER = 0
EYE_FPS_THRESHOLD = 30 # Approx 1-1.5 seconds of closure
EAR_THRESHOLD = 0.15   # Set slightly below your 'looking down' value

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    img_h, img_w, _ = frame.shape 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mesh_points = [(int(point.x * img_w), int(point.y * img_h)) 
                           for point in face_landmarks.landmark]
            
            # EAR Calculation
            left_ear = get_ear(L_EYE, mesh_points)
            right_ear = get_ear(R_EYE, mesh_points)
            avg_ear = (left_ear + right_ear) / 2
            
            # Calibration Display
            cv2.putText(frame, f"Live EAR: {avg_ear:.2f}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Alert Logic with Counter
            if avg_ear < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= EYE_FPS_THRESHOLD:
                    cv2.putText(frame, "!!! DROWSY ALERT !!!", (30, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                COUNTER = 0
            
            # Visual Feedback: Draw Eyes (Green) and Irises (Yellow)
            for idx in L_EYE + R_EYE:
                cv2.circle(frame, mesh_points[idx], 1, (0, 255, 0), -1)
            for idx in IRIS:
                cv2.circle(frame, mesh_points[idx], 1, (0, 255, 255), -1)

    cv2.imshow('Drowsiness Monitor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import winsound
import config
import utils

#1. Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye & Iris landmark indices
COUNTER = 0 #count frames
Yawn_counter = 0 #count farmes
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
            left_ear = utils.get_ear(config.L_EYE, mesh_points)
            right_ear = utils.get_ear(config.R_EYE, mesh_points)
            avg_ear = (left_ear + right_ear) / 2
            mar = utils.get_mar(mesh_points)
            
            # Calibration Display
            cv2.putText(frame, f"Live EAR: {avg_ear:.2f}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Alert Logic with Counter
            if avg_ear < config.EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= config.EYE_FPS_THRESHOLD:
                    cv2.putText(frame, "!!! DROWSY ALERT !!!", (30, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    winsound.Beep(200, 500) # 200 -> very low frequency  2500->high freq
            else:
                COUNTER = 0
            
            #Yawn logic
            if mar > config.MAR_THRESHOLD:
                Yawn_counter += 1
                if Yawn_counter >= config.MAR_FPS_THRESHOLD:
                    cv2.putText(frame, "YAWN DETECTED", (30, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    print(f"Mouth Open Frames: {Yawn_counter}")
                    
                    if not config.YAWN_ACTIVE:    #one yawn should not count morethan one time
                        config.YAWN_TOTAL += 1
                        config.YAWN_ACTIVE = True
            else:
                Yawn_counter = 0
                config.YAWN_ACTIVE = False   #resent flag when mouth close

            cv2.putText(frame, f"Total Yawns: {config.YAWN_TOTAL}", (30, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


            if config.YAWN_TOTAL >= 5:
                cv2.putText(frame, "ADVICE: PLEASE TAKE A BREAK", (30, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3) # Orange text

            # --- HEAD POSE / DISTRACTION LOGIC ---
            head_ratio = utils.get_head_direction(mesh_points)
            
            if head_ratio > config.LOOK_LEFT_THRESH or head_ratio < config.LOOK_RIGHT_THRESH:
                config.DISTRACTION_COUNTER += 1
                if config.DISTRACTION_COUNTER >= config.DISTRACTION_FPS_THRESHOLD:
                    cv2.putText(frame, "!!! LOOK AT THE ROAD !!!", (30, 350), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    # Optional: winsound.Beep(800, 200)
            else:
                config.DISTRACTION_COUNTER = 0  

            # # Visual Feedback: Draw Eyes (Green) and Irises (Yellow)
            # for idx in config.L_EYE + config.R_EYE:
            #     cv2.circle(frame, mesh_points[idx], 1, (0, 255, 0), -1)
            # for idx in config.IRIS:
            #     cv2.circle(frame, mesh_points[idx], 1, (0, 255, 255), -1)

    cv2.imshow('Drowsiness Monitor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
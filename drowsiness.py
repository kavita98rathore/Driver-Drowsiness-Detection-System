import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

EYE_THRESHOLD = 0.10
MOUTH_THRESHOLD = 0.6

eye_counter = 0
yawn_counter = 0


def eye_aspect_ratio(eye_points, landmarks, w, h):
    points = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_points]

    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    return (A + B) / (2.0 * C)


def mouth_ratio(mouth_points, landmarks, w, h):
    points = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in mouth_points]

    vertical = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
    horizontal = np.linalg.norm(np.array(points[2]) - np.array(points[3]))

    return vertical / horizontal


def process_frame(frame):
    global eye_counter, yawn_counter

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    h, w, _ = frame.shape

    status = "AWAKE"

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            landmarks = face_landmarks.landmark

            leftEAR = eye_aspect_ratio(LEFT_EYE, landmarks, w, h)
            rightEAR = eye_aspect_ratio(RIGHT_EYE, landmarks, w, h)
            ear = (leftEAR + rightEAR) / 2

            mar = mouth_ratio(MOUTH, landmarks, w, h)

            if ear < EYE_THRESHOLD:
                eye_counter += 1

                if eye_counter > 20:
                    status = "DROWSY"
                    cv2.putText(frame, "DROWSINESS ALERT!", (100, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            elif mar > MOUTH_THRESHOLD:
                yawn_counter += 1

                if yawn_counter > 10:
                    status = "YAWNING"
                    cv2.putText(frame, "YAWNING ALERT!", (100, 420),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            else:
                eye_counter = 0
                yawn_counter = 0

            cv2.putText(frame, f"EAR: {ear:.2f}", (480, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"MAR: {mar:.2f}", (480, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Status: {status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    return frame, status
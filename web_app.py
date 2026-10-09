from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import threading
import time

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13,14,78,308]

EYE_THRESHOLD = 0.20
MOUTH_THRESHOLD = 0.07

eye_counter = 0
last_drowsy = False
last_yawning = False

def eye_aspect_ratio(eye_points, landmarks, w, h):
    points = []
    for p in eye_points:
        x = int(landmarks[p].x * w)
        y = int(landmarks[p].y * h)
        points.append((x,y))
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    return (A + B) / (2.0 * C)

def mouth_ratio(mouth_points, landmarks, w, h):
    points = []
    for p in mouth_points:
        x = int(landmarks[p].x * w)
        y = int(landmarks[p].y * h)
        points.append((x,y))
    vertical = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
    horizontal = np.linalg.norm(np.array(points[2]) - np.array(points[3]))
    return vertical / horizontal

def process_frame(frame):
    global eye_counter, last_drowsy, last_yawning

    frame = cv2.resize(frame, (640,480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    h, w, _ = frame.shape
    drowsy = False
    yawning = False
    ear_value = 0.0
    mar_value = 0.0

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            leftEAR = eye_aspect_ratio(LEFT_EYE, landmarks, w, h)
            rightEAR = eye_aspect_ratio(RIGHT_EYE, landmarks, w, h)
            ear_value = (leftEAR + rightEAR) / 2
            mar_value = mouth_ratio(MOUTH, landmarks, w, h)

            if ear_value < EYE_THRESHOLD:
                eye_counter += 1
                if eye_counter > 20:
                    drowsy = True
                    cv2.putText(frame, "DROWSINESS ALERT!", (150,80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            else:
                eye_counter = 0

            if mar_value > MOUTH_THRESHOLD:
                yawning = True
                cv2.putText(frame, "YAWNING DETECTED", (150,420),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            cv2.putText(frame, f"EAR:{ear_value:.2f}", (500,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"MAR:{mar_value:.2f}", (500,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    last_drowsy = drowsy
    last_yawning = yawning

    # Browser-side JS reads these text markers to trigger the alarm.
    cv2.putText(frame, "DROWSY=1" if drowsy else "DROWSY=0",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    cv2.putText(frame, "YAWN=1" if yawning else "YAWN=0",
                (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return buffer.tobytes()

def generate():
    while True:
        # This server does NOT use cv2.VideoCapture(0).
        # The browser sends webcam frames to /process_frame.
        time.sleep(0.03)
        yield b""

@app.route("/")
def index():
    return render_template("index.html")

@app.post("/process_frame")
def process():
    from flask import request
    data = request.data
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return ("Invalid frame", 400)
    output = process_frame(frame)
    resp = Response(output, mimetype="image/jpeg")
    resp.headers["X-Drowsy"] = "1" if last_drowsy else "0"
    resp.headers["X-Yawn"] = "1" if last_yawning else "0"
    return resp

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

import cv2
import numpy as np
import mediapipe as mp
import os
import pyttsx3
from tensorflow.keras.models import load_model

# ===== NEW IMPORTS FOR FRONTEND (Flask) =====
from flask import Flask, render_template, Response

# ===================== PATHS =====================
MODEL_PATH = "../model/sign_language_model.h5"
DATA_PATH = "../backend/data"

# ===================== SETTINGS =====================
SEQUENCE_LENGTH = 30
THRESHOLD = 0.8

# ===================== LOAD ACTIONS =====================
actions = sorted(os.listdir(DATA_PATH))

# ===================== LOAD MODEL =====================
model = load_model(MODEL_PATH)

# ===================== MEDIAPIPE =====================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===================== CAMERA =====================
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: Camera not accessible")
    exit()

print("Camera opened successfully.")

sequence = []
prediction_text = ""

# ===================== FRAME GENERATOR (FOR WEB) =====================
def generate_frames():
    global sequence, prediction_text

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            landmarks = []

            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            sequence.append(landmarks)
            sequence = sequence[-SEQUENCE_LENGTH:]

            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            if len(sequence) == SEQUENCE_LENGTH:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                if np.max(res) > THRESHOLD:
                    prediction_text = actions[np.argmax(res)]

        # ===================== DISPLAY TEXT =====================
        cv2.putText(
            frame,
            f"Prediction: {prediction_text}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Convert frame for web streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ===================== FLASK APP =====================
app = Flask(__name__, template_folder="../templates")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ===================== RUN SERVER =====================
if __name__ == "__main__":
    app.run(debug=True)

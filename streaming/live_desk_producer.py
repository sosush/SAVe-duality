import cv2
import json
import time
import base64
import sounddevice as sd
import numpy as np
from kafka import KafkaProducer

PRODUCER = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

SAMPLE_RATE = 16000
AUDIO_BUFFER = []

def audio_callback(indata, frames, time_info, status):
    global AUDIO_BUFFER
    volume = np.linalg.norm(indata) * 10
    AUDIO_BUFFER.append(float(volume))
    if len(AUDIO_BUFFER) > 50:
        AUDIO_BUFFER.pop(0)

# Start live microphone capture
audio_stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE)
audio_stream.start()

cap = cv2.VideoCapture(0)
frame_idx = 0
tamper_mode = False

print("\n" + "="*80)
print(" 🎥 KAFKA DUAL-FEED PRODUCER (LIVE SENSOR INGESTION)")
print(" [CONTROLS]: Press 'T' to Toggle Desync Tampering | 'Q' to Quit")
print("="*80 + "\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    realtime_pts = int(time.time() * 1000)
    video_pts = realtime_pts + (3500 if tamper_mode else 0)

    # Encode video frame for Kafka
    small_frame = cv2.resize(frame, (112, 112))
    _, buffer = cv2.imencode('.jpg', small_frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')

    # Emit to Kafka topics
    PRODUCER.send('video-stream', {
        "camera_id": "CUBICLE_CAM_01",
        "pts": video_pts,
        "frame_idx": frame_idx,
        "frame_b64": frame_b64
    })

    mic_val = AUDIO_BUFFER[-1] if AUDIO_BUFFER else 0.05
    PRODUCER.send('audio-stream', {
        "camera_id": "CUBICLE_CAM_01",
        "pts": realtime_pts,
        "frame_idx": frame_idx,
        "audio_chunk": [mic_val] * 10
    })

    cv2.putText(frame, f"TAMPER MODE: {tamper_mode}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if tamper_mode else (0, 255, 0), 2)
    cv2.imshow("Kafka Producer Feed (Press 'T' to Tamper, 'Q' to Quit)", frame)
    key = cv2.waitKey(30) & 0xFF
    if key in [ord('t'), ord('T')]:
        tamper_mode = not tamper_mode
        print(f"[*] Tamper state toggled: {tamper_mode}")
    elif key in [ord('q'), ord('Q')]:
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
audio_stream.stop()
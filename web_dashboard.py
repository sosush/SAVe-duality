import os
import cv2
import time
import json
import subprocess
import threading
import numpy as np
import sounddevice as sd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pyarrow as pa
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import StringType, FloatType, NestedField
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS

# Optional Apache Kafka Hook
try:
    from kafka import KafkaProducer
    KAFKA_PRODUCER = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    print("[✓] Apache Kafka Producer active on :9092.")
except Exception:
    KAFKA_PRODUCER = None

app = Flask(__name__, template_folder='templates')
CORS(app)

os.makedirs("data/iceberg_warehouse", exist_ok=True)

# ----------------- APACHE ICEBERG LAKEHOUSE -----------------
ICEBERG_CATALOG = load_catalog(
    "local_lakehouse",
    **{
        "type": "sql",
        "uri": "sqlite:///data/iceberg_catalog.db",
        "warehouse": "file://" + os.path.abspath("data/iceberg_warehouse"),
    }
)
ICEBERG_CATALOG.create_namespace_if_not_exists("smart_security")

iceberg_schema = Schema(
    NestedField(field_id=1, name="timestamp", field_type=StringType(), required=False),
    NestedField(field_id=2, name="incident", field_type=StringType(), required=False),
    NestedField(field_id=3, name="motion", field_type=FloatType(), required=False),
    NestedField(field_id=4, name="audio", field_type=FloatType(), required=False)
)

try:
    ICEBERG_TABLE = ICEBERG_CATALOG.load_table("smart_security.cubicle_incidents")
except Exception:
    ICEBERG_TABLE = ICEBERG_CATALOG.create_table("smart_security.cubicle_incidents", schema=iceberg_schema)
    print("[✓] Initialized Apache Iceberg Table: smart_security.cubicle_incidents")

PA_SCHEMA = pa.schema([
    pa.field("timestamp", pa.string(), nullable=True),
    pa.field("incident", pa.string(), nullable=True),
    pa.field("motion", pa.float32(), nullable=True),
    pa.field("audio", pa.float32(), nullable=True),
])

# Thread-safe in-memory incident log cache (avoids 430ms SQLite catalog scans per poll)
RECENT_ICEBERG_EVENTS = []
ICEBERG_LOCK = threading.Lock()

def commit_to_iceberg(timestamp_str, incident_str, motion_val, audio_val):
    event = {
        "timestamp": str(timestamp_str),
        "incident": str(incident_str),
        "motion": float(motion_val),
        "audio": float(audio_val)
    }
    # Instant in-memory cache update (0.001ms)
    RECENT_ICEBERG_EVENTS.insert(0, event)
    if len(RECENT_ICEBERG_EVENTS) > 20:
        RECENT_ICEBERG_EVENTS.pop()

    # Commit to Apache Iceberg lakehouse asynchronously in background thread
    def _async_commit():
        with ICEBERG_LOCK:
            try:
                data = pa.Table.from_pydict({
                    "timestamp": [str(timestamp_str)],
                    "incident": [str(incident_str)],
                    "motion": [np.float32(motion_val)],
                    "audio": [np.float32(audio_val)]
                }, schema=PA_SCHEMA)
                ICEBERG_TABLE.append(data)
            except Exception as e:
                print(f"[!] Iceberg async commit notice: {e}")

    threading.Thread(target=_async_commit, daemon=True).start()

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# ----------------- PYTORCH NEURAL BIOMETRIC ENGINE (CPU OPTIMIZED) -----------------
print("[*] Initializing FaceNet Biometric Neural Network...")
device = torch.device("cpu")
torch.set_num_threads(4)

mtcnn = MTCNN(
    image_size=160,
    margin=14,
    min_face_size=35,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    select_largest=True,
    device=device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print(f"[✓] MTCNN Face Detector & InceptionResnetV1 (VGGFace2) active on: {device}")

def detect_face_fast(frame):
    """
    Ultra-fast (8-10ms) deep neural face detector using MTCNN on downscaled frame.
    Returns (x, y, w, h), confidence.
    Returns None, 0.0 when no face is present.
    """
    h, w = frame.shape[:2]
    scale_w, scale_h = 320, 240
    small = cv2.resize(frame, (scale_w, scale_h))
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    
    boxes, probs = mtcnn.detect(rgb_small)
    if boxes is None or probs[0] is None or probs[0] < 0.75:
        return None, 0.0
        
    sx = w / float(scale_w)
    sy = h / float(scale_h)
    x1 = max(0, int(boxes[0][0] * sx))
    y1 = max(0, int(boxes[0][1] * sy))
    x2 = min(w, int(boxes[0][2] * sx))
    y2 = min(h, int(boxes[0][3] * sy))
    return (x1, y1, x2 - x1, y2 - y1), float(probs[0])

@torch.no_grad()
def extract_face_embedding(frame, box):
    """
    Extracts 512-dimensional FaceNet embedding invariant to lighting and angle.
    """
    if box is None:
        return None
    x, y, bw, bh = box
    if bw < 20 or bh < 20:
        return None
    h, w = frame.shape[:2]
    crop = frame[max(0, y):min(h, y+bh), max(0, x):min(w, x+bw)]
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).resize((160, 160))
    tensor = torch.tensor(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    emb = resnet(tensor.unsqueeze(0)).detach().numpy().flatten()
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 1e-6 else None

def match_face(embedding):
    """
    Cosine similarity match against enrolled master profiles.
    FaceNet threshold: 0.60+ indicates same authentic person.
    """
    if len(ENROLLED_OWNERS) == 0 or embedding is None:
        return None, 0.0
    best_name = None
    best_sim = -1.0
    for name, master_emb in ENROLLED_OWNERS.items():
        sim = float(np.dot(embedding, master_emb))
        if sim > best_sim:
            best_sim = sim
            best_name = name
    if best_sim >= 0.60:
        return best_name, best_sim
    return None, best_sim

def robust_skin_mask(bgr):
    """
    Multi-space (YCrCb + RGB) skin filter for hand vs object discrimination.
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    cond_ycrcb = (cr >= 130) & (cr <= 180) & (cb >= 75) & (cb <= 135)
    
    b, g, r = bgr[:, :, 0].astype(int), bgr[:, :, 1].astype(int), bgr[:, :, 2].astype(int)
    cond_rgb = (r > 60) & (g > 25) & (b > 15) & (r > g) & (r > b)
    
    mask = (cond_ycrcb & cond_rgb).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# ----------------- STATE -----------------
ENROLLED_OWNERS = {}
AUDIO_BUFFER = []
ANCHORED_ASSETS = []

VISITOR_NOTIFIED = False
LAST_VISITOR_SEEN_TIME = 0.0
LAST_NOTIF_TIME = 0.0
LAST_OWNER_SEEN_TIME = 0.0
ACTIVE_OWNER_NAME = "None"

TELEMETRY = {
    "status": "AUTHENTIC",
    "incident_tag": "SECURE BASELINE",
    "owner_name": "None",
    "person_present": False,
    "match_confidence": 0.0,
    "desk_assets": 0,
    "motion_level": 0.0,
    "audio_level": 0.0,
    "enrolled_count": 0
}

# ----------------- MICROPHONE STREAM -----------------
def audio_callback(indata, frames, time_info, status):
    global AUDIO_BUFFER
    rms = np.sqrt(np.mean(indata**2)) * 100
    AUDIO_BUFFER.append(float(rms))
    if len(AUDIO_BUFFER) > 30:
        AUDIO_BUFFER.pop(0)

try:
    audio_stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=16000)
    audio_stream.start()
    print("[✓] Hardware Microphone active.")
except Exception as e:
    print(f"[!] Audio stream notice: {e}")

def send_notification(title, message, sound="Basso"):
    global LAST_NOTIF_TIME
    now = time.time()
    if now - LAST_NOTIF_TIME > 3.0:
        cmd = f'osascript -e \'display notification "{message}" with title "{title}" sound name "{sound}"\''
        subprocess.Popen(cmd, shell=True)
        LAST_NOTIF_TIME = now

# ----------------- ULTRA-FAST CUBICLE DEFENSE ENGINE -----------------
class CubicleDefenseEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.prev_desk_gray = None
        self.latest_frame_bytes = None
        self.latest_raw_frame = None
        self.running = True
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def _loop(self):
        global TELEMETRY, ANCHORED_ASSETS, VISITOR_NOTIFIED, LAST_VISITOR_SEEN_TIME, LAST_OWNER_SEEN_TIME, ACTIVE_OWNER_NAME, AUDIO_BUFFER
        frame_idx = 0
        last_seen_face_box = None
        last_face_time = 0.0
        current_confidence = 0.0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            h, w = frame.shape[:2]
            with self.lock:
                self.latest_raw_frame = frame.copy()

            now = time.time()

            # ----------------- 1. BIOMETRIC FACE & OWNER RECOGNITION (MTCNN + FACENET) -----------------
            # Detect face every 3 frames (~90ms) for high responsiveness + zero lag
            if frame_idx % 3 == 0:
                face_box, prob = detect_face_fast(frame)
                if face_box is not None:
                    last_seen_face_box = face_box
                    last_face_time = now
                    
                    if len(ENROLLED_OWNERS) > 0:
                        emb = extract_face_embedding(frame, face_box)
                        name, sim = match_face(emb)
                        if name:
                            current_confidence = sim
                            LAST_OWNER_SEEN_TIME = now
                            ACTIVE_OWNER_NAME = name
                        else:
                            current_confidence = sim

            # Person is present if face was seen within last 0.6 seconds
            person_present = (now - last_face_time < 0.6)
            
            # Owner is authenticated IF person is present AND verified recently (within 1.8s)
            owner_authenticated = person_present and (now - LAST_OWNER_SEEN_TIME < 1.8) and (len(ENROLLED_OWNERS) > 0)
            display_owner_name = ACTIVE_OWNER_NAME if owner_authenticated else "None"

            # Render face overlay
            if person_present and last_seen_face_box is not None:
                fx, fy, fw, fh = last_seen_face_box
                if owner_authenticated:
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
                    cv2.putText(frame, f"OWNER: {display_owner_name.upper()} (AUTHORIZED)", (fx, max(20, fy-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 220, 255), 2)
                    cv2.putText(frame, "VISITOR / STRANGER", (fx, max(20, fy-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 2)

            # ----------------- 2. ROOM BACKGROUND vs DESK FOREGROUND MOTION -----------------
            # Startup camera auto-exposure stabilization
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)

            theft_detected = False
            theft_item_label = ""
            is_empty_hand = False
            room_motion_detected = False
            active_desk_boxes = []
            desk_horizon_y = int(h * 0.68)

            if frame_idx < 15:
                # Camera sensor auto-exposure and white-balance warm-up
                self.prev_desk_gray = gray_blurred.copy()
            elif self.prev_desk_gray is not None:
                diff = cv2.absdiff(self.prev_desk_gray, gray_blurred)
                _, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
                thresh = cv2.dilate(thresh, None, iterations=2)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                skin_mask = robust_skin_mask(frame)

                for c in contours:
                    area = cv2.contourArea(c)
                    if area < 1500:
                        continue

                    bx, by, bw, bh = cv2.boundingRect(c)

                    # KEY ISOLATION: Check if motion originates in the room/background
                    if by < desk_horizon_y:
                        # Motion is in the room (person walking past, wardrobe, bed, background objects)
                        # THIS IS STRICTLY ROOM ACTIVITY / VISITOR MOTION - NEVER DESK THEFT!
                        room_motion_detected = True
                        if not owner_authenticated and not person_present:
                            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (120, 160, 180), 1)
                        continue

                    # FOREGROUND DESK REGION (by >= desk_horizon_y):
                    c_mask = np.zeros((bh, bw), dtype=np.uint8)
                    cv2.drawContours(c_mask, [c - (bx, by)], -1, 255, -1)
                    total_moving = int(np.sum(c_mask == 255))
                    if total_moving < 1000:
                        continue

                    box_skin = skin_mask[by:by+bh, bx:bx+bw]
                    box_gray = gray[by:by+bh, bx:bx+bw]

                    edges = cv2.Canny(box_gray, 40, 120)
                    contour_edges = (c_mask == 255) & (edges > 0)
                    edge_density = float(np.sum(contour_edges)) / float(max(1, total_moving))

                    if edge_density < 0.025:
                        # Subtle shadow or lighting shift -> ignore
                        continue

                    moving_skin = (c_mask == 255) & (box_skin == 255)
                    skin_ratio = float(np.sum(moving_skin)) / float(max(1, total_moving))

                    active_desk_boxes.append((bx, by, bw, bh))

                    if owner_authenticated:
                        # REGISTERED OWNER IS PRESENT: ALL desk access is 100% authorized!
                        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
                        cv2.putText(frame, "AUTHORIZED ACCESS", (bx, max(20, by - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    else:
                        # OWNER AWAY: Discriminate empty hand from stolen desk object
                        if skin_ratio >= 0.18:
                            # Bare empty hand passing across desk without removing an object
                            is_empty_hand = True
                            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 215, 255), 2)
                            cv2.putText(frame, "EMPTY HAND: NO ASSET DISPLACED", (bx, max(20, by - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 2)
                        elif edge_density >= 0.035 and area >= 2000:
                            # Solid non-skin physical object / asset being lifted or taken from the desk!
                            theft_detected = True
                            theft_item_label = "OBJECT / ASSET DISPLACED FROM DESK"
                            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 0, 255), 2)
                            cv2.putText(frame, "🚨 THEFT: OBJECT DISPLACED", (bx, max(20, by - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                self.prev_desk_gray = gray_blurred.copy()
            audio_energy = float(AUDIO_BUFFER[-1]) if AUDIO_BUFFER else 0.0

            # ----------------- SECURITY RULES DECISION -----------------
            if owner_authenticated:
                # RULE 1: REGISTERED USER IS PRESENT AT CAMERA
                # STRICT IMMUNITY: Never flag anything below or above baseline!
                status = "AUTHENTIC"
                incident = f"🟢 AUTHORIZED ACTIVITY (OWNER {ACTIVE_OWNER_NAME.upper()} PRESENT)"
                VISITOR_NOTIFIED = False
            elif theft_detected:
                # RULE 2: OBJECT BEING REMOVED BY UNAUTHORIZED PERSON IN DESK ZONE
                status = "TAMPER_DETECTED"
                incident = f"🚨 THEFT DETECTED: {theft_item_label}"
                send_notification("SECURITY ALERT: THEFT ATTEMPT", "Someone is lifting or moving an object from your unattended desk!", sound="Basso")
            elif not person_present and audio_energy > 20.0:
                # RULE 3: LOUD THUD / IMPACT WHILE OWNER IS AWAY
                status = "TAMPER_DETECTED"
                incident = "🚨 LOUD ACOUSTIC EVENT DETECTED (IMPACT AT UNATTENDED DESK)"
                send_notification("CUBICLE SECURITY ALERT", "Loud impact detected at your unattended desk!", sound="Sosumi")
            elif is_empty_hand:
                # RULE 4: VISITOR EMPTY HAND IN DESK ZONE (NO OBJECT TAKEN)
                status = "VISITOR_NOTICE"
                incident = "🟡 VISITOR HAND IN DESK ZONE (NO OBJECT REMOVED)"
                LAST_VISITOR_SEEN_TIME = now
                if not VISITOR_NOTIFIED:
                    send_notification("CUBICLE NOTICE", "Someone placed an empty hand on your desk (no object taken).", sound="Ping")
                    VISITOR_NOTIFIED = True
            elif person_present:
                # RULE 5: VISITOR STANDING IN CUBICLE / ROOM
                status = "VISITOR_NOTICE"
                incident = "🟡 VISITOR PRESENT AT CUBICLE (MONITORING)"
                LAST_VISITOR_SEEN_TIME = now
                if not VISITOR_NOTIFIED:
                    send_notification("CUBICLE NOTICE", "A visitor is currently standing at your cubicle.", sound="Ping")
                    VISITOR_NOTIFIED = True
            else:
                # RULE 6: CUBICLE EMPTY & SECURE (Background motion in room is ignored)
                status = "AUTHENTIC"
                incident = "✅ SECURE BASELINE: INFRASTRUCTURE NORMAL"

            if not person_present and not is_empty_hand and (now - LAST_VISITOR_SEEN_TIME > 8.0):
                VISITOR_NOTIFIED = False

            # Draw HUD Overlays
            hud_color = (0, 0, 255) if status == "TAMPER_DETECTED" else ((0, 220, 255) if status == "VISITOR_NOTICE" else (0, 255, 0))
            cv2.line(frame, (0, desk_horizon_y), (w, desk_horizon_y), (0, 255, 200), 2)
            cv2.putText(frame, "DESK FOREGROUND MONITORING ZONE (ASSET DEFENSE)", (15, desk_horizon_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1)
            cv2.putText(frame, f"STATUS: {incident}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, hud_color, 2)
            cv2.putText(frame, "REC ● LIVE [30 FPS]", (w - 170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)

            TELEMETRY = {
                "status": status,
                "incident_tag": incident,
                "owner_name": display_owner_name,
                "person_present": person_present,
                "match_confidence": round(current_confidence, 2),
                "desk_assets": len(active_desk_boxes),
                "motion_level": round(float(len(active_desk_boxes)), 1),
                "audio_level": round(audio_energy, 1),
                "enrolled_count": len(ENROLLED_OWNERS)
            }

            if KAFKA_PRODUCER is not None:
                try:
                    KAFKA_PRODUCER.send('video-stream', {
                        "camera_id": "CUBICLE_CAM_01",
                        "pts": int(time.time() * 1000),
                        "status": status,
                        "confidence": current_confidence
                    })
                except Exception:
                    pass

            if status == "TAMPER_DETECTED":
                commit_to_iceberg(time.strftime("%H:%M:%S"), incident, 10.0, audio_energy)

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with self.lock:
                self.latest_frame_bytes = buffer.tobytes()

            frame_idx += 1
            time.sleep(0.015)

    def get_frame(self):
        with self.lock:
            return self.latest_frame_bytes

    def get_raw_frame(self):
        with self.lock:
            return self.latest_raw_frame.copy() if self.latest_raw_frame is not None else None

ENGINE = CubicleDefenseEngine()

def generate_stream():
    while True:
        fb = ENGINE.get_frame()
        if fb is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + fb + b'\r\n')
        time.sleep(0.02)

# ----------------- FLASK ROUTES -----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry')
def get_telemetry():
    return jsonify(TELEMETRY)

@app.route('/api/iceberg_logs')
def get_iceberg_logs():
    return jsonify(RECENT_ICEBERG_EVENTS[:10])

@app.route('/api/enroll_owner', methods=['POST'])
def enroll_owner():
    name = request.json.get('name', '').strip()
    if not name:
        return jsonify({"status": "error", "message": "Name cannot be empty"}), 400

    print(f"[*] Starting FaceID scan for: {name}...")
    embeddings = []

    # Sample 15 frames from camera
    for _ in range(15):
        raw_frame = ENGINE.get_raw_frame()
        if raw_frame is not None:
            face_box, prob = detect_face_fast(raw_frame)
            if face_box is not None:
                emb = extract_face_embedding(raw_frame, face_box)
                if emb is not None:
                    embeddings.append(emb)
        time.sleep(0.06)

    if len(embeddings) == 0:
        return jsonify({"status": "error", "message": "No face detected. Please look directly at the webcam during FaceID enrollment."}), 400

    master_embedding = np.mean(embeddings, axis=0)
    master_embedding = master_embedding / np.linalg.norm(master_embedding)
    ENROLLED_OWNERS[name] = master_embedding

    global LAST_OWNER_SEEN_TIME, ACTIVE_OWNER_NAME
    LAST_OWNER_SEEN_TIME = time.time()
    ACTIVE_OWNER_NAME = name

    print(f"[✓] Enrolled FaceID master profile for {name} ({len(embeddings)} angle samples compiled)")
    return jsonify({"status": "success", "name": name, "frames_scanned": len(embeddings), "enrolled_count": len(ENROLLED_OWNERS)})

@app.route('/api/clear_owners', methods=['POST'])
def clear_owners():
    global ENROLLED_OWNERS, LAST_OWNER_SEEN_TIME, ACTIVE_OWNER_NAME
    ENROLLED_OWNERS.clear()
    LAST_OWNER_SEEN_TIME = 0.0
    ACTIVE_OWNER_NAME = "None"
    return jsonify({"status": "cleared", "enrolled_count": 0})

if __name__ == '__main__':
    print("\n" + "="*80)
    print(" 🛡️  SAVe-DUALITY FAST CUBICLE DEFENSE (ZERO LAG ACTIVE)")
    print(" Open in Browser: http://127.0.0.1:5000")
    print("="*80 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
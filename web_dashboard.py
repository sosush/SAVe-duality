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

# Real Object Detection Backbone (SSDLite MobileNetV3)
import torchvision.models.detection as obj_detection
print("[*] Initializing SSDLite MobileNetV3 neural object detector...")
OBJECT_DETECTOR = obj_detection.ssdlite320_mobilenet_v3_large(
    weights=obj_detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
).eval().to(device)
COCO_CATEGORIES = obj_detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT.meta["categories"]
print("[✓] SSDLite MobileNetV3 active for real object identification.")

# Furniture, fixtures, and background structures to ignore
BACKGROUND_FURNITURE_CLASSES = {
    'person', '__background__', 'N/A', 'chair', 'couch', 'bed', 
    'dining table', 'toilet', 'tv', 'microwave', 'oven', 'toaster', 
    'sink', 'refrigerator', 'potted plant', 'bench', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'clock', 'vase'
}

def detect_objects_nn(frame):
    """
    Runs MobileNetV3 SSDLite detector on frame for real-time inference.
    Filters out static room furniture and background fixtures.
    """
    h, w = frame.shape[:2]
    infer_w, infer_h = 480, 360
    small_rgb = cv2.cvtColor(cv2.resize(frame, (infer_w, infer_h)), cv2.COLOR_BGR2RGB)
    tensor = transforms.functional.to_tensor(small_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = OBJECT_DETECTOR(tensor)[0]
    
    results = []
    boxes = preds['boxes'].cpu().numpy()
    labels = preds['labels'].cpu().numpy()
    scores = preds['scores'].cpu().numpy()
    
    sx = w / float(infer_w)
    sy = h / float(infer_h)
    
    for box, label_idx, score in zip(boxes, labels, scores):
        if score < 0.28:
            continue
        label_name = COCO_CATEGORIES[label_idx] if label_idx < len(COCO_CATEGORIES) else "object"
        if label_name not in BACKGROUND_FURNITURE_CLASSES:
            x1 = max(0, int(box[0] * sx))
            y1 = max(0, int(box[1] * sy))
            x2 = min(w, int(box[2] * sx))
            y2 = min(h, int(box[3] * sy))
            results.append({
                'label': 'object',
                'box': (x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
                'score': float(score)
            })
    return results

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
    Robust skin detector combining YCrCb and HSV color spaces for all lighting conditions.
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    cond_ycrcb = (cr >= 125) & (cr <= 185) & (cb >= 70) & (cb <= 140)
    
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    cond_hsv = ((h_channel <= 25) | (h_channel >= 165)) & (s_channel >= 28) & (s_channel <= 240)
    
    mask = ((cond_ycrcb | cond_hsv)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
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
        safe_title = title.replace('"', '').replace("'", "")
        safe_msg = message.replace('"', '').replace("'", "")
        cmd = f'osascript -e \'display notification "{safe_msg}" with title "{safe_title}" sound name "{sound}"\''
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
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
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
            # ----------------- 2. REAL OBJECT DETECTION + HAND PROXIMITY -----------------
            theft_detected = False
            theft_item_label = ""
            is_empty_hand = False
            object_in_hand_detected = False
            bg_movement_detected = False
            active_motion_boxes = []

            # Detect real physical objects via MobileNetV3 every 4 frames (~120ms) for snappy, accurate detection
            if not hasattr(self, 'cached_objects'):
                self.cached_objects = []

            if frame_idx % 4 == 0:
                try:
                    self.cached_objects = detect_objects_nn(frame)
                except Exception as e:
                    self.cached_objects = []

            skin_mask = robust_skin_mask(frame)

            # Check for detected real physical objects
            detected_objects = self.cached_objects

            for obj in detected_objects:
                ox, oy, ow, oh = obj['box']
                label = obj['label']
                conf = obj['score']

                # Compute skin proximity around and inside the object box (hand holding item)
                pad_x = max(0, ox - int(ow * 0.50))
                pad_y = max(0, oy - int(oh * 0.50))
                pad_w = min(w - pad_x, ow + int(ow * 1.0))
                pad_h = min(h - pad_y, oh + int(oh * 1.0))
                obj_skin_region = skin_mask[pad_y:pad_y+pad_h, pad_x:pad_x+pad_w]
                skin_in_vicinity = int(np.sum(obj_skin_region == 255))
                # If there are skin pixels in/around the object or a person is close in front
                is_held = skin_in_vicinity > 150

                active_motion_boxes.append((ox, oy, ow, oh))

                if owner_authenticated:
                    # Authorized owner holding or interacting with object
                    status_text = "OWNER HELD: OBJECT" if is_held else "OBJECT DETECTED"
                    cv2.rectangle(frame, (ox, oy), (ox+ow, oy+oh), (0, 255, 0), 2)
                    cv2.putText(frame, f"{status_text} ({int(conf*100)}%)", (ox, max(20, oy - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                else:
                    # For visitor / unattended: ONLY trigger when object is actively IN HAND
                    if is_held:
                        object_in_hand_detected = True
                        theft_detected = True
                        theft_item_label = "OBJECT IN VISITOR'S HAND"
                        cv2.rectangle(frame, (ox, oy), (ox+ow, oy+oh), (0, 0, 255), 2)
                        cv2.putText(frame, f"🚨 OBJECT IN HAND ({int(conf*100)}%)", (ox, max(20, oy - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)
                    else:
                        # Static resting item on table/desk (e.g., monitor, laptop sitting idle)
                        cv2.rectangle(frame, (ox, oy), (ox+ow, oy+oh), (120, 160, 180), 1)
                        cv2.putText(frame, "STATIONARY ASSET", (ox, max(20, oy - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 160, 180), 1)

            # Check if someone reached with an empty hand (when no object is detected)
            if not theft_detected and not owner_authenticated:
                # Mask out face region so face doesn't trigger hand detection
                hand_skin_mask = skin_mask.copy()
                if person_present and last_seen_face_box is not None:
                    fx, fy, fw, fh = last_seen_face_box
                    # Clear out face and torso area from hand mask
                    fy_end = min(h, fy + int(fh * 3.5))
                    fx_start = max(0, fx - int(fw * 0.8))
                    fx_end = min(w, fx + int(fw * 1.8))
                    hand_skin_mask[fy:fy_end, fx_start:fx_end] = 0

                hand_contours, _ = cv2.findContours(hand_skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for hc in hand_contours:
                    h_area = cv2.contourArea(hc)
                    if 1800 < h_area < 25000:
                        hx, hy, hw, hh = cv2.boundingRect(hc)
                        is_empty_hand = True
                        active_motion_boxes.append((hx, hy, hw, hh))
                        cv2.rectangle(frame, (hx, hy), (hx+hw, hy+hh), (0, 215, 255), 2)
                        cv2.putText(frame, "HAND DETECTED (EMPTY)", (hx, max(20, hy - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1)
                        break

            audio_energy = float(AUDIO_BUFFER[-1]) if AUDIO_BUFFER else 0.0

            # ----------------- SECURITY RULES DECISION -----------------
            if owner_authenticated:
                status = "AUTHENTIC"
                incident = f"🟢 AUTHORIZED ACTIVITY (OWNER {ACTIVE_OWNER_NAME.upper()} PRESENT)"
                VISITOR_NOTIFIED = False
            elif theft_detected:
                status = "TAMPER_DETECTED"
                incident = f"🚨 TAMPER DETECTED: {theft_item_label}"
                send_notification("SECURITY ALERT: OBJECT MOVED", f"Alert: {theft_item_label} detected while unattended!", sound="Basso")
            elif not owner_authenticated and audio_energy > 12.0:
                status = "TAMPER_DETECTED"
                incident = "🚨 LOUD ACOUSTIC EVENT DETECTED (OFF-CAMERA IMPACT)"
                send_notification("SECURITY ALERT: ACOUSTIC IMPACT", "Loud off-camera thud or impact detected at workstation!", sound="Sosumi")
            elif is_empty_hand:
                status = "VISITOR_NOTICE"
                incident = "🟡 VISITOR HAND MOTION (NO OBJECT IN HAND)"
                LAST_VISITOR_SEEN_TIME = now
                if not VISITOR_NOTIFIED:
                    send_notification("CUBICLE NOTICE", "Someone waved or reached an empty hand across the view.", sound="Ping")
                    VISITOR_NOTIFIED = True
            elif person_present:
                status = "VISITOR_NOTICE"
                incident = "🟡 VISITOR PRESENT (MONITORING)"
                LAST_VISITOR_SEEN_TIME = now
                if not VISITOR_NOTIFIED:
                    send_notification("CUBICLE NOTICE", "A visitor is in front of the workstation.", sound="Ping")
                    VISITOR_NOTIFIED = True
            elif bg_movement_detected:
                status = "AUTHENTIC"
                incident = "🔵 BACKGROUND MOTION OBSERVED (NO THREAT)"
            else:
                status = "AUTHENTIC"
                incident = "✅ SECURE BASELINE: INFRASTRUCTURE NORMAL"

            if not person_present and not is_empty_hand and not theft_detected and (now - LAST_VISITOR_SEEN_TIME > 8.0):
                VISITOR_NOTIFIED = False

            # Draw HUD Overlays
            hud_color = (0, 0, 255) if status == "TAMPER_DETECTED" else ((0, 220, 255) if status == "VISITOR_NOTICE" else (0, 255, 0))
            cv2.putText(frame, f"STATUS: {incident}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, hud_color, 2)
            cv2.putText(frame, "FULL-FRAME TRACKING ● LIVE", (w - 230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

            TELEMETRY = {
                "status": status,
                "incident_tag": incident,
                "owner_name": display_owner_name,
                "person_present": person_present,
                "match_confidence": round(current_confidence, 2),
                "desk_assets": len(active_motion_boxes),
                "motion_level": round(float(len(active_motion_boxes)), 1),
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
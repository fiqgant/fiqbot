import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
import threading
import subprocess
import random
import cv2
import numpy as np
import onnxruntime as ort

from gpiozero import Device, Motor
from gpiozero.pins.lgpio import LGPIOFactory

# =========================================================
# CONFIG
# =========================================================

# ---------- Voice (Output Only) ----------
TTS_ENABLE = True
TTS_LANG = "en-us"

# ---------- Follow (YOLO ONNX) ----------
ONNX_PATH = "yolo11n.onnx"
YOLO_IMG_SIZE = 160

CAM_INDEX = 0
FRAME_W, FRAME_H = 512, 288
CAM_FPS = 60

CONF_TH = 0.35
NMS_TH = 0.45
PERSON_CLASS_ID = 0

# ---------- L298N pins BCM ----------
IN1, IN2, IN3, IN4 = 18, 19, 20, 21
ENA, ENB = 12, 13

MOTOR_B_FORWARD = IN4
MOTOR_B_BACKWARD = IN3

# ---------- Motor tuning ----------
MAX_SPEED = 0.80
MIN_MOVE = 0.25

# ---------- Follow control ----------
KP_TURN = 1.2
KP_FWD = 1.4

TARGET_AREA = 0.22
AREA_DEADBAND = 0.02
X_DEADBAND = 0.06

TURN_LIMIT = 0.75
TURN_MIN = 0.22
INVERT_TURN = False

TARGET_LOST_GRACE = 1.5

# ---------- Performance ----------
INFER_EVERY_N_FRAMES = 1

# ---------- UI ----------
SHOW_UI = True
WINDOW_NAME = "Neo Robot"


# =========================================================
# UTILS
# =========================================================

def clamp(x, lo, hi):
    # Clamp x between lo and hi. Also force small values to 0.0
    val = lo if x < lo else hi if x > hi else x
    if abs(val) < 0.01:
        return 0.0
    return val

def sign(x):
    return 1.0 if x > 0 else -1.0 if x < 0 else 0.0

def apply_min(v: float) -> float:
    if abs(v) < 0.01:
        return 0.0
    if abs(v) < MIN_MOVE:
        return MIN_MOVE if v > 0 else -MIN_MOVE
    return v

def fit_to_window(frame, win_w, win_h):
    h, w = frame.shape[:2]
    win_w = int(win_w)
    win_h = int(win_h)
    if win_w <= 0 or win_h <= 0:
        return frame
    scale = min(win_w / w, win_h / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    x0 = (win_w - nw) // 2
    y0 = (win_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas

# ----------------- TTS -----------------
def say(text: str):
    if not TTS_ENABLE:
        return
    text = (text or "").strip()
    if not text:
        return
    threading.Thread(target=_say_worker, args=(text,), daemon=True).start()

def _say_worker(text: str):
    try:
        subprocess.Popen(
            ["espeak-ng", "-v", TTS_LANG, "-s", "160", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        ).wait()
    except Exception:
        pass

NEO_RESPONSES = {
    "found": [
        "Target found.",
        "I see you.",
        "Hello again.",
        "Target acquired."
    ],
    "lost": [
        "Target lost.",
        "Where did you go?",
        "Searching.",
        "Scanning."
    ],
    "idle": [
        "Patrol mode.",
        "Systems idle.",
        "Waiting."
    ],
    "quit": [
        "Shutting down.",
        "Goodbye.",
        "Offline."
    ]
}

def speak_response(key):
    if key in NEO_RESPONSES:
        say(random.choice(NEO_RESPONSES[key]))

# ----------------- Motors -----------------
Device.pin_factory = LGPIOFactory()
motor_a = Motor(forward=IN1, backward=IN2, enable=ENA, pwm=True)
motor_b = Motor(forward=MOTOR_B_FORWARD, backward=MOTOR_B_BACKWARD, enable=ENB, pwm=True)

def stop_all():
    motor_a.stop()
    motor_b.stop()

def set_motor(motor: Motor, v: float):
    v = clamp(v, -MAX_SPEED, MAX_SPEED)
    if abs(v) < 0.01:
        motor.stop()
        return
    if v >= 0:
        motor.forward(v)
    else:
        motor.backward(-v)

# ----------------- EYES UI -----------------
class FaceUI:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.state = "neutral"  # neutral, happy, sad
        self.blink_timer = 0.0
        self.next_blink = time.time() + random.uniform(2, 5)

    def draw(self, canvas):
        # Background black
        canvas[:] = (0, 0, 0)
        
        now = time.time()
        
        # Blink Logic
        is_blink = False
        if self.state == "neutral":
            if now > self.next_blink:
                self.blink_timer = now
                self.next_blink = now + random.uniform(2, 8)
            
            if now - self.blink_timer < 0.15:
                is_blink = True # closed eyes

        # Colors
        color = (255, 255, 255) # White
        if self.state == "happy":
            color = (0, 255, 255) # Yellow/Cyan
        elif self.state == "sad":
            color = (0, 0, 255) # Red

        # Draw Eyes
        cy = self.h // 2
        cx1 = self.w // 3
        cx2 = (self.w * 2) // 3
        
        eye_w = 60
        eye_h = 80
        
        if is_blink:
             cv2.line(canvas, (cx1 - 50, cy), (cx1 + 50, cy), color, 4)
             cv2.line(canvas, (cx2 - 50, cy), (cx2 + 50, cy), color, 4)
             return

        if self.state == "neutral":
            cv2.ellipse(canvas, (cx1, cy), (eye_w, eye_h), 0, 0, 360, color, -1)
            cv2.ellipse(canvas, (cx2, cy), (eye_w, eye_h), 0, 0, 360, color, -1)
            
        elif self.state == "happy":
            # ^ ^ shape (inverted parabola approx)
            cv2.ellipse(canvas, (cx1, cy), (eye_w, eye_h), 0, 180, 360, color, -1)
            cv2.ellipse(canvas, (cx2, cy), (eye_w, eye_h), 0, 180, 360, color, -1)
            
        elif self.state == "sad":
            # T_T shape (flat top, or small circles)
            cv2.circle(canvas, (cx1, cy), 30, color, -1)
            cv2.circle(canvas, (cx2, cy), 30, color, -1)
            
            # Searching animation?
            offset = int(30 * np.sin(now * 5))
            canvas[:] = (0,0,0)
            cv2.circle(canvas, (cx1 + offset, cy), 35, color, -1)
            cv2.circle(canvas, (cx2 + offset, cy), 35, color, -1)

    def set_state(self, s):
        self.state = s

# ----------------- Camera Thread -----------------
class CamThread:
    def __init__(self, index, w, h, fps):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
             print("Warning: Camera not found")
        
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.stopped = False
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(1)
                continue
            self.cap.grab()
            ok, frm = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = frm
                self.ok = True

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def release(self):
        self.stopped = True
        try: self.t.join(timeout=1.0) 
        except: pass
        self.cap.release()

# ----------------- YOLO -----------------
def create_ort_session(path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name, [o.name for o in sess.get_outputs()]

def letterbox(image, new_shape=(320, 320), color=(114, 114, 114)):
    h, w = image.shape[:2]
    nh, nw = new_shape
    scale = min(nw / w, nh / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((nh, nw, 3), color, dtype=np.uint8)
    pad_w = (nw - new_w) // 2
    pad_h = (nh - new_h) // 2
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return canvas, scale, pad_w, pad_h

def to_blob(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0)

def parse_output(outs, conf_th):
    out = np.array(outs[0])
    if out.ndim == 3: out = out[0]
    # N,6 format expected from YOLO11 usually
    if out.ndim == 2 and out.shape[1] >= 6:
        boxes_xyxy = out[:, 0:4].astype(np.float32)
        scores = out[:, 4].astype(np.float32)
        cls = out[:, 5].astype(np.int32)
        keep = scores > conf_th
        return boxes_xyxy[keep], scores[keep], cls[keep]
    # Fallback/Other formats omitted for brevity unless needed
    return np.zeros((0,4)), np.zeros((0,)), np.zeros((0,))


# ----------------- MAIN -----------------
def main():
    if not os.path.isfile(ONNX_PATH):
        raise RuntimeError(f"ONNX model not found: {ONNX_PATH}")

    cam = CamThread(CAM_INDEX, FRAME_W, FRAME_H, CAM_FPS)
    sess, in_name, out_names = create_ort_session(ONNX_PATH)
    print(f"Neo loaded. Input size: {YOLO_IMG_SIZE}")

    if SHOW_UI:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        # Default typical screen size
        cv2.resizeWindow(WINDOW_NAME, 800, 480) 
        cv2.moveWindow(WINDOW_NAME, 0, 0)

    # Face UI
    face = FaceUI(FRAME_W, FRAME_H)
    face_img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    last_seen = 0.0
    target_locked = False
    
    speak_response("idle")

    try:
        while True:
            # 1. Update Face (Independent of camera sometimes)
            face.draw(face_img)

            # 2. Camera & Inference
            ok, frame = cam.read()
            if ok and frame is not None:
                h0, w0 = frame.shape[:2]
                
                # YOLO Inference
                img, scale, pad_w, pad_h = letterbox(frame, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
                blob = to_blob(img)
                outs = sess.run(out_names, {in_name: blob})
                boxes, scores, cls = parse_output(outs, CONF_TH)
                
                # Filter Persons
                mask = (cls == PERSON_CLASS_ID)
                boxes = boxes[mask]
                scores = scores[mask]

                # AUTO FOLLOW LOGIC
                if len(boxes) > 0:
                    # Pick best target (largest area)
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    best_idx = np.argmax(areas)
                    x1_l, y1_l, x2_l, y2_l = boxes[best_idx]
                    
                    # Transform back coords
                    x1 = (x1_l - pad_w) / scale
                    x2 = (x2_l - pad_w) / scale
                    cx_l = (x1 + x2) * 0.5
                    
                    # Logic
                    if not target_locked:
                        target_locked = True
                        speak_response("found")
                    
                    face.set_state("happy")
                    last_seen = time.time()
                    
                    # Motion Control
                    err_x = (cx_l - (w0 * 0.5)) / (w0 * 0.5)
                    # Approx area norm
                    area_norm = areas[best_idx] / (scale * scale) / (w0 * h0)
                    err_a = (TARGET_AREA - area_norm)

                    # Simple P-control
                    turn = KP_TURN * err_x
                    fwd = KP_FWD * err_a
                    
                    # Clamp
                    turn = clamp(turn, -TURN_LIMIT, TURN_LIMIT)
                    if abs(fwd) < AREA_DEADBAND: fwd = 0.0
                    
                    left = apply_min(fwd + turn)
                    right = apply_min(fwd - turn)
                    set_motor(motor_a, left)
                    set_motor(motor_b, right)

                else:
                    # No person found
                    if target_locked:
                        if time.time() - last_seen > TARGET_LOST_GRACE:
                            target_locked = False
                            speak_response("lost")
                            face.set_state("sad")
                            stop_all()
                    else:
                        face.set_state("neutral")
                        stop_all()
            
            # Show UI
            if SHOW_UI:
                # Decide what to show. 
                # Option A: Only Face.
                # Option B: Camera overlay.
                # Let's show Face primarily, maybe small cam in corner?
                
                final_vis = face_img.copy()
                
                # Small camera PIP
                if ok and frame is not None:
                     small_cam = cv2.resize(frame, (160, 90))
                     final_vis[0:90, 0:160] = small_cam

                show = fit_to_window(final_vis, 800, 480)
                cv2.imshow(WINDOW_NAME, show)
                
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    speak_response("quit")
                    break

    except KeyboardInterrupt:
        pass
    finally:
        stop_all()
        # Clean shutdown
        try: motor_a.close() 
        except: pass
        try: motor_b.close() 
        except: pass
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

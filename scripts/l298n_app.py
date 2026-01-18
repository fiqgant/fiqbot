import os
import json
import time
import queue
import threading
import subprocess

import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp

import sounddevice as sd
from vosk import Model as VoskModel, KaldiRecognizer

from gpiozero import Device, Motor
from gpiozero.pins.lgpio import LGPIOFactory

# =========================================================
# CONFIG
# =========================================================

# ---------- Voice (offline) ----------
VOSK_MODEL_PATH = "/home/fiq/fiqbot/models/vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000
MIC_DEVICE_INDEX = None  # None = default mic

WAKE_WORDS = ["robot", "fiqbot", "fobot"]
WAKE_TIMEOUT_S = 5.0
MIN_CMD_GAP_S = 0.25

TTS_ENABLE = True
TTS_LANG = "en"  # espeak-ng voice id: try "id" if available, else "en"

# ---------- Follow (YOLO ONNX) ----------
ONNX_PATH = "yolo11n.onnx"

CAM_INDEX = 0
FRAME_W, FRAME_H = 512, 288
CAM_FPS = 60

CONF_TH = 0.35
NMS_TH = 0.45
PERSON_CLASS_ID = 0

# ---------- L298N pins BCM (punyamu) ----------
IN1, IN2, IN3, IN4 = 18, 19, 20, 21
ENA, ENB = 12, 13

# Motor B wiring kamu: forward=IN4 backward=IN3
MOTOR_B_FORWARD = IN4
MOTOR_B_BACKWARD = IN3

# ---------- Motor tuning ----------
MAX_SPEED = 0.80
MIN_MOVE = 0.25

STEP_SEC = 0.35       # manual step duration
TURN_STEP_SEC = 0.30  # manual turn duration

# ---------- Follow control ----------
KP_TURN = 1.2
KP_FWD = 1.4

TARGET_AREA = 0.22
AREA_DEADBAND = 0.02

X_DEADBAND = 0.06

TURN_LIMIT = 0.75
TURN_MIN = 0.22
INVERT_TURN = False  # kalau belok kebalik -> True

TARGET_MATCH_IOU = 0.20
TARGET_LOST_GRACE = 0.8

# ---------- Gesture lock ----------
LOCK_HOLD_FRAMES = 6
UNLOCK_HOLD_FRAMES = 6
GESTURE_COOLDOWN_S = 1.0
REQUIRE_HAND_IN_BOX = True

# ---------- Performance ----------
INFER_EVERY_N_FRAMES = 1  # naikkan jadi 2 kalau mau lebih ringan

# ---------- UI ----------
SHOW_UI = True
WINDOW_NAME = "Robot All-in-One"
DRAW_ALL_PERSONS = False

# =========================================================


# ----------------- Utils -----------------
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def sign(x):
    return 1.0 if x > 0 else -1.0 if x < 0 else 0.0


def apply_min(v: float) -> float:
    if abs(v) < 0.01:
        return 0.0
    if abs(v) < MIN_MOVE:
        return MIN_MOVE if v > 0 else -MIN_MOVE
    return v


def box_area_xyxy(b):
    x1, y1, x2, y2 = b
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = box_area_xyxy(a)
    area_b = box_area_xyxy(b)
    return inter / (area_a + area_b - inter + 1e-6)


def point_in_box(px, py, b):
    x1, y1, x2, y2 = b
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


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
    try:
        subprocess.Popen(
            ["espeak-ng", "-v", TTS_LANG, "-s", "160", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        pass


# ----------------- Motors (L298N) -----------------
Device.pin_factory = LGPIOFactory()
motor_a = Motor(forward=IN1, backward=IN2, enable=ENA, pwm=True)  # kiri
motor_b = Motor(forward=MOTOR_B_FORWARD, backward=MOTOR_B_BACKWARD, enable=ENB, pwm=True)  # kanan

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

def manual_forward(speed=0.7, sec=STEP_SEC):
    set_motor(motor_a, speed)
    set_motor(motor_b, speed)
    time.sleep(sec)
    stop_all()

def manual_backward(speed=0.7, sec=STEP_SEC):
    set_motor(motor_a, -speed)
    set_motor(motor_b, -speed)
    time.sleep(sec)
    stop_all()

def manual_spin_left(speed=0.6, sec=TURN_STEP_SEC):
    set_motor(motor_a, -speed)
    set_motor(motor_b, speed)
    time.sleep(sec)
    stop_all()

def manual_spin_right(speed=0.6, sec=TURN_STEP_SEC):
    set_motor(motor_a, speed)
    set_motor(motor_b, -speed)
    time.sleep(sec)
    stop_all()


# ----------------- Camera thread -----------------
class CamThread:
    def __init__(self, index, w, h, fps):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("Kamera tidak kebuka. Coba ganti CAM_INDEX atau cek /dev/video*")
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.stopped = False
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while not self.stopped:
            self.cap.grab()
            ok, frm = self.cap.read()
            if not ok:
                time.sleep(0.002)
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
        try:
            self.t.join(timeout=1.0)
        except Exception:
            pass
        self.cap.release()


# ----------------- YOLO (ONNXRuntime) -----------------
def create_ort_session(path: str):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1

    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    # Auto-detect input size (avoid 256 vs 320 mismatch)
    shp = sess.get_inputs()[0].shape  # [1,3,H,W] or dynamic
    img_size = 320
    if isinstance(shp, list) and len(shp) == 4:
        h = shp[2]
        w = shp[3]
        if isinstance(h, int) and isinstance(w, int) and h == w:
            img_size = int(h)

    return sess, in_name, out_names, img_size


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


def xywh_to_xyxy(xywh):
    x, y, w, h = xywh
    return (x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0)


def parse_output_any(outs, conf_th):
    out = np.array(outs[0])
    if out.ndim == 3:
        out = out[0]

    # case (N,6): x1,y1,x2,y2,score,cls
    if out.ndim == 2 and out.shape[1] == 6:
        boxes_xyxy = out[:, 0:4].astype(np.float32)
        scores = out[:, 4].astype(np.float32)
        cls = out[:, 5].astype(np.int32)
        keep = scores > conf_th
        return boxes_xyxy[keep], scores[keep], cls[keep]

    # case (C,N) or (N,C): xywh + class scores
    if out.ndim == 2:
        if out.shape[0] < out.shape[1]:
            out = out.transpose(1, 0)

        boxes_xywh = out[:, 0:4]
        scores_all = out[:, 4:]
        cls = np.argmax(scores_all, axis=1).astype(np.int32)
        scores = scores_all[np.arange(scores_all.shape[0]), cls].astype(np.float32)

        keep = scores > conf_th
        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        cls = cls[keep]

        boxes_xyxy = np.array([xywh_to_xyxy(b) for b in boxes_xywh], dtype=np.float32)
        return boxes_xyxy, scores, cls

    return (np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32))


# ----------------- MediaPipe Hands -----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
MCP_THUMB = 2

def finger_count(hand_lms):
    lm = hand_lms.landmark
    c = 0
    for f in ["index", "middle", "ring", "pinky"]:
        if lm[TIP[f]].y < lm[PIP[f]].y:
            c += 1
    if abs(lm[TIP["thumb"]].x - lm[MCP_THUMB].x) > 0.06:
        c += 1
    return c

def classify_gesture(hand_lms):
    c = finger_count(hand_lms)
    if c >= 4:
        return "OPEN"
    if c <= 1:
        return "FIST"
    return "OTHER"

def hand_point_px(hand_lms, w, h):
    lm = hand_lms.landmark
    x = int(((lm[0].x + lm[5].x) * 0.5) * w)
    y = int(((lm[0].y + lm[5].y) * 0.5) * h)
    return x, y

def choose_target_strict(person_boxes, hx, hy):
    inside = [b for b in person_boxes if point_in_box(hx, hy, b)]
    if not inside:
        return None
    inside.sort(key=box_area_xyxy, reverse=True)
    return inside[0]


# ----------------- Voice (Vosk) background -----------------
voice_audio_q = queue.Queue()
voice_event_q = queue.Queue()

def audio_callback(indata, frames, t, status):
    voice_audio_q.put(bytes(indata))

def normalize_text(s: str) -> str:
    return (s or "").strip().lower()

def has_wake(text: str) -> bool:
    t = normalize_text(text)
    for w in WAKE_WORDS:
        if w in t.split() or w in t:
            return True
    return False

def parse_intent(text: str):
    t = normalize_text(text)
    if not t:
        return None

    if "quit" in t or "exit" in t or "keluar" in t:
        return "quit"

    if "follow" in t or "ikuti" in t:
        return "mode_follow"
    if "manual" in t:
        return "mode_manual"
    if "idle" in t or "standby" in t:
        return "mode_idle"

    if "stop" in t or "berhenti" in t or "halt" in t or "diam" in t:
        return "stop"

    if "forward" in t or "maju" in t or ("go" in t and "back" not in t):
        return "forward"
    if "backward" in t or "mundur" in t or ("back" in t):
        return "backward"
    if "left" in t or "kiri" in t:
        return "left"
    if "right" in t or "kanan" in t:
        return "right"

    if "status" in t or "kabar" in t:
        return "status"

    return None

class VoiceEngine:
    def __init__(self):
        if not os.path.isdir(VOSK_MODEL_PATH):
            raise RuntimeError(f"Vosk model folder tidak ditemukan: {VOSK_MODEL_PATH}")
        self.model = VoskModel(VOSK_MODEL_PATH)
        self.rec = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.rec.SetWords(False)

        self.awake_until = 0.0
        self.last_cmd_t = 0.0
        self.stopped = False

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=8000,
            device=MIC_DEVICE_INDEX,
            dtype="int16",
            channels=1,
            callback=audio_callback
        ):
            while not self.stopped:
                data = voice_audio_q.get()
                if self.rec.AcceptWaveform(data):
                    res = json.loads(self.rec.Result())
                    text = normalize_text(res.get("text", ""))
                    if not text:
                        continue

                    if has_wake(text):
                        self.awake_until = time.time() + WAKE_TIMEOUT_S
                        voice_event_q.put(("wake", text))
                        continue

                    if time.time() > self.awake_until:
                        continue

                    if time.time() - self.last_cmd_t < MIN_CMD_GAP_S:
                        continue

                    intent = parse_intent(text)
                    if intent:
                        self.last_cmd_t = time.time()
                        voice_event_q.put((intent, text))

    def stop(self):
        self.stopped = True


# ----------------- MAIN -----------------
MODE_IDLE = "IDLE"
MODE_MANUAL = "MANUAL"
MODE_FOLLOW = "FOLLOW"

def main():
    if not os.path.isfile(ONNX_PATH):
        raise RuntimeError(f"ONNX model tidak ditemukan: {ONNX_PATH}")

    cam = CamThread(CAM_INDEX, FRAME_W, FRAME_H, CAM_FPS)
    sess, in_name, out_names, img_size = create_ort_session(ONNX_PATH)
    print("ONNX loaded. input IMG_SIZE =", img_size)

    if SHOW_UI:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)
        cv2.moveWindow(WINDOW_NAME, 0, 0)
        cv2.imshow(WINDOW_NAME, np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8))
        cv2.waitKey(1)

    print("Loading Vosk voice...")
    ve = VoiceEngine()
    say("Ready")

    # FOLLOW state
    locked = False
    target_box = None
    target_last_seen = 0.0
    open_count = 0
    fist_count = 0
    cooldown_until = 0.0

    mode = MODE_IDLE

    frame_id = 0
    ui_fps = 0.0
    infer_fps = 0.0
    prev_ui_t = time.time()
    prev_infer_t = time.time()

    try:
        while True:
            # ---- handle voice events ----
            while True:
                try:
                    intent, raw = voice_event_q.get_nowait()
                except queue.Empty:
                    break

                if intent == "wake":
                    say("Yes")
                    continue

                if intent == "quit":
                    say("Bye")
                    return

                if intent == "status":
                    say(f"Mode {mode.lower()}")
                    continue

                if intent == "mode_idle":
                    mode = MODE_IDLE
                    locked = False
                    target_box = None
                    stop_all()
                    say("Idle")
                    continue

                if intent == "mode_manual":
                    mode = MODE_MANUAL
                    locked = False
                    target_box = None
                    stop_all()
                    say("Manual")
                    continue

                if intent == "mode_follow":
                    mode = MODE_FOLLOW
                    locked = False
                    target_box = None
                    stop_all()
                    say("Follow")
                    continue

                # manual motion commands only in MANUAL
                if mode == MODE_MANUAL:
                    if intent == "forward":
                        say("Forward")
                        manual_forward(0.7, STEP_SEC)
                    elif intent == "backward":
                        say("Backward")
                        manual_backward(0.7, STEP_SEC)
                    elif intent == "left":
                        say("Left")
                        manual_spin_left(0.6, TURN_STEP_SEC)
                    elif intent == "right":
                        say("Right")
                        manual_spin_right(0.6, TURN_STEP_SEC)
                    elif intent == "stop":
                        say("Stop")
                        stop_all()

            # ---- get latest frame ----
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.002)
                continue

            now = time.time()
            h0, w0 = frame.shape[:2]
            frame_id += 1

            # window dims for scaling
            if SHOW_UI:
                try:
                    _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
                except Exception:
                    win_w, win_h = 0, 0

            # ---- FOLLOW pipeline ----
            gesture = "NONE"
            hx = hy = None
            person_boxes = []
            tracked = None

            if mode == MODE_FOLLOW:
                # hands
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res_h = hands.process(rgb)
                if res_h.multi_hand_landmarks:
                    hand_lms = res_h.multi_hand_landmarks[0]
                    gesture = classify_gesture(hand_lms)
                    hx, hy = hand_point_px(hand_lms, w0, h0)

                if now < cooldown_until:
                    open_count = 0
                    fist_count = 0
                else:
                    open_count = open_count + 1 if gesture == "OPEN" else 0
                    fist_count = fist_count + 1 if gesture == "FIST" else 0

                # inference
                do_infer = (frame_id % INFER_EVERY_N_FRAMES == 0)
                if do_infer:
                    img, scale, pad_w, pad_h = letterbox(frame, (img_size, img_size))
                    blob = to_blob(img)
                    outs = sess.run(out_names, {in_name: blob})

                    boxes_lb, scores, cls = parse_output_any(outs, CONF_TH)
                    mask = (cls == PERSON_CLASS_ID)
                    boxes_lb = boxes_lb[mask]
                    scores = scores[mask]

                    if boxes_lb.shape[0] > 0:
                        nms_boxes = []
                        nms_scores = []
                        for i in range(boxes_lb.shape[0]):
                            x1, y1, x2, y2 = boxes_lb[i]
                            nms_boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
                            nms_scores.append(float(scores[i]))

                        idxs = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, CONF_TH, NMS_TH)
                        if len(idxs) > 0:
                            for j in idxs.flatten().tolist():
                                x1_l, y1_l, w_l, h_l = nms_boxes[j]
                                x2_l = x1_l + w_l
                                y2_l = y1_l + h_l

                                x1 = (x1_l - pad_w) / scale
                                y1 = (y1_l - pad_h) / scale
                                x2 = (x2_l - pad_w) / scale
                                y2 = (y2_l - pad_h) / scale

                                x1 = clamp(x1, 0, w0 - 1)
                                x2 = clamp(x2, 0, w0 - 1)
                                y1 = clamp(y1, 0, h0 - 1)
                                y2 = clamp(y2, 0, h0 - 1)
                                person_boxes.append((x1, y1, x2, y2))

                    dt = now - prev_infer_t
                    if dt > 0:
                        infer_fps = 0.9 * infer_fps + 0.1 * (1.0 / dt) if infer_fps > 0 else (1.0 / dt)
                    prev_infer_t = now

                # lock/unlock
                if not locked:
                    if open_count >= LOCK_HOLD_FRAMES and hx is not None:
                        cand = choose_target_strict(person_boxes, hx, hy) if REQUIRE_HAND_IN_BOX else None
                        if cand is not None:
                            locked = True
                            target_box = cand
                            target_last_seen = now
                            cooldown_until = now + GESTURE_COOLDOWN_S
                            open_count = 0
                            say("Locked")
                else:
                    if fist_count >= UNLOCK_HOLD_FRAMES:
                        locked = False
                        target_box = None
                        cooldown_until = now + GESTURE_COOLDOWN_S
                        fist_count = 0
                        stop_all()
                        say("Unlocked")

                # track target
                if locked and target_box is not None and person_boxes:
                    best = None
                    best_iou = 0.0
                    for b in person_boxes:
                        v = iou_xyxy(target_box, b)
                        if v > best_iou:
                            best_iou = v
                            best = b
                    if best is not None and best_iou >= TARGET_MATCH_IOU:
                        tracked = best
                        target_box = best
                        target_last_seen = now

                # control
                if not locked:
                    stop_all()
                else:
                    if tracked is None:
                        if (now - target_last_seen) > TARGET_LOST_GRACE:
                            stop_all()
                    else:
                        x1, y1, x2, y2 = tracked
                        cx = (x1 + x2) * 0.5
                        err_x = (cx - (w0 * 0.5)) / (w0 * 0.5)
                        if INVERT_TURN:
                            err_x = -err_x
                        if abs(err_x) < X_DEADBAND:
                            err_x = 0.0

                        area_px = max(1.0, (x2 - x1) * (y2 - y1))
                        area_norm = area_px / float(w0 * h0)
                        err_a = (TARGET_AREA - area_norm)
                        if abs(err_a) < AREA_DEADBAND:
                            err_a = 0.0

                        base = KP_FWD * err_a
                        turn = KP_TURN * err_x

                        if err_x != 0.0 and abs(turn) < TURN_MIN:
                            turn = TURN_MIN * sign(err_x)
                        turn = clamp(turn, -TURN_LIMIT, TURN_LIMIT)

                        left = apply_min(base + turn)
                        right = apply_min(base - turn)

                        set_motor(motor_a, left)
                        set_motor(motor_b, right)

            # ---- UI ----
            if SHOW_UI:
                vis = frame.copy()

                cv2.line(vis, (w0 // 2, 0), (w0 // 2, h0), (0, 255, 255), 2)

                if mode == MODE_FOLLOW:
                    if DRAW_ALL_PERSONS:
                        for b in person_boxes:
                            x1, y1, x2, y2 = map(int, b)
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (70, 70, 70), 1)

                    if locked and target_box is not None:
                        x1, y1, x2, y2 = map(int, target_box)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    if hx is not None:
                        cv2.circle(vis, (hx, hy), 7, (255, 0, 0), -1)

                dt = now - prev_ui_t
                if dt > 0:
                    ui_fps = 0.9 * ui_fps + 0.1 * (1.0 / dt) if ui_fps > 0 else (1.0 / dt)
                prev_ui_t = now

                cv2.putText(vis, f"MODE {mode} | UI {ui_fps:.1f} | INFER {infer_fps:.1f}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if mode == MODE_FOLLOW:
                    st = "LOCKED" if locked else "UNLOCKED"
                    cv2.putText(vis, f"FOLLOW {st} | Gesture {gesture}",
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(vis, "OPEN palm=LOCK | FIST=UNLOCK | say 'robot' then 'follow/manual/idle'",
                                (10, h0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    cv2.putText(vis, "Say 'robot' then: follow/manual/idle/forward/back/left/right/stop",
                                (10, h0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                show = fit_to_window(vis, win_w, win_h)
                cv2.imshow(WINDOW_NAME, show)

                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    say("Bye")
                    return

    finally:
        stop_all()
        motor_a.close()
        motor_b.close()
        cam.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            ve.stop()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        stop_all()
        raise

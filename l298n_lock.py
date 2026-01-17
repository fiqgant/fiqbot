import time
import threading
import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp

from gpiozero import Device, Motor
from gpiozero.pins.lgpio import LGPIOFactory

# =========================
# CONFIGURATION
# =========================
ONNX_PATH = "yolo11n.onnx"

CAM_INDEX = 0
FRAME_W, FRAME_H = 512, 288
CAM_FPS = 60

IMG_SIZE = 320
CONF_TH = 0.35
NMS_TH = 0.45
PERSON_CLASS_ID = 0

# BCM Pin Configuration
IN1, IN2, IN3, IN4 = 18, 19, 20, 21
ENA, ENB = 12, 13

# PID / Motion Tuning
KP_TURN = 1.2
KP_FWD = 1.4
MAX_SPEED = 0.80
MIN_MOVE = 0.25

TARGET_AREA = 0.22
AREA_DEADBAND = 0.02

# Turning Tuning
TURN_LIMIT = 0.75
TURN_MIN = 0.22
X_DEADBAND = 0.06
INVERT_TURN = False

# Gesture / Lock Configuration
LOCK_HOLD_FRAMES = 6
UNLOCK_HOLD_FRAMES = 6
GESTURE_COOLDOWN_S = 1.0

TARGET_MATCH_IOU = 0.20
TARGET_LOST_GRACE = 0.8
REQUIRE_HAND_IN_BOX = True

INFER_EVERY_N_FRAMES = 1

SHOW_UI = True
DRAW_ALL_PERSONS = False

# Wayland Support: Use borderless window maximizing strategy
TARGET_SCREEN_W = 1920
TARGET_SCREEN_H = 1080
# =========================


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


# ===== Motor Initialization (Pi 5 optimized) =====
Device.pin_factory = LGPIOFactory()
motor_left = Motor(forward=IN1, backward=IN2, enable=ENA, pwm=True)
motor_right = Motor(forward=IN4, backward=IN3, enable=ENB, pwm=True)

def stop_motors():
    motor_left.stop()
    motor_right.stop()

def set_motor(motor: Motor, v: float):
    v = clamp(v, -MAX_SPEED, MAX_SPEED)
    if abs(v) < 0.01:
        motor.stop()
        return
    if v >= 0:
        motor.forward(v)
    else:
        motor.backward(-v)


# ===== Camera Thread =====
class CamThread:
    def __init__(self, index, w, h, fps):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("Camera failed to open. Check CAM_INDEX or permissions.")
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
        except:
            pass
        try:
            self.cap.release()
        except:
            pass


cam = CamThread(CAM_INDEX, FRAME_W, FRAME_H, CAM_FPS)


# ===== ONNX Runtime Initialization =====
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 4
so.inter_op_num_threads = 1

sess = ort.InferenceSession(ONNX_PATH, sess_options=so, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
out_names = [o.name for o in sess.get_outputs()]


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

    if out.ndim == 2 and out.shape[1] == 6:
        boxes_xyxy = out[:, 0:4].astype(np.float32)
        scores = out[:, 4].astype(np.float32)
        cls = out[:, 5].astype(np.int32)
        keep = scores > conf_th
        return boxes_xyxy[keep], scores[keep], cls[keep]

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

    return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)


# ===== MediaPipe Hands =====
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


# ===== UI (Wayland-friendly maximize) =====
WIN = "FiqBot - Gesture Lock Follow"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, TARGET_SCREEN_W, TARGET_SCREEN_H)
cv2.moveWindow(WIN, 0, 0)
cv2.imshow(WIN, np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8))
cv2.waitKey(1)


locked = False
target_box = None
target_last_seen = 0.0

open_count = 0
fist_count = 0
cooldown_until = 0.0

frame_id = 0
ui_fps = 0.0
infer_fps = 0.0
prev_ui_t = time.time()
prev_infer_t = time.time()

try:
    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            time.sleep(0.002)
            continue

        now = time.time()
        h0, w0 = frame.shape[:2]
        frame_id += 1

        # Get actual window size if possible
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(WIN)
            if win_w <= 0 or win_h <= 0:
                win_w, win_h = TARGET_SCREEN_W, TARGET_SCREEN_H
        except:
            win_w, win_h = TARGET_SCREEN_W, TARGET_SCREEN_H

        # ---- Hands Detection ----
        gesture = "NONE"
        hx = hy = None

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

        # ---- YOLOv11 Detection ----
        person_boxes = []
        tracked = None

        do_infer = (frame_id % INFER_EVERY_N_FRAMES == 0)
        if do_infer:
            img, scale, pad_w, pad_h = letterbox(frame, (IMG_SIZE, IMG_SIZE))
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

        # ---- Lock / Unlock Logic ----
        if not locked:
            if open_count >= LOCK_HOLD_FRAMES and hx is not None:
                cand = choose_target_strict(person_boxes, hx, hy) if REQUIRE_HAND_IN_BOX else None
                if cand is not None:
                    locked = True
                    target_box = cand
                    target_last_seen = now
                    cooldown_until = now + GESTURE_COOLDOWN_S
                    open_count = 0
        else:
            if fist_count >= UNLOCK_HOLD_FRAMES:
                locked = False
                target_box = None
                cooldown_until = now + GESTURE_COOLDOWN_S
                fist_count = 0
                stop_motors()

        # ---- Track Target ----
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

        # ---- Motion Control ----
        if not locked:
            stop_motors()
        else:
            if tracked is None:
                if (now - target_last_seen) > TARGET_LOST_GRACE:
                    stop_motors()
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

                set_motor(motor_left, left)
                set_motor(motor_right, right)

        # ---- UI Visualization ----
        vis = frame
        cv2.line(vis, (w0 // 2, 0), (w0 // 2, h0), (0, 255, 255), 2)

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

        status = "LOCKED" if locked else "IDLE"
        cv2.putText(vis, f"UI {ui_fps:.1f} | INFER {infer_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"{status} | Gesture {gesture}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, "Wayland Mode: Press Q to Quit", (10, h0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        show = fit_to_window(vis, win_w, win_h)
        cv2.imshow(WIN, show)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break

except KeyboardInterrupt:
    pass
finally:
    stop_motors()
    motor_left.close()
    motor_right.close()
    cam.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass


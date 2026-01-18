import os
import time
import cv2
import numpy as np
import onnxruntime as ort

from gpiozero import Device, Motor
from gpiozero.pins.lgpio import LGPIOFactory

# =========================
# CONFIGURATION
# =========================

# Helper: Find model source
def get_model_path(filename):
    # Check same directory as script
    p = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(p):
        return p
    # Check parent directory (project root)
    p = os.path.join(os.path.dirname(os.path.dirname(__file__)), filename)
    if os.path.exists(p):
        return p
    # Default to current dir if not found (let ONNX error out naturally later)
    return filename

ONNX_PATH = get_model_path("yolo11n.onnx")
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 360
CAM_FPS = 30

IMG_SIZE = 640
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
LOST_TIMEOUT = 0.6

# Turning Tuning
TURN_LIMIT = 0.75
TURN_MIN = 0.22
X_DEADBAND = 0.06
INVERT_TURN = False   # Set True if the robot turns in the opposite direction

# UI Options
SHOW_UI = True
UI_SCALE = 1.0
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


# ===== Camera Initialization =====
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise RuntimeError("Camera failed to open. Check CAM_INDEX or /dev/video* permissions.")


# ===== ONNX Runtime Initialization =====
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
out_names = [o.name for o in sess.get_outputs()]


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
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
    img = np.transpose(img, (2, 0, 1))      # CHW
    return np.expand_dims(img, 0)           # NCHW


def xywh_to_xyxy(xywh):
    x, y, w, h = xywh
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return x1, y1, x2, y2


def parse_output_any(outs, conf_th):
    """
    Parses YOLO output for multiple shapes:
    A) (C,N)/(N,C): [x,y,w,h, cls...]
    B) (N,6): [x1,y1,x2,y2,score,cls]
    Returns: boxes_xyxy (N,4), scores (N,), cls (N,)
    """
    out = np.array(outs[0])

    if out.ndim == 3:
        out = out[0]

    # Case B
    if out.ndim == 2 and out.shape[1] == 6:
        boxes_xyxy = out[:, 0:4].astype(np.float32)
        scores = out[:, 4].astype(np.float32)
        cls = out[:, 5].astype(np.int32)
        keep = scores > conf_th
        return boxes_xyxy[keep], scores[keep], cls[keep]

    # Case A
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

        boxes_xyxy = []
        for b in boxes_xywh:
            boxes_xyxy.append(xywh_to_xyxy(b))
        return np.array(boxes_xyxy, dtype=np.float32), scores, cls

    return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)


last_seen = 0.0
t0 = time.time()
fps = 0.0
best_box = None
best_score = 0.0

try:
    while True:
        cap.grab()
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        h0, w0 = frame.shape[:2]

        img, scale, pad_w, pad_h = letterbox(frame, (IMG_SIZE, IMG_SIZE))
        blob = to_blob(img)

        outs = sess.run(out_names, {in_name: blob})
        boxes_lb, scores, cls = parse_output_any(outs, CONF_TH)

        # Filter for Person class only
        mask = (cls == PERSON_CLASS_ID)
        boxes_lb = boxes_lb[mask]
        scores = scores[mask]

        now = time.time()
        best_box = None
        best_score = 0.0

        if boxes_lb.shape[0] == 0:
            if (now - last_seen) > LOST_TIMEOUT:
                stop_motors()
        else:
            last_seen = now

            # Non-Maximum Suppression (NMS)
            nms_boxes = []
            nms_scores = []
            for i in range(boxes_lb.shape[0]):
                x1, y1, x2, y2 = boxes_lb[i]
                nms_boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
                nms_scores.append(float(scores[i]))

            idxs = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, CONF_TH, NMS_TH)
            if len(idxs) > 0:
                idxs = idxs.flatten().tolist()
                best_i = max(idxs, key=lambda j: nms_scores[j])

                x1_l, y1_l, w_l, h_l = nms_boxes[best_i]
                x2_l = x1_l + w_l
                y2_l = y1_l + h_l

                # Map back to original image coordinates
                x1 = (x1_l - pad_w) / scale
                y1 = (y1_l - pad_h) / scale
                x2 = (x2_l - pad_w) / scale
                y2 = (y2_l - pad_h) / scale

                x1 = clamp(x1, 0, w0 - 1)
                x2 = clamp(x2, 0, w0 - 1)
                y1 = clamp(y1, 0, h0 - 1)
                y2 = clamp(y2, 0, h0 - 1)

                best_box = (x1, y1, x2, y2)
                best_score = nms_scores[best_i]

                # Motion Control Logic
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

                # Force minimal turn if target is off-center to avoid stalling
                if err_x != 0.0 and abs(turn) < TURN_MIN:
                    turn = TURN_MIN * sign(err_x)
                turn = clamp(turn, -TURN_LIMIT, TURN_LIMIT)

                left = apply_min(base + turn)
                right = apply_min(base - turn)

                set_motor(motor_left, left)
                set_motor(motor_right, right)

        # FPS Calculation
        dt = now - t0
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
        t0 = now

        # UI Visualization
        if SHOW_UI:
            vis = frame
            if UI_SCALE != 1.0:
                vis = cv2.resize(vis, None, fx=UI_SCALE, fy=UI_SCALE, interpolation=cv2.INTER_LINEAR)

            hh, ww = vis.shape[:2]
            # Draw centerline
            cv2.line(vis, (ww // 2, 0), (ww // 2, hh), (0, 255, 255), 2)

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                if UI_SCALE != 1.0:
                    x1 *= UI_SCALE; y1 *= UI_SCALE; x2 *= UI_SCALE; y2 *= UI_SCALE
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(vis, f"Person {best_score:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, "Press Q to Quit", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("FiqBot - YOLO Following", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

except KeyboardInterrupt:
    pass
finally:
    stop_motors()
    motor_left.close()
    motor_right.close()
    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass

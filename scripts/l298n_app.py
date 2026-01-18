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

# ---------- TTS ----------
TTS_ENABLE = True
TTS_LANG = "en-us"
TTS_MIN_GAP = 2.0  # seconds between TTS to avoid spam

# ---------- YOLO ONNX ----------
ONNX_PATH = "yolo11n.onnx"

CAM_INDEX = 0
FRAME_W, FRAME_H = 512, 288
CAM_FPS = 60

CONF_TH = 0.25       # start with 0.25; for debugging try 0.15
NMS_TH = 0.45
PERSON_CLASS_ID = 0

# ---------- L298N pins BCM ----------
IN1, IN2, IN3, IN4 = 18, 19, 20, 21
ENA, ENB = 12, 13
MOTOR_B_FORWARD = IN4
MOTOR_B_BACKWARD = IN3

# ---------- Motor ----------
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
INFER_EVERY_N_FRAMES = 1  # set 2 kalau berat

# ---------- UI ----------
SHOW_UI = True
WINDOW_NAME = "Neo Robot"
UI_W, UI_H = 1280, 720     # window size (resizable)
PIP_W, PIP_H = 320, 180    # camera picture-in-picture size

# =========================================================
# UTILS
# =========================================================
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


# =========================================================
# TTS
# =========================================================
_last_tts_t = 0.0

NEO_RESPONSES = {
    "found": ["Target found.", "I see you.", "Hello again.", "Target acquired."],
    "lost":  ["Target lost.", "Where did you go?", "Searching.", "Scanning."],
    "idle":  ["Systems idle.", "Waiting.", "Standing by."],
    "quit":  ["Shutting down.", "Goodbye.", "Offline."]
}

def _say_worker(text: str):
    try:
        subprocess.Popen(
            ["espeak-ng", "-v", TTS_LANG, "-s", "160", text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).wait()
    except Exception:
        pass

def say(text: str):
    global _last_tts_t
    if not TTS_ENABLE:
        return
    text = (text or "").strip()
    if not text:
        return
    now = time.time()
    if now - _last_tts_t < TTS_MIN_GAP:
        return
    _last_tts_t = now
    threading.Thread(target=_say_worker, args=(text,), daemon=True).start()

def speak_response(key: str):
    if key in NEO_RESPONSES:
        say(random.choice(NEO_RESPONSES[key]))


# =========================================================
# MOTORS (L298N)
# =========================================================
Device.pin_factory = LGPIOFactory()
motor_a = Motor(forward=IN1, backward=IN2, enable=ENA, pwm=True)  # left
motor_b = Motor(forward=MOTOR_B_FORWARD, backward=MOTOR_B_BACKWARD, enable=ENB, pwm=True)  # right

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


# =========================================================
# EYES UI
# =========================================================
class FaceUI:
    def __init__(self, w, h):
        self.w = int(w)
        self.h = int(h)
        self.state = "neutral"  # neutral, happy, sad
        self.blink_t = 0.0
        self.next_blink = time.time() + random.uniform(2, 6)
        self.target_x = 0.0

    def set_state(self, s: str):
        self.state = s

    def draw(self, canvas, target_x=0.0):
        canvas[:] = (0, 0, 0)
        now = time.time()
        self.target_x = float(clamp(target_x, -1.0, 1.0))

        # blink
        blink = False
        if now > self.next_blink:
            self.blink_t = now
            self.next_blink = now + random.uniform(2, 7)
        if now - self.blink_t < 0.12:
            blink = True

        # colors
        if self.state == "happy":
            eye_color = (120, 255, 255)
            pupil_color = (0, 220, 220)
        elif self.state == "sad":
            eye_color = (150, 150, 255)
            pupil_color = (60, 60, 180)
        else:
            eye_color = (255, 255, 255)
            pupil_color = (60, 60, 60)

        cy = self.h // 2
        cx1 = self.w // 4
        cx2 = (self.w * 3) // 4

        eye_w = int(120)
        eye_h = int(160)

        if blink:
            cv2.line(canvas, (cx1 - eye_w, cy), (cx1 + eye_w, cy), eye_color, 8)
            cv2.line(canvas, (cx2 - eye_w, cy), (cx2 + eye_w, cy), eye_color, 8)
            return

        # eyes
        cv2.ellipse(canvas, (cx1, cy), (eye_w, eye_h), 0, 0, 360, eye_color, -1)
        cv2.ellipse(canvas, (cx2, cy), (eye_w, eye_h), 0, 0, 360, eye_color, -1)

        # pupils follow target
        pupil_dx = int(40 * self.target_x)
        pupil_r = 30
        cv2.circle(canvas, (cx1 + pupil_dx, cy), pupil_r, pupil_color, -1)
        cv2.circle(canvas, (cx2 + pupil_dx, cy), pupil_r, pupil_color, -1)


# =========================================================
# CAMERA THREAD
# =========================================================
class CamThread:
    def __init__(self, index, w, h, fps):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not opened. Check CAM_INDEX or /dev/video*")

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
        try:
            self.t.join(timeout=1.0)
        except Exception:
            pass
        self.cap.release()


# =========================================================
# YOLO
# =========================================================
def create_ort_session(path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    # auto detect input size from model input shape [1,3,H,W]
    shp = sess.get_inputs()[0].shape
    img_size = 320
    if isinstance(shp, list) and len(shp) == 4:
        h = shp[2]
        w = shp[3]
        if isinstance(h, int) and isinstance(w, int) and h == w:
            img_size = int(h)

    return sess, in_name, out_names, img_size


def letterbox(image, new_shape, color=(114, 114, 114)):
    nh, nw = new_shape
    h, w = image.shape[:2]
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
    img = np.transpose(img, (2, 0, 1))  # CHW
    return np.expand_dims(img, 0)       # NCHW


def parse_output_any(outs, conf_th):
    """
    Supports:
    - (N,6): x1,y1,x2,y2,score,cls
    - (1,C,N) or (C,N) or (N,C): xywh + class scores
    Returns boxes_xyxy in letterbox coords.
    """
    out = outs[0]
    out = np.array(out)

    if out.ndim == 3:
        out = out[0]

    # (N,6)
    if out.ndim == 2 and out.shape[1] >= 6:
        boxes = out[:, 0:4].astype(np.float32)
        scores = out[:, 4].astype(np.float32)
        cls = out[:, 5].astype(np.int32)
        keep = scores > conf_th
        return boxes[keep], scores[keep], cls[keep]

    # (C,N) or (N,C)
    if out.ndim == 2:
        if out.shape[0] < out.shape[1]:
            out = out.transpose(1, 0)  # (N,C)

        boxes_xywh = out[:, 0:4].astype(np.float32)
        scores_all = out[:, 4:].astype(np.float32)
        cls = np.argmax(scores_all, axis=1).astype(np.int32)
        scores = scores_all[np.arange(scores_all.shape[0]), cls].astype(np.float32)
        keep = scores > conf_th

        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        cls = cls[keep]

        boxes_xyxy = np.zeros((boxes_xywh.shape[0], 4), dtype=np.float32)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
        return boxes_xyxy, scores, cls

    return (np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32))


# =========================================================
# MAIN
# =========================================================
def main():
    if not os.path.isfile(ONNX_PATH):
        raise RuntimeError(f"ONNX model not found: {ONNX_PATH}")

    cam = CamThread(CAM_INDEX, FRAME_W, FRAME_H, CAM_FPS)
    sess, in_name, out_names, yolo_img = create_ort_session(ONNX_PATH)
    print(f"Neo loaded. ONNX input size: {yolo_img}")

    if SHOW_UI:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, UI_W, UI_H)
        cv2.moveWindow(WINDOW_NAME, 0, 0)

    face = FaceUI(UI_W, UI_H)
    face_img = np.zeros((UI_H, UI_W, 3), dtype=np.uint8)

    last_seen = 0.0
    locked = False
    speak_response("idle")

    frame_id = 0
    ui_fps = 0.0
    infer_fps = 0.0
    prev_ui = time.time()
    prev_inf = time.time()

    try:
        while True:
            frame_id += 1
            now = time.time()
            target_dir = 0.0

            ok, frame = cam.read()
            if not ok or frame is None:
                stop_all()
                face.set_state("neutral")
                face.draw(face_img, 0.0)
                if SHOW_UI:
                    cv2.imshow(WINDOW_NAME, face_img)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        speak_response("quit")
                        break
                continue

            h0, w0 = frame.shape[:2]

            # default state
            if not locked:
                face.set_state("neutral")

            # inference
            person_boxes = []
            person_scores = []

            if (frame_id % INFER_EVERY_N_FRAMES) == 0:
                img, scale, pad_w, pad_h = letterbox(frame, (yolo_img, yolo_img))
                blob = to_blob(img)
                outs = sess.run(out_names, {in_name: blob})
                boxes_lb, scores, cls = parse_output_any(outs, CONF_TH)

                # filter person
                mask = (cls == PERSON_CLASS_ID)
                boxes_lb = boxes_lb[mask]
                scores = scores[mask]

                # NMS
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

                            # back to original coords
                            x1 = (x1_l - pad_w) / scale
                            y1 = (y1_l - pad_h) / scale
                            x2 = (x2_l - pad_w) / scale
                            y2 = (y2_l - pad_h) / scale

                            x1 = clamp(x1, 0, w0 - 1)
                            x2 = clamp(x2, 0, w0 - 1)
                            y1 = clamp(y1, 0, h0 - 1)
                            y2 = clamp(y2, 0, h0 - 1)

                            person_boxes.append((float(x1), float(y1), float(x2), float(y2)))
                            person_scores.append(nms_scores[j])

                dt_inf = now - prev_inf
                if dt_inf > 0:
                    infer_fps = 0.9 * infer_fps + 0.1 * (1.0 / dt_inf) if infer_fps > 0 else (1.0 / dt_inf)
                prev_inf = now

            # choose target: largest area
            if person_boxes:
                areas = [ (b[2]-b[0]) * (b[3]-b[1]) for b in person_boxes ]
                best_i = int(np.argmax(np.array(areas)))
                x1, y1, x2, y2 = person_boxes[best_i]
                cx = (x1 + x2) * 0.5

                err_x = (cx - (w0 * 0.5)) / (w0 * 0.5)  # -1..1
                if INVERT_TURN:
                    err_x = -err_x
                if abs(err_x) < X_DEADBAND:
                    err_x = 0.0

                target_dir = float(clamp(err_x, -1.0, 1.0))

                area_norm = ((x2 - x1) * (y2 - y1)) / float(w0 * h0)
                err_a = (TARGET_AREA - area_norm)
                if abs(err_a) < AREA_DEADBAND:
                    err_a = 0.0

                base = KP_FWD * err_a
                turn = KP_TURN * err_x

                # ensure turning shows up
                if err_x != 0.0 and abs(turn) < TURN_MIN:
                    turn = TURN_MIN * sign(err_x)
                turn = clamp(turn, -TURN_LIMIT, TURN_LIMIT)

                left = apply_min(base + turn)
                right = apply_min(base - turn)

                set_motor(motor_a, left)
                set_motor(motor_b, right)

                last_seen = now
                if not locked:
                    locked = True
                    face.set_state("happy")
                    speak_response("found")
                else:
                    face.set_state("happy")

            else:
                # lost logic
                if locked and (now - last_seen) > TARGET_LOST_GRACE:
                    locked = False
                    stop_all()
                    face.set_state("sad")
                    speak_response("lost")
                elif not locked:
                    stop_all()
                    face.set_state("neutral")

            # draw face
            face.draw(face_img, target_dir)

            # draw PIP camera + bbox
            if SHOW_UI:
                pip = frame.copy()
                # draw center line
                cv2.line(pip, (w0 // 2, 0), (w0 // 2, h0), (0, 255, 255), 2)

                # draw persons
                for b in person_boxes:
                    x1, y1, x2, y2 = map(int, b)
                    cv2.rectangle(pip, (x1, y1), (x2, y2), (0, 255, 0), 2)

                pip_small = cv2.resize(pip, (PIP_W, PIP_H), interpolation=cv2.INTER_LINEAR)
                face_img[0:PIP_H, 0:PIP_W] = pip_small

                # HUD
                dt_ui = now - prev_ui
                if dt_ui > 0:
                    ui_fps = 0.9 * ui_fps + 0.1 * (1.0 / dt_ui) if ui_fps > 0 else (1.0 / dt_ui)
                prev_ui = now

                status = "TRACK" if locked else "IDLE"
                cv2.putText(face_img, f"{status} | UI {ui_fps:.1f} | INFER {infer_fps:.1f}",
                            (10, UI_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.imshow(WINDOW_NAME, face_img)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    speak_response("quit")
                    break

    except KeyboardInterrupt:
        pass
    finally:
        stop_all()
        try:
            motor_a.close()
            motor_b.close()
        except Exception:
            pass
        cam.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()

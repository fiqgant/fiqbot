# neo_follow_ui_piper_kawaii_full.py
# - Kawaii HDMI RoboEyes-style UI (OpenCV fullscreen)
# - Piper TTS via python -m piper, with QUEUE + CACHE (no cut, supports long speech)
# - Audio playback prefers pw-play (PipeWire) then paplay then aplay
# - Robust ONNX output parsing (supports (N,6) and Ultralytics (1,C,N)/(1,N,C))
# - Auto-detect ONNX input size
# - Debug overlay + periodic terminal prints
# - L298N motors via gpiozero + LGPIOFactory
# - Camera PIP + boxes
# - NEW: random expressions + random quotes + random idle chatter
#
# Run:
#   source ~/fiqbot/robot/bin/activate
#   python neo_follow_ui_piper_kawaii_full.py

import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
import threading
import subprocess
import random
import math
import hashlib
import queue

import cv2
import numpy as np
import onnxruntime as ort

from gpiozero import Device, Motor
from gpiozero.pins.lgpio import LGPIOFactory

# =========================================================
# CONFIG
# =========================================================

# ---------- TTS (Piper) ----------
TTS_ENABLE = True
TTS_MIN_GAP = 1.5           # smaller = more talkative, but still avoid spam
TTS_MAX_CHARS = 220         # split long text into chunks (natural + safe)
TTS_CACHE_DIR = "/tmp/neo_tts_cache"

PIPER_MODEL = os.path.expanduser("~/voices/piper/en_US-lessac-medium.onnx")
PIPER_CONFIG = os.path.expanduser("~/voices/piper/en_US-lessac-medium.onnx.json")
PIPER_SPEAKER = None

PIPER_LENGTH_SCALE = 1.05   # >1 slower, <1 faster
PIPER_NOISE_SCALE = 0.667
PIPER_NOISE_W = 0.8

# ---------- YOLO ONNX ----------
ONNX_PATH = "yolo11n.onnx"

CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 360
CAM_FPS = 60

CONF_TH = 0.15
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
KP_TURN = 1.4
KP_FWD = 1.4

TARGET_AREA = 0.22
AREA_DEADBAND = 0.02

X_DEADBAND = 0.05
TURN_LIMIT = 0.80
TURN_MIN = 0.20
INVERT_TURN = False

TARGET_LOST_GRACE = 1.5

# ---------- Performance ----------
INFER_EVERY_N_FRAMES = 2    # lighter CPU -> TTS feels faster; set 1 if still OK

# ---------- UI ----------
SHOW_UI = True
WINDOW_NAME = "Neo Robot"
UI_W, UI_H = 1280, 720
PIP_W, PIP_H = 280, 158
PIP_PAD = 18 
FULLSCREEN = True

# ---------- Debug ----------
DEBUG_PRINT_EVERY = 30
DRAW_ALL_PERSONS = True

# ---------- Random personality ----------
RANDOM_IDLE_CHATTER = True
IDLE_CHATTER_MIN_S = 18
IDLE_CHATTER_MAX_S = 45
EXPRESSIONS_ENABLE = True

# =========================================================


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

def box_area_xyxy(b):
    x1, y1, x2, y2 = b
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))


# =========================================================
# AUDIO PLAYBACK (prefer PipeWire on Pi 5 HDMI)
# =========================================================
def _cmd_exists(cmd0: str) -> bool:
    try:
        r = subprocess.run([cmd0, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0
    except Exception:
        return False

def play_wav(path: str) -> bool:
    players = [
        ["pw-play", path],
        ["paplay", path],
        ["aplay", "-D", "default", path],
    ]
    for cmd in players:
        if not _cmd_exists(cmd[0]):
            continue
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).wait(timeout=60)
            return True
        except Exception:
            continue
    return False


# =========================================================
# PIPER TTS (QUEUE + CACHE)
# =========================================================
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

_tts_q = queue.Queue(maxsize=12)
_tts_worker_started = False
_last_tts_t = 0.0

# --- Cute random quotes / phrases ---
NEO_QUOTES = [
    "Beep beep. I'm trying my best!",
    "If lost, please reboot human.",
    "I run on snacks... and electricity.",
    "I see with my eyes. I follow with my heart.",
    "Tiny robot, big dreams.",
    "Hold still—I'm focusing!",
    "I'm not nervous. You're nervous.",
    "I have one job: follow politely.",
    "If you smile, my motors smile too.",
    "Soft turns only, okay?",
]

NEO_EXPRESSIONS = {
    "giggle": [
        "Hehe!", "Hihi!", "Ehehe!", "Tehehe!"
    ],
    "wow": [
        "Whoa!", "Waaah!", "Wowza!", "Ohhh!"
    ],
    "curious": [
        "Hmm?", "Ooh?", "What is that?", "Interesting..."
    ],
    "encourage": [
        "You got this!", "Let's go!", "I'm with you!", "Okay okay!"
    ],
    "apologize": [
        "Sorry sorry!", "Oops!", "My bad!", "I didn't mean to!"
    ],
    "robot": [
        "Beep.", "Boop.", "Beep boop!", "Bweep!"
    ]
}

NEO_RESPONSES = {
    "found": [
        "Hi hi! I see you clearly now. I will follow you gently. Please move slowly so I can keep you in my camera view.",
        "Target acquired! If you step to the left, I will turn left. If you step to the right, I will turn right. If you get too close, I will slow down."
    ],
    "lost": [
        "Uh oh, I lost you. Please come back in front of me so I can see you again.",
        "Where did you go? I'm scanning. Try standing in the center of my camera view."
    ],
    "idle": [
        "Standing by! Wave at me, and I will start following you.",
        "I'm ready. Please step in front of the camera so I can lock on to you."
    ],
    "quit": ["Shutting down. Goodbye!"]
}

def _hash_key(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def _split_text(text: str, max_chars: int):
    text = " ".join((text or "").strip().split())
    if not text:
        return []

    parts = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in ".!?":
            parts.append(buf.strip())
            buf = ""
    if buf.strip():
        parts.append(buf.strip())

    merged = []
    cur = ""
    for p in parts:
        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= max_chars:
            cur = cur + " " + p
        else:
            merged.append(cur)
            cur = p
    if cur:
        merged.append(cur)

    final = []
    for m in merged:
        if len(m) <= max_chars:
            final.append(m)
        else:
            for i in range(0, len(m), max_chars):
                final.append(m[i:i+max_chars])
    return final

def _piper_cmd(out_path: str):
    cmd = ["python", "-m", "piper", "--model", PIPER_MODEL, "--output_file", out_path]
    if PIPER_CONFIG and os.path.isfile(PIPER_CONFIG):
        cmd += ["--config", PIPER_CONFIG]
    cmd += ["--length_scale", str(PIPER_LENGTH_SCALE)]
    cmd += ["--noise_scale", str(PIPER_NOISE_SCALE)]
    cmd += ["--noise_w", str(PIPER_NOISE_W)]
    if PIPER_SPEAKER is not None:
        cmd += ["--speaker", str(int(PIPER_SPEAKER))]
    return cmd

def _ensure_wav(text: str) -> str:
    key = _hash_key(
        f"{PIPER_MODEL}|{PIPER_CONFIG}|{PIPER_LENGTH_SCALE}|{PIPER_NOISE_SCALE}|{PIPER_NOISE_W}|{PIPER_SPEAKER}|{text}"
    )
    out_path = os.path.join(TTS_CACHE_DIR, f"{key}.wav")
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 2000:
        return out_path

    try:
        p = subprocess.Popen(
            _piper_cmd(out_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True
        )
        try:
            p.stdin.write(text)
            p.stdin.flush()
            p.stdin.close()
        except Exception:
            pass
        p.wait(timeout=90)
    except Exception:
        return ""

    if os.path.isfile(out_path) and os.path.getsize(out_path) > 2000:
        return out_path
    return ""

def _tts_worker_loop():
    while True:
        item = _tts_q.get()
        if item is None:
            break
        text = item
        try:
            chunks = _split_text(text, TTS_MAX_CHARS)
            for c in chunks:
                wav = _ensure_wav(c)
                if wav:
                    play_wav(wav)
        except Exception:
            pass
        _tts_q.task_done()

def _start_tts_worker_once():
    global _tts_worker_started
    if _tts_worker_started:
        return
    _tts_worker_started = True
    threading.Thread(target=_tts_worker_loop, daemon=True).start()

def _decorate(text: str) -> str:
    if not EXPRESSIONS_ENABLE:
        return text
    # add short prefix/suffix sometimes (keeps it cute but not too spammy)
    prefix = ""
    suffix = ""
    r = random.random()
    if r < 0.25:
        prefix = random.choice(NEO_EXPRESSIONS["robot"]) + " "
    elif r < 0.45:
        prefix = random.choice(NEO_EXPRESSIONS["giggle"]) + " "
    elif r < 0.60:
        prefix = random.choice(NEO_EXPRESSIONS["curious"]) + " "
    elif r < 0.70:
        prefix = random.choice(NEO_EXPRESSIONS["wow"]) + " "

    if random.random() < 0.25:
        suffix = " " + random.choice(NEO_QUOTES)

    return (prefix + text + suffix).strip()

def say(text: str):
    global _last_tts_t
    if not TTS_ENABLE:
        return
    if not os.path.isfile(PIPER_MODEL):
        return

    text = (text or "").strip()
    if not text:
        return

    now = time.time()
    if now - _last_tts_t < TTS_MIN_GAP:
        return
    _last_tts_t = now

    _start_tts_worker_once()

    text = _decorate(text)

    try:
        _tts_q.put_nowait(text)
    except queue.Full:
        # drop if too chatty, avoid blocking robot loop
        pass

def speak_response(key: str):
    if key in NEO_RESPONSES:
        say(random.choice(NEO_RESPONSES[key]))

def maybe_idle_chatter(tracking: bool, next_idle_t: float) -> float:
    if not RANDOM_IDLE_CHATTER:
        return next_idle_t
    now = time.time()
    if tracking:
        return now + random.uniform(IDLE_CHATTER_MIN_S, IDLE_CHATTER_MAX_S)
    if now >= next_idle_t and random.random() < 0.75:
        # mix of short expression + quote + small instruction
        idle_lines = [
            "I'm here. Wave at me!",
            "If you want me to follow you, stand in the middle.",
            "I can see best when the light is nice and bright.",
            "Try moving slowly. I get dizzy easily.",
            "I am scanning for a human-shaped friend.",
        ]
        line = random.choice(idle_lines)
        # sometimes just a quote
        if random.random() < 0.35:
            line = random.choice(NEO_QUOTES)
        say(line)
        return now + random.uniform(IDLE_CHATTER_MIN_S, IDLE_CHATTER_MAX_S)
    return next_idle_t


# =========================================================
# MOTORS
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
# KAWAII HDMI ROBOEYES-STYLE UI (with extra moods)
# =========================================================
# PATCH: smoother pupil movement (low-pass filter) + optional speed limit
# Drop-in replacement for KawaiiEyesUI class in your current file.
# (Everything else stays the same.)

class KawaiiEyesUI:
    def __init__(self, w, h):
        self.w = int(w)
        self.h = int(h)
        self.state = "neutral"

        # --- target / smoothed gaze ---
        self.target_x = 0.0          # raw input (-1..1)
        self.gaze_x = 0.0            # smoothed
        self.gaze_goal = 0.0         # desired gaze (tracking or idle)
        self.gaze_v = 0.0            # (optional) velocity for smoothstep-ish feel

        # smoothing knobs (tune)
        self.GAZE_ALPHA = 0.12       # 0.05 smoother/slower, 0.20 faster
        self.GAZE_MAX_STEP = 0.04    # max change per frame (limit jitter)

        # blink / wink
        self.blink_t = 0.0
        self.next_blink = time.time() + random.uniform(2.0, 5.0)
        self.wink_t = 0.0
        self.wink_side = None

        # idle drift
        self.idle_x = 0.0
        self.idle_goal = 0.0
        self.next_idle_shift = time.time() + random.uniform(0.7, 1.7)

        # anim phases
        self.phase = 0.0
        self.sparkle_phase = 0.0
        self.mouth_phase = 0.0

        # extra mood
        self.extra_mood = "none"
        self.next_mood = time.time() + random.uniform(4, 9)

    def set_state(self, s: str):
        s = s or "neutral"
        if s != self.state:
            self.state = s
            if s == "happy" and random.random() < 0.35:
                self.wink_side = "L" if random.random() < 0.5 else "R"
                self.wink_t = time.time()

    def _blink_now(self, now):
        if now > self.next_blink:
            self.blink_t = now
            self.next_blink = now + random.uniform(2.2, 6.0)
        return (now - self.blink_t) < 0.12

    def _wink_side_now(self, now):
        if self.wink_side is None:
            return None
        if (now - self.wink_t) < 0.18:
            return self.wink_side
        self.wink_side = None
        return None

    def _idle_target_x(self, now):
        if now > self.next_idle_shift:
            self.next_idle_shift = now + random.uniform(0.8, 2.0)
            self.idle_goal = random.uniform(-0.8, 0.8)
        self.idle_x += (self.idle_goal - self.idle_x) * 0.06
        return float(clamp(self.idle_x, -1.0, 1.0))

    def _update_extra_mood(self, tracking: bool):
        now = time.time()
        if now > self.next_mood:
            self.next_mood = now + random.uniform(5, 12)
            if tracking and random.random() < 0.55:
                self.extra_mood = random.choice(["love", "shock", "none"])
            elif not tracking and random.random() < 0.55:
                self.extra_mood = random.choice(["sleepy", "none", "none"])
            else:
                self.extra_mood = "none"

    def _smooth_gaze(self, desired: float):
        """
        Low-pass with step limit:
          - desired: [-1..1]
          - gaze_x changes smoothly even if detection jumps
        """
        desired = float(clamp(desired, -1.0, 1.0))

        # standard low-pass
        proposed = self.gaze_x + (desired - self.gaze_x) * self.GAZE_ALPHA

        # step limit for sudden jumps (kills jitter)
        step = proposed - self.gaze_x
        if abs(step) > self.GAZE_MAX_STEP:
            proposed = self.gaze_x + self.GAZE_MAX_STEP * sign(step)

        self.gaze_x = float(clamp(proposed, -1.0, 1.0))
        return self.gaze_x

    def draw(self, canvas, target_x=0.0, tracking=False):
        canvas[:] = (0, 0, 0)
        now = time.time()

        self._update_extra_mood(tracking)

        self.phase += 0.07
        self.sparkle_phase += 0.12
        self.mouth_phase += 0.10

        # --- choose desired gaze ---
        if tracking:
            self.gaze_goal = float(clamp(target_x, -1.0, 1.0))
        else:
            self.gaze_goal = self._idle_target_x(now)

        # --- smooth it ---
        gaze = self._smooth_gaze(self.gaze_goal)

        # palette (BGR)
        if self.state == "happy":
            eye_fill = (240, 250, 255)
            outline = (170, 220, 255)
            pupil = (25, 25, 25)
            blush = (140, 120, 255)
            mouth = (230, 230, 230)
        elif self.state == "sad":
            eye_fill = (230, 235, 255)
            outline = (255, 190, 190)
            pupil = (60, 60, 110)
            blush = (120, 110, 180)
            mouth = (170, 170, 170)
        else:
            eye_fill = (255, 255, 255)
            outline = (200, 200, 200)
            pupil = (55, 55, 55)
            blush = (90, 80, 140)
            mouth = (190, 190, 190)

        cx1 = self.w // 4
        cx2 = (self.w * 3) // 4
        cy = self.h // 2

        bob = int(8 * math.sin(self.phase))
        cy_eyes = cy + bob - 20

        eye_w = 150
        eye_h = 185

        # <<< SMOOTH pupil dx uses `gaze` not raw target >>>
        pupil_dx = int(55 * gaze)
        pupil_dy = int(10 * math.sin(self.phase * 0.7))  # slightly smaller vertical wobble

        blink = self._blink_now(now)
        wink = self._wink_side_now(now)
        left_closed = blink or (wink == "L")
        right_closed = blink or (wink == "R")

        def draw_closed(cx, cy_):
            cv2.line(canvas, (cx - eye_w + 10, cy_), (cx + eye_w - 10, cy_), eye_fill, 10)
            cv2.line(canvas, (cx - 40, cy_ - 10), (cx - 10, cy_ - 4), eye_fill, 6)

        def draw_open(cx, cy_):
            cv2.ellipse(canvas, (cx, cy_), (eye_w, eye_h), 0, 0, 360, eye_fill, -1)
            cv2.ellipse(canvas, (cx, cy_), (eye_w, eye_h), 0, 0, 360, outline, 3)

            px = cx + pupil_dx
            py = cy_ + pupil_dy

            # special pupils
            if self.extra_mood == "love":
                cv2.circle(canvas, (px - 10, py), 18, (0, 0, 255), -1)
                cv2.circle(canvas, (px + 10, py), 18, (0, 0, 255), -1)
                cv2.ellipse(canvas, (px, py + 10), (26, 18), 0, 0, 180, (0, 0, 255), -1)
            elif self.extra_mood == "shock":
                cv2.circle(canvas, (px, py), 44, pupil, -1)
                cv2.circle(canvas, (px, py), 18, (255, 255, 255), -1)
            else:
                cv2.circle(canvas, (px, py), 38, pupil, -1)

            # highlights
            s1 = int(10 + 4 * (0.5 + 0.5 * math.sin(self.sparkle_phase)))
            s2 = int(6 + 3 * (0.5 + 0.5 * math.cos(self.sparkle_phase)))
            cv2.circle(canvas, (px - 14, py - 14), s1, (255, 255, 255), -1)
            cv2.circle(canvas, (px + 12, py - 20), s2, (255, 255, 255), -1)

            if self.extra_mood == "sleepy":
                overlay = canvas.copy()
                cv2.ellipse(overlay, (cx, cy_ - 20), (eye_w, eye_h), 0, 0, 360, (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

        if left_closed:
            draw_closed(cx1, cy_eyes)
        else:
            draw_open(cx1, cy_eyes)

        if right_closed:
            draw_closed(cx2, cy_eyes)
        else:
            draw_open(cx2, cy_eyes)

        # blush
        blush_alpha = 0.55 if self.state == "happy" else (0.25 if self.state == "neutral" else 0.18)
        overlay = canvas.copy()
        r = 38
        cv2.circle(overlay, (cx1 + 135, cy_eyes + 95), r, blush, -1)
        cv2.circle(overlay, (cx2 - 135, cy_eyes + 95), r, blush, -1)
        cv2.addWeighted(overlay, blush_alpha, canvas, 1.0 - blush_alpha, 0, canvas)

        # mouth
        mx = self.w // 2
        my = cy + 210 + bob
        if self.extra_mood == "shock":
            cv2.circle(canvas, (mx, my), 10, mouth, 3)
        elif self.state == "happy":
            w_amp = int(10 + 4 * (0.5 + 0.5 * math.sin(self.mouth_phase)))
            cv2.ellipse(canvas, (mx - 18, my), (16, w_amp), 0, 10, 170, mouth, 4)
            cv2.ellipse(canvas, (mx + 18, my), (16, w_amp), 0, 10, 170, mouth, 4)
        elif self.state == "sad":
            cv2.ellipse(canvas, (mx, my + 10), (26, 14), 0, 200, 340, mouth, 4)
        else:
            cv2.circle(canvas, (mx, my), 6, mouth, -1)

        label = "TRACKING!" if tracking else "IDLE..."
        cv2.putText(canvas, label, (20, self.h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)

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

    shp = sess.get_inputs()[0].shape
    img_size = 320
    if isinstance(shp, list) and len(shp) == 4:
        h = shp[2]
        w = shp[3]
        if isinstance(h, int) and isinstance(w, int) and h == w:
            img_size = int(h)

    print("ONNX inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("ONNX outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])
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
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0)

def parse_output_any(outs, conf_th):
    out = np.array(outs[0])
    if out.ndim == 3:
        out = out[0]

    if out.ndim == 2 and out.shape[1] == 6:
        boxes = out[:, 0:4].astype(np.float32)
        scores = out[:, 4].astype(np.float32)
        cls = out[:, 5].astype(np.int32)
        keep = scores > conf_th
        return boxes[keep], scores[keep], cls[keep]

    if out.ndim == 2:
        if out.shape[0] < out.shape[1]:
            out = out.transpose(1, 0)
        if out.shape[1] < 6:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32))

        boxes_xywh = out[:, 0:4].astype(np.float32)
        scores_all = out[:, 4:].astype(np.float32)
        cls = np.argmax(scores_all, axis=1).astype(np.int32)
        scores = scores_all[np.arange(scores_all.shape[0]), cls].astype(np.float32)

        keep = scores > conf_th
        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        cls = cls[keep]

        boxes = np.zeros((boxes_xywh.shape[0], 4), dtype=np.float32)
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
        return boxes, scores, cls

    return (np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32))


# =========================================================
# MAIN
# =========================================================
def main():
    if not os.path.isfile(ONNX_PATH):
        raise RuntimeError(f"ONNX model not found: {ONNX_PATH}")

    if TTS_ENABLE:
        if not os.path.isfile(PIPER_MODEL):
            print("[WARN] Piper model not found:", PIPER_MODEL)
        elif not os.path.isfile(PIPER_CONFIG):
            print("[WARN] Piper config json not found (still OK):", PIPER_CONFIG)

        if not (_cmd_exists("pw-play") or _cmd_exists("paplay") or _cmd_exists("aplay")):
            print("[WARN] No audio player found. Install pipewire-audio/pipewire-pulse or alsa-utils.")

    # start TTS worker
    global _tts_worker_started
    if not _tts_worker_started:
        _tts_worker_started = True
        threading.Thread(target=_tts_worker_loop, daemon=True).start()

    cam = CamThread(CAM_INDEX, FRAME_W, FRAME_H, CAM_FPS)
    sess, in_name, out_names, yolo_img = create_ort_session(ONNX_PATH)
    print(f"Neo loaded. ONNX input size: {yolo_img}")

    if SHOW_UI:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, UI_W, UI_H)
        cv2.moveWindow(WINDOW_NAME, 0, 0)
        if FULLSCREEN:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    ui = KawaiiEyesUI(UI_W, UI_H)
    face_img = np.zeros((UI_H, UI_W, 3), dtype=np.uint8)

    speak_response("idle")

    last_seen = 0.0
    tracking = False

    frame_id = 0
    ui_fps = 0.0
    infer_fps = 0.0
    prev_ui = time.time()
    prev_inf = time.time()

    next_idle_t = time.time() + random.uniform(IDLE_CHATTER_MIN_S, IDLE_CHATTER_MAX_S)

    try:
        while True:
            now = time.time()
            frame_id += 1

            ok, frame = cam.read()
            if not ok or frame is None:
                stop_all()
                ui.set_state("neutral")
                ui.draw(face_img, 0.0, tracking=False)
                if SHOW_UI:
                    cv2.imshow(WINDOW_NAME, face_img)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        speak_response("quit")
                        break
                continue

            h0, w0 = frame.shape[:2]
            person_boxes = []
            target_dir = 0.0
            target_box = None

            if (frame_id % INFER_EVERY_N_FRAMES) == 0:
                img, scale, pad_w, pad_h = letterbox(frame, (yolo_img, yolo_img))
                blob = to_blob(img)
                outs = sess.run(out_names, {in_name: blob})
                boxes_lb, scores, cls = parse_output_any(outs, CONF_TH)

                if frame_id % DEBUG_PRINT_EVERY == 0:
                    a = np.array(outs[0])
                    print(f"[DBG] out0 shape={a.shape} conf_th={CONF_TH} raw_det={len(boxes_lb)}")

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

                            person_boxes.append((float(x1), float(y1), float(x2), float(y2)))

                dt_inf = now - prev_inf
                if dt_inf > 0:
                    infer_fps = 0.9 * infer_fps + 0.1 * (1.0 / dt_inf) if infer_fps > 0 else (1.0 / dt_inf)
                prev_inf = now

            if person_boxes:
                areas = [box_area_xyxy(b) for b in person_boxes]
                best_i = int(np.argmax(np.array(areas)))
                x1, y1, x2, y2 = person_boxes[best_i]
                target_box = (x1, y1, x2, y2)

                cx = (x1 + x2) * 0.5
                err_x = (cx - (w0 * 0.5)) / (w0 * 0.5)
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

                if err_x != 0.0 and abs(turn) < TURN_MIN:
                    turn = TURN_MIN * sign(err_x)
                turn = clamp(turn, -TURN_LIMIT, TURN_LIMIT)

                left = apply_min(base + turn)
                right = apply_min(base - turn)

                set_motor(motor_a, left)
                set_motor(motor_b, right)

                last_seen = now
                if not tracking:
                    tracking = True
                    speak_response("found")
                ui.set_state("happy")

                next_idle_t = now + random.uniform(IDLE_CHATTER_MIN_S, IDLE_CHATTER_MAX_S)

                if frame_id % DEBUG_PRINT_EVERY == 0:
                    print(f"[DBG] persons={len(person_boxes)} best_area_norm={area_norm:.3f} err_x={err_x:.2f} base={base:.2f} turn={turn:.2f}")

            else:
                if tracking and (now - last_seen) > TARGET_LOST_GRACE:
                    tracking = False
                    stop_all()
                    speak_response("lost")
                    ui.set_state("sad")
                    next_idle_t = now + random.uniform(IDLE_CHATTER_MIN_S, IDLE_CHATTER_MAX_S)
                elif not tracking:
                    stop_all()
                    ui.set_state("neutral")
                    next_idle_t = maybe_idle_chatter(tracking=False, next_idle_t=next_idle_t)

            ui.draw(face_img, target_dir, tracking=tracking)

            if SHOW_UI:
                pip = frame.copy()
                cv2.line(pip, (w0 // 2, 0), (w0 // 2, h0), (0, 255, 255), 2)

                if DRAW_ALL_PERSONS:
                    for b in person_boxes:
                        x1, y1, x2, y2 = map(int, b)
                        cv2.rectangle(pip, (x1, y1), (x2, y2), (0, 200, 0), 2)

                if target_box is not None:
                    x1, y1, x2, y2 = map(int, target_box)
                    cv2.rectangle(pip, (x1, y1), (x2, y2), (0, 255, 255), 3)

                pip_small = cv2.resize(pip, (PIP_W, PIP_H), interpolation=cv2.INTER_LINEAR)

                # --- place PIP bottom-right with padding ---
                x0 = UI_W - PIP_W - PIP_PAD
                y0 = UI_H - PIP_H - PIP_PAD

                # safety clamp (just in case)
                x0 = max(0, min(UI_W - PIP_W, x0))
                y0 = max(0, min(UI_H - PIP_H, y0))

                face_img[y0:y0 + PIP_H, x0:x0 + PIP_W] = pip_small

                # optional: add a thin border so it looks neat
                cv2.rectangle(face_img, (x0 - 2, y0 - 2), (x0 + PIP_W + 2, y0 + PIP_H + 2), (255, 255, 255), 2)

                dt_ui = now - prev_ui
                if dt_ui > 0:
                    ui_fps = 0.9 * ui_fps + 0.1 * (1.0 / dt_ui) if ui_fps > 0 else (1.0 / dt_ui)
                prev_ui = now

                status = "TRACK" if tracking else "IDLE"
                cv2.putText(face_img, f"{status} | UI {ui_fps:.1f} | INFER {infer_fps:.1f} | CONF {CONF_TH}",
                            (10, UI_H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2)

                if EXPRESSIONS_ENABLE:
                    cv2.putText(face_img, f"mood: {ui.state} | extra: {ui.extra_mood}",
                                (10, UI_H - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2)

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

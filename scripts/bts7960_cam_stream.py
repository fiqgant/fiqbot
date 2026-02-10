import time
import threading
from dataclasses import dataclass

import cv2
from flask import Flask, request, jsonify, Response

from gpiozero import PWMOutputDevice, Device
from gpiozero.pins.pigpio import PiGPIOFactory

# =========================
# FORCE gpiozero WITHOUT lgpio
# =========================
Device.pin_factory = PiGPIOFactory()  # requires pigpiod running

# =========================
# CONFIG
# =========================
HOST = "0.0.0.0"
PORT = 5000

# Camera
CAM_INDEX = 0
FRAME_W = 640
FRAME_H = 480
JPEG_QUALITY = 80

# BTS7960 pins (BCM)
L_RPWM = 18
L_LPWM = 23
R_RPWM = 13
R_LPWM = 24

PWM_FREQ = 200
MAX_SPEED = 0.85
TURN_RATIO = 0.75
WATCHDOG_TIMEOUT = 0.6

# =========================
# Helpers
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# =========================
# Motor driver
# =========================
class BTS7960:
    def __init__(self, rpwm_pin, lpwm_pin, freq=200):
        self.rpwm = PWMOutputDevice(rpwm_pin, frequency=freq, initial_value=0.0)
        self.lpwm = PWMOutputDevice(lpwm_pin, frequency=freq, initial_value=0.0)

    def set_speed(self, s):
        s = clamp(s, -1.0, 1.0)
        if s >= 0:
            self.rpwm.value = s
            self.lpwm.value = 0.0
        else:
            self.rpwm.value = 0.0
            self.lpwm.value = -s

    def stop(self):
        self.rpwm.value = 0.0
        self.lpwm.value = 0.0

left_motor = BTS7960(L_RPWM, L_LPWM, PWM_FREQ)
right_motor = BTS7960(R_RPWM, R_LPWM, PWM_FREQ)

def drive_tank(speed, turn):
    speed = clamp(speed, -1.0, 1.0)
    turn = clamp(turn, -1.0, 1.0)

    left = clamp(speed - turn, -MAX_SPEED, MAX_SPEED)
    right = clamp(speed + turn, -MAX_SPEED, MAX_SPEED)

    left_motor.set_speed(left)
    right_motor.set_speed(right)

def stop_all():
    left_motor.stop()
    right_motor.stop()

# =========================
# State + Watchdog
# =========================
@dataclass
class ControlState:
    speed: float = 0.35
    last_cmd_ts: float = 0.0
    last_cmd: str = "stop"

state = ControlState()
lock = threading.Lock()

def touch(cmd):
    with lock:
        state.last_cmd = cmd
        state.last_cmd_ts = time.time()

def watchdog_loop():
    while True:
        time.sleep(0.05)
        with lock:
            last = state.last_cmd_ts
        if last and (time.time() - last > WATCHDOG_TIMEOUT):
            stop_all()

threading.Thread(target=watchdog_loop, daemon=True).start()

# =========================
# Camera thread
# =========================
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

frame_lock = threading.Lock()
latest_frame = None
camera_ok = False

def camera_loop():
    global latest_frame, camera_ok
    while True:
        ok, frame = cap.read()
        if not ok:
            camera_ok = False
            time.sleep(0.1)
            continue
        camera_ok = True
        with frame_lock:
            latest_frame = frame

threading.Thread(target=camera_loop, daemon=True).start()

def mjpeg_generator():
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        ok, jpg = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

# =========================
# Flask app
# =========================
app = Flask(__name__)

@app.get("/")
def index():
    return """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rover Control</title>
  <style>
    body { font-family: system-ui, Arial; margin: 14px; }
    .wrap { max-width: 760px; margin: auto; }
    .cam { width: 100%; border-radius: 14px; border:1px solid #ccc; }
    .grid { display:grid; grid-template-columns: 1fr 1fr 1fr; gap:10px; max-width:420px; }
    button { padding: 16px 12px; font-size: 18px; border-radius: 12px; border:1px solid #ccc; background:#f7f7f7; }
    button:active { transform: scale(0.98); }
    input[type=range] { width: 100%; max-width:420px; }
    .status { margin-top:10px; padding:10px; border-radius:12px; background:#f2f2ff; }
  </style>
</head>
<body>
<div class="wrap">
  <h2>Rover Tank Control + Camera</h2>
  <img class="cam" src="/video" />

  <div style="margin:12px 0;">
    <label>Speed: <span id="spdVal">0.35</span></label><br>
    <input id="spd" type="range" min="0" max="1" step="0.01" value="0.35">
  </div>

  <div class="grid">
    <div></div>
    <button onpointerdown="send('forward')" onpointerup="send('stop')" onpointercancel="send('stop')">▲ Forward</button>
    <div></div>

    <button onpointerdown="send('left')" onpointerup="send('stop')" onpointercancel="send('stop')">◀ Left</button>
    <button onclick="send('stop')">■ Stop</button>
    <button onpointerdown="send('right')" onpointerup="send('stop')" onpointercancel="send('stop')">Right ▶</button>

    <div></div>
    <button onpointerdown="send('back')" onpointerup="send('stop')" onpointercancel="send('stop')">▼ Back</button>
    <div></div>
  </div>

  <div class="status" id="status">Status: -</div>
</div>

<script>
  const spd = document.getElementById('spd');
  const spdVal = document.getElementById('spdVal');
  const statusBox = document.getElementById('status');

  spd.addEventListener('input', async () => {
    spdVal.textContent = spd.value;
    await fetch('/speed', {
      method:'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({speed: parseFloat(spd.value)})
    });
  });

  async function send(cmd) {
    const r = await fetch('/cmd', {
      method:'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({cmd})
    });
    const j = await r.json();
    statusBox.textContent = `Status: ${j.cmd} | speed=${j.speed.toFixed(2)} | camera=${j.camera_ok}`;
  }

  setInterval(async () => {
    const r = await fetch('/status');
    const j = await r.json();
    statusBox.textContent = `Status: ${j.last_cmd} | speed=${j.speed.toFixed(2)} | camera=${j.camera_ok}`;
  }, 500);
</script>
</body>
</html>
"""

@app.get("/video")
def video():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
def status():
    with lock:
        spd = state.speed
        last_cmd = state.last_cmd
        age = time.time() - state.last_cmd_ts if state.last_cmd_ts else None
    return jsonify(speed=spd, last_cmd=last_cmd, last_cmd_age_s=age, camera_ok=camera_ok)

@app.post("/speed")
def api_speed():
    data = request.get_json(force=True, silent=True) or {}
    with lock:
        state.speed = clamp(float(data.get("speed", 0.35)), 0.0, 1.0)
    return jsonify(ok=True, speed=state.speed)

@app.post("/cmd")
def api_cmd():
    data = request.get_json(force=True, silent=True) or {}
    cmd = str(data.get("cmd", "stop")).lower()

    with lock:
        spd = clamp(state.speed, 0.0, 1.0)

    if cmd == "forward":
        drive_tank(spd, 0.0)
    elif cmd == "back":
        drive_tank(-spd, 0.0)
    elif cmd == "left":
        drive_tank(0.0, -spd * TURN_RATIO)
    elif cmd == "right":
        drive_tank(0.0, spd * TURN_RATIO)
    else:
        cmd = "stop"
        stop_all()

    touch(cmd)
    return jsonify(ok=True, cmd=cmd, speed=spd, camera_ok=camera_ok)

if __name__ == "__main__":
    stop_all()
    print(f"Open: http://<IP_RPI>:{PORT}")
    try:
        app.run(host=HOST, port=PORT, debug=False, threaded=True)
    finally:
        stop_all()
        cap.release()

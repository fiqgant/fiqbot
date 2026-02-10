import time
import threading
from flask import Flask, request, jsonify, Response
import cv2

from gpiozero import DigitalOutputDevice

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
# LEFT
L_RPWM = 18
L_LPWM = 23
# RIGHT
R_RPWM = 13
R_LPWM = 24

# Safety watchdog
WATCHDOG_TIMEOUT = 0.6  # seconds

# =========================
# MOTOR DRIVER (NO PWM)
# =========================
class BTS7960_ONOFF:
    """
    Control BTS7960 using ON/OFF only:
    - Forward: RPWM=1, LPWM=0
    - Reverse: RPWM=0, LPWM=1
    - Stop:    RPWM=0, LPWM=0
    """
    def __init__(self, rpwm_pin: int, lpwm_pin: int):
        self.rpwm = DigitalOutputDevice(rpwm_pin, initial_value=False)
        self.lpwm = DigitalOutputDevice(lpwm_pin, initial_value=False)

    def forward(self):
        self.rpwm.on()
        self.lpwm.off()

    def reverse(self):
        self.rpwm.off()
        self.lpwm.on()

    def stop(self):
        self.rpwm.off()
        self.lpwm.off()

left_motor = BTS7960_ONOFF(L_RPWM, L_LPWM)
right_motor = BTS7960_ONOFF(R_RPWM, R_LPWM)

def stop_all():
    left_motor.stop()
    right_motor.stop()

def cmd_forward():
    left_motor.forward()
    right_motor.forward()

def cmd_back():
    left_motor.reverse()
    right_motor.reverse()

def cmd_left_spin():
    # spin left: left reverse, right forward
    left_motor.reverse()
    right_motor.forward()

def cmd_right_spin():
    left_motor.forward()
    right_motor.reverse()

# =========================
# WATCHDOG
# =========================
last_cmd_ts = 0.0
last_cmd = "stop"
lock = threading.Lock()

def touch(cmd: str):
    global last_cmd_ts, last_cmd
    with lock:
        last_cmd = cmd
        last_cmd_ts = time.time()

def watchdog_loop():
    while True:
        time.sleep(0.05)
        with lock:
            ts = last_cmd_ts
        if ts and (time.time() - ts > WATCHDOG_TIMEOUT):
            stop_all()

threading.Thread(target=watchdog_loop, daemon=True).start()

# =========================
# CAMERA THREAD
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

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

# =========================
# FLASK APP
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
    .status { margin-top:10px; padding:10px; border-radius:12px; background:#f2f2ff; }
    .hint { font-size: 13px; color:#555; margin-top:6px; }
  </style>
</head>
<body>
<div class="wrap">
  <h2>Rover Control + Camera (No PWM)</h2>
  <img class="cam" src="/video" />
  <div class="hint">Tekan & tahan tombol untuk jalan, lepas = stop.</div>

  <div class="grid" style="margin-top:12px;">
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
  const statusBox = document.getElementById('status');

  async function send(cmd) {
    const r = await fetch('/cmd', {
      method:'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({cmd})
    });
    const j = await r.json();
    statusBox.textContent = `Status: ${j.cmd} | camera=${j.camera_ok}`;
  }

  setInterval(async () => {
    const r = await fetch('/status');
    const j = await r.json();
    statusBox.textContent = `Status: ${j.last_cmd} | camera=${j.camera_ok}`;
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
        cmd = last_cmd
        age = time.time() - last_cmd_ts if last_cmd_ts else None
    return jsonify(last_cmd=cmd, last_cmd_age_s=age, camera_ok=camera_ok)

@app.post("/cmd")
def api_cmd():
    data = request.get_json(force=True, silent=True) or {}
    cmd = str(data.get("cmd", "stop")).lower()

    if cmd == "forward":
        cmd_forward()
    elif cmd == "back":
        cmd_back()
    elif cmd == "left":
        cmd_left_spin()
    elif cmd == "right":
        cmd_right_spin()
    else:
        cmd = "stop"
        stop_all()

    touch(cmd)
    return jsonify(ok=True, cmd=cmd, camera_ok=camera_ok)

if __name__ == "__main__":
    stop_all()
    print(f"Open: http://<IP_RPI>:{PORT}")
    try:
        app.run(host=HOST, port=PORT, debug=False, threaded=True)
    finally:
        stop_all()
        cap.release()

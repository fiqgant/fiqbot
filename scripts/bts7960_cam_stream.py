import time
import threading
from dataclasses import dataclass

import cv2
from flask import Flask, request, jsonify, Response
from gpiozero import PWMOutputDevice, Device
from gpiozero.pins.lgpio import LGPIOFactory

# =========================
# FORCE SOFTWARE PWM (PENTING UNTUK RPI 5)
# =========================
Device.pin_factory = LGPIOFactory(pwm=True)

# =========================
# CONFIG
# =========================
HOST = "0.0.0.0"
PORT = 5000

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
# HELPERS
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# =========================
# MOTOR DRIVER
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
    left = clamp(speed - turn, -MAX_SPEED, MAX_SPEED)
    right = clamp(speed + turn, -MAX_SPEED, MAX_SPEED)
    left_motor.set_speed(left)
    right_motor.set_speed(right)

def stop_all():
    left_motor.stop()
    right_motor.stop()

# =========================
# STATE + WATCHDOG
# =========================
@dataclass
class ControlState:
    speed: float = 0.35
    last_cmd_ts: float = 0.0
    last_cmd: str = "stop"

state = ControlState()
lock = threading.Lock()

def watchdog():
    while True:
        time.sleep(0.05)
        with lock:
            if state.last_cmd_ts and time.time() - state.last_cmd_ts > WATCHDOG_TIMEOUT:
                stop_all()

threading.Thread(target=watchdog, daemon=True).start()

def touch(cmd):
    with lock:
        state.last_cmd = cmd
        state.last_cmd_ts = time.time()

# =========================
# CAMERA THREAD
# =========================
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

latest_frame = None
frame_lock = threading.Lock()
camera_ok = False

def cam_loop():
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

threading.Thread(target=cam_loop, daemon=True).start()

def mjpeg():
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue
        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpg.tobytes() + b"\r\n")

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

@app.route("/")
def index():
    return """
    <h2>Rover Control</h2>
    <img src="/video"><br><br>
    <button onmousedown="send('forward')" onmouseup="send('stop')">Forward</button>
    <button onmousedown="send('left')" onmouseup="send('stop')">Left</button>
    <button onclick="send('stop')">Stop</button>
    <button onmousedown="send('right')" onmouseup="send('stop')">Right</button>
    <button onmousedown="send('back')" onmouseup="send('stop')">Back</button>
    <script>
      async function send(cmd){
        await fetch('/cmd', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify({cmd})
        });
      }
    </script>
    """

@app.route("/video")
def video():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.post("/cmd")
def cmd():
    data = request.json
    cmd = data.get("cmd", "stop")
    spd = state.speed

    if cmd == "forward":
        drive_tank(spd, 0)
    elif cmd == "back":
        drive_tank(-spd, 0)
    elif cmd == "left":
        drive_tank(0, -spd * TURN_RATIO)
    elif cmd == "right":
        drive_tank(0, spd * TURN_RATIO)
    else:
        stop_all()
        cmd = "stop"

    touch(cmd)
    return jsonify(ok=True, cmd=cmd)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    stop_all()
    print("Open: http://IP_RPI:5000")
    app.run(host=HOST, port=PORT)

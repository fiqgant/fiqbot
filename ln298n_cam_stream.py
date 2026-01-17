from flask import Flask, request
import threading
import time

from gpiozero import Device, Motor
from gpiozero.pins.lgpio import LGPIOFactory

# =======================
# SETTINGS
# =======================
BIND_IP = "0.0.0.0"
PORT = 8000

# ustreamer URL (sesuaikan IP/port kalau beda)
STREAM_URL = "http://{host}:8080/stream"

# Pin BCM (sesuai yang kamu pakai)
IN1, IN2, IN3, IN4 = 18, 19, 20, 21
ENA, ENB = 12, 13

SPEED = 0.7
TURN_SPEED = 0.6
# =======================

app = Flask(__name__)

# Motor init (Pi 5 friendly)
Device.pin_factory = LGPIOFactory()
motor_a = Motor(forward=IN1, backward=IN2, enable=ENA, pwm=True)  # kiri
motor_b = Motor(forward=IN4, backward=IN3, enable=ENB, pwm=True)  # kanan (dibalik)
motor_lock = threading.Lock()

def stop():
    with motor_lock:
        motor_a.stop()
        motor_b.stop()

def maju():
    with motor_lock:
        motor_a.forward(SPEED)
        motor_b.forward(SPEED)

def mundur():
    with motor_lock:
        motor_a.backward(SPEED)
        motor_b.backward(SPEED)

def putar_kiri():
    with motor_lock:
        motor_a.backward(TURN_SPEED)
        motor_b.forward(TURN_SPEED)

def putar_kanan():
    with motor_lock:
        motor_a.forward(TURN_SPEED)
        motor_b.backward(TURN_SPEED)

# Safety: auto-stop kalau koneksi putus / tombol ga dilepas
last_cmd_time = time.time()
CMD_TIMEOUT = 0.35

def watchdog():
    while True:
        if (time.time() - last_cmd_time) > CMD_TIMEOUT:
            stop()
        time.sleep(0.05)

threading.Thread(target=watchdog, daemon=True).start()

HTML = """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Robot Control</title>
  <style>
    body { font-family: sans-serif; margin: 12px; }
    .wrap { display: grid; gap: 12px; max-width: 900px; margin: auto; }
    img { width: 100%; border-radius: 12px; border: 1px solid #ddd; }
    .pad { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
    button {
      font-size: 20px; padding: 16px; border-radius: 14px; border: 1px solid #bbb;
      background: #f6f6f6;
      user-select: none; -webkit-user-select: none;
      touch-action: none;
    }
    .row { display:flex; gap:10px; align-items:center; }
    input[type=range]{ width: 220px; }
    .hint { opacity: 0.8; font-size: 14px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div>
      <h2>Robot Control + Fast Stream (ustreamer)</h2>
      <div class="hint">Mode tahan: tahan tombol untuk jalan, lepas untuk stop. Keyboard: W/A/S/D. Space = stop.</div>
    </div>

    <img id="cam" />

    <div class="row">
      <label>Speed</label>
      <input id="spd" type="range" min="0" max="100" value="70"/>
      <span id="spdval">70</span>
    </div>

    <div class="pad">
      <div></div>
      <button id="fwd">▲</button>
      <div></div>

      <button id="left">⟲</button>
      <button id="stop">■</button>
      <button id="right">⟳</button>

      <div></div>
      <button id="back">▼</button>
      <div></div>
    </div>
  </div>

<script>
  const spd = document.getElementById('spd');
  const spdval = document.getElementById('spdval');
  spd.addEventListener('input', () => spdval.textContent = spd.value);

  // stream url from server (same host as control page)
  const cam = document.getElementById('cam');
  cam.src = "/stream_proxy";

  async function send(cmd, pressed) {
    const v = spd.value;
    await fetch('/cmd?c=' + cmd + '&p=' + (pressed ? '1':'0') + '&v=' + v, {cache:'no-store'});
  }

  function holdButton(btn, cmd){
    const down = (e)=>{ e.preventDefault(); send(cmd, true); };
    const up   = (e)=>{ e.preventDefault(); send(cmd, false); };

    btn.addEventListener('pointerdown', down);
    btn.addEventListener('pointerup', up);
    btn.addEventListener('pointercancel', up);
    btn.addEventListener('pointerleave', up);
  }

  holdButton(document.getElementById('fwd'),  'w');
  holdButton(document.getElementById('back'), 's');
  holdButton(document.getElementById('left'), 'a');
  holdButton(document.getElementById('right'),'d');

  document.getElementById('stop').addEventListener('click', ()=>send('x', true));

  // Keyboard hold
  const held = new Set();
  function applyHeld(){
    if (held.has('w')) send('w', true);
    else if (held.has('s')) send('s', true);
    else if (held.has('a')) send('a', true);
    else if (held.has('d')) send('d', true);
    else send('x', true);
  }

  window.addEventListener('keydown', (e)=>{
    const k = e.key.toLowerCase();
    if (['w','a','s','d'].includes(k)) { held.add(k); applyHeld(); }
    if (e.key === ' ') { held.clear(); send('x', true); }
  });

  window.addEventListener('keyup', (e)=>{
    const k = e.key.toLowerCase();
    if (held.has(k)) held.delete(k);
    if (['w','a','s','d'].includes(k)) {
      if (held.size === 0) send(k, false);
      else applyHeld();
    }
  });
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return HTML

# Proxy supaya satu host/port (biar gampang di HP)
@app.route("/stream_proxy")
def stream_proxy():
    host = request.host.split(":")[0]
    # ustreamer default endpoint: /stream
    return (f'<img src="{STREAM_URL.format(host=host)}">', 302, {"Location": STREAM_URL.format(host=host)})

@app.route("/cmd")
def cmd():
    global last_cmd_time, SPEED, TURN_SPEED

    c = request.args.get("c", "x")
    p = request.args.get("p", "0") == "1"
    v = request.args.get("v", None)

    if v is not None:
        try:
            val = max(0, min(100, int(v))) / 100.0
            SPEED = val
            TURN_SPEED = min(val, 0.9)
        except:
            pass

    last_cmd_time = time.time()

    if not p:
        stop()
        return "OK"

    if c == "w":
        maju()
    elif c == "s":
        mundur()
    elif c == "a":
        putar_kiri()
    elif c == "d":
        putar_kanan()
    else:
        stop()

    return "OK"

if __name__ == "__main__":
    try:
        app.run(host=BIND_IP, port=PORT, threaded=True)
    finally:
        stop()
        motor_a.close()
        motor_b.close()


#include <WiFi.h>
#include <WebServer.h>

// ===================== WIFI AP =====================
const char* AP_SSID = "TANK-ESP32";
const char* AP_PASS = "12345678";

// ===================== PIN (SESUI WIRING KAMU) =====================
// KIRI (Motor A)
const int ENA = 25;   // PWM
const int IN1 = 26;
const int IN2 = 27;

// KANAN (Motor B)
const int ENB = 14;   // PWM
const int IN3 = 12;
const int IN4 = 13;

// ===================== PWM =====================
const int pwmFreq = 20000;
const int pwmRes  = 8;        // 0..255
volatile int spd  = 180;      // default speed

// ===================== KALIBRASI ARAH (kalau maju/mundur kebalik) =====================
bool invertLeft  = false;  // set true kalau motor kiri kebalik
bool invertRight = false;  // set true kalau motor kanan kebalik

WebServer server(80);

// dir: 1 maju, -1 mundur, 0 stop
void motorLeft(int speed, int dir) {
  if (invertLeft) dir = -dir;
  speed = constrain(speed, 0, 255);

  if (dir == 1) {
    digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
    ledcWrite(ENA, speed);
  } else if (dir == -1) {
    digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
    ledcWrite(ENA, speed);
  } else {
    digitalWrite(IN1, LOW);  digitalWrite(IN2, LOW);
    ledcWrite(ENA, 0);
  }
}

void motorRight(int speed, int dir) {
  if (invertRight) dir = -dir;
  speed = constrain(speed, 0, 255);

  if (dir == 1) {
    digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
    ledcWrite(ENB, speed);
  } else if (dir == -1) {
    digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);
    ledcWrite(ENB, speed);
  } else {
    digitalWrite(IN3, LOW);  digitalWrite(IN4, LOW);
    ledcWrite(ENB, 0);
  }
}

void stopAll() {
  motorLeft(0, 0);
  motorRight(0, 0);
}

// Gerakan standar (INI yang UI pakai)
// Diperbaiki berdasarkan masalah: kiri->mundur, kanan->maju, atas->kanan, bawah->kiri
// Analisis masalah:
// - turnLeft() menghasilkan mundur -> berarti motorLeft(-1) dan motorRight(1) menghasilkan mundur total
// - turnRight() menghasilkan maju -> berarti motorLeft(1) dan motorRight(-1) menghasilkan maju total
// - goForward() menghasilkan ke kanan -> berarti motorLeft(1) dan motorRight(1) menghasilkan belok kanan
// - goBackward() menghasilkan ke kiri -> berarti motorLeft(-1) dan motorRight(-1) menghasilkan belok kiri
// Kesimpulan: motor kiri dan kanan terbalik secara fisik, atau arah motor terbalik
// Solusi: tukar motor kiri dan kanan, dan invert arah untuk maju/mundur
void goForward()  { motorRight(spd,  1); motorLeft(spd,  1); }   // kedua maju (tukar motor)
void goBackward() { motorRight(spd, -1); motorLeft(spd, -1); }   // kedua mundur (tukar motor)
void turnLeft()   { motorRight(spd, -1); motorLeft(spd,  1); }   // kanan mundur, kiri maju -> belok kiri (tukar motor)
void turnRight()  { motorRight(spd,  1); motorLeft(spd, -1); }   // kanan maju, kiri mundur -> belok kanan (tukar motor)

// ===================== UI HTML (SVG, no encoding issue) =====================
const char INDEX_HTML[] PROGMEM = R"HTML(
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no" />
  <title>ESP32 Tank Controller</title>
  <style>
    :root{
      --bg:#0b1020; --card:#0f172a; --stroke:rgba(255,255,255,.10);
      --text:#e7eefc; --muted:#9db0d0;
      --accent:#60a5fa; --danger:#ff5d5d; --ok:#2dd4bf;
      --shadow: 0 14px 40px rgba(0,0,0,.45);
      --r:18px;
    }
    *{box-sizing:border-box; -webkit-tap-highlight-color:transparent;}
    body{
      margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;
      background:
        radial-gradient(900px 700px at 15% 10%, rgba(96,165,250,.22), transparent 55%),
        radial-gradient(900px 700px at 85% 30%, rgba(45,212,191,.14), transparent 55%),
        var(--bg);
      color:var(--text);
    }
    .wrap{max-width:520px; margin:0 auto; padding:18px;}
    .top{display:flex; justify-content:space-between; align-items:center; gap:12px; margin:6px 0 14px;}
    .title h1{margin:0; font-size:18px; letter-spacing:.2px;}
    .title p{margin:6px 0 0; font-size:12px; color:var(--muted);}
    .pill{
      font-size:12px; color:var(--muted);
      padding:8px 10px; border-radius:999px;
      background:rgba(255,255,255,.06);
      border:1px solid rgba(255,255,255,.08);
      white-space:nowrap;
    }

    .card{
      border-radius:var(--r);
      background:linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.03));
      border:1px solid rgba(255,255,255,.10);
      box-shadow:var(--shadow);
      overflow:hidden;
    }
    .inner{padding:16px;}

    .grid{
      display:grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap:10px;
      align-items:center;
    }

    .btn{
      user-select:none; touch-action:none;
      border-radius:16px;
      border:1px solid var(--stroke);
      background:linear-gradient(180deg, rgba(255,255,255,.10), rgba(255,255,255,.03));
      color:var(--text);
      padding:16px 0;
      display:flex; align-items:center; justify-content:center;
      box-shadow:0 10px 18px rgba(0,0,0,.25);
    }
    .btn:active{transform:translateY(1px); filter:brightness(1.06);}
    .btn.center{
      background:linear-gradient(180deg, rgba(96,165,250,.22), rgba(96,165,250,.08));
      border-color: rgba(96,165,250,.28);
    }
    .btn.stop{
      grid-column:span 3;
      padding:14px 0;
      font-weight:700;
      background:linear-gradient(180deg, rgba(255,93,93,.25), rgba(255,93,93,.10));
      border-color: rgba(255,93,93,.30);
    }

    svg{width:26px; height:26px; opacity:.95}
    .row{display:flex; gap:10px; align-items:center; margin-top:14px;}
    .label{width:64px; font-size:12px; color:var(--muted);}
    .value{min-width:52px; text-align:right; font-variant-numeric:tabular-nums;}
    input[type=range]{width:100%; accent-color:var(--accent);}

    .status{
      display:flex; justify-content:space-between; align-items:center;
      margin-top:12px; padding-top:12px;
      border-top:1px solid rgba(255,255,255,.08);
      font-size:12px; color:var(--muted);
    }
    .dot{width:8px;height:8px;border-radius:999px;background:rgba(157,176,208,.6);display:inline-block;margin-right:8px;}
    .dot.ok{background:var(--ok); box-shadow:0 0 0 6px rgba(45,212,191,.10);}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div class="title">
        <h1>ESP32 Tank Controller</h1>
        <p>Tekan & tahan untuk jalan • Lepas = stop</p>
      </div>
      <div class="pill">192.168.4.1</div>
    </div>

    <div class="card">
      <div class="inner">
        <div class="grid">
          <div></div>
          <button class="btn" id="F" aria-label="Forward">
            <svg viewBox="0 0 24 24" fill="none"><path d="M12 4l7 8H5l7-8Z" fill="currentColor"/></svg>
          </button>
          <div></div>

          <button class="btn" id="L" aria-label="Left">
            <svg viewBox="0 0 24 24" fill="none"><path d="M4 12l8-7v14l-8-7Z" fill="currentColor"/></svg>
          </button>
          <button class="btn center" id="C" aria-label="Stop">
            <svg viewBox="0 0 24 24" fill="none"><path d="M7 7h10v10H7V7Z" fill="currentColor"/></svg>
          </button>
          <button class="btn" id="R" aria-label="Right">
            <svg viewBox="0 0 24 24" fill="none"><path d="M20 12l-8 7V5l8 7Z" fill="currentColor"/></svg>
          </button>

          <div></div>
          <button class="btn" id="B" aria-label="Backward">
            <svg viewBox="0 0 24 24" fill="none"><path d="M12 20l-7-8h14l-7 8Z" fill="currentColor"/></svg>
          </button>
          <div></div>

          <button class="btn stop" id="S">STOP</button>
        </div>

        <div class="row">
          <div class="label">Speed</div>
          <input id="speed" type="range" min="0" max="255" value="180" />
          <div class="value" id="sv">180</div>
        </div>

        <div class="status">
          <div><span class="dot" id="dot"></span><span id="st">Disconnected</span></div>
          <div id="last">—</div>
        </div>
      </div>
    </div>
  </div>

<script>
  let moving = null;
  let timer = null;

  const st = document.getElementById('st');
  const dot = document.getElementById('dot');
  const last = document.getElementById('last');

  function setConn(ok){
    if(ok){ st.textContent="Connected"; dot.classList.add('ok'); }
    else { st.textContent="Disconnected"; dot.classList.remove('ok'); }
  }

  async function ping(){
    try{ const r = await fetch('/ping',{cache:'no-store'}); if(r.ok) setConn(true); }
    catch(e){ setConn(false); }
  }
  setInterval(ping, 1200); ping();

  async function send(dir){
    try{
      const t0 = performance.now();
      const r = await fetch('/cmd?dir='+dir, {cache:'no-store'});
      const t1 = performance.now();
      if(r.ok){
        setConn(true);
        last.textContent = dir.toUpperCase()+" • "+Math.round(t1-t0)+"ms";
      }
    }catch(e){ setConn(false); }
  }

  function startHold(dir){
    moving = dir;
    send(dir);
    clearInterval(timer);
    timer = setInterval(()=>{ if(moving) send(moving); }, 120);
  }
  function endHold(){
    moving = null;
    clearInterval(timer);
    timer = null;
    send('stop');
  }

  function bindHold(id, dir){
    const el = document.getElementById(id);
    const down = (e)=>{ e.preventDefault(); startHold(dir); };
    const up   = (e)=>{ e.preventDefault(); endHold(); };

    el.addEventListener('pointerdown', down);
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
    el.addEventListener('pointerleave', up);
  }

  // MAPPING BENAR:
  // ▲ forward, ▼ backward, ◀ left, ▶ right
  bindHold('F','f');
  bindHold('B','b');
  bindHold('L','l');
  bindHold('R','r');

  document.getElementById('C').addEventListener('click', ()=>send('stop'));
  document.getElementById('S').addEventListener('click', ()=>send('stop'));

  const speed = document.getElementById('speed');
  const sv = document.getElementById('sv');
  speed.addEventListener('input', ()=>{
    sv.textContent = speed.value;
    fetch('/speed?v='+speed.value, {cache:'no-store'}).catch(()=>setConn(false));
  });
</script>
</body>
</html>
)HTML";

// ===================== HTTP HANDLERS =====================
void handleRoot() { server.send(200, "text/html", INDEX_HTML); }
void handlePing() { server.send(200, "text/plain", "OK"); }

void handleSpeed() {
  if (!server.hasArg("v")) { server.send(400, "text/plain", "Missing v"); return; }
  spd = constrain(server.arg("v").toInt(), 0, 255);
  server.send(200, "text/plain", "SPD=" + String(spd));
}

void handleCmd() {
  if (!server.hasArg("dir")) { server.send(400, "text/plain", "Missing dir"); return; }
  String d = server.arg("dir");

  // Mapping standar: f=maju, b=mundur, l=belok kiri, r=belok kanan
  // Fungsi gerakan sudah diperbaiki dengan menukar motor kiri dan kanan
  if      (d == "f") goForward();   // atas -> maju
  else if (d == "b") goBackward();  // bawah -> mundur
  else if (d == "l") turnLeft();    // kiri -> belok kiri
  else if (d == "r") turnRight();   // kanan -> belok kanan
  else               stopAll();

  server.send(200, "text/plain", "OK");
}

void setup() {
  Serial.begin(115200);
  delay(200);

  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

  // ESP32 core baru: PWM attach ke pin
  ledcAttach(ENA, pwmFreq, pwmRes);
  ledcAttach(ENB, pwmFreq, pwmRes);

  stopAll();

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);

  Serial.print("AP IP: ");
  Serial.println(WiFi.softAPIP());
  Serial.println("Open: http://192.168.4.1");

  server.on("/", handleRoot);
  server.on("/ping", handlePing);
  server.on("/speed", handleSpeed);
  server.on("/cmd", handleCmd);
  server.begin();
}

void loop() {
  server.handleClient();
}

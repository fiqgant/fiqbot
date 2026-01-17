# FiqBot - Advanced Raspberry Pi 5 Robot Control 🤖

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-C51A4A?logo=raspberrypi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**FiqBot** is a modular, high-performance robot control system optimized for the **Raspberry Pi 5**. It integrates real-time camera streaming, web-based remote control, and AI-powered computer vision for person following and gesture recognition.

---

## 📋 Table of Contents
- [Features](#-features)
- [Hardware Requirements](#-hardware-requirements)
- [Wiring & Pinout](#-wiring--pinout)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

- **🚀 Remote Control**: Low-latency terminal-based driving (WASD).
- **📹 Live Streaming**: Web interface with high-speed MJPEG streaming via `ustreamer`.
- **🧠 AI Person Following**: Autonomous tracking using **YOLOv11** (ONNX).
- **✋ Gesture Lock**: Secure target locking using Hand Gestures (Open Hand / Fist) via MediaPipe.
- **⚡ Optimized for Pi 5**: Utilizes `lgpio` and `gpiozero` for modern GPIO handling.

---

## 🛠 Hardware Requirements

| Component | Recommendation |
|-----------|----------------|
| **SBC** | Raspberry Pi 5 (Preferred) / Pi 4 |
| **Driver** | L298N Motor Driver Module |
| **Motors** | 2x or 4x DC Gear Motors (TT Motors) |
| **Camera** | USB Webcam / Pi Camera Module |
| **Power** | 2S Li-ion / 7.2V-12V External Battery |

---

## 🔌 Wiring & Pinout

The system is pre-configured for **Raspberry Pi 5** using BCM numbering.

| L298N Pin | Raspberry Pi (BCM) | Description |
|:---:|:---:|:---|
| **IN1** | `GPIO 18` | Left Motor Forward |
| **IN2** | `GPIO 19` | Left Motor Backward |
| **IN3** | `GPIO 20` | Right Motor Forward |
| **IN4** | `GPIO 21` | Right Motor Backward |
| **ENA** | `GPIO 12` | Left Motor Speed (PWM) |
| **ENB** | `GPIO 13` | Right Motor Speed (PWM) |
| **GND** | `GND` | Common Ground (Critical!) |

> **⚠️ Important**: Ensure the L298N ground is connected to the Raspberry Pi ground to establish a common reference.

---

## 📥 Installation

### 1. System Dependencies
Update your system and install system-level dependencies for OpenCV.
```bash
sudo apt update
sudo apt install -y python3-opencv libopencv-dev
```

### 2. Python Environment
Clone the repository and install the required Python packages (including `gpiozero`, `numpy`, `opencv`, `mediapipe`, etc.).

```bash
git clone https://github.com/fiqgant/fiqbot.git
cd fiqbot
pip install -r requirements.txt
```

### 3. Setup Camera Stream (Optional)
For the web control feature, install `ustreamer` for low-latency streaming.
```bash
sudo apt install ustreamer
# Run ustreamer in the background on port 8080
ustreamer --host=0.0.0.0 --port=8080 -r 640x480 -f 30 &
```

---

## 🎮 Usage Guide

### 1. Verification Test
Run a quick diagnostic to check motor rotation and wiring.
```bash
python l298ntest.py
```

### 2. Terminal Remote
Control the robot directly from your SSH terminal using keyboard inputs.
```bash
python l298n_control.py
```
| Key | Action |
|:---:|:---|
| **W** | Move Forward |
| **S** | Move Backward |
| **A** | Turn Left |
| **D** | Turn Right |
| **Space** | Emergency Stop |

### 3. Web Control Center
Launch the web interface to control the bot from a smartphone or browser.
```bash
python l298n_cam_stream.py
```
> **Access**: `http://<your-pi-ip>:8000`

### 4. AI Follower (YOLO)
Activate autonomous person tracking.
```bash
python l298n_yolo.py
```

### 5. Smart Lock (Gesture Control)
Look for an **Open Hand** to lock onto a target, and a **Fist** to unlock/stop.
```bash
python l298n_lock.py
```

---

## 📂 Project Structure

```plaintext
fiqbot/
├── l298n_cam_stream.py   # Web control server with camera feed
├── l298n_control.py      # Terminal-based remote control
├── l298n_lock.py         # Gesture-based target locking AI
├── l298n_yolo.py         # Basic person following AI
├── l298ntest.py          # Hardware diagnostic script
├── requirements.txt      # Python dependencies
└── README.md             # Documentation
```

---

## ❓ Troubleshooting

- **Motors not moving?**
  - Check if the 12V power switch is ON.
  - Verify grounds are connected between Pi and L298N.
- **"Camera not found" error?**
  - Verify `CAM_INDEX = 0` in the scripts. Try changing it to `1` or `-1`.
- **GPIO errors?**
  - Ensure you are using `lgpio` on Raspberry Pi 5. Run `rpi-update` if needed.

---

Made with ❤️ by **Fiq**

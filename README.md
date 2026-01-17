# FiqBot - Raspberry Pi 5 Robot Control System

A modular Python-based robot control system designed for Raspberry Pi 5 using the L298N motor driver. This project features multiple control modes including terminal remote, web-based control with camera streaming, and AI-powered person following with gesture locking.

## Hardware Requirements

- **Raspberry Pi 5** (Compatible with Pi 4/Zero with updated pin config)
- **L298N Motor Driver Module**
- **DC Motors + Chassis** (2WD or 4WD)
- **USB Webcam** (for AI and Streaming features)
- **Power Supply** (Separate power recommended for motors)

## Wiring / Pinout

The following wiring is configured for **Raspberry Pi 5** (BCM Pin numbering):

| L298N Pin | Raspberry Pi Pin (BCM) | Function                  |
|:---------:|:----------------------:|:-------------------------:|
| **IN1**   | GPIO 18                | Motor A (Left) Direction  |
| **IN2**   | GPIO 19                | Motor A (Left) Direction  |
| **IN3**   | GPIO 20                | Motor B (Right) Direction |
| **IN4**   | GPIO 21                | Motor B (Right) Direction |
| **ENA**   | GPIO 12                | Motor A Speed (PWM)       |
| **ENB**   | GPIO 13                | Motor B Speed (PWM)       |
| **GND**   | GND                    | Common Ground             |

> **Note**: This project uses `gpiozero` and `lgpio` which is the recommended GPIO library for Raspberry Pi 5.

## Installation

1. **Install System Dependencies** (if needed for OpenCV/MediaPipe):
   ```bash
   sudo apt update
   sudo apt install -y python3-opencv
   ```

2. **Install Python Libraries**:
   ```bash
   pip install gpiozero lgpio opencv-python numpy onnxruntime mediapipe flask
   ```

   *Note: For the web stream feature, you may need `ustreamer` installed and running on port 8080.*

3. **Download Model File**:
   Ensure `yolo11n.onnx` is present in the directory for AI features.

## Usage Guide

### 1. Basic Motor Test
Run a simple sequence to verify wiring and motor rotation direction.
```bash
python l298ntest.py
```

### 2. Terminal Remote Control
Control the robot using your keyboard (WASD).
```bash
python l298n_control.py
```
- **W/S**: Forward / Backward
- **A/D**: Spot Turn Left / Right
- **X**: Stop
- **Q**: Quit

### 3. Web Control + Camera Stream
Starts a Flask web server for controlling the robot from a phone or browser.
```bash
python l298n_cam_stream.py
```
- Access via `http://<your-pi-ip>:8000`
- **Requires**: `ustreamer` running on port 8080 for the video feed.

### 4. AI Person Follow
Uses YOLOv11 (ONNX) to detect and follow a person.
```bash
python l298n_yolo.py
```

### 5. Gesture Lock + Follow
Advanced mode that "locks" onto a target using a hand gesture (Open Hand to lock, Fist to unlock).
```bash
python l298n_lock.py
```

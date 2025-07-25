import cv2
import mediapipe as mp
import Jetson.GPIO as GPIO
import time
import os

# === Text-to-Speech ===
def speak(text):
    os.system(f'espeak "{text}" --stdout | aplay')

# === GPIO Motor Setup ===
GPIO.setmode(GPIO.BOARD)
IN1, IN2, IN3, IN4 = 21, 22, 23, 24
ENA, ENB = 32, 33
motor_pins = [IN1, IN2, IN3, IN4, ENA, ENB]

for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)
pwmA.start(70)
pwmB.start(70)

# === Motor Control Functions ===
def move_forward():
    speak("Moving forward")
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def turn_left():
    speak("Turning left")
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def turn_right():
    speak("Turning right")
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

def stop():
    speak("Stopping")
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

# === Camera & Pose Estimation ===
cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# === Tracking Variables ===
target_x, target_y = None, None
tracking_tolerance = 80
turn_tolerance = 50
last_seen_time = time.time()

# === Voice Status Flags ===
last_speak_no_person = 0
no_person_interval = 5
target_locked = False
target_lost = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture camera")
            break

        h, w, _ = frame.shape
        center_screen = w // 2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

            x_center = int(((left_hip.x + right_hip.x) / 2) * w)
            y_center = int(((left_hip.y + right_hip.y) / 2) * h)
            body_height = abs(left_shoulder.y - left_hip.y)

            # Lock target if not set
            if not target_locked:
                target_x, target_y = x_center, y_center
                target_locked = True
                target_lost = False
                speak("Person detected, following")
                print("Target locked at:", target_x, target_y)

            # Check if current person is still the target
            if abs(x_center - target_x) < tracking_tolerance and abs(y_center - target_y) < tracking_tolerance:
                last_seen_time = time.time()

                if body_height < 0.06:
                    print("Too far, moving forward")
                    move_forward()
                elif body_height > 0.08:
                    print("Too close, stopping")
                    stop()
                else:
                    if x_center < center_screen - turn_tolerance:
                        print("Turning left")
                        turn_left()
                    elif x_center > center_screen + turn_tolerance:
                        print("Turning right")
                        turn_right()
                    else:
                        print("Straight ahead")
                        move_forward()
            else:
                print("Person is not target, ignoring")
                stop()

            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        else:
            print("No person detected")
            stop()

            # Handle lost target
            if target_locked and (time.time() - last_seen_time > 5):
                if not target_lost:
                    speak("Target lost")
                    print("Target lost. Resetting.")
                    target_x, target_y = None, None
                    target_locked = False
                    target_lost = True

            # Periodic voice feedback if no one is around
            if not target_locked:
                now = time.time()
                if now - last_speak_no_person > no_person_interval:
                    speak("No person detected")
                    last_speak_no_person = now

        # Show camera frame
        cv2.imshow("Robot Follower", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()


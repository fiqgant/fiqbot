#!/usr/bin/env python3
"""
Hand Gesture Controlled JetBot
Controls robot using hand gestures:
- Open palm = robot follows hand position
- Closed fist = robot stops
"""

import cv2
import numpy as np
import time
import sys
import os

# Add jetbot to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jetbot.robot import Robot
from jetbot.camera import Camera

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Error: mediapipe not installed. Install with: pip3 install mediapipe")
    sys.exit(1)


class HandGestureController:
    def __init__(self, camera_width=640, camera_height=480):
        self.robot = Robot()
        self.camera = Camera(width=camera_width, height=camera_height)
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Control parameters
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.center_x = camera_width // 2
        self.center_y = camera_height // 2
        self.dead_zone_x = 60  # Dead zone in pixels
        self.dead_zone_y = 60
        self.max_speed = 0.6  # Maximum motor speed (0.0 to 1.0)
        self.min_speed = 0.25  # Minimum motor speed to start moving
        self.kp_x = 0.004  # Proportional gain for X-axis (horizontal)
        self.kp_y = 0.003  # Proportional gain for Y-axis (vertical)
        
        # State
        self.last_hand_time = time.time()
        self.hand_timeout = 1.5  # Stop if no hand detected for 1.5 seconds
        self.running = True
        self.current_gesture = "None"
        
        print("Hand Gesture Controller initialized")
        print("Controls:")
        print("  - Open palm: Robot follows your hand")
        print("  - Closed fist: Robot stops")
        print("  - Press 'q' to quit")
    
    def count_fingers(self, landmarks):
        """Count how many fingers are up"""
        # Landmark indices for finger tips and joints
        # Thumb: 4 (tip), 3 (joint)
        # Index: 8 (tip), 6 (joint)
        # Middle: 12 (tip), 10 (joint)
        # Ring: 16 (tip), 14 (joint)
        # Pinky: 20 (tip), 18 (joint)
        
        fingers = []
        
        # Thumb (check x coordinate for left/right hand)
        if landmarks[self.mp_hands.HandLandmark.THUMB_TIP].x > landmarks[self.mp_hands.HandLandmark.THUMB_IP].x:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers (check y coordinate)
        finger_tips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        finger_pips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y < landmarks[pip].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return sum(fingers)
    
    def detect_gesture(self, landmarks):
        """Detect if hand is open (palm) or closed (fist)"""
        fingers_up = self.count_fingers(landmarks)
        
        # Open palm: 4-5 fingers up
        # Closed fist: 0-1 fingers up
        if fingers_up >= 4:
            return "open"
        elif fingers_up <= 1:
            return "closed"
        else:
            return "partial"
    
    def get_hand_center(self, landmarks):
        """Get center point of hand (wrist)"""
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        x = int(wrist.x * self.camera_width)
        y = int(wrist.y * self.camera_height)
        return x, y
    
    def calculate_control(self, hand_x, hand_y):
        """Calculate motor control based on hand position"""
        # Calculate error (distance from frame center)
        error_x = hand_x - self.center_x
        error_y = hand_y - self.center_y
        
        # Calculate motor speeds
        # Left/Right control (X-axis)
        if abs(error_x) < self.dead_zone_x:
            left_speed = 0.0
            right_speed = 0.0
        else:
            # Turn left if hand is on the left (negative error_x)
            # Turn right if hand is on the right (positive error_x)
            control_x = error_x * self.kp_x
            base_speed = self.min_speed + abs(control_x) * self.max_speed
            base_speed = min(base_speed, self.max_speed)
            
            if error_x < 0:  # Hand on left, turn left
                left_speed = -base_speed
                right_speed = base_speed
            else:  # Hand on right, turn right
                left_speed = base_speed
                right_speed = -base_speed
        
        # Forward/Backward control (Y-axis)
        if abs(error_y) < self.dead_zone_y:
            # No forward/backward movement if in dead zone
            pass
        else:
            control_y = error_y * self.kp_y
            forward_speed = abs(control_y) * self.max_speed * 0.5  # Slower forward/backward
            
            if error_y < 0:  # Hand above center, move forward
                left_speed = (left_speed + forward_speed) if left_speed >= 0 else (left_speed - forward_speed)
                right_speed = (right_speed + forward_speed) if right_speed >= 0 else (right_speed - forward_speed)
            else:  # Hand below center, move backward
                left_speed = (left_speed - forward_speed) if left_speed >= 0 else (left_speed + forward_speed)
                right_speed = (right_speed - forward_speed) if right_speed >= 0 else (right_speed + forward_speed)
        
        # Clamp speeds
        left_speed = max(-self.max_speed, min(self.max_speed, left_speed))
        right_speed = max(-self.max_speed, min(self.max_speed, right_speed))
        
        return left_speed, right_speed
    
    def run(self):
        """Main loop"""
        try:
            while self.running:
                frame = self.camera.read()
                
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                # Draw center crosshair
                cv2.line(frame, (self.center_x - 30, self.center_y), 
                        (self.center_x + 30, self.center_y), (0, 255, 0), 2)
                cv2.line(frame, (self.center_x, self.center_y - 30), 
                        (self.center_x, self.center_y + 30), (0, 255, 0), 2)
                
                # Draw dead zone circle
                cv2.circle(frame, (self.center_x, self.center_y), 
                          self.dead_zone_x, (0, 255, 0), 1)
                
                if results.multi_hand_landmarks:
                    self.last_hand_time = time.time()
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                        
                        # Detect gesture
                        gesture = self.detect_gesture(hand_landmarks.landmark)
                        self.current_gesture = gesture
                        
                        # Get hand center
                        hand_x, hand_y = self.get_hand_center(hand_landmarks.landmark)
                        
                        # Draw hand center
                        cv2.circle(frame, (hand_x, hand_y), 10, (255, 0, 255), -1)
                        
                        if gesture == "open":
                            # Calculate control and move robot
                            left_speed, right_speed = self.calculate_control(hand_x, hand_y)
                            self.robot.set_motors(left_speed, right_speed)
                            
                            # Display info
                            cv2.putText(frame, "OPEN PALM - Following", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(frame, f"L: {left_speed:.2f} R: {right_speed:.2f}", 
                                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Draw line from center to hand
                            cv2.line(frame, (self.center_x, self.center_y), 
                                    (hand_x, hand_y), (0, 255, 0), 2)
                        
                        elif gesture == "closed":
                            # Stop robot
                            self.robot.stop()
                            
                            cv2.putText(frame, "CLOSED FIST - STOPPED", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        else:
                            # Partial gesture - stop for safety
                            self.robot.stop()
                            cv2.putText(frame, "PARTIAL - STOPPED", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                else:
                    # No hand detected
                    elapsed = time.time() - self.last_hand_time
                    if elapsed > self.hand_timeout:
                        self.robot.stop()
                        cv2.putText(frame, "No hand - Stopped", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, f"Show your hand... ({elapsed:.1f}s)", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Display gesture status
                cv2.putText(frame, f"Gesture: {self.current_gesture.upper()}", 
                          (10, self.camera_height - 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow('Hand Gesture Controller', frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                time.sleep(0.03)  # ~30 FPS
        
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.running = False
        self.robot.stop()
        self.robot.cleanup()
        self.camera.release()
        self.hands.close()
        cv2.destroyAllWindows()
        print("Done")


def enable_pwm_pins():
    """Enable PWM pins on Jetson Nano"""
    print("Enabling PWM pins...")
    try:
        import subprocess
        # Enable Pin 32 / PWM0
        subprocess.run(['busybox', 'devmem', '0x700031fc', '32', '0x45'], check=True)
        subprocess.run(['busybox', 'devmem', '0x6000d504', '32', '0x2'], check=True)
        
        # Enable Pin 33 / PWM2
        subprocess.run(['busybox', 'devmem', '0x70003248', '32', '0x46'], check=True)
        subprocess.run(['busybox', 'devmem', '0x6000d100', '32', '0x00'], check=True)
        print("PWM pins enabled successfully")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not enable PWM pins: {e}")
        print("You may need to run this script with sudo")
    except FileNotFoundError:
        print("Warning: busybox not found. Make sure you're running on Jetson Nano")


if __name__ == '__main__':
    # Enable PWM pins (requires sudo)
    enable_pwm_pins()
    
    # Create and run hand gesture controller
    controller = HandGestureController()
    controller.run()


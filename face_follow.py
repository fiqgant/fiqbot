#!/usr/bin/env python3
"""
Face Following JetBot
Detects faces using OpenCV and controls the robot to follow them.
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


class FaceFollower:
    def __init__(self, camera_width=640, camera_height=480):
        self.robot = Robot()
        self.camera = Camera(width=camera_width, height=camera_height)
        
        # Load face detection model (Haar Cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("Error: Could not load face cascade classifier")
            sys.exit(1)
        
        # Control parameters - dynamically set based on camera resolution
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.center_x = camera_width // 2
        self.center_y = camera_height // 2
        self.dead_zone_x = 50  # Dead zone in pixels (no movement if face is within this)
        self.dead_zone_y = 50
        self.max_speed = 0.5  # Maximum motor speed (0.0 to 1.0)
        self.min_speed = 0.2  # Minimum motor speed to start moving
        self.kp_x = 0.003  # Proportional gain for X-axis (horizontal)
        self.kp_y = 0.002  # Proportional gain for Y-axis (vertical)
        
        # State
        self.last_face_time = time.time()
        self.face_timeout = 2.0  # Stop if no face detected for 2 seconds
        self.running = True
        
        print("Face Follower initialized")
        print("Press 'q' to quit")
    
    def detect_face(self, frame):
        """Detect faces in the frame and return the largest one"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Return the largest face (by area)
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            return largest_face
        return None
    
    def calculate_control(self, face_rect):
        """Calculate motor control based on face position"""
        x, y, w, h = face_rect
        
        # Calculate face center
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Calculate error (distance from frame center)
        error_x = face_center_x - self.center_x
        error_y = face_center_y - self.center_y
        
        # Calculate control signals
        control_x = error_x * self.kp_x
        control_y = error_y * self.kp_y
        
        # Calculate motor speeds
        # Left/Right control (X-axis)
        if abs(error_x) < self.dead_zone_x:
            left_speed = 0.0
            right_speed = 0.0
        else:
            # Turn left if face is on the left (negative error_x)
            # Turn right if face is on the right (positive error_x)
            base_speed = self.min_speed + abs(control_x) * self.max_speed
            base_speed = min(base_speed, self.max_speed)
            
            if error_x < 0:  # Face on left, turn left
                left_speed = -base_speed
                right_speed = base_speed
            else:  # Face on right, turn right
                left_speed = base_speed
                right_speed = -base_speed
        
        # Forward/Backward control (Y-axis) - optional
        # Uncomment if you want the robot to move forward/backward based on face size
        # For now, we'll just track horizontally
        
        return left_speed, right_speed, face_center_x, face_center_y
    
    def run(self):
        """Main loop"""
        try:
            while self.running:
                frame = self.camera.read()
                
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Detect face
                face = self.detect_face(frame)
                
                # Draw center crosshair
                cv2.line(frame, (self.center_x - 20, self.center_y), 
                        (self.center_x + 20, self.center_y), (0, 255, 0), 2)
                cv2.line(frame, (self.center_x, self.center_y - 20), 
                        (self.center_x, self.center_y + 20), (0, 255, 0), 2)
                
                if face is not None:
                    self.last_face_time = time.time()
                    x, y, w, h = face
                    
                    # Calculate control
                    left_speed, right_speed, face_x, face_y = self.calculate_control(face)
                    
                    # Control robot
                    self.robot.set_motors(left_speed, right_speed)
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.circle(frame, (face_x, face_y), 5, (0, 0, 255), -1)
                    
                    # Display info
                    cv2.putText(frame, f"Face detected", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"L: {left_speed:.2f} R: {right_speed:.2f}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # No face detected
                    elapsed = time.time() - self.last_face_time
                    if elapsed > self.face_timeout:
                        self.robot.stop()
                        cv2.putText(frame, "No face - Stopped", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, f"Searching... ({elapsed:.1f}s)", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Face Follower', frame)
                
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
    
    # Create and run face follower
    follower = FaceFollower()
    follower.run()


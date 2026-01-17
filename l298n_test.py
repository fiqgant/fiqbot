from gpiozero import Device, Motor
from gpiozero.pins.lgpio import LGPIOFactory
import time

Device.pin_factory = LGPIOFactory()

# BCM Pin Configuration
IN1, IN2, IN3, IN4 = 18, 19, 20, 21
ENA, ENB = 12, 13

# Motor Initialization
# Right motor is inverted to ensure correct forward direction
motor_left = Motor(forward=IN1, backward=IN2, enable=ENA, pwm=True)
motor_right = Motor(forward=IN4, backward=IN3, enable=ENB, pwm=True)

speed = 0.7
turn_speed = 0.6
turn_duration = 0.7  # Adjust validation turn duration (e.g., 0.3-1.2s)

def stop():
    motor_left.stop()
    motor_right.stop()

def move_forward(v=speed):
    motor_left.forward(v)
    motor_right.forward(v)

def move_backward(v=speed):
    motor_left.backward(v)
    motor_right.backward(v)

def spin_left(v=turn_speed):
    # Left motor backward, Right motor forward -> Spin Left
    motor_left.backward(v)
    motor_right.forward(v)

def spin_right(v=turn_speed):
    # Left motor forward, Right motor backward -> Spin Right
    motor_left.forward(v)
    motor_right.backward(v)

try:
    print("Testing: Forward...")
    move_forward()
    time.sleep(2)
    stop()
    time.sleep(0.5)

    print("Testing: Spin Left...")
    spin_left()
    time.sleep(turn_duration)
    stop()
    time.sleep(0.5)

    print("Testing: Spin Right...")
    spin_right()
    time.sleep(turn_duration)
    stop()
    time.sleep(0.5)

    print("Testing: Backward...")
    move_backward()
    time.sleep(2)

finally:
    stop()
    motor_left.close()
    motor_right.close()
    print("Test Complete.")

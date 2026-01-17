from gpiozero import Device, Motor
from gpiozero.pins.lgpio import LGPIOFactory
import time
import curses

Device.pin_factory = LGPIOFactory()

# BCM Pin Configuration
IN1, IN2, IN3, IN4 = 18, 19, 20, 21
ENA, ENB = 12, 13

# Motor Initialization
# Right motor is inverted to match chassis direction
motor_left = Motor(forward=IN1, backward=IN2, enable=ENA, pwm=True)
motor_right = Motor(forward=IN4, backward=IN3, enable=ENB, pwm=True)

speed = 0.7
turn_speed = 0.6

def stop():
    motor_left.stop()
    motor_right.stop()

def move_forward():
    motor_left.forward(speed)
    motor_right.forward(speed)

def move_backward():
    motor_left.backward(speed)
    motor_right.backward(speed)

def spin_left():
    motor_left.backward(turn_speed)
    motor_right.forward(turn_speed)

def spin_right():
    motor_left.forward(turn_speed)
    motor_right.backward(turn_speed)

def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)   # Non-blocking input
    stdscr.keypad(True)
    stdscr.addstr(0, 0, "Hold Mode (Terminal): Hold W/A/S/D to move | X to Stop | Q to Quit")
    stdscr.refresh()

    last_key_time = 0
    held = None  # 'w','a','s','d' or None
    timeout_release = 0.12  # Threshold to detect key release (no repeat signal)

    while True:
        ch = stdscr.getch()
        now = time.time()

        if ch != -1:
            # Convert to lowercase
            try:
                k = chr(ch).lower()
            except ValueError:
                k = ""

            if k == "q":
                break
            elif k == "x":
                held = None
                stop()
            elif k in ("w", "a", "s", "d"):
                held = k
                last_key_time = now

                if held == "w":
                    move_forward()
                elif held == "s":
                    move_backward()
                elif held == "a":
                    spin_left()
                elif held == "d":
                    spin_right()

        # Detect key release via timeout (lack of repeat signal)
        if held is not None and (now - last_key_time) > timeout_release:
            held = None
            stop()

        time.sleep(0.01)

try:
    curses.wrapper(main)
finally:
    stop()
    motor_left.close()
    motor_right.close()

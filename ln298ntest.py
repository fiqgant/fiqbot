from gpiozero import Device, Motor
from gpiozero.pins.lgpio import LGPIOFactory
import time

Device.pin_factory = LGPIOFactory()

IN1, IN2, IN3, IN4 = 18, 19, 20, 21
ENA, ENB = 12, 13

# Motor kanan dibalik (sesuai wiring kamu)
motor_a = Motor(forward=IN1, backward=IN2, enable=ENA, pwm=True)      # kiri
motor_b = Motor(forward=IN4, backward=IN3, enable=ENB, pwm=True)      # kanan

speed = 0.7
turn_speed = 0.6
turn_time = 0.7  # atur ini biar pas derajat putarnya (mis. 0.3-1.2)

def stop():
    motor_a.stop()
    motor_b.stop()

def maju(v=speed):
    motor_a.forward(v)
    motor_b.forward(v)

def mundur(v=speed):
    motor_a.backward(v)
    motor_b.backward(v)

def putar_kiri(v=turn_speed):
    # kiri mundur, kanan maju -> spin left
    motor_a.backward(v)
    motor_b.forward(v)

def putar_kanan(v=turn_speed):
    # kiri maju, kanan mundur -> spin right
    motor_a.forward(v)
    motor_b.backward(v)

try:
    maju()
    time.sleep(2)
    stop()
    time.sleep(0.5)

    putar_kiri()
    time.sleep(turn_time)
    stop()
    time.sleep(0.5)

    putar_kanan()
    time.sleep(turn_time)
    stop()
    time.sleep(0.5)

    mundur()
    time.sleep(2)

finally:
    stop()
    motor_a.close()
    motor_b.close()

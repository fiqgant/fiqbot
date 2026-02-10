import time
from gpiozero import DigitalOutputDevice

# BCM pins
L_RPWM, L_LPWM = 18, 23
R_RPWM, R_LPWM = 13, 24

l_r = DigitalOutputDevice(L_RPWM, initial_value=False)
l_l = DigitalOutputDevice(L_LPWM, initial_value=False)
r_r = DigitalOutputDevice(R_RPWM, initial_value=False)
r_l = DigitalOutputDevice(R_LPWM, initial_value=False)

def stop_all():
    l_r.off(); l_l.off(); r_r.off(); r_l.off()

def left_forward():
    l_r.on(); l_l.off()

def right_forward():
    r_r.on(); r_l.off()

def left_reverse():
    l_r.off(); l_l.on()

def right_reverse():
    r_r.off(); r_l.on()

stop_all()
time.sleep(1)

print("LEFT forward 2s")
left_forward()
time.sleep(2)
stop_all()
time.sleep(1)

print("RIGHT forward 2s")
right_forward()
time.sleep(2)
stop_all()
time.sleep(1)

print("RIGHT reverse 2s")
right_reverse()
time.sleep(2)
stop_all()
print("DONE")

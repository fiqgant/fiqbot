from gpiozero import Device, Motor
from gpiozero.pins.lgpio import LGPIOFactory
import time
import curses

Device.pin_factory = LGPIOFactory()

IN1, IN2, IN3, IN4 = 18, 19, 20, 21
ENA, ENB = 12, 13

motor_a = Motor(forward=IN1, backward=IN2, enable=ENA, pwm=True)  # kiri
motor_b = Motor(forward=IN4, backward=IN3, enable=ENB, pwm=True)  # kanan (dibalik)

speed = 0.7
turn_speed = 0.6

def stop():
    motor_a.stop()
    motor_b.stop()

def maju():
    motor_a.forward(speed)
    motor_b.forward(speed)

def mundur():
    motor_a.backward(speed)
    motor_b.backward(speed)

def putar_kiri():
    motor_a.backward(turn_speed)
    motor_b.forward(turn_speed)

def putar_kanan():
    motor_a.forward(turn_speed)
    motor_b.backward(turn_speed)

def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)   # non-blocking
    stdscr.keypad(True)
    stdscr.addstr(0, 0, "Mode tahan (terminal): tahan W/A/S/D | X stop | Q keluar")
    stdscr.refresh()

    last_key_time = 0
    held = None  # 'w','a','s','d' or None
    timeout_release = 0.12  # kalau tidak ada repeat input, anggap dilepas

    while True:
        ch = stdscr.getch()
        now = time.time()

        if ch != -1:
            # konversi ke huruf kecil
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
                    maju()
                elif held == "s":
                    mundur()
                elif held == "a":
                    putar_kiri()
                elif held == "d":
                    putar_kanan()

        # ?lepas tombol? di terminal dideteksi dari tidak adanya repeat input
        if held is not None and (now - last_key_time) > timeout_release:
            held = None
            stop()

        time.sleep(0.01)

try:
    curses.wrapper(main)
finally:
    stop()
    motor_a.close()
    motor_b.close()

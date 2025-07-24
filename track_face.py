import cv2
import mediapipe as mp
import Jetson.GPIO as GPIO
import time

# Konfigurasi GPIO
GPIO.setmode(GPIO.BOARD)
IN1 = 21
IN2 = 22
IN3 = 23
IN4 = 24
ENA = 32
ENB = 33

motor_pins = [IN1, IN2, IN3, IN4, ENA, ENB]
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)
pwmA.start(100)
pwmB.start(100)

# Fungsi gerakan motor
def maju():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def kiri():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def kanan():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

def berhenti():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Inisialisasi MediaPipe
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal ambil video")
            break

        h, w, _ = frame.shape
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x_mid = int((bbox.xmin + bbox.width / 2) * w)

                # Garis tengah layar
                tengah = w // 2
                toleransi = 50

                # Gerak berdasarkan posisi wajah
                if x_mid < tengah - toleransi:
                    print("Belok Kiri")
                    kiri()
                elif x_mid > tengah + toleransi:
                    print("Belok Kanan")
                    kanan()
                else:
                    print("Maju")
                    maju()
        else:
            print("Tidak terdeteksi wajah. Berhenti.")
            berhenti()

        # Tampilkan video
        cv2.imshow("Tracking Manusia", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()

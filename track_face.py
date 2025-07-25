import cv2
import mediapipe as mp
import Jetson.GPIO as GPIO
import time

# === Konfigurasi GPIO ===
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

# === Fungsi Motor ===
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

# === Inisialisasi Kamera dan MediaPipe ===
cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# === Variabel Tracking Target ===
target_x, target_y = None, None
tolerance_track = 80    # toleransi posisi target
tengah = 320            # asumsi tengah frame (640x480)
toleransi_x = 50        # toleransi belok kiri/kanan
last_seen_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera gagal terbaca")
            break

        h, w, _ = frame.shape
        tengah = w // 2
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

            x_center = int(((left_hip.x + right_hip.x) / 2) * w)
            y_center = int(((left_hip.y + right_hip.y) / 2) * h)
            body_height = abs(left_shoulder.y - left_hip.y)

            # Inisialisasi target
            if target_x is None and target_y is None:
                target_x, target_y = x_center, y_center
                print(f"Target dikunci di posisi ({target_x}, {target_y})")

            # Cek apakah orang ini masih target
            if abs(x_center - target_x) < tolerance_track and abs(y_center - target_y) < tolerance_track:
                last_seen_time = time.time()
                print("Mengikuti target")

                # === Logika Jarak (berdasarkan body_height) ===
                if body_height < 0.06:
                    print("Terlalu jauh (>2m), maju")
                    maju()
                elif body_height > 0.08:
                    print("Terlalu dekat (<2m), berhenti")
                    berhenti()
                else:
                    # Arah kiri-kanan
                    if x_center < tengah - toleransi_x:
                        print("Belok kiri")
                        kiri()
                    elif x_center > tengah + toleransi_x:
                        print("Belok kanan")
                        kanan()
                    else:
                        print("Lurus")
                        maju()
            else:
                print("Orang bukan target. Berhenti.")
                berhenti()

            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            print("Tidak ada tubuh terdeteksi")
            berhenti()
            # Reset target jika tidak terlihat selama >5 detik
            if time.time() - last_seen_time > 5:
                print("Target hilang. Reset.")
                target_x, target_y = None, None

        # Tampilkan frame
        cv2.imshow("Follower Robot", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()

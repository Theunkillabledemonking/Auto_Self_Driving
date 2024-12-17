import Jetson.GPIO as GPIO
import torch
import cv2
import numpy as np
import time

# **1. GPIO 설정**
GPIO.setmode(GPIO.BOARD)

# 서보 모터 설정
servo_pin = 32  # 서보 모터 PWM 핀
GPIO.setup(servo_pin, GPIO.OUT)
servo = GPIO.PWM(servo_pin, 50)  # 50Hz PWM
servo.start(7.5)  # 초기값 (90도)

# DC 모터 설정
dir_pin = 29  # IN1
in2_pin = 31  # IN2
pwm_pin = 33  # ENA (속도 제어 핀)
GPIO.setup(dir_pin, GPIO.OUT)
GPIO.setup(in2_pin, GPIO.OUT)
GPIO.setup(pwm_pin, GPIO.OUT)
dc_motor = GPIO.PWM(pwm_pin, 1000)  # 1kHz PWM
dc_motor.start(70)  # 항상 70% 속도로 작동

# **2. 서보 모터 각도 설정 함수**
def set_servo_angle(angle):
    duty_cycle = 2.5 + (angle / 180.0) * 10
    print(f"Setting servo angle to {angle}, Duty Cycle: {duty_cycle:.2f}")
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.2)
    servo.ChangeDutyCycle(0)

# **3. DC 모터 제어 함수**
def control_dc_motor():
    # DC 모터를 항상 앞으로 70% 속도로 이동하도록 설정
    GPIO.output(dir_pin, GPIO.LOW)
    GPIO.output(in2_pin, GPIO.HIGH)
    dc_motor.ChangeDutyCycle(70)

# **4. PilotNet 모델 정의**
class PilotNet(torch.nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 24, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(24, 36, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(36, 48, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(48, 64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 1 * 18, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 5)  # 5개의 범주로 분류
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# **5. 모델 로드**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PilotNet().to(device)

try:
    model.load_state_dict(torch.load("best_pilotnet_model.pth", map_location=device))
    model.eval()
    print("모델 로드 성공!")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    exit()

# **6. 카메라 초기화**
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# **7. RC 카 제어 메인 루프**
categories = [30, 60, 90, 120, 150]  # 예측 각도 범주
try:
    print("RC 카 제어 시작")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 영상을 가져올 수 없습니다.")
            break

        # **ROI 설정 (상단 20% 제거)**
        height, width, _ = frame.shape
        roi_top = int(height * 0.2)
        roi = frame[roi_top:, :]
        print(f"ROI 크기: {roi.shape[1]}x{roi.shape[0]}")

        # **이미지 전처리**
        frame_resized = cv2.resize(roi, (200, 66))
        frame_normalized = frame_resized / 255.0
        frame_transposed = np.transpose(frame_normalized, (2, 0, 1))  # HWC → CHW
        frame_tensor = torch.tensor(frame_transposed, dtype=torch.float32).unsqueeze(0).to(device)

        # **모델 예측**
        try:
            outputs = model(frame_tensor)
            predicted_category = torch.argmax(outputs, dim=1).item()
            predicted_angle = categories[predicted_category]
            print(f"실시간 예측된 조향 각도: {predicted_angle}도")

            # **서보 모터 제어**
            set_servo_angle(predicted_angle)

            # **DC 모터 제어**
            control_dc_motor()

        except Exception as e:
            print(f"모델 예측 중 오류 발생: {e}")

        if cv2.waitKey(1) & 0xFF == 27:
            print("RC 카 제어 종료")
            break

except KeyboardInterrupt:
    print("프로그램 강제 종료")
finally:
    cap.release()
    servo.stop()
    dc_motor.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()

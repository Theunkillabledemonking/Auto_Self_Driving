import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import datetime
import Jetson.GPIO as GPIO
import time
import logging
import subprocess
# ============================
# Pilonet 모델 정의 (분류용)
# ============================
class Pilonet(nn.Module):
    def __init__(self, num_classes=5):
        super(Pilonet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, num_classes)  # 분류용 출력층
        )

    def forward(self, x):
        return self.model(x)
# ============================
# 각도 범주 설정
# ============================
categories = [30, 60, 90, 120, 150]  # 분류된 각도 범주

def process_frame(self, frame):
    # 프레임 전처리 및 모델 예측
    resized_frame = cv2.resize(frame, (200, 66))
    input_tensor = torch.tensor(resized_frame / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

    with torch.no_grad():
        outputs = self.model(input_tensor)
        _, predicted_class = torch.max(outputs, 1)  # 가장 높은 확률의 클래스 선택
        predicted_angle = categories[predicted_class.item()]  # 클래스를 각도로 변환

    self.set_servo_angle(predicted_angle)
    self.display_frame(frame, predicted_angle)

# ============================
# 메인 앱 클래스
# ============================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera and Autonomous Control")

        # GPIO 초기화
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        self.servo_pin = 33
        GPIO.setup(self.servo_pin, GPIO.OUT)
        self.servo = GPIO.PWM(self.servo_pin, 50)
        self.servo.start(0)

        # 모델 로드
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Pilonet().to(self.device)
        self.model.load_state_dict(torch.load("pilonet_model.pth", map_location=self.device))
        self.model.eval()
        print("Pilonet 모델 로드 완료!")

        # 카메라 초기화
        self.cap = cv2.VideoCapture(0)
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()
        self.camera_active = True
        self.camera_thread = CameraThread(self)
        self.camera_thread.start()

        # 현재 각도
        self.current_angle = 90

    def process_frame(self, frame):
        # 프레임 전처리 및 모델 예측
        resized_frame = cv2.resize(frame, (200, 66))
        input_tensor = torch.tensor(resized_frame / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predicted_angle = self.model(input_tensor).item()

        self.set_servo_angle(predicted_angle)

        # 화면 출력
        self.display_frame(frame, predicted_angle)

    def set_servo_angle(self, angle):
        self.current_angle = max(30, min(150, angle))
        duty_cycle = 2 + (self.current_angle / 18)
        self.servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.1)

    def display_frame(self, frame, angle):
        # 각도 및 화면에 표시
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 20)
        draw.text((10, 10), f"Predicted Angle: {int(angle)}°", fill="yellow", font=font)

        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def quit(self):
        self.camera_active = False
        self.servo.stop()
        GPIO.cleanup()
        self.root.destroy()

# ============================
# 메인 실행 함수
# ============================
def main():
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.quit)
    root.mainloop()

if __name__ == "__main__":
    main()

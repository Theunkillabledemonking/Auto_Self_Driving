import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import threading
import datetime
import Jetson.GPIO as GPIO
import time
import logging
import os


class CameraThread(threading.Thread):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.app.cap.read()
            if ret:
                self.app.process_frame(frame)
            time.sleep(0.03)  # 프레임 읽기 간격 조정

    def stop(self):
        self.running = False


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera and Motor Control")

        # 카메라 초기화
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("카메라를 열 수 없습니다.")
            exit()

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 이미지를 표시할 캔버스 생성
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        # GPIO 초기화
        self.servo_pin = 33
        self.dc_motor_pwm_pin = 32
        self.dc_motor_dir_pin1 = 29
        self.dc_motor_dir_pin2 = 31

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.servo_pin, GPIO.OUT)
        GPIO.setup(self.dc_motor_pwm_pin, GPIO.OUT)
        GPIO.setup(self.dc_motor_dir_pin1, GPIO.OUT)
        GPIO.setup(self.dc_motor_dir_pin2, GPIO.OUT)

        self.servo = GPIO.PWM(self.servo_pin, 50)
        self.dc_motor_pwm = GPIO.PWM(self.dc_motor_pwm_pin, 1000)
        self.servo.start(0)
        self.dc_motor_pwm.start(0)

        # 기본 모터 속도 및 서보 각도
        self.current_speed = 60
        self.current_servo_angle = 90

        # 키 바인딩
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.bind('<KeyRelease>', self.on_key_release)

        # 설정 초기화
        self.keys_pressed = set()
        self.setup_logging()
        self.start_forward_motion()

        # 화살표 방향 표시
        self.direction_label = tk.Label(self.root, text="", font=("Helvetica", 16))
        self.direction_label.pack(pady=10)

        # 카메라 스레드 실행
        self.camera_thread = CameraThread(self)
        self.camera_thread.start()

    def setup_logging(self):
        if not os.path.exists('logs'):
            os.makedirs('logs')
        log_filename = datetime.datetime.now().strftime('logs/log_%Y-%m-%d_%H-%M-%S.log')
        logging.basicConfig(filename=log_filename, level=logging.INFO,
                            format='%(asctime)s %(levelname)s: %(message)s')
        logging.info("로그 초기화 완료.")

    def start_forward_motion(self):
        self.set_dc_motor(self.current_speed, "forward")
        logging.info("기본 전진 상태 시작.")

    def on_key_press(self, event):
        self.keys_pressed.add(event.keysym)
        if event.keysym == 'q':
            self.quit()
        elif event.keysym == 'a':
            self.current_servo_angle = max(30, self.current_servo_angle - 30)
            self.set_servo_angle(self.current_servo_angle)
            self.direction_label.config(text="← 좌회전", fg="blue")
            logging.info(f"좌회전: 각도 {self.current_servo_angle}°")
        elif event.keysym == 'd':
            self.current_servo_angle = min(150, self.current_servo_angle + 30)
            self.set_servo_angle(self.current_servo_angle)
            self.direction_label.config(text="→ 우회전", fg="red")
            logging.info(f"우회전: 각도 {self.current_servo_angle}°")

    def on_key_release(self, event):
        if event.keysym in self.keys_pressed:
            self.keys_pressed.remove(event.keysym)
        self.direction_label.config(text="")

    def process_frame(self, frame):
        # 이미지 처리
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)

        # 각도 및 기타 정보 추가
        draw = ImageDraw.Draw(img)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font = ImageFont.truetype(font_path, 20)
        draw.text((10, 10), f"각도: {self.current_servo_angle}°", fill="yellow", font=font)

        # 프레임 저장
        if not hasattr(self, 'frame_count'):
            self.frame_count = 0
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            img = img.convert("RGB")  
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"images/frame_{timestamp}_angle_{self.current_servo_angle}.jpg"
            if not os.path.exists('images'):
                os.makedirs('images')
            img.save(filename)
            logging.info(f"이미지 저장: {filename}")

    def set_dc_motor(self, speed, direction):
        try:
            if direction == "forward":
                GPIO.output(self.dc_motor_dir_pin1, GPIO.HIGH)
                GPIO.output(self.dc_motor_dir_pin2, GPIO.LOW)
            elif direction == "backward":
                GPIO.output(self.dc_motor_dir_pin1, GPIO.LOW)
                GPIO.output(self.dc_motor_dir_pin2, GPIO.HIGH)
            elif direction == "stop":
                GPIO.output(self.dc_motor_dir_pin1, GPIO.LOW)
                GPIO.output(self.dc_motor_dir_pin2, GPIO.LOW)
            self.dc_motor_pwm.ChangeDutyCycle(speed)
        except Exception as e:
            logging.error(f"DC 모터 설정 중 오류: {e}")

    def set_servo_angle(self, angle):
        try:
            duty_cycle = 2 + (angle / 18)
            self.servo.ChangeDutyCycle(duty_cycle)
            time.sleep(0.1)
            self.servo.ChangeDutyCycle(0)
        except Exception as e:
            logging.error(f"서보 각도 설정 중 오류: {e}")

    def quit(self):
        self.camera_thread.stop()
        self.camera_thread.join()
        logging.info("프로그램 종료 중...")
        self.cap.release()
        self.servo.stop()
        self.dc_motor_pwm.stop()
        GPIO.cleanup()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == '__main__':
    main()

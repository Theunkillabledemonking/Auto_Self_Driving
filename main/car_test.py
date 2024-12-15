import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import threading
import datetime
import Jetson.GPIO as GPIO
import time
import logging
import os
import subprocess

class CameraThread(threading.Thread):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.running = False

    def run(self):
        # 카메라 프레임 처리 스레드
        while self.app.camera_active:
            start_time = time.time()
            ret, frame = self.app.cap.read()
            if ret:
                self.app.process_frame(frame)
                self.app.save_frame(frame)  # 프레임 저장 추가
            # 0.1초 주기로 프레임 처리
            time.sleep(max(0, 0.1 - (time.time() - start_time)))

    def start_recording(self):
        if not self.running:
            self.running = True
            self.app.camera_active = True
            self.start()

    def stop_recording(self):
        if self.running:
            self.running = False
            self.app.camera_active = False

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera and Motor Control")

        # GPIO 경고 비활성화
        GPIO.setwarnings(False)

        # 하드웨어 초기화 (devmem 설정 주석 처리 가능)
        self.setup_hardware()

        # 카메라 초기화
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("카메라를 열 수 없습니다.")
            exit()

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Tkinter 캔버스
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        # GPIO 핀 설정
        self.servo_pin = 33
        self.dc_motor_pwm_pin = 32
        self.dc_motor_dir_pin1 = 29
        self.dc_motor_dir_pin2 = 31

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.servo_pin, GPIO.OUT)
        GPIO.setup(self.dc_motor_pwm_pin, GPIO.OUT)
        GPIO.setup(self.dc_motor_dir_pin1, GPIO.OUT)
        GPIO.setup(self.dc_motor_dir_pin2, GPIO.OUT)

        # PWM 설정
        self.servo = GPIO.PWM(self.servo_pin, 50)  # 서보: 50Hz
        self.dc_motor_pwm = GPIO.PWM(self.dc_motor_pwm_pin, 1000)  # DC 모터: 1kHz
        self.servo.start(0)
        self.dc_motor_pwm.start(0)

        # 기본 속도 및 각도
        self.current_speed = 65  # 속도 기본값 낮춤
        self.current_servo_angle = 90
        self.set_servo_angle(self.current_servo_angle)

        # 종료 상태 관리
        self.running = True
        self.camera_active = False

        # 키 이벤트 바인딩
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.bind('<KeyRelease>', self.on_key_release)

        self.keys_pressed = set()
        self.setup_logging()

        # 방향 표시 라벨
        self.direction_label = tk.Label(self.root, text="", font=("Helvetica", 30))
        self.direction_label.pack(pady=10)

        # 각도 표시 라벨
        self.angle_label = tk.Label(self.root, text=f"현재 각도: {self.current_servo_angle}°", font=("Helvetica", 14))
        self.angle_label.pack(pady=10)

        # 각도에 따른 화살표 매핑
        self.angle_to_arrow = {
            30: '←',
            60: '↖',
            90: '↑',
            120: '↗',
            150: '→'
        }

        self.frame_count = 0
        self.update_arrow_direction()

        # 카메라 처리 스레드 초기화
        self.camera_thread = CameraThread(self)

        # 기본적으로 앞으로 이동
        self.set_dc_motor(self.current_speed, "forward")
        logging.info("기본적으로 앞으로 이동 시작")

    def run_command(self, command):
        subprocess.run(command, shell=True, check=True)

    def setup_hardware(self):
        try:
            subprocess.run("busybox --help", shell=True, check=True)
        except subprocess.CalledProcessError:
            self.run_command("apt update && apt install -y busybox")

        commands = [
            "busybox devmem 0x700031fc 32 0x45",
            "busybox devmem 0x6000d504 32 0x2",
            "busybox devmem 0x70003248 32 0x46",
            "busybox devmem 0x6000d100 32 0x00"
        ]
        for cmd in commands:
            self.run_command(cmd)

    def setup_logging(self):
        if not os.path.exists('logs'):
            os.makedirs('logs')
        log_filename = datetime.datetime.now().strftime('logs/log_%Y-%m-%d_%H-%M-%S.log')
        logging.basicConfig(filename=log_filename, level=logging.INFO,
                            format='%(asctime)s %(levelname)s: %(message)s')
        logging.info("로그 초기화 완료.")

    def on_key_press(self, event):
        self.keys_pressed.add(event.keysym)
        if event.keysym == 'q':
            logging.info("Q 키 입력: 프로그램 종료 요청")
            self.quit()
        elif event.keysym == 'r':
            if self.camera_active:
                self.camera_thread.stop_recording()
                logging.info("R 키 입력: 촬영 중지")
            else:
                self.camera_thread.start_recording()
                logging.info("R 키 입력: 촬영 시작")
        elif event.keysym == 'w':
            self.set_dc_motor(self.current_speed, "forward")
            logging.info(f"W 키 입력: 전진 시작 (속도: {self.current_speed})")
        elif event.keysym == 's':
            self.set_dc_motor(self.current_speed, "backward")
            logging.info(f"S 키 입력: 후진 시작 (속도: {self.current_speed})")
        elif event.keysym == 'a':
            new_angle = max(30, self.current_servo_angle - 30)
            self.current_servo_angle = new_angle
            self.set_servo_angle(self.current_servo_angle)
            self.angle_label.config(text=f"현재 각도: {self.current_servo_angle}°")
            logging.info(f"좌회전: 각도 {self.current_servo_angle}°")
            self.update_arrow_direction()
        elif event.keysym == 'd':
            new_angle = min(150, self.current_servo_angle + 30)
            self.current_servo_angle = new_angle
            self.set_servo_angle(self.current_servo_angle)
            self.angle_label.config(text=f"현재 각도: {self.current_servo_angle}°")
            logging.info(f"우회전: 각도 {self.current_servo_angle}°")
            self.update_arrow_direction()

    def on_key_release(self, event):
        if event.keysym in self.keys_pressed:
            self.keys_pressed.remove(event.keysym)
            # 키에서 손을 떼었을 때 모터 정지
            if event.keysym in ['w', 's']:
                self.stop_motors()
                logging.info(f"{event.keysym.upper()} 키 해제: 모터 정지")

    def update_arrow_direction(self):
        arrow = self.angle_to_arrow.get(self.current_servo_angle, '↑')
        self.direction_label.config(text=arrow)

    def process_frame(self, frame):
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)

        draw = ImageDraw.Draw(img)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font = ImageFont.truetype(font_path, 20) if os.path.exists(font_path) else ImageFont.load_default()
        draw.text((10, 10), f"각도: {self.current_servo_angle}°", fill="yellow", font=font)

        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def save_frame(self, frame):
        if not os.path.exists('data/images'):
            os.makedirs('data/images')
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filepath = f"data/images/frame_{timestamp}.jpg"
        cv2.imwrite(filepath, frame)
        logging.info(f"프레임 저장: {filepath}")

    def set_servo_angle(self, angle):
        angle = max(0, min(180, angle))
        duty_cycle = 2 + (angle / 18)
        self.servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)
        self.servo.ChangeDutyCycle(0)

    def set_dc_motor(self, speed, direction):
        try:
            speed = max(0, min(100, speed))
            if direction == "forward":
                GPIO.output(self.dc_motor_dir_pin1, GPIO.HIGH)
                GPIO.output(self.dc_motor_dir_pin2, GPIO.LOW)
            elif direction == "backward":
                GPIO.output(self.dc_motor_dir_pin1, GPIO.LOW)
                GPIO.output(self.dc_motor_dir_pin2, GPIO.HIGH)
            self.dc_motor_pwm.ChangeDutyCycle(speed)
            logging.info(f"모터 설정: 속도 {speed}, 방향 {direction}")
        except Exception as e:
            logging.error(f"DC 모터 설정 오류: {e}")

    def stop_motors(self):
        try:
            self.dc_motor_pwm.ChangeDutyCycle(0)
            logging.info("모터 정지")
        except Exception as e:
            logging.error(f"모터 정지 오류: {e}")

    def quit(self):
        if not self.running:
            return
        self.running = False
        logging.info("프로그램 종료 중...")
        try:
            self.camera_thread.stop_recording()
            self.cap.release()
            self.servo.stop()
            self.dc_motor_pwm.stop()
            GPIO.cleanup()
            logging.info("GPIO 및 하드웨어 정리 완료.")
        except Exception as e:
            logging.error(f"종료 중 오류: {e}")
        finally:
            self.root.destroy()
            logging.info("프로그램 종료 완료.")


def main():
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.quit)  # 창 닫기 버튼 클릭 시 종료 함수 호출
    root.mainloop()


if __name__ == '__main__':
    main()

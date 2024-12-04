import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading
import datetime
import Jetson.GPIO as GPIO
import time
import logging
import os

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

        # 모터 및 서보 파라미터 초기화
        self.current_speed = 60  # 기본 속도
        self.current_servo_angle = 90  # 중립 각도
        self.booster_speed = 80
        self.keys_pressed = set()
        self.photo_count = 0  # 사진 개수 초기화

        # 속도, 각도, 사진 개수를 표시할 레이블 생성
        self.info_frame = tk.Frame(root)
        self.info_frame.pack(pady=10)

        self.speed_label = tk.Label(self.info_frame, text=f"속도: {self.current_speed}%", font=("Helvetica", 12))
        self.speed_label.pack(side=tk.LEFT, padx=10)

        self.angle_label = tk.Label(self.info_frame, text=f"서보 각도: {self.current_servo_angle}°", font=("Helvetica", 12))
        self.angle_label.pack(side=tk.LEFT, padx=10)

        self.photo_label = tk.Label(self.info_frame, text=f"촬영된 사진 수: {self.photo_count}", font=("Helvetica", 12))
        self.photo_label.pack(side=tk.LEFT, padx=10)

        # 로그 설정
        self.setup_logging()

        # 키 바인딩
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.bind('<KeyRelease>', self.on_key_release)

        # 업데이트 시작
        self.update_frame()
        self.update_motors()

        # 연속 이미지 촬영 플래그
        self.capturing = False

    def setup_logging(self):
        # logs 디렉토리 생성
        if not os.path.exists('logs'):
            os.makedirs('logs')
        # 로그 설정
        log_filename = datetime.datetime.now().strftime('logs/log_%Y-%m-%d_%H-%M-%S.log')
        logging.basicConfig(filename=log_filename, level=logging.INFO,
                            format='%(asctime)s %(levelname)s: %(message)s')
        logging.info("로그 초기화 완료.")

    def on_key_press(self, event):
        self.keys_pressed.add(event.keysym)
        if event.keysym == 'q':
            self.quit()
        elif event.keysym == 'r':
            if not self.capturing:
                self.capturing = True
                logging.info("연속 사진 촬영 시작.")
                print("연속 사진 촬영 시작.")
            else:
                self.capturing = False
                logging.info("연속 사진 촬영 중지.")
                print("연속 사진 촬영 중지.")

    def on_key_release(self, event):
        if event.keysym in self.keys_pressed:
            self.keys_pressed.remove(event.keysym)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 프레임을 ImageTk 형식으로 변환
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor='nw', image=imgtk)
            self.root.image = imgtk  # 참조 유지

            if self.capturing:
                # 이미지 저장 (메타데이터 포함)
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
                speed = self.current_speed if self.current_speed != self.booster_speed else self.booster_speed
                angle = self.current_servo_angle
                filename = f"image_{timestamp}_speed_{speed}_angle_{angle}.jpg"
                if not os.path.exists('images'):
                    os.makedirs('images')
                filepath = os.path.join('images', filename)
                cv2.imwrite(filepath, frame)
                self.photo_count += 1
                self.photo_label.config(text=f"촬영된 사진 수: {self.photo_count}")
                logging.info(f"사진 저장됨: {filename} (속도: {speed}%, 각도: {angle}°)")
                print(f"사진 저장됨: {filename}")
        else:
            logging.error("카메라에서 프레임을 읽을 수 없습니다.")

        self.root.after(10, self.update_frame)

    def update_motors(self):
        # 키 입력에 따른 모터 제어
        if 'w' in self.keys_pressed and 'a' in self.keys_pressed:
            # 전진 + 좌회전
            self.current_servo_angle = max(80, self.current_servo_angle - 2)
            self.set_servo_angle(self.current_servo_angle)
            self.set_dc_motor(self.current_speed, "forward")
            logging.info(f"좌회전 전진: 각도 {self.current_servo_angle}°")
            print(f"좌회전 전진: 각도 {self.current_servo_angle}°")
        elif 'w' in self.keys_pressed and 'd' in self.keys_pressed:
            # 전진 + 우회전
            self.current_servo_angle = min(100, self.current_servo_angle + 2)
            self.set_servo_angle(self.current_servo_angle)
            self.set_dc_motor(self.current_speed, "forward")
            logging.info(f"우회전 전진: 각도 {self.current_servo_angle}°")
            print(f"우회전 전진: 각도 {self.current_servo_angle}°")
        elif 'w' in self.keys_pressed:
            # 전진
            self.set_servo_angle(90)
            self.set_dc_motor(self.current_speed, "forward")
            logging.info(f"직진: 속도 {self.current_speed}%")
            print(f"직진: 속도 {self.current_speed}%")
        elif 's' in self.keys_pressed:
            # 후진
            self.set_servo_angle(90)
            self.set_dc_motor(55, "backward")
            logging.info("후진 중: 속도 55%")
            print("후진 중: 속도 55%")
        elif 'Shift_L' in self.keys_pressed or 'Shift_R' in self.keys_pressed:
            # 부스터
            self.set_dc_motor(self.booster_speed, "forward")
            logging.info(f"부스터 사용: 속도 {self.booster_speed}%")
            print(f"부스터 사용: 속도 {self.booster_speed}%")
        elif 'a' in self.keys_pressed:
            # 좌회전
            self.current_servo_angle = max(80, self.current_servo_angle - 2)
            self.set_servo_angle(self.current_servo_angle)
            logging.info(f"좌회전 중: 각도 {self.current_servo_angle}°")
            print(f"좌회전 중: 각도 {self.current_servo_angle}°")
        elif 'd' in self.keys_pressed:
            # 우회전
            self.current_servo_angle = min(100, self.current_servo_angle + 2)
            self.set_servo_angle(self.current_servo_angle)
            logging.info(f"우회전 중: 각도 {self.current_servo_angle}°")
            print(f"우회전 중: 각도 {self.current_servo_angle}°")
        else:
            # 정지
            self.set_dc_motor(0, "stop")
            logging.info("정지 또는 감속 중")
            print("정지 또는 감속 중")

        # 레이블 업데이트
        speed = self.current_speed if self.current_speed != self.booster_speed else self.booster_speed
        self.speed_label.config(text=f"속도: {speed}%")
        self.angle_label.config(text=f"서보 각도: {self.current_servo_angle}°")

        self.root.after(100, self.update_motors)

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
            logging.info(f"서보 각도 설정: {angle}°")
            self.angle_label.config(text=f"서보 각도: {angle}°")
        except Exception as e:
            logging.error(f"서보 각도 설정 중 오류: {e}")

    def quit(self):
        # 종료 처리
        logging.info("프로그램 종료 중...")
        print("프로그램 종료 중...")
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

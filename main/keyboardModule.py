import cv2
import threading
import datetime
import Jetson.GPIO as GPIO
import time
import logging
import os
import subprocess
import keyboard  # keyboard 모듈 사용

class CameraThread(threading.Thread):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.running = True

    def run(self):
        while self.running:
            start_time = time.time()
            ret, frame = self.app.cap.read()
            if ret:
                self.app.process_frame(frame)
            time.sleep(max(0, 0.1 - (time.time() - start_time)))

    def stop(self):
        self.running = False

class App:
    def __init__(self):
        # GPIO 경고 비활성화
        GPIO.setwarnings(False)

        # 하드웨어 초기화 (필요 시 주석)
        self.setup_hardware()

        # 카메라 초기화
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("카메라를 열 수 없습니다.")
            exit()

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
        self.servo = GPIO.PWM(self.servo_pin, 50)     # 서보: 50Hz
        self.dc_motor_pwm = GPIO.PWM(self.dc_motor_pwm_pin, 1000)  # DC 모터: 1kHz
        self.servo.start(0)
        self.dc_motor_pwm.start(0)

        self.current_speed = 75
        self.current_servo_angle = 90
        self.set_servo_angle(self.current_servo_angle)  # 초기 각도 90도

        self.setup_logging()
        self.frame_count = 0

        # 카메라 스레드 시작
        self.camera_thread = CameraThread(self)
        self.camera_thread.start()

    def run_command(self, command):
        subprocess.run(command, shell=True, check=True)

    def setup_hardware(self):
        # 필요하다면 devmem을 통한 핀mux 설정
        try:
            subprocess.run("busybox --help", shell=True, check=True)
            print("busybox is already installed.")
        except subprocess.CalledProcessError:
            print("busybox not found. Installing...")
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

    def process_frame(self, frame):
        # 프레임 처리: 각도 표시 및 저장
        # tkinter 제거하였으므로 이미지 디스플레이는 없음, 단순히 이미지 저장
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.imencode('.jpg', cv2image)[1].tobytes()

        # 이미지 저장
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
        if not os.path.exists('images'):
            os.makedirs('images')
        filename = f"images/frame_{timestamp}_angle_{self.current_servo_angle}.jpg"
        
        # OpenCV imwrite 사용
        cv2.imwrite(filename, cv2image)
        logging.info(f"이미지 저장: {filename}")
        self.frame_count += 1

    def capture_and_save_frame(self):
        # W키 입력 시 즉시 캡처
        ret, frame = self.cap.read()
        if ret:
            if not os.path.exists('images'):
                os.makedirs('images')
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
            filename = f"images/capture_{timestamp}_angle_{self.current_servo_angle}.jpg"
            cv2.imwrite(filename, frame)
            logging.info(f"W키 입력 시 사진 저장: {filename}")

    def set_servo_angle(self, angle):
        if angle < 30: angle = 30
        if angle > 150: angle = 150
        self.current_servo_angle = angle

        duty_cycle = 2 + (angle / 18)
        self.servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)
        self.servo.ChangeDutyCycle(0)

    def set_dc_motor(self, speed, direction):
        try:
            if direction == "forward":
                GPIO.output(self.dc_motor_dir_pin1, GPIO.HIGH)
                GPIO.output(self.dc_motor_dir_pin2, GPIO.LOW)
            elif direction == "backward":
                GPIO.output(self.dc_motor_dir_pin1, GPIO.LOW)
                GPIO.output(self.dc_motor_dir_pin2, GPIO.HIGH)
            self.dc_motor_pwm.ChangeDutyCycle(speed)
        except Exception as e:
            logging.error(f"DC 모터 설정 중 오류: {e}")

    def quit(self):
        self.camera_thread.stop()
        self.camera_thread.join()
        logging.info("프로그램 종료 중...")
        self.cap.release()
        self.servo.stop()
        self.dc_motor_pwm.stop()
        GPIO.cleanup()
        exit(0)

def main():
    app = App()

    # 키 상태 기반 제어 루프
    # q : 종료
    # w : 전진
    # a : 각도 왼쪽(감소)
    # d : 각도 오른쪽(증가)
    
    last_a_pressed = False
    last_d_pressed = False
    while True:
        try:
            if keyboard.is_pressed('q'):
                app.quit()

            if keyboard.is_pressed('w'):
                # 전진 중
                app.set_dc_motor(app.current_speed, "forward")
                app.capture_and_save_frame()
            else:
                # 정지 상태
                app.set_dc_motor(0, "forward")

            # a키 각도 변경 (누를 때마다 한번씩 -30도)
            # 키를 계속 누르고 있으면 각도를 너무 많이 바꾸지 않도록,
            # 한 번 눌렀다 떼는 시점에만 변화하도록 함.
            currently_a_pressed = keyboard.is_pressed('a')
            if currently_a_pressed and not last_a_pressed:
                new_angle = app.current_servo_angle - 30
                app.set_servo_angle(new_angle)
                logging.info(f"좌회전: 각도 {app.current_servo_angle}°")
            last_a_pressed = currently_a_pressed

            # d키 각도 변경
            currently_d_pressed = keyboard.is_pressed('d')
            if currently_d_pressed and not last_d_pressed:
                new_angle = app.current_servo_angle + 30
                app.set_servo_angle(new_angle)
                logging.info(f"우회전: 각도 {app.current_servo_angle}°")
            last_d_pressed = currently_d_pressed

            time.sleep(0.1)

        except KeyboardInterrupt:
            app.quit()

if __name__ == '__main__':
    main()

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
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.app.cap.read()
            if ret:
                self.app.process_frame(frame)
            time.sleep(0.03)

    def stop(self):
        self.running = False

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera and Motor Control")

        # 패스워드 관련 변수 제거
        # self.sudo_password = "your_password_here"

        self.setup_hardware()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Camera could not be opened.")
            exit()

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

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

        self.current_speed = 60
        self.current_servo_angle = 90

        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.bind('<KeyRelease>', self.on_key_release)

        self.keys_pressed = set()
        self.setup_logging()
        self.start_forward_motion()

        self.direction_label = tk.Label(self.root, text="", font=("Helvetica", 30))
        self.direction_label.pack(pady=10)

        self.angle_label = tk.Label(self.root, text=f"Current Angle: {self.current_servo_angle}", font=("Helvetica", 14))
        self.angle_label.pack(pady=10)

        self.angle_to_arrow = {
            30: '←',
            60: '↖',
            90: '↑',
            120: '↗',
            150: '→'
        }

        self.frame_count = 0

        self.update_arrow_direction()

        self.camera_thread = CameraThread(self)
        self.camera_thread.start()

    def run_command(self, command):
        # 패스워드 및 sudo 제거
        subprocess.run(command, shell=True, check=True)

    def setup_hardware(self):
        try:
            subprocess.run("busybox --help", shell=True, check=True)
            print("busybox is already installed.")
        except subprocess.CalledProcessError:
            print("busybox not found. Installing...")
            # sudo 제거
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
        logging.info("Logging initialized.")

    def start_forward_motion(self):
        self.set_dc_motor(self.current_speed, "forward")
        logging.info("Default forward motion started.")

    def on_key_press(self, event):
        self.keys_pressed.add(event.keysym)
        if event.keysym == 'q':
            self.quit()
        elif event.keysym == 'a':
            new_angle = self.current_servo_angle - 30
            if new_angle < 30:
                new_angle = 30
            self.current_servo_angle = new_angle
            self.set_servo_angle(self.current_servo_angle)
            self.angle_label.config(text=f"Current Angle: {self.current_servo_angle}")
            logging.info(f"Left Turn: Angle {self.current_servo_angle}")
            self.update_arrow_direction()

        elif event.keysym == 'd':
            new_angle = self.current_servo_angle + 30
            if new_angle > 150:
                new_angle = 150
            self.current_servo_angle = new_angle
            self.set_servo_angle(self.current_servo_angle)
            self.angle_label.config(text=f"Current Angle: {self.current_servo_angle}")
            logging.info(f"Right Turn: Angle {self.current_servo_angle}")
            self.update_arrow_direction()

    def on_key_release(self, event):
        if event.keysym in self.keys_pressed:
            self.keys_pressed.remove(event.keysym)

    def update_arrow_direction(self):
        arrow = self.angle_to_arrow.get(self.current_servo_angle, '↑')
        self.direction_label.config(text=arrow)

    def process_frame(self, frame):
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)

        draw = ImageDraw.Draw(img)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font = ImageFont.truetype(font_path, 20)
        draw.text((10, 10), f"Angle: {self.current_servo_angle}", fill="yellow", font=font)

        self.frame_count += 1
        if self.frame_count % 10 == 0:
            img = img.convert("RGB")
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            if not os.path.exists('/workspace/images'):
                os.makedirs('/workspace/images')
            filename = f"/workspace/images/frame_{timestamp}_angle_{self.current_servo_angle}.jpg"
            img.save(filename)
            logging.info(f"Image saved: {filename}")

        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def set_servo_angle(self, angle):
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
            logging.error(f"Error setting DC motor: {e}")

    def quit(self):
        self.camera_thread.stop()
        self.camera_thread.join()
        logging.info("Program exiting...")
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

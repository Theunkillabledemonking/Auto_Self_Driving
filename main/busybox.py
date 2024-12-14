import os
import Jetson.GPIO as GPIO
import time
import logging
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont, ImageTk
import cv2
import datetime
import subprocess

# Pin configuration
SERVO_PIN = 33  # The GPIO pin connected to the servo motor (PWM signal)
DC_MOTOR_PWM_PIN = 32
DC_MOTOR_DIR_PIN1 = 29
DC_MOTOR_DIR_PIN2 = 31

# Function to run shell commands
def run_command(cmd):
    """
    Executes a system command.
    :param cmd: The command string to execute.
    """
    try:
        os.system(cmd)
    except Exception as e:
        print(f"Error running command '{cmd}': {e}")

# Register configuration commands
def configure_hardware_registers():
    """
    Configure hardware registers using devmem for specific GPIO/pin settings.
    """
    commands = [
        "busybox devmem 0x700031fc 32 0x45",  # Example hardware configuration
        "busybox devmem 0x6000d504 32 0x2",
        "busybox devmem 0x70003248 32 0x46",
        "busybox devmem 0x6000d100 32 0x00"
    ]
    for cmd in commands:
        run_command(cmd)

# GPIO initialization
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(DC_MOTOR_PWM_PIN, GPIO.OUT)
GPIO.setup(DC_MOTOR_DIR_PIN1, GPIO.OUT)
GPIO.setup(DC_MOTOR_DIR_PIN2, GPIO.OUT)

# PWM setup
servo = GPIO.PWM(SERVO_PIN, 50)  # PWM frequency: 50Hz
servo.start(0)  # Initialize with a duty cycle of 0
dc_motor_pwm = GPIO.PWM(DC_MOTOR_PWM_PIN, 1000)  # DC Motor: 1kHz
dc_motor_pwm.start(0)

class App:
    def __init__(self, root):
        self.root = root
        self.current_speed = 75
        self.current_servo_angle = 90
        self.keys_pressed = set()
        self.setup_logging()

        self.angle_to_arrow = {
            30: '←',
            60: '↖',
            90: '↑',
            120: '↗',
            150: '→'
        }

        self.setup_hardware()
        self.create_ui()

    def setup_logging(self):
        if not os.path.exists('logs'):
            os.makedirs('logs')
        log_filename = datetime.datetime.now().strftime('logs/log_%Y-%m-%d_%H-%M-%S.log')
        logging.basicConfig(filename=log_filename, level=logging.INFO,
                            format='%(asctime)s %(levelname)s: %(message)s')

    def setup_hardware(self):
        try:
            subprocess.run("busybox --help", shell=True, check=True)
        except subprocess.CalledProcessError:
            self.run_command("apt update && apt install -y busybox")
        configure_hardware_registers()

    def create_ui(self):
        self.direction_label = tk.Label(self.root, text="", font=("Helvetica", 30))
        self.direction_label.pack(pady=10)

        self.angle_label = tk.Label(self.root, text=f"Angle: {self.current_servo_angle}°", font=("Helvetica", 14))
        self.angle_label.pack(pady=10)

        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.bind('<KeyRelease>', self.on_key_release)
        self.update_arrow_direction()

    def run_command(self, command):
        subprocess.run(command, shell=True, check=True)

    def set_servo_angle(self, angle):
        if angle < 0: angle = 0
        if angle > 180: angle = 180
        duty_cycle = 2 + (angle / 18)
        servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)
        servo.ChangeDutyCycle(0)

    def set_dc_motor(self, speed, direction):
        try:
            if direction == "forward":
                GPIO.output(DC_MOTOR_DIR_PIN1, GPIO.HIGH)
                GPIO.output(DC_MOTOR_DIR_PIN2, GPIO.LOW)
            elif direction == "backward":
                GPIO.output(DC_MOTOR_DIR_PIN1, GPIO.LOW)
                GPIO.output(DC_MOTOR_DIR_PIN2, GPIO.HIGH)
            dc_motor_pwm.ChangeDutyCycle(speed)
        except Exception as e:
            logging.error(f"DC motor error: {e}")

    def on_key_press(self, event):
        self.keys_pressed.add(event.keysym)
        if event.keysym == 'q':
            self.quit()
        elif event.keysym == 'w':
            self.set_dc_motor(self.current_speed, "forward")
        elif event.keysym == 'a':
            self.current_servo_angle = max(30, self.current_servo_angle - 30)
            self.set_servo_angle(self.current_servo_angle)
            self.angle_label.config(text=f"Angle: {self.current_servo_angle}°")
            self.update_arrow_direction()
        elif event.keysym == 'd':
            self.current_servo_angle = min(150, self.current_servo_angle + 30)
            self.set_servo_angle(self.current_servo_angle)
            self.angle_label.config(text=f"Angle: {self.current_servo_angle}°")
            self.update_arrow_direction()

    def on_key_release(self, event):
        if event.keysym in self.keys_pressed:
            self.keys_pressed.remove(event.keysym)

    def update_arrow_direction(self):
        arrow = self.angle_to_arrow.get(self.current_servo_angle, '↑')
        self.direction_label.config(text=arrow)

    def quit(self):
        dc_motor_pwm.stop()
        servo.stop()
        GPIO.cleanup()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()

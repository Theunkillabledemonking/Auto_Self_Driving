import Jetson.GPIO as GPIO
import time
import subprocess

# Set up GPIO pins
servo_pin = 33  # Servo PWM pin
dc_motor_pwm_pin = 32  # DC motor PWM pin
dc_motor_dir_pin1 = 29
dc_motor_dir_pin2 = 31

GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

# Initialize PWM for servo and DC motor
servo = GPIO.PWM(servo_pin, 50)  # Servo PWM frequency: 50 Hz
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # DC motor PWM frequency: 1 kHz

servo.start(0)
dc_motor_pwm.start(0)

# Function to initialize busybox and configure PWM pins
def setup_pwm_pins():
    commands = [
        "busybox devmem 0x700031fc 32 0x45",
        "busybox devmem 0x6000d504 32 0x2",
        "busybox devmem 0x70003248 32 0x46",
        "busybox devmem 0x6000d100 32 0x00",
    ]
    for cmd in commands:
        subprocess.run(cmd, shell=True, check=True)

# Function to set the servo angle
def set_servo_angle(angle):
    try:
        duty_cycle = max(2, min(12.5, 2 + (angle / 18)))  # Restrict duty cycle to valid range
        servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)  # Allow servo to reach the position
        servo.ChangeDutyCycle(0)  # Stop PWM signal to avoid jitter
    except Exception as e:
        print(f"Error setting servo angle: {e}")

# Function to control the DC motor
def set_dc_motor(speed, direction):
    try:
        if direction == "forward":
            GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
            GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
        elif direction == "backward":
            GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
            GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
        dc_motor_pwm.ChangeDutyCycle(speed)
    except Exception as e:
        print(f"Error controlling DC motor: {e}")

# Main logic
try:
    setup_pwm_pins()

    # Test servo and DC motor
    print("Testing servo motor...")
    for angle in range(0, 181, 30):
        set_servo_angle(angle)
    for angle in range(180, -1, -30):
        set_servo_angle(angle)

    print("Testing DC motor...")
    set_dc_motor(50, "forward")
    time.sleep(2)
    set_dc_motor(50, "backward")
    time.sleep(2)

finally:
    print("Cleaning up GPIO...")
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()

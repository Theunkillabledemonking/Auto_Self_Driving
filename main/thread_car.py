import Jetson.GPIO as GPIO
import time
import keyboard
import subprocess
import cv2
import threading
import datetime

# Set the sudo password for running commands with sudo
sudo_password = "12"

# Function to run shell commands with the sudo password
def run_command(command):
    full_command = f"echo {sudo_password} | sudo -S {command}"
    subprocess.run(full_command, shell=True, check=True)
    
servo_pin = 33 # PWM pin for servo motor
dc_motor_pwm_pin = 32 # PWM pin for DC motor speed
dc_motor_dir_pin1 = 29 # Direction control pin 1
dc_motor_dir_pin2 = 31 # Direction control pin 2

# GPIO 모드 설정 및 핀 초기화
GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

servo = GPIO.PWM(servo_pin, 50) # 50Hz for servo motor
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # 1kHz for DC motor
servo.start(0)
dc_motor_pwm.start(0)

fixed_speed = 75 # Fixed speed at 75%
current_speed = 0 # Start speed at 0 for smooth acceleration
current_angle = 90 # Middle position for straight driving

def set_servo_angle(angle):
    duty_cycle = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.1)
    servo.ChangeDutyCycle(0) # Turn off signal to avoid jitter
    
def set_dc_motor(speed, direction="forward"):
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
    
    dc_motor_pwm.ChangeDutyCycle(speed)
    
def record_video():
    cap = cv2.VideoCapture(0)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{timestamp}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
    
    while True:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Recording', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved as {video_filename}")

# Function to start recording in a separate thread
def start_recording():
    video_thread = threading.Thread(target=record_video)
    video_thread.start()
    return video_thread

try:
    set_servo_angle(current_angle)
    print("Use W/S to control speed, A/D for steering. Press 'q' to quit.")
    
    video_thread = start_recording() # camera start
    
    while True:
        # Speed control with W and S keys
        if keyboard.is_pressed('w'):
            current_speed = min(current_speed + 5, fixed_speed) # Accelerate
            set_dc_motor(current_speed, "forward")
            print(f"Speed increased to: {current_speed}%")
        elif keyboard.is_pressed('s'):
            current_speed = max(current_speed - 5, 0) # Decelerate
            set_dc_motor(current_speed, "forward" if current_speed > 0 else "backward")
            print(f"Speed decreased to: {current_speed}%")
            
        # Steering control with A and D keys
        if keyboard.is_pressed('a'):
            current_angle = max(0, current_angle - 5) # Steer left
            set_servo_angle(current_angle)
            print(f"Steering angle: {current_angle} (Left)")
        elif keyboard.is_pressed('d'):
            current_angle = min(180, current_angle + 5) # Steer right
            set_servo_angle(current_angle)
            print(f"Steering angle: {current_angle} (Right)")
            
        # Quit
        if keyboard.is_pressed('q'):
            print("Quit key pressed")
            break
        
        time.sleep(0.1)
        
    # Join video thread after quitting
    if video_thread.is_alive():
        video_thread.join()
        
finally:
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
    print("Cleaned up GPIO and stopped recording.")
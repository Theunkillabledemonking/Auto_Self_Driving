import Jetson.GPIO as GPIO
import time
import subprocess

# sudo_password 제거
# sudo, password 없이 바로 명령어 실행할 함수
def run_command(command):
    # sudo 없이 바로 명령 실행
    subprocess.run(command, shell=True, check=True)

# busybox 설치 확인
try:
    subprocess.run("busybox --help", shell=True, check=True)
    print("busybox is already installed.")
except subprocess.CalledProcessError:
    print("busybox not found. Installing...")
    # sudo 없이 apt 명령어 사용 가능하도록 가정(루트 권한 환경)
    run_command("apt update && apt install -y busybox")

# devmem 명령어들
commands = [
    "busybox devmem 0x700031fc 32 0x45",
    "busybox devmem 0x6000d504 32 0x2",
    "busybox devmem 0x70003248 32 0x46",
    "busybox devmem 0x6000d100 32 0x00"
]

for command in commands:
    run_command(command)

# GPIO 설정
servo_pin = 33  # PWM 핀(서보)
dc_motor_pwm_pin = 32  # PWM 핀(DC 모터)
dc_motor_dir_pin1 = 29
dc_motor_dir_pin2 = 31

GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

servo = GPIO.PWM(servo_pin, 50)   # 서보용 50Hz
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # DC 모터용 1kHz
servo.start(0)
dc_motor_pwm.start(0)

def set_servo_angle(angle):
    duty_cycle = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)  # 서보 이동 시간
    servo.ChangeDutyCycle(0)  # 지터 방지용 신호 끄기

def set_dc_motor(speed, direction):
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
    dc_motor_pwm.ChangeDutyCycle(speed)

try:
    # 서보 0도~180도 갔다가 돌아오기
    for angle in range(0, 181, 10):
        set_servo_angle(angle)
    for angle in range(180, -1, -10):
        set_servo_angle(angle)
    
    # DC 모터 전진 50% -> 2초 대기 -> 후진 50% -> 2초 대기
    set_dc_motor(50, "forward")
    time.sleep(2)
    set_dc_motor(50, "backward")
    time.sleep(2)
finally:
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()

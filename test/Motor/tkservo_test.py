import tkinter as tk
import Jetson.GPIO as GPIO
import time

# GPIO 초기 설정 및 오류 방지를 위한 초기화
GPIO.cleanup() # 이전 설정을 정리하여 충돌 방지
servo_pin = 33 # 서보 모터 핀 번호
GPIO.setmode(GPIO.BOARD) 
GPIO.setup(servo_pin, GPIO.OUT)

# 서보 모터 PWM 설정
servo = GPIO.PWM(servo_pin, 50) # 서보 모터용 50hz PWM 생성
servo.start(0)

# 기본 조향 각도 설정 (90도는 직진)
current_angle = 90

# 서보 모터 각도 설정 함수
def set_servo_angle(angle):
    # 각도 범위 확인 (0~180도 범위 내로 제한)
    if angle < 0 or angle > 180:
        print(f"Error : Angle out of range (0 to 180 degrees)")
        return
    try:
        duty_cycle = 2 + (angle / 18)
        servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.1)
        servo.ChangeDutyCycle(0) # 신호를 꺼서 떨림 방지
    except Exception as e:
        print(f"An error occurred while setting servo angle: {e}")
        
# 'a' 버튼 클릭 시 왼쪽으로 조향
def steer_left():
    global current_angle
    current_angle = max(0, current_angle - 5) # 각도를 줄여 왼쪽으로 이동 (0도가 최대 왼쪽)
    set_servo_angle(current_angle)
    angle_label.config(text=f"Steering Angle: {current_angle}")
    
# 'd' 버튼 클릭 시 오른쪽으로 조향
def steer_right():
    global current_angle
    current_angle = min(180, current_angle + 5) # 각도를 늘려 오른쪽으로 이동 (180도가 최대 오른쪽)
    set_servo_angle(current_angle)
    angle_label.config(text=f"Steering Angle: {current_angle}")
    
# tkinter Gui 설정
root = tk.Tk()
root.title("Servo Motor Steering Control")
root.geometry("300x200")

# 조향 각도 레이블
angle_label = tk.Label(root, text=f"Steering Angle: {current_angle}", font=("Arial", 14))
angle_label.pack(pady=10)

# '왼쪽' 버튼
left_button = tk.Button(root, text="<- Left (A)", command=steer_left, font=("Arial", 12), width=10)
left_button.pack(side="left", padx=20, pady=20)

# '오른쪽' 버튼
right_button = tk.Button(root, text="Right (D) ->", command=steer_right, font=("Arial", 12), width=10)
right_button.pack(side="right", padx=20, pady=20)

# GUI 종료 시 GPIO 정리
def on_closing():
    servo.stop()
    GPIO.cleanup()
    root.destroy()
    
root.protocol("WM_DELETE_WINDOW", on_closing)

# tiinter GUI 실행
root.mainloop()
import os
import Jetson.GPIO as GPIO
import time

# Pin configuration
SERVO_PIN = 33  # The GPIO pin connected to the servo motor (PWM signal)

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

# PWM setup
servo = GPIO.PWM(SERVO_PIN, 50)  # PWM frequency: 50Hz
servo.start(0)  # Initialize with a duty cycle of 0

def set_servo_angle(angle):
    """
    Adjust the servo motor angle.
    :param angle: The angle to set (range: 0 to 180 degrees)
    """
    try:
        duty_cycle = 2 + (angle / 18)  # Calculate duty cycle for the angle
        servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)  # Allow the servo to reach the target position
        servo.ChangeDutyCycle(0)  # Stop sending signal after adjustment
    except Exception as e:
        print(f"Error setting servo angle: {e}")

try:
    print("Configuring hardware registers...")
    configure_hardware_registers()  # Execute register configuration

    print("Testing servo motor...")
    # Rotate the servo motor from 0° to 180° in steps of 30°
    for angle in range(0, 181, 30):
        set_servo_angle(angle)

    # Rotate the servo motor back from 180° to 0°
    for angle in range(180, -1, -30):
        set_servo_angle(angle)

<<<<<<< HEAD
finally:
    print("Cleaning up GPIO...")
    servo.stop()
    GPIO.cleanup()
=======
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

        # 기본 속도 및 각도
        self.current_speed = 75
        self.current_servo_angle = 90
        self.set_servo_angle(self.current_servo_angle)  # 초기값 90도

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

        # 카메라 처리 스레드 시작
        self.camera_thread = CameraThread(self)
        self.camera_thread.start()

    def run_command(self, command):
        # 명령어 실행
        subprocess.run(command, shell=True, check=True)

    def setup_hardware(self):
        # devmem 명령어로 핀mux를 조정하는 부분(필요 시 주석 해제)
        # 이 부분이 PWM 신호 설정에 영향을 줄 수 있으므로 문제 발생 시 주석 처리 후 테스트
        
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
       
        pass

    def setup_logging(self):
        # 로깅 설정
        if not os.path.exists('logs'):
            os.makedirs('logs')
        log_filename = datetime.datetime.now().strftime('logs/log_%Y-%m-%d_%H-%M-%S.log')
        logging.basicConfig(filename=log_filename, level=logging.INFO,
                            format='%(asctime)s %(levelname)s: %(message)s')
        logging.info("로그 초기화 완료.")

    def on_key_press(self, event):
        self.keys_pressed.add(event.keysym)
        if event.keysym == 'q':
            # 프로그램 종료
            self.quit()
        elif event.keysym == 'w':
            # 전진
            self.set_dc_motor(self.current_speed, "forward")
            logging.info("W 키 입력: 전진 시작")
            self.capture_and_save_frame()
        elif event.keysym == 'a':
            # 왼쪽(각도 감소)
            new_angle = self.current_servo_angle - 30
            if new_angle < 30:
                new_angle = 30
            self.current_servo_angle = new_angle
            self.set_servo_angle(self.current_servo_angle)
            self.angle_label.config(text=f"현재 각도: {self.current_servo_angle}°")
            logging.info(f"좌회전: 각도 {self.current_servo_angle}°")
            self.update_arrow_direction()
        elif event.keysym == 'd':
            # 오른쪽(각도 증가)
            new_angle = self.current_servo_angle + 30
            if new_angle > 150:
                new_angle = 150
            self.current_servo_angle = new_angle
            self.set_servo_angle(self.current_servo_angle)
            self.angle_label.config(text=f"현재 각도: {self.current_servo_angle}°")
            logging.info(f"우회전: 각도 {self.current_servo_angle}°")
            self.update_arrow_direction()

    def on_key_release(self, event):
        if event.keysym in self.keys_pressed:
            self.keys_pressed.remove(event.keysym)

    def update_arrow_direction(self):
        arrow = self.angle_to_arrow.get(self.current_servo_angle, '↑')
        self.direction_label.config(text=arrow)

    def process_frame(self, frame):
        # 프레임 처리: 각도 표시 및 저장
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)

        draw = ImageDraw.Draw(img)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, 20)
        else:
            font = ImageFont.load_default()

        draw.text((10, 10), f"각도: {self.current_servo_angle}°", fill="yellow", font=font)

        self.frame_count += 1

        img = img.convert("RGB")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
        if not os.path.exists('images'):
            os.makedirs('images')
        filename = f"images/frame_{timestamp}_angle_{self.current_servo_angle}.jpg"
        img.save(filename)
        logging.info(f"이미지 저장: {filename}")

        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def capture_and_save_frame(self):
        # W키 입력 시 즉시 캡처
        ret, frame = self.cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not os.path.exists('images'):
                os.makedirs('images')
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
            filename = f"images/capture_{timestamp}_angle_{self.current_servo_angle}.jpg"
            img.save(filename)
            logging.info(f"W키 입력 시 사진 저장: {filename}")

    def set_servo_angle(self, angle):
        # 범위 체크(필요하다면)
        if angle < 0: angle = 0
        if angle > 180: angle = 180

        duty_cycle = 2 + (angle / 18)
        self.servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)
        self.servo.ChangeDutyCycle(0)

    def set_dc_motor(self, speed, direction):
        try:
            # 방향 설정
            if direction == "forward":
                GPIO.output(self.dc_motor_dir_pin1, GPIO.HIGH)
                GPIO.output(self.dc_motor_dir_pin2, GPIO.LOW)
            elif direction == "backward":
                GPIO.output(self.dc_motor_dir_pin1, GPIO.LOW)
                GPIO.output(self.dc_motor_dir_pin2, GPIO.HIGH)
            # 속도(PWM)
            self.dc_motor_pwm.ChangeDutyCycle(speed)
        except Exception as e:
            logging.error(f"DC 모터 설정 중 오류: {e}")

    def quit(self):
        # 종료 시 안전하게 리소스 해제
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
>>>>>>> fc359192e1d5d7efd5ac8d82033ad58988bd3b80

import os
import shutil
import tkinter as tk
from tkinter import messagebox, Canvas
from PIL import Image, ImageTk, UnidentifiedImageError
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def crop_and_resize_images(input_folder, output_folder, crop_percentage=40, target_size=(200, 66)):
    """
    이미지 파일을 크롭하고 리사이즈하며, 파일 이름에서 각도 정보를 자동 추출하여 저장합니다.

    :param input_folder: 원본 이미지 폴더 경로
    :param output_folder: 결과 이미지 저장 폴더 경로
    :param crop_percentage: 잘라낼 상단 비율
    :param target_size: 리사이즈할 크기 (width, height)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    valid_angles = [30, 60, 90, 120, 150]

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_extensions):
            input_path = os.path.join(input_folder, filename)

            # 파일 이름에서 각도 추출 (없으면 기본값 90 사용)
            base_filename = os.path.splitext(filename)[0]
            angle = 90  # 기본값
            if "angle_" in base_filename:
                try:
                    angle_part = base_filename.split("angle_")[1]
                    angle = int(angle_part.split("_")[0])
                    if angle not in valid_angles:
                        angle = 90  # 유효하지 않은 각도 처리
                except ValueError:
                    angle = 90
            else:
                # 파일명에 각도 정보가 없으면 자동으로 설정
                angle = valid_angles[(hash(base_filename) % len(valid_angles))]
            
            # 중복된 각도 태그를 제거하고 새로운 각도만 추가
            base_filename = base_filename.split("_angle_")[0]
            output_filename = f"{base_filename}_angle_{angle}.jpg"
            output_path = os.path.join(output_folder, output_filename)

            try:
                with Image.open(input_path) as img:
                    width, height = img.size
                    crop_height = int(height * (crop_percentage / 100))
                    cropped_img = img.crop((0, crop_height, width, height))
                    resized_img = cropped_img.resize(target_size, Image.LANCZOS)
                    resized_img.save(output_path)
                    print(f"크롭 및 리사이즈 완료: {output_filename}")
            except Exception as e:
                print(f"이미지 처리 중 오류 발생 ({filename}): {e}")

class ServoAngleGUI:
    def __init__(self, root, data_folder="data/images", temp_folder="data/temp", processed_folder="data/processed"):
        self.root = root
        self.root.title("Servo Angle File Manager")

        self.data_folder = data_folder
        self.temp_folder = temp_folder
        self.processed_folder = processed_folder
        self.cropped_folder = os.path.join(self.processed_folder, "cropped_resized")

        for folder in [self.temp_folder, self.processed_folder, self.cropped_folder]:
            os.makedirs(folder, exist_ok=True)

        self.file_list = []
        self.current_index = 0

        self.create_widgets()
        self.update_file_list()
        self.display_current_photo()
        self.auto_crop_and_resize()

    def create_widgets(self):
        self.photo_frame = tk.Frame(self.root)
        self.photo_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.photo_canvas = Canvas(self.photo_frame, width=500, height=500, bg="white")
        self.photo_canvas.pack()

        self.delete_button = tk.Button(self.photo_frame, text="Delete", command=self.delete_current_photo)
        self.delete_button.pack(side=tk.TOP, pady=5)

    def auto_crop_and_resize(self):
        """자동으로 이미지 크롭 및 리사이즈 수행"""
        crop_and_resize_images(self.data_folder, self.cropped_folder, crop_percentage=40, target_size=(200, 66))
        messagebox.showinfo("Auto Crop Complete", "All images have been cropped, resized, and angle added.")
        print("모든 이미지 크롭, 리사이즈 및 각도 추가 완료")

    def delete_current_photo(self, event=None):
        if self.file_list:
            current_file = self.file_list[self.current_index]
            src_path = os.path.join(self.data_folder, current_file)
            dst_path = os.path.join(self.temp_folder, current_file)
            shutil.move(src_path, dst_path)
            messagebox.showinfo("Photo Deleted", f"Deleted: {current_file}")
            self.update_file_list()
            self.display_current_photo()

    def update_file_list(self):
        supported_extensions = ('.jpg', '.jpeg', '.png')
        self.file_list = [file for file in os.listdir(self.data_folder) if file.lower().endswith(supported_extensions)]
        self.current_index = max(0, min(self.current_index, len(self.file_list) - 1))

    def display_current_photo(self):
        self.photo_canvas.delete("all")
        if self.file_list:
            current_file = self.file_list[self.current_index]
            file_path = os.path.join(self.data_folder, current_file)
            try:
                # 파일 이름에서 각도 추출
                base_filename = os.path.splitext(current_file)[0]
                angle = 90  # 기본값
                if "angle_" in base_filename:
                    try:
                        angle_part = base_filename.split("angle_")[1]
                        angle = int(angle_part.split("_")[0])
                        if angle not in [30, 60, 90, 120, 150]:
                            angle = 90
                    except ValueError:
                        angle = 90
                
                image = Image.open(file_path).resize((500, 500), Image.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.photo_canvas.image = photo
                self.photo_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.photo_canvas.create_text(250, 480, text=f"Angle: {angle}", font=("Helvetica", 12), fill="red")
            except UnidentifiedImageError:
                print(f"손상된 이미지 건너뛰기: {current_file}")
        else:
            self.photo_canvas.create_text(250, 250, text="No photos available", font=("Helvetica", 16), fill="gray")

if __name__ == "__main__":
    root = tk.Tk()
    app = ServoAngleGUI(root, data_folder="data/images", temp_folder="data/temp", processed_folder="data/processed")
    root.mainloop()

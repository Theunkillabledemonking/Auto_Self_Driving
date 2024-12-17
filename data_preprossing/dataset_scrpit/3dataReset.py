import pandas as pd
import cv2
import os

# CSV 파일 경로
csv_path = r"C:\Users\USER\Desktop\programing\code\data\processed\training_data_updated.csv"

# CSV 파일 불러오기
df = pd.read_csv(csv_path)

# 이미지가 없을 경우 예외 처리
if df.empty:
    print("CSV 파일에 데이터가 없습니다.")
    exit()

print("이미지 확인 및 삭제 기능 시작")
print("키 설명: A(이전), D(다음), C(삭제), Q(종료)")

index = 0  # 현재 이미지 인덱스

while True:
    # 이미지 경로 가져오기
    image_path = df.iloc[index]["frame_path"]

    # 이미지 로드
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
    else:
        print(f"이미지 없음: {image_path}")
        index = (index + 1) % len(df)
        continue

    # 이미지 표시
    cv2.imshow("Image Viewer", img)
    print(f"현재 이미지: {image_path} (인덱스 {index+1}/{len(df)})")

    key = cv2.waitKey(0) & 0xFF  # 키 입력 대기

    if key == ord('a'):  # A: 이전 이미지
        index = (index - 1) % len(df)

    elif key == ord('d'):  # D: 다음 이미지
        index = (index + 1) % len(df)

    elif key == ord('c'):  # C: 이미지 삭제
        # 이미지 파일 삭제
        os.remove(image_path)
        print(f"이미지 삭제됨: {image_path}")

        # 데이터프레임에서 해당 행 제거
        df = df.drop(index).reset_index(drop=True)
        print("CSV 데이터 업데이트 완료.")

        # 현재 인덱스 조정
        if index >= len(df):
            index = 0

    elif key == ord('q'):  # Q: 종료
        print("이미지 확인 및 삭제를 종료합니다.")
        break

cv2.destroyAllWindows()

# 최종 수정된 CSV 파일 저장
df.to_csv(csv_path, index=False)
print(f"최종 CSV 파일이 저장되었습니다: {csv_path}")

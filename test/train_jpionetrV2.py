import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

# ==============================
# 1. 데이터셋 정의
# ==============================
class SteeringDataset(Dataset):
    def __init__(self, csv_file, categories, transform=None):
        self.data = pd.read_csv(csv_file)
        self.categories = categories
        self.transform = transform

        # 각도를 범주로 변환
        self.data['steering_category'] = self.data['angle'].apply(
            lambda angle: self.categories.index(min(self.categories, key=lambda x: abs(x - angle)))
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['frame_path']
        category = int(row['steering_category'])

        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지 로드 실패: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 데이터 증강 및 정규화
        if self.transform:
            image = self.transform(image)
        else:
            image = cv2.resize(image, (200, 66))
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float32)

        return image, torch.tensor(category, dtype=torch.long)

# ==============================
# 2. PilotNet 모델 정의
# ==============================
class PilotNet(nn.Module):
    def __init__(self, num_classes=5):
        super(PilotNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 1 * 18, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ==============================
# 3. 데이터 준비
# ==============================
csv_path = "data/processed/training_data_resized.csv"
categories = [30, 60, 90, 120, 150]

# 데이터셋 분할
df = pd.read_csv(csv_path)
train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.3333, random_state=42)

train_data.to_csv("train.csv", index=False)
val_data.to_csv("val.csv", index=False)
test_data.to_csv("test.csv", index=False)

# 데이터 증강 설정
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 데이터 로더
train_dataset = SteeringDataset("train.csv", categories, transform=train_transform)
val_dataset = SteeringDataset("val.csv", categories)
test_dataset = SteeringDataset("test.csv", categories)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# ==============================
# 4. 학습 설정
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = PilotNet(num_classes=len(categories)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# ==============================
# 5. 학습 및 검증
# ==============================
train_losses, val_losses = [], []
best_val_loss = float('inf')
model_save_path = "best_pilotnet_model.pth"

for epoch in range(30):  # 반드시 30 Epoch 실행
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch [{epoch+1}/30], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    scheduler.step()

    # Best Model 저장
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        torch.save(model.state_dict(), model_save_path)
        print(f"Best Model 저장됨: {best_val_loss:.4f}")

# ==============================
# 6. 테스트 평가
# ==============================
model.load_state_dict(torch.load(model_save_path))
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# ==============================
# 7. 학습 결과 시각화
# ==============================
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.show()

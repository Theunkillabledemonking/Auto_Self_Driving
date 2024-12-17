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
from torch.optim import lr_scheduler

# ==============================
# 1. Early Stopping 클래스 정의
# ==============================
class EarlyStopping:
    def __init__(self, patience=10, min_epochs=15):
        self.patience = patience
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, epoch):
        if epoch < self.min_epochs:
            return  # 최소 에폭 수 보장
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early Stopping Triggered!")

# ==============================
# 2. 데이터셋 정의
# ==============================
class SteeringDataset(Dataset):
    def __init__(self, csv_file, categories, transform=None):
        self.data = pd.read_csv(csv_file)
        self.categories = categories
        self.transform = transform

        # steering_angle을 가장 가까운 범주로 변환
        self.data['steering_category'] = self.data['angle'].apply(
            lambda angle: self.categories.index(min(self.categories, key=lambda x: abs(x - angle)))
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['frame_path']
        category = int(row['steering_category'])

        # 이미지 로드 및 전처리
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)
        else:
            image = cv2.resize(image, (200, 66))
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float32)

        return image, torch.tensor(category, dtype=torch.long)

# ==============================
# 3. PilotNet 모델 정의
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
            nn.Linear(64 * 1 * 18, 256), nn.ReLU(), nn.Dropout(0.6),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ==============================
# 4. 데이터 로더 및 경로 설정
# ==============================
csv_path = "data/processed/training_data_resized.csv"
categories = [30, 60, 90, 120, 150]

# 데이터 분할
df = pd.read_csv(csv_path)
train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.3333, random_state=42)

train_data.to_csv("train.csv", index=False)
val_data.to_csv("val.csv", index=False)
test_data.to_csv("test.csv", index=False)

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop((66, 200), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = SteeringDataset("train.csv", categories, transform=train_transform)
val_dataset = SteeringDataset("val.csv", categories)
test_dataset = SteeringDataset("test.csv", categories)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# ==============================
# 5. 학습 설정
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = PilotNet(num_classes=len(categories)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
early_stopping = EarlyStopping(patience=10, min_epochs=15)
best_val_loss = float('inf')
model_save_path = "best_pilotnet_model.pth"

# ==============================
# 6. 학습 및 검증
# ==============================
train_losses, val_losses = [], []

for epoch in range(30):
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
    early_stopping(val_losses[-1], epoch)
    if early_stopping.early_stop:
        break

torch.save(model.state_dict(), model_save_path)
print(f"최적 모델 저장됨: {model_save_path}")

# ==============================
# 7. 테스트 평가
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
# 8. 학습 결과 시각화
# ==============================
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.show()

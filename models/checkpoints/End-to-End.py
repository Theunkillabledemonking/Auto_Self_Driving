import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure 'models' folder exists
os.makedirs("models", exist_ok=True)

# Dataset Class
class SteeringAngleDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 0])
        steering_angle = self.data.iloc[idx, 1]
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(steering_angle, dtype=torch.float32)

# CNN Model Definition
class NVIDIAEndToEndModel(nn.Module):
    def __init__(self):
        super(NVIDIAEndToEndModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

# Hyperparameters
batch_size = 641
learning_rate = 0.001
num_epochs = 50

# Paths
train_csv = "data/processed/train_labels.csv"
train_folder = "data/processed/balanced"
val_csv = "data/processed/val_labels.csv"
val_folder = "data/processed/balanced"
test_csv = "data/processed/test_labels.csv"
test_folder = "data/processed/balanced"

# Transforms
transform = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Datasets
train_dataset = SteeringAngleDataset(train_csv, train_folder, transform=transform)
val_dataset = SteeringAngleDataset(val_csv, val_folder, transform=transform)
test_dataset = SteeringAngleDataset(test_csv, test_folder, transform=transform)

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
model = NVIDIAEndToEndModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Function
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, angles in loader:
        images, angles = images.to(device), angles.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), angles)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

# Validation Function
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, angles in loader:
            images, angles = images.to(device), angles.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), angles)
            running_loss += loss.item()
    return running_loss / len(loader)

# Prediction Function
def predict(model, loader, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for images, angles in loader:
            images = images.to(device)
            outputs = model(images).squeeze().cpu().numpy()
            predictions.extend(outputs)
            actuals.extend(angles.numpy())
    return predictions, actuals

# Training Loop
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save Model
torch.save(model.state_dict(), "models/end_to_end_model.pth")
print("Model saved successfully.")

# Test Dataset Performance
predictions, actuals = predict(model, test_loader, device)
rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
print(f"Test RMSE: {rmse:.4f}")

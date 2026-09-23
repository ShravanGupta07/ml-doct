import os
import math
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("[*] Fetching California Housing dataset (20,640 samples)...")
raw_data = fetch_california_housing(as_frame=True)
df = raw_data.frame

# Feature engineering to make the tabular space complex
df["RoomsPerHousehold"] = df["AveRooms"] / (df["AveOccup"] + 1e-5)
df["BedroomsRatio"] = df["AveBedrms"] / (df["AveRooms"] + 1e-5)
df["IncomePerOccupant"] = df["MedInc"] / (df["AveOccup"] + 1e-5)

features = [col for col in df.columns if col != "MedHouseVal"]
target = "MedHouseVal"

X = df[features].values
y = df[target].values.reshape(-1, 1)

# Train/Val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_val = scaler_x.transform(X_val)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.transform(y_val)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

# Deep Residual MLP Architecture
class DeepTabularResNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, 128)
        self.block1 = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )
        self.block2 = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = x + self.block1(x)  # Residual skip
        x = x + self.block2(x)  # Residual skip
        return self.head(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Initialized Tabular ResNet on engine: {device}")
model = DeepTabularResNet(X.shape[1]).to(device)

criterion = nn.MSELoss()
# High learning rate to stress numerical stability
optimizer = optim.AdamW(model.parameters(), lr=0.08, weight_decay=1e-4)

os.makedirs("experiment_logs", exist_ok=True)
log_records = []
step_idx = 0
total_loss_accum = 0.0

print("[*] Launching training across 10 epochs...")

for epoch in range(10):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Optimization step
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()

        # BUG 1: Accumulating raw tensor without .item() (OOM Memory Leak trigger)
        total_loss_accum += loss

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = total_norm ** 0.5

        optimizer.step()
        optimizer.zero_grad()

        # Validation step every 20 batches
        if step_idx % 20 == 0:
            # BUG 2: Missing model.eval() before torch.no_grad()
            model.eval()  # [Auto-fixed by ML Doctor]
            with torch.no_grad():
                val_x, val_y = next(iter(val_loader))
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_preds = model(val_x)
                v_loss = criterion(val_preds, val_y).item()

            train_loss_scalar = loss.item()
            if math.isnan(train_loss_scalar) or math.isinf(train_loss_scalar):
                train_loss_scalar = 99.0
            if math.isnan(v_loss) or math.isinf(v_loss):
                v_loss = 99.0

            log_records.append({
                "step": step_idx,
                "train_loss": round(train_loss_scalar, 4),
                "val_loss": round(v_loss, 4),
                "grad_norm": round(grad_norm, 4)
            })
            print(f"Epoch {epoch} | Step {step_idx:03d} | Train Loss: {train_loss_scalar:.4f} | Val Loss: {v_loss:.4f} | Grad Norm: {grad_norm:.2f}")

        step_idx += 1

csv_path = "experiment_logs/tabular_resnet_metrics.csv"
pd.DataFrame(log_records).to_csv(csv_path, index=False)
print(f"[+] Run complete. Metrics written to '{csv_path}'.")
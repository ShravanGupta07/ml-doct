import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import math

# 1. Load Real-World Data (20,640 rows, 8 features)
print("Loading California Housing dataset...")
data = fetch_california_housing()
X, y = data.data, data.target

# BUG 1: We forgot to use StandardScaler! Neural networks hate raw, unscaled data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).view(-1, 1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).view(-1, 1)

# 2. Deep Regression Model
model = nn.Sequential(
    nn.Linear(8, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

criterion = nn.MSELoss()
# BUG 2: Learning rate is absurdly high for unscaled data -> Exploding Gradients
optimizer = optim.Adam(model.parameters(), lr=0.1)

logs = []

print("Training model...")
for step in range(50):
    model.train()
    optimizer.zero_grad()
    
    preds = model(X_train_t)
    loss = criterion(preds, y_train_t)
    
    # Catch NaN to prevent the script from crashing so we can log it
    loss_val = loss.item()
    if math.isnan(loss_val) or math.isinf(loss_val):
        loss_val = 99.0
        grad_norm = 999.0
    else:
        loss.backward()
        
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = total_norm ** 0.5
        optimizer.step()
    
    # BUG 3: Missing model.eval() before torch.no_grad()
    with torch.no_grad():
        val_preds = model(X_test_t)
        val_loss = criterion(val_preds, y_test_t).item()
        if math.isnan(val_loss) or math.isinf(val_loss):
            val_loss = 99.0
        
    logs.append({
        "step": step,
        "train_loss": round(loss_val, 4),
        "val_loss": round(val_loss, 4),
        "grad_norm": round(grad_norm, 4)
    })

pd.DataFrame(logs).to_csv("complex_logs.csv", index=False)
print("Saved 'complex_logs.csv'")
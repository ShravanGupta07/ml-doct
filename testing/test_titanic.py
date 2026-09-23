import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the Titanic Dataset
print("Loading Titanic data...")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Keep only useful columns and drop missing data
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

X = df[['Pclass', 'Sex', 'Age', 'Fare']].values
y = df['Survived'].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==========================================
# THE BUG: We purposely use only 40 passengers for training!
# This forces the massive model to memorize them (Overfitting).
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=40, random_state=42)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

# 2. Define a MASSIVE model (way too big for 40 rows)
model = nn.Sequential(
    nn.Linear(4, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 2)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

logs = []

print("Training model for 100 steps...")
# 3. The Training Loop
for step in range(100):
    model.train()
    optimizer.zero_grad()
    
    preds = model(X_train_t)
    loss = criterion(preds, y_train_t)
    loss.backward()
    
    # Calculate gradient norm for logging
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    grad_norm = total_norm ** 0.5
    
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_test_t)
        val_loss = criterion(val_preds, y_test_t)
        
    # Log the metrics exactly how mldoctor expects them
    logs.append({
        "step": step,
        "train_loss": round(loss.item(), 4),
        "val_loss": round(val_loss.item(), 4),
        "grad_norm": round(grad_norm, 4)
    })

# 4. Save the evidence
df_logs = pd.DataFrame(logs)
df_logs.to_csv("titanic_logs.csv", index=False)
print("Finished! Saved training curves to 'titanic_logs.csv'")
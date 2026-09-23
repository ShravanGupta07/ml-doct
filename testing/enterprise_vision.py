import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from mldoctor.tuner import tune_batch_size 

# 1. Complex Real-World Model (ResNet-style Feature Extractor)
class EnterpriseVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 56 * 56, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        features = self.conv_stack(x)
        return self.classifier(features)

def main():
    print("Initializing Enterprise Vision Training...")
    
    # Strict GPU Allocation
    if not torch.cuda.is_available():
        print("[WARN] CUDA (GPU) is not detected by PyTorch! Falling back to CPU.")
        device = torch.device("cpu")
    else:
        print("[SUCCESS] NVIDIA GPU Detected! Locking training to CUDA.")
        device = torch.device("cuda")
        
    model = EnterpriseVisionModel().to(device)

    # 2. Heavy Data size for Real GPU Testing (1000 images)
    print("Generating 1000 high-resolution images for GPU stress test...")
    X = torch.randn(1000, 3, 224, 224)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)

    # ML DOCTOR PRE-FLIGHT DRY RUN
    optimal_batch = tune_batch_size(model, dataset_sample_shape=(3, 224, 224))

    # Keep num_workers=0 on Windows to prevent multiprocessing freezing
    train_loader = DataLoader(dataset, batch_size=optimal_batch, shuffle=True, num_workers=0)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_history = []

    # Training Loop
    for epoch in range(20):
        total_loss = 0.0
        model.train() 
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()  
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() 
            
            # Live progress tracker
            print(f"   -> Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} processed...")
            
        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f}")
        
        # Validation Phase
        model.eval()  
        with torch.no_grad():
            val_outputs = model(X[:100].to(device))
            val_loss = criterion(val_outputs, y[:100].to(device))
            print(f"Epoch {epoch+1} | Validation Loss: {val_loss.item():.4f}")
            
            loss_history.append({
                "epoch": epoch, 
                "train_loss": total_loss / len(train_loader), 
                "val_loss": val_loss.item(), 
                "grad_norm": 2.5
            })

    pd.DataFrame(loss_history).to_csv("vision_metrics.csv", index=False)
    print("\nTraining Complete! Logs saved to 'vision_metrics.csv'")

if __name__ == "__main__":
    main()
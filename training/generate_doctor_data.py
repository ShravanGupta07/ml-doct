import os
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

# Use progress bar if available
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, desc="": x

# -----------------------------------------------------------------------------
# 1. Hardware & Simulation Settings
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_STEPS = 100        # Number of logging checkpoints per training run
RUNS_PER_CLASS = 70      # 10 classes * 70 runs = 700 total training curves
OUTPUT_FILE = "ml_training_failure_dataset.csv"

print(f"[*] Initialized ML Data Generator")
print(f"[*] Target Compute Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

# -----------------------------------------------------------------------------
# 2. Diagnosed Scenarios (10 Classes)
# -----------------------------------------------------------------------------
FAILURE_MODES = [
    "healthy",
    "lr_too_high",
    "lr_too_low",
    "overfitting",
    "underfitting",
    "exploding_gradients",
    "vanishing_gradients",
    "missing_zero_grad",
    "label_noise",
    "high_variance_batch"
]

# -----------------------------------------------------------------------------
# 3. Dynamic Neural Architecture for Simulation
# -----------------------------------------------------------------------------
class SimulationNet(nn.Module):
    def __init__(self, in_features=64, hidden_dim=128, out_features=10, 
                 depth=3, activation="relu", bad_init=False, init_scale=1.0):
        super().__init__()
        layers = []
        curr_dim = in_features
        
        for _ in range(depth):
            linear = nn.Linear(curr_dim, hidden_dim)
            if bad_init:
                # Deliberate poor initialization to trigger gradient explosions
                nn.init.normal_(linear.weight, mean=0.0, std=init_scale)
                nn.init.constant_(linear.bias, 0.0)
            layers.append(linear)
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation == "none":
                pass
            curr_dim = hidden_dim
            
        final_layer = nn.Linear(curr_dim, out_features)
        if bad_init:
            nn.init.normal_(final_layer.weight, mean=0.0, std=init_scale)
        layers.append(final_layer)
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# 4. Synthetic Data Synthesizer
# -----------------------------------------------------------------------------
def create_dataset(num_samples=1500, in_features=64, num_classes=10, noise_ratio=0.0, batch_size=32):
    # Generates standard feature clusters
    X = torch.randn(num_samples, in_features)
    true_matrix = torch.randn(in_features, num_classes)
    logits = X @ true_matrix
    y = torch.argmax(logits, dim=1)
    
    # Inject deliberate label corruption
    if noise_ratio > 0.0:
        noise_mask = torch.rand(num_samples) < noise_ratio
        random_y = torch.randint(0, num_classes, (int(noise_mask.sum()),))
        y[noise_mask] = random_y

    split = int(0.8 * num_samples)
    train_ds = TensorDataset(X[:split], y[:split])
    val_ds = TensorDataset(X[split:], y[split:])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    return train_loader, val_loader

# -----------------------------------------------------------------------------
# 5. Core Simulation Runner for a Single Curve
# -----------------------------------------------------------------------------
def simulate_single_run(run_id, mode):
    # Default Hyperparameters
    lr = 1e-3
    batch_size = 32
    activation = "relu"
    depth = 3
    bad_init = False
    init_scale = 1.0
    zero_grad_enabled = True
    noise_ratio = 0.0
    sample_count = 1600
    optimizer_choice = "adamw"

    # Inject deliberate real-world failure dynamics
    if mode == "healthy":
        lr = random.uniform(5e-4, 2e-3)
        depth = random.choice([2, 3, 4])
        optimizer_choice = random.choice(["adamw", "sgd"])
        
    elif mode == "lr_too_high":
        lr = random.uniform(0.15, 1.8) # Jumps wildly, overshooting local minima
        
    elif mode == "lr_too_low":
        lr = random.uniform(1e-8, 1e-6) # Microscopic steps; curve flatlines high
        
    elif mode == "overfitting":
        sample_count = 80              # Very few samples
        depth = 6                      # High network capacity (memorizes data)
        lr = 2e-3
        
    elif mode == "underfitting":
        depth = 1                      # 1 linear layer without activations
        activation = "none"
        
    elif mode == "exploding_gradients":
        bad_init = True
        init_scale = 7.5               # Massive weights blow up gradients
        lr = 0.02
        
    elif mode == "vanishing_gradients":
        depth = 12                     # Deep chain of unnormalized Sigmoids
        activation = "sigmoid"
        lr = 1e-3
        
    elif mode == "missing_zero_grad":
        zero_grad_enabled = False       # Accumulates past gradients endlessly
        lr = 1e-3
        
    elif mode == "label_noise":
        noise_ratio = random.uniform(0.75, 0.95) # 80%+ wrong labels
        
    elif mode == "high_variance_batch":
        batch_size = 2                  # Tiny batch size causes high trajectory noise
        lr = 0.01

    # Prepare data loaders
    train_loader, val_loader = create_dataset(
        num_samples=sample_count,
        noise_ratio=noise_ratio,
        batch_size=batch_size
    )

    # Instantiate model & optimizer
    model = SimulationNet(
        depth=depth,
        activation=activation,
        bad_init=bad_init,
        init_scale=init_scale
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    if optimizer_choice == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    records = []
    train_iter = iter(train_loader)
    has_collapsed = False

    for step in range(TOTAL_STEPS):
        if has_collapsed:
            # If math has already exploded, record sentinel collapsed values
            records.append({
                "run_id": run_id,
                "step": step,
                "train_loss": 99.0,
                "val_loss": 99.0,
                "grad_norm": 999.0,
                "label": mode
            })
            continue

        model.train()
        try:
            bx, by = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            bx, by = next(train_iter)

        bx, by = bx.to(DEVICE), by.to(DEVICE)

        if zero_grad_enabled:
            optimizer.zero_grad()

        output = model(bx)
        loss = criterion(output, by)
        loss_float = loss.item()

        # Guard against NaN/Inf crashes
        if math.isnan(loss_float) or math.isinf(loss_float) or loss_float > 50.0:
            has_collapsed = True
            loss_float = 99.0
            grad_norm = 999.0
        else:
            loss.backward()

            # Compute Gradient Norm across all parameters
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    p_norm = p.grad.data.norm(2)
                    total_norm += p_norm.item() ** 2
            grad_norm = total_norm ** 0.5

            if math.isnan(grad_norm) or math.isinf(grad_norm) or grad_norm > 500.0:
                has_collapsed = True
                grad_norm = 999.0
            else:
                optimizer.step()

        # Compute validation loss
        model.eval()
        val_accum = 0.0
        val_count = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                v_loss = criterion(model(vx), vy).item()
                if math.isnan(v_loss) or math.isinf(v_loss) or v_loss > 50.0:
                    val_accum += 99.0
                else:
                    val_accum += v_loss
                val_count += 1
                if val_count >= 2: # Keep simulation fast
                    break

        val_loss_float = val_accum / max(1, val_count)

        records.append({
            "run_id": run_id,
            "step": step,
            "train_loss": round(float(loss_float), 4),
            "val_loss": round(float(val_loss_float), 4),
            "grad_norm": round(float(grad_norm), 4),
            "label": mode
        })

    return records

# -----------------------------------------------------------------------------
# 6. Main Orchestrator
# -----------------------------------------------------------------------------
def main():
    total_runs = len(FAILURE_MODES) * RUNS_PER_CLASS
    print(f"[*] Commencing simulation: {len(FAILURE_MODES)} classes x {RUNS_PER_CLASS} runs = {total_runs} trajectories.")
    
    all_rows = []
    run_id = 0

    pbar = tqdm(total=total_runs, desc="Simulating Runs")
    for mode in FAILURE_MODES:
        for _ in range(RUNS_PER_CLASS):
            run_data = simulate_single_run(run_id, mode)
            all_rows.extend(run_data)
            run_id += 1
            pbar.update(1)
    pbar.close()

    # Save to CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n[+] Data generation complete!")
    print(f"[+] Output saved: {OUTPUT_FILE}")
    print(f"[+] Total rows logged: {len(df)} ({total_runs} runs x {TOTAL_STEPS} steps)")

if __name__ == "__main__":
    main()
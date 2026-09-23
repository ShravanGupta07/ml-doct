import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

DATA_FILE = "ml_training_failure_dataset.csv"
MODEL_OUTPUT = "doctor_model.pkl"

print(f"[*] Loading raw dataset from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

# -----------------------------------------------------------------------------
# 1. Feature Extraction (The "Translator")
# -----------------------------------------------------------------------------
print("[*] Extracting diagnostic features from time-series curves...")
features_list = []

# Group the 70,000 rows into the 700 individual training runs
grouped = df.groupby('run_id')

for run_id, group in grouped:
    # Sort by step to ensure chronological order
    group = group.sort_values('step')
    
    train_loss = group['train_loss'].values
    val_loss = group['val_loss'].values
    grad_norm = group['grad_norm'].values
    label = group['label'].iloc[0]
    
    # 1. Starting vs Ending metrics
    t_loss_start = train_loss[0]
    t_loss_end = train_loss[-1]
    v_loss_start = val_loss[0]
    v_loss_end = val_loss[-1]
    
    # 2. Progress Ratios (Did it actually learn?)
    # Avoid division by zero
    t_loss_drop = (t_loss_start - t_loss_end) / (t_loss_start + 1e-8)
    
    # 3. Overfitting Gap (Did validation diverge from train?)
    overfit_gap = v_loss_end - t_loss_end
    
    # 4. Spikiness and Variance (Sign of bad learning rates)
    t_loss_var = np.var(train_loss)
    grad_norm_var = np.var(grad_norm)
    
    # 5. Gradient Extremes (Exploding vs Vanishing)
    grad_max = np.max(grad_norm)
    grad_mean = np.mean(grad_norm)
    grad_min = np.min(grad_norm)
    
    # 6. Collapse Indicator (Did math break?)
    has_collapsed = 1.0 if (t_loss_end == 99.0 or grad_max == 999.0) else 0.0

    # Build the feature row
    features_list.append({
        't_loss_start': t_loss_start,
        't_loss_end': t_loss_end,
        'v_loss_start': v_loss_start,
        'v_loss_end': v_loss_end,
        't_loss_drop': t_loss_drop,
        'overfit_gap': overfit_gap,
        't_loss_var': t_loss_var,
        'grad_max': grad_max,
        'grad_mean': grad_mean,
        'grad_min': grad_min,
        'has_collapsed': has_collapsed,
        'label': label
    })

# Convert extracted features into a clean DataFrame
feature_df = pd.DataFrame(features_list)
print(f"[+] Extraction complete! Squished 70,000 steps into {len(feature_df)} feature profiles.")

# -----------------------------------------------------------------------------
# 2. Training the Diagnostic AI
# -----------------------------------------------------------------------------
# Separate our inputs (X) from the answer key (y)
X = feature_df.drop('label', axis=1)
y = feature_df['label']

# Split: 80% for studying, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("[*] Training HistGradientBoostingClassifier (Tree-based model)...")
clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42)
clf.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 3. Evaluating Accuracy
# -----------------------------------------------------------------------------
print("[*] Running final exam on test data...")
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n[+] Model Accuracy: {accuracy * 100:.2f}%\n")
print("=== Detailed Diagnosis Report ===")
print(classification_report(y_test, y_pred))

# -----------------------------------------------------------------------------
# 4. Save the Model Artifact
# -----------------------------------------------------------------------------
with open(MODEL_OUTPUT, 'wb') as f:
    pickle.dump(clf, f)

print(f"[+] Success! Diagnostic model saved as '{MODEL_OUTPUT}'")
print(f"    This file is tiny and ready to be packaged into your CLI.")
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import itertools
import random

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 1. Load data
df = pd.read_csv("cnn_ready_with_pheno.csv", index_col=0)
X = df.drop(columns=["phenotype"]).values.astype(np.float32)
y = df["phenotype"].values.astype(np.float32)

# Print first 20 rows of features and labels
print("First 20 feature rows:\n", X[:20])
print("\nFirst 20 labels:\n", y[:20])

# 2. Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Convert to torch tensors
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y).unsqueeze(1)  # (n_samples, 1)

# 4. Dataset and DataLoader
dataset = TensorDataset(X_tensor, y_tensor)

# Train/test split (80/20)
n_train = int(0.8 * len(dataset))
n_test = len(dataset) - n_train
train_ds, test_ds = random_split(dataset, [n_train, n_test])

# Model definition with configurable dropout
class RegularizedImprovedCNN(nn.Module):
    def __init__(self, n_features, dropout1=0.3, dropout2=0.4, dropout_fc=0.5):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(0.2),  # Use Dropout1d for 1D data
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout1)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout2)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, n_features)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.global_pool(x)  # (batch, channels, 1)
        x = x.view(x.size(0), -1)  # (batch, channels)
        x = self.fc(x)
        return x

# Hyperparameter grid
param_grid = {
    "lr": [0.0001, 0.00005],
    "batch_size": [16, 32],
    "dropout1": [0.3, 0.4],
    "dropout2": [0.4, 0.5],
    "dropout_fc": [0.5, 0.6],
    "weight_decay": [0, 1e-4],
}

keys = list(param_grid.keys())
grid = list(itertools.product(*param_grid.values()))

best_mse = float("inf")
best_params = None
results = []

for values in grid:
    params = dict(zip(keys, values))
    print(f"Testing: {params}")
    # DataLoaders for this run
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=params["batch_size"])
    # Model/optimizer
    model = RegularizedImprovedCNN(
        X.shape[1], 
        dropout1=params["dropout1"], 
        dropout2=params["dropout2"], 
        dropout_fc=params["dropout_fc"]
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    n_epochs = 12  # Reduce for faster grid search, increase for final training

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        running_loss /= n_train
    # Evaluate
    model.eval()
    with torch.no_grad():
        all_preds, all_targets = [], []
        for xb, yb in test_loader:
            preds = model(xb)
            all_preds.append(preds.numpy())
            all_targets.append(yb.numpy())
        pred = np.concatenate(all_preds)
        true = np.concatenate(all_targets)
        mse = mean_squared_error(true, pred)
        r2 = r2_score(true, pred)
        print(f"Test MSE: {mse:.4f}, Test R2: {r2:.4f}")
        results.append((params, mse, r2))
        if mse < best_mse:
            best_mse = mse
            best_params = params

print("\nBest parameters:", best_params)
print("Best Test MSE:", best_mse)
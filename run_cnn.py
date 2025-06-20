import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim

# 1. Load data
df = pd.read_csv("cnn_ready_with_pheno.csv", index_col=0)
X = df.drop(columns=["phenotype"]).values.astype(np.float32)
y = df["phenotype"].values.astype(np.float32)

# 2. Optional: Standardize features
from sklearn.preprocessing import StandardScaler
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
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)

# 5. Build a simple 1D CNN
class SimpleCNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear((n_features//2)*16, 32)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, n_features)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleCNN(X.shape[1])

# 6. Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Training loop
n_epochs = 20
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
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {running_loss:.4f}")

# 8. Evaluate
model.eval()
with torch.no_grad():
    all_preds, all_targets = [], []
    for xb, yb in test_loader:
        preds = model(xb)
        all_preds.append(preds.numpy())
        all_targets.append(yb.numpy())
    pred = np.concatenate(all_preds)
    true = np.concatenate(all_targets)
    mse = np.mean((pred - true) ** 2)
    print(f"Test MSE: {mse:.4f}")
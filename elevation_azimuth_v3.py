# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 17:40:26 2025

@author: lab619
"""

# -*- coding: utf-8 -*-
"""
Direct PITT regression: time -> [elevation, azimuth]
(Train to match Excel angle values directly)
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 讀取 CSV
# ============================================================
df = pd.read_csv("satellite_angles.csv")
time = df.iloc[:, 0].to_numpy(dtype=np.float32)
time = time - time[0]                         # 從 0 開始計算 elapsed time
raw_data = df.iloc[:, 1:3].to_numpy(dtype=np.float32)  # [Elevation, Azimuth]

# 正規化
t_mean, t_std = time.mean(), time.std()
x_time = (time - t_mean) / t_std

mean = raw_data.mean(axis=0, keepdims=True)
std = raw_data.std(axis=0, keepdims=True)
y_data = (raw_data - mean) / std              # y 標準化

# ============================================================
# 2. Dataset
# ============================================================
class TimeDataset(Dataset):
    def __init__(self, t, y):
        self.t = t
        self.y = y
    def __len__(self):
        return len(self.t)
    def __getitem__(self, i):
        return torch.tensor([self.t[i]]), torch.tensor(self.y[i])

ds = TimeDataset(x_time, y_data)
n_train = int(len(ds) * 0.8)
train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, len(ds) - n_train])
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64)

# ============================================================
# 3. 模型定義 (MLP or simple transformer encoder)
# ============================================================
class PITTRegressor(nn.Module):
    def __init__(self, d_model=64, nlayers=3, dff=128, dim_out=2, dropout=0.8):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, d_model))
        layers.append(nn.GELU())
        for _ in range(nlayers - 1):
            layers.append(nn.Linear(d_model, dff))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))   # ✨ dropout 在這裡
            layers.append(nn.Linear(dff, d_model))
            layers.append(nn.GELU())
        layers.append(nn.Linear(d_model, dim_out))
        self.net = nn.Sequential(*layers)
    def forward(self, t):
        return self.net(t)

# ============================================================
# 4. 訓練
# ============================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PITTRegressor().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

for epoch in range(500):
    model.train()
    total_loss = 0
    for t_batch, y_batch in train_dl:
        t_batch, y_batch = t_batch.to(device), y_batch.to(device)
        pred = model(t_batch)
        loss = loss_fn(pred, y_batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:03d} | Loss = {total_loss/len(train_dl):.6e}")

# ============================================================
# 5. 評估
# ============================================================
model.eval()
t_tensor = torch.tensor(x_time).unsqueeze(1).to(device)
with torch.no_grad():
    pred_norm = model(t_tensor).cpu().numpy()

# 反標準化
pred_raw = pred_norm * std + mean

# ============================================================
# 6. 視覺化
# ============================================================
plt.figure(figsize=(10,5))
plt.plot(time, raw_data[:,0], label='True Elevation (deg)')
plt.plot(time, pred_raw[:,0], '--', label='Pred Elevation (deg)')
plt.xlabel('Elapsed Time (s)')
plt.ylabel('Elevation (deg)')
plt.title('Direct Regression of Elevation')
plt.legend(); plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(time, raw_data[:,1], label='True Azimuth (deg)')
plt.plot(time, pred_raw[:,1], '--', label='Pred Azimuth (deg)')
plt.xlabel('Elapsed Time (s)')
plt.ylabel('Azimuth (deg)')
plt.title('Direct Regression of Azimuth')
plt.legend(); plt.grid(True)
plt.show()

# 誤差統計
mse = np.mean((pred_raw - raw_data)**2, axis=0)
mae = np.mean(np.abs(pred_raw - raw_data), axis=0)
print(f"\nElevation MSE={mse[0]:.6f}, MAE={mae[0]:.6f}")
print(f"Azimuth   MSE={mse[1]:.6f}, MAE={mae[1]:.6f}")

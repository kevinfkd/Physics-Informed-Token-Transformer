# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 20:02:43 2025

@author: lab619
"""

# -*- coding: utf-8 -*-
"""
PITT-like ODE physics-informed simulation (no external function calls)
Created on Mon Oct 13 2025
@author: lab619
"""

# ============================================================
# === Imports 與隨機種子 ===
# ============================================================
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# === 系統參數設定 ===
# ============================================================
num_steps = 150
d  = 2
nx = 4    # [aR, aI, theta, phi]
ny = 2    # [rR, rI]
sigma_v = 1
sigma_n = 2

# ============================================================
# === 系統矩陣 A (block companion form)
# ============================================================
I4 = np.eye(nx)
Z4 = np.zeros((nx, nx))
g  = [0.8, 0.15, 0.05]
A_top = np.hstack([g[0]*I4, g[1]*I4, g[2]*I4])
A_mid = np.hstack([I4, Z4, Z4])
A_bot = np.hstack([Z4, I4, Z4])
A = np.vstack([A_top, A_mid, A_bot])      # 12×12 離散 A

# 連續化近似
Ts = 0.05
Ac = (A - np.eye(A.shape[0]))/Ts
Ac = torch.tensor(Ac, dtype=torch.float32, device=device)

# ============================================================
# === 生成連續系統真解 (RK4 展開式)
# ============================================================
T  = num_steps * Ts
t_grid = np.linspace(0, T, num_steps)

X_true = np.zeros((num_steps, nx*(d+1)))
X_true[0,:] = np.array([1.0, 0.5, 0.2, 0.1,
                        1.0, 0.5, 0.2, 0.1,
                        1.0, 0.5, 0.2, 0.1]) + 0.002*np.random.randn(nx*(d+1))

Ac_np = Ac.detach().cpu().numpy()

alpha = 0.6
L = 8
for k in range(num_steps-1):
    x = X_true[k,:]
    # ---- RK4 手動展開 ----
    k1 = Ac_np @ x
    k2 = Ac_np @ (x + 0.5*Ts*k1)
    k3 = Ac_np @ (x + 0.5*Ts*k2)
    k4 = Ac_np @ (x + Ts*k3)
    x_next = x + (Ts/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    x_next += sigma_v*np.random.randn(nx*(d+1))
    
    #if k >= 2:
        #x_prev1 = X_true[k-1,:]
        #x_prev2 = X_true[k-2,:]
        #x_next = (x_next + x_prev1 + x_prev2) / 3.0
     # ---- 加權滑動平滑 (Weighted Smoothing) ----
    if k >= L:
        weights = np.linspace(1, L, L)
        weights /= weights.sum()       # 正規化成權重
        past_vals = np.stack([X_true[k-i,:] for i in range(L)], axis=0)
        smoothed = np.average(past_vals, axis=0, weights=weights)
        # 混合指數平滑 + 加權平均
        x_next = alpha * x_next + (1 - alpha) * smoothed


    X_true[k+1,:] = x_next

# ============================================================
# === 非線性觀測 (展開式)
# ============================================================
Y_meas = np.zeros((num_steps, ny))
for k in range(num_steps):
    aR, aI, theta, phi = X_true[k,0], X_true[k,1], X_true[k,2], X_true[k,3]
    rR = aR*np.cos(theta) - aI*np.sin(phi)
    rI = aR*np.sin(theta) + aI*np.cos(phi)
    Y_meas[k,0] = rR + sigma_n*np.random.randn()
    Y_meas[k,1] = rI + sigma_n*np.random.randn()

# ============================================================
# === PITT-like model 定義 (內建在主體內)
# ============================================================
class PITTlite(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, nx*(d+1))
        )
    def forward(self, t):
        return self.net(t)

model = PITTlite().to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

# ============================================================
# === 轉成 Torch tensor
# ============================================================
t_torch   = torch.tensor(t_grid, dtype=torch.float32, device=device).unsqueeze(-1)
X_true_th = torch.tensor(X_true, dtype=torch.float32, device=device)
Y_meas_th = torch.tensor(Y_meas, dtype=torch.float32, device=device)
Ac        = Ac.float().to(device)
model     = model.float()   # ✅ 這一行非常關鍵

# Loss 權重
w_ode = 1.0
w_ic  = 5.0
w_dat = 0.5

# ============================================================
# === 訓練
# ============================================================
history = []
for epoch in range(2000):
    t_torch.requires_grad_(True)
    X_hat = model(t_torch)  # [N,12]

    # ---- Physics ODE loss ----
    dXdt = torch.autograd.grad(X_hat, t_torch,
                               grad_outputs=torch.ones_like(X_hat),
                               create_graph=True)[0]
    ode_res = dXdt - (X_hat @ Ac.T)
    loss_ode = torch.mean(ode_res**2)

    # ---- 初值 loss ----
    loss_ic = torch.mean((X_hat[0,:] - X_true_th[0,:])**2)

    # ---- 資料 loss ----
    aR = X_hat[:,0]; aI = X_hat[:,1]; theta = X_hat[:,2]; phi = X_hat[:,3]
    rR = aR*torch.cos(theta) - aI*torch.sin(phi)
    rI = aR*torch.sin(theta) + aI*torch.cos(phi)
    y_hat = torch.stack([rR, rI], dim=-1)
    loss_dat = torch.mean((y_hat - Y_meas_th)**2)

    # ---- 總 loss ----
    loss = w_ode*loss_ode + w_ic*loss_ic + w_dat*loss_dat
    opt.zero_grad()
    loss.backward()
    opt.step()

    history.append([loss.item(), loss_ode.item(), loss_ic.item(), loss_dat.item()])

    if (epoch+1) % 500 == 0:
        print(f"Epoch {epoch+1:4d} | total={loss.item():.3e} | ode={loss_ode.item():.3e} | ic={loss_ic.item():.3e} | data={loss_dat.item():.3e}")

# ============================================================
# === 評估與誤差分析 ===
# ============================================================
X_hat_eval = model(t_torch.detach()).detach()
err = X_hat_eval - X_true_th
rel_L2  = torch.norm(err)/torch.norm(X_true_th)
Linf    = torch.max(torch.abs(err))
print(f"\nRelative L2 = {rel_L2.item():.3e},  Linf = {Linf.item():.3e}")

# ============================================================
# === 畫圖 ===
# ============================================================
idxs = [0,1,2,3]
plt.figure(figsize=(11,6))
for i,idx in enumerate(idxs,1):
    plt.subplot(2,2,i)
    plt.plot(t_grid, X_true[:,idx], label='True')
    plt.plot(t_grid, X_hat_eval[:,idx].cpu(), '--', label='PITT-like')
    plt.title(f"State dim {idx}")
    plt.grid(True)
    if i==1: plt.legend()
plt.tight_layout(); plt.show()

# ---- 觀測對齊 ----
aR = X_hat_eval[:,0]; aI = X_hat_eval[:,1]; theta = X_hat_eval[:,2]; phi = X_hat_eval[:,3]
rR = aR*torch.cos(theta) - aI*torch.sin(phi)
rI = aR*torch.sin(theta) + aI*torch.cos(phi)
y_hat_eval = torch.stack([rR, rI], dim=-1).detach().cpu().numpy()

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(t_grid, Y_meas[:,0], label='y_meas rR')
plt.plot(t_grid, y_hat_eval[:,0], '--', label='y_hat rR')
plt.grid(True); plt.legend()
plt.subplot(1,2,2)
plt.plot(t_grid, Y_meas[:,1], label='y_meas rI')
plt.plot(t_grid, y_hat_eval[:,1], '--', label='y_hat rI')
plt.grid(True); plt.legend()
plt.tight_layout(); plt.show()

# ---- 損失曲線 ----
history = np.array(history)
plt.figure(figsize=(9,4))
plt.plot(history[:,0], label='total')
plt.plot(history[:,1], label='ode')
plt.plot(history[:,2], label='ic')
plt.plot(history[:,3], label='data')
plt.yscale('log'); plt.grid(True); plt.legend(); plt.title("Loss curves"); plt.show()

# ============================================================
# PITT-like Estimation for a_real, a_imag (0~1200 sec)
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

# ============================================================
# Time Axis (1200 sec)
# ============================================================
T_final = 1200
num_steps = 3000  # 降解析但可貼合
t_grid = np.linspace(0, T_final, num_steps)

# Normalize time for NN
t_norm = (t_grid - t_grid[0]) / (t_grid[-1] - t_grid[0])
T_total = t_grid[-1] - t_grid[0]

# Torch time tensor
t_torch = torch.tensor(t_norm, dtype=torch.float32, device=device).unsqueeze(-1)

# ============================================================
# True θ(t), φ(t) (從你 elevation/azimuth 模擬而來)
# ============================================================

theta_true = 35*np.sin(np.pi * t_grid / 1200) * np.pi/180  # rad
phi_true   = (160 - 150*np.sin(np.pi * t_grid / 1200)) * np.pi/180  # rad

# ============================================================
# Generate a_real[k], a_imag[k] (Damped oscillation + noise)
# ============================================================

rng = np.random.default_rng(0)
t = t_grid

# ---- a_real ----
osc_real = np.exp(-t/150.0) * np.sin(2*np.pi*t/40.0)
slow_real = 0.2 * np.sin(2*np.pi*t/600.0)
noise_real = 0.1 * rng.standard_normal(num_steps)

a_real = 4.0 + 1.2*osc_real + slow_real + noise_real

# ---- a_imag ----
osc_imag = np.exp(-t/180.0) * np.cos(2*np.pi*t/55.0 + 0.5)
slow_imag = 0.2 * np.cos(2*np.pi*t/500.0 + 0.7)
noise_imag = 0.1 * rng.standard_normal(num_steps)

a_imag = 2.5 + 0.9*osc_imag + slow_imag + noise_imag

# Stack true state
X_true = np.stack([a_real, a_imag], axis=1)
X_true_th = torch.tensor(X_true, dtype=torch.float32, device=device)

# ============================================================
# Nonlinear Measurement r_real[k], r_imag[k]
# ============================================================

sigma_n = 0.1
Y_meas = np.zeros((num_steps, 2))

for k in range(num_steps):
    aR = a_real[k]
    aI = a_imag[k]
    th = theta_true[k]
    ph = phi_true[k]

    rR = aR*np.cos(th) - aI*np.sin(ph)
    rI = aR*np.sin(th) + aI*np.cos(ph)

    Y_meas[k,0] = rR + sigma_n*rng.standard_normal()
    Y_meas[k,1] = rI + sigma_n*rng.standard_normal()

Y_meas_th = torch.tensor(Y_meas, dtype=torch.float32, device=device)

# ============================================================
# PITT-like Neural Network
# ============================================================

class PITTlite(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2)     # Output: a_real, a_imag
        )
    def forward(self, t):
        return self.net(t)

model = PITTlite().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

# Data weight
w_dat = 1.0
w_ic = 5.0

# ============================================================
# Train
# ============================================================

history = []
epochs = 1500

for ep in range(epochs):
    optimizer.zero_grad()

    X_hat = model(t_torch)  # [N, 2]

    # Initial condition loss
    loss_ic = torch.mean((X_hat[0] - X_true_th[0])**2)

    # Compute predicted measurement
    aR = X_hat[:,0]
    aI = X_hat[:,1]

    theta_t = torch.tensor(theta_true, dtype=torch.float32, device=device)
    phi_t   = torch.tensor(phi_true, dtype=torch.float32, device=device)

    rR_hat = aR*torch.cos(theta_t) - aI*torch.sin(phi_t)
    rI_hat = aR*torch.sin(theta_t) + aI*torch.cos(phi_t)
    Y_hat  = torch.stack([rR_hat, rI_hat], dim=1)

    # Measurement loss
    loss_dat = torch.mean((Y_hat - Y_meas_th)**2)

    # Total loss
    loss = w_ic*loss_ic + w_dat*loss_dat
    loss.backward()
    optimizer.step()

    history.append(loss.item())

    if (ep+1) % 200 == 0:
        print(f"Epoch {ep+1} / {epochs}, Loss = {loss.item():.4e}")

# ============================================================
# Evaluate
# ============================================================

with torch.no_grad():
    X_hat_eval = model(t_torch).cpu().numpy()

# Recompute output
aR_hat = X_hat_eval[:,0]
aI_hat = X_hat_eval[:,1]

rR_hat = aR_hat*np.cos(theta_true) - aI_hat*np.sin(phi_true)
rI_hat = aR_hat*np.sin(theta_true) + aI_hat*np.cos(phi_true)

# ============================================================
# Plot 1: True vs PITT (State)
# ============================================================
# ============================================================
# Plot 1: a_real (True vs PITT)
# ============================================================
plt.figure(figsize=(10,4))
plt.plot(t_grid, a_real, label="True a_real", linewidth=2)
plt.plot(t_grid, X_hat_eval[:,0], '--', label="PITT a_real", linewidth=2)
plt.title("State Estimation: a_real")
plt.xlabel("Time (sec)")
plt.ylabel("a_real")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# Plot 2: a_imag (True vs PITT)
# ============================================================
plt.figure(figsize=(10,4))
plt.plot(t_grid, a_imag, label="True a_imag", linewidth=2)
plt.plot(t_grid, X_hat_eval[:,1], '--', label="PITT a_imag", linewidth=2)
plt.title("State Estimation: a_imag")
plt.xlabel("Time (sec)")
plt.ylabel("a_imag")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# Plot 3: r_real (True vs PITT)
# ============================================================
plt.figure(figsize=(10,4))
plt.plot(t_grid, Y_meas[:,0], label="True r_real", linewidth=2)
plt.plot(t_grid, rR_hat, '--', label="PITT r_real", linewidth=2)
plt.title("Output Estimation: r_real")
plt.xlabel("Time (sec)")
plt.ylabel("r_real")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# Plot 4: r_imag (True vs PITT)
# ============================================================
plt.figure(figsize=(10,4))
plt.plot(t_grid, Y_meas[:,1], label="True r_imag", linewidth=2)
plt.plot(t_grid, rI_hat, '--', label="PITT r_imag", linewidth=2)
plt.title("Output Estimation: r_imag")
plt.xlabel("Time (sec)")
plt.ylabel("r_imag")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
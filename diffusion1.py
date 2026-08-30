import numpy as np
import matplotlib.pyplot as plt

# ========== 1. set coefficient ==========
D = 1.0    # diffusion coefficient
L = 10.0   # total length of space
nx = 100   # number of spatial points
dx = L / nx
dt = 0.001 # time step
nt = 1000  # number of steps

# stability condition
print(f"stability check: D*dt/dx^2 = {D*dt/dx**2:.4f} (need < 0.5)")

# ========== 2. initial condition ==========
x = np.linspace(0, L, nx)
C = np.zeros(nx)
C[40:60] = 1.0   # initial high concentration region

# ========== 3. time stepping ==========
snapshots = [0, 200, 400, 600, 800, 999]  # time steps to plot

for n in range(nt):
    C_new = C.copy()
    for i in range(1, nx - 1):  # boundary points remain unchanged (zero)
        C_new[i] = C[i] + D * dt / dx**2 * (C[i+1] - 2*C[i] + C[i-1])
    C = C_new

    if n in snapshots:
        plt.plot(x, C, label=f't = {n*dt:.2f}')

# ========== 4. plot ==========
plt.xlabel('position x')
plt.ylabel('concentration c')
plt.title("one-dimensional diffusion simulation")
plt.legend()
plt.show()
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parameters (modifiable)
# ============================================================
porosity = 0.343                # porosity from Boolean model
D0_water = 1.33e-9              # diffusivity of Na+ in free water (m^2/s)
q_max = 2.0                     # maximum adsorption capacity (dimensionless)
K = 2.0                         # Langmuir equilibrium constant (dimensionless)
C_inlet = 1.0                   # surface liquid concentration (dimensionless)
L = 1.0                         # space length (dimensionless)
T = 5.0                         # simulation time (enough to approach equilibrium)

# Effective diffusivity and dimensionless mapping
porosity_ref = 0.343
D_base = 0.1
D = D_base * (porosity / porosity_ref) ** 1.5
D_eff_actual = D0_water * porosity ** 1.5

print(f"Actual effective diffusivity D_eff = {D_eff_actual:.3e} m^2/s")
print(f"PINN dimensionless diffusivity D = {D:.4f}")

# ============================================================
# PDE definition (with Langmuir adsorption)
# ============================================================
def pde(x, u):
    C = u[:, 0:1]
    dC_dt = dde.grad.jacobian(u, x, i=0, j=1)
    d2C_dx2 = dde.grad.hessian(u, x, i=0, j=0)
    # Langmuir adsorption term: dq/dt = dq/dC * dC/dt
    dq_dt = (q_max * K / (1 + K * C) ** 2) * dC_dt
    return dC_dt - D * d2C_dx2 + dq_dt   # rho = 1 (can be adjusted)

# ============================================================
# Geometry and boundary/initial conditions
# ============================================================
geom = dde.geometry.Interval(0, L)
timedomain = dde.geometry.TimeDomain(0, T)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Left boundary (x=0): zero flux (symmetry center)
def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)
bc_left = dde.icbc.NeumannBC(geomtime, lambda x: 0, boundary_left)

# Right boundary (x=L): fixed concentration C_inlet
def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], L)
bc_right = dde.icbc.DirichletBC(geomtime, lambda x: C_inlet, boundary_right)

# Initial condition: smooth transition near surface to avoid discontinuity
def init_cond(x):
    # Use a wider smooth step compared to previous version
    return C_inlet * 0.5 * (1 + np.tanh((x[:, 0:1] - 0.9) / 0.05))

ic = dde.icbc.IC(geomtime, init_cond, lambda x, on_initial: on_initial)

# ============================================================
# Training data settings (balance accuracy and cost)
# ============================================================
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_left, bc_right, ic],
    num_domain=2048,               # more internal points for stability
    num_boundary=256,
    num_initial=512,
    train_distribution="Sobol"
)

# ============================================================
# Neural network (moderate size)
# ============================================================
net = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)

# ============================================================
# Training strategy
# ============================================================
# Adam pre-training
model.compile("adam", lr=1e-3, loss_weights=[1, 1, 1, 1])
# Increase iterations for better convergence
losshistory, train_state = model.train(iterations=15000, display_every=1000)

# Optional L-BFGS refinement (uncomment if needed)
# model.compile("L-BFGS")
# losshistory, train_state = model.train()

# ============================================================
# Prediction
# ============================================================
x = np.linspace(0, L, 200)
t = np.linspace(0, T, 200)
X, T_grid = np.meshgrid(x, t)
X_star = np.column_stack([X.ravel(), T_grid.ravel()])
u_pred = model.predict(X_star).reshape(X.shape)

# Clip negative values (only for display; physically should be non-negative)
u_pred = np.clip(u_pred, 0, None)

# ============================================================
# Extract adsorption kinetics (average adsorbed amount vs time)
# ============================================================
# Langmuir adsorption amount q = q_max * K * C / (1 + K * C)
q_pred = q_max * K * u_pred / (1 + K * u_pred)

# Spatial average using trapezoidal integration
avg_q = np.trapz(q_pred, x, axis=1) / L

# Plot kinetics
plt.figure(figsize=(8, 5))
plt.plot(t, avg_q, linewidth=2, label="Average adsorbed amount")
plt.xlabel("Time (dimensionless)")
plt.ylabel("Average adsorbed amount q_avg")
plt.title("Adsorption kinetics")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Print theoretical equilibrium adsorbed amount
q_eq_theory = q_max * K * C_inlet / (1 + K * C_inlet)
print(f"Theoretical equilibrium q_eq = {q_eq_theory:.3f}")
print(f"Simulated final average q = {avg_q[-1]:.3f}")

# ============================================================
# Optional: plot concentration profiles at selected times
# ============================================================
plt.figure(figsize=(8, 5))
plot_times = [0, 0.5, 1, 2, 3, 5]
for ti in plot_times:
    idx = np.argmin(np.abs(t - ti))
    plt.plot(x, u_pred[idx], label=f"t = {t[idx]:.2f}", linewidth=1.5)
plt.xlabel("x (particle center → surface)")
plt.ylabel("C")
plt.title("Concentration profiles at different times")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
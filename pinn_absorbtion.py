import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 物理参数与有效扩散系数计算
# ============================================================
porosity = 0.343          # 来自孔隙生成模块
D0_water = 1.33e-9        # Na+ 在自由水中的扩散系数 (m^2/s)
D_eff_actual = D0_water * porosity ** 1.5   # 实际有效扩散系数 (m^2/s)

# 无量纲映射（保持训练稳定）
porosity_ref = 0.343      # 参考孔隙率
D_base = 0.1              # 该孔隙率下训练稳定的无量纲 D
D = D_base * (porosity / porosity_ref) ** 1.5

print(f"实际有效扩散系数 D_eff = {D_eff_actual:.3e} m^2/s")
print(f"PINN 无量纲扩散系数 D = {D:.4f}")

rho = 1.0
q_max = 4.5
K = 10.0
L = 1.0
T = 1.0

# ============================================================
# PDE 定义（含 Langmuir 吸附项）
# ============================================================
def pde(x, u):
    C = u[:, 0:1]
    dC_dt = dde.grad.jacobian(u, x, i=0, j=1)
    d2C_dx2 = dde.grad.hessian(u, x, i=0, j=0)

    # Langmuir 吸附项: dq/dt = dq/dC * dC/dt
    dq_dt = (q_max * K / (1 + K * C) ** 2) * dC_dt

    return dC_dt - D * d2C_dx2 + rho * dq_dt

# ============================================================
# 几何定义
# ============================================================
geom = dde.geometry.Interval(0, L)
timedomain = dde.geometry.TimeDomain(0, T)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# ============================================================
# 边界条件
# ============================================================
# 左边界：零通量 (Neumann)
def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

bc_left = dde.icbc.NeumannBC(geomtime, lambda x: 0, boundary_left)

# 右边界：固定浓度 1 (Dirichlet)
def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], L)

bc_right = dde.icbc.DirichletBC(geomtime, lambda x: 1.0, boundary_right)

# ============================================================
# 初始条件（软化过渡，避免阶跃冲突）
# ============================================================
def init_cond(x):
    return 0.5 * (1 + np.tanh((x[:, 0:1] - 0.95) / 0.02))

ic = dde.icbc.IC(geomtime, init_cond, lambda x, on_initial: on_initial)

# ============================================================
# 训练数据设置
# ============================================================
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_left, bc_right, ic],
    num_domain=2048,
    num_boundary=256,
    num_initial=512,
    train_distribution="Sobol"
)

# ============================================================
# 神经网络
# ============================================================
net = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot normal")

# ============================================================
# 模型与训练
# ============================================================
model = dde.Model(data, net)

# Adam 训练（带损失权重）
model.compile("adam", lr=1e-3, loss_weights=[10, 1, 1, 1])

# PDE 点重采样（避免边界点数量变化错误）
resampler = dde.callbacks.PDEPointResampler(
    period=1000,
    pde_points=True,
    bc_points=False
)

losshistory_adam, train_state_adam = model.train(
    iterations=12000,
    display_every=1000,
    disregard_previous_best=True,
    callbacks=[resampler]
)

# L-BFGS 精调
model.compile("L-BFGS")
losshistory_lbfgs, train_state_lbfgs = model.train()

# ============================================================
# 预测
# ============================================================
x = np.linspace(0, L, 300)
t = np.linspace(0, T, 200)
X, T_grid = np.meshgrid(x, t)
X_star = np.column_stack([X.ravel(), T_grid.ravel()])
u_pred = model.predict(X_star).reshape(X.shape)

# ============================================================
# 绘制浓度分布图
# ============================================================
plt.figure(figsize=(10, 6))

plot_times = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
for ti in plot_times:
    idx = np.argmin(np.abs(t - ti))
    plt.plot(x, u_pred[idx], label=f"t = {t[idx]:.3f}", linewidth=2)

plt.xlabel("x (particle center → surface)")
plt.ylabel("C")
plt.title("Diffusion–adsorption concentration profile")
plt.legend(loc="center left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
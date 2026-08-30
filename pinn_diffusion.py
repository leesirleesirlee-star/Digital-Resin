import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Parameters
# ============================================================
D = 0.1
L = 1.0
T = 1.0

# ============================================================
# 2. PDE: u_t = D u_xx
# ============================================================
def pde(x, u):
    u_t = dde.grad.jacobian(u, x, i=0, j=1)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)

    return u_t - D * u_xx


# ============================================================
# 3. Geometry
# ============================================================
geom = dde.geometry.Interval(0, L)
timedomain = dde.geometry.TimeDomain(0, T)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# ============================================================
# 4. Boundary conditions
# ============================================================
def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], L)


bc_left = dde.icbc.DirichletBC(
    geomtime,
    lambda x: 0,
    boundary_left,
)

bc_right = dde.icbc.DirichletBC(
    geomtime,
    lambda x: 0,
    boundary_right,
)


# ============================================================
# 5. Initial condition
# ============================================================
def init_cond(x):
    return np.exp(-((x[:, 0:1] - 0.5) ** 2) / 0.01)


ic = dde.icbc.IC(
    geomtime,
    init_cond,
    lambda x, on_initial: on_initial,
)


# ============================================================
# 6. Training data
# ============================================================
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_left, bc_right, ic],

    # Start smaller than your original configuration.
    num_domain=1500,
    num_boundary=100,
    num_initial=500,

    # Better than repeatedly generating exactly the same points.
    train_distribution="Hammersley",
)


# ============================================================
# 7. Neural network
# ============================================================
net = dde.nn.FNN(
    [2] + [64] * 4 + [1],
    "tanh",
    "Glorot normal",
)


# ============================================================
# 8. Model
# ============================================================
model = dde.Model(data, net)


# ============================================================
# 9. Adam pre-training
# ============================================================
model.compile(
    "adam",
    lr=1e-3,
)

losshistory, train_state = model.train(
    iterations=10000,
    display_every=1000,
)


# ============================================================
# 10. L-BFGS refinement
# ============================================================
model.compile("L-BFGS")

losshistory, train_state = model.train(
    display_every=1000,
)


# ============================================================
# 11. Prediction
# ============================================================
x = np.linspace(0, L, 200)
t = np.linspace(0, T, 200)

X, T_grid = np.meshgrid(x, t)

X_star = np.column_stack(
    (
        X.ravel(),
        T_grid.ravel(),
    )
)

u_pred = model.predict(X_star).reshape(X.shape)


# ============================================================
# 12. Plot loss
# ============================================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.semilogy(
    np.sum(losshistory.loss_train, axis=1),
)

plt.xlabel("Training step")
plt.ylabel("Total loss")
plt.title("PINN training loss")


# ============================================================
# 13. Plot solution at different times
# ============================================================
plt.subplot(1, 2, 2)

for ti in [0, 0.1, 0.25, 0.5, 1.0]:

    idx = np.argmin(np.abs(t - ti))

    plt.plot(
        x,
        u_pred[idx],
        label=f"t = {t[idx]:.2f}",
    )

plt.xlabel("x")
plt.ylabel("u")
plt.title("PINN solution")

plt.legend()
plt.tight_layout()
plt.show()
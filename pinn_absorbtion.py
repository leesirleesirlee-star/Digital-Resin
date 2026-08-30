import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Font settings
# ============================================================

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# Physical parameters
# ============================================================

D = 0.1
rho = 1.0
q_max = 4.5
K = 10.0

L = 1.0
T = 1.0


# ============================================================
# PDE
# ============================================================

def pde(x, u):

    # Concentration
    C = u[:, 0:1]

    # dC/dt
    dC_dt = dde.grad.jacobian(
        u,
        x,
        i=0,
        j=1
    )

    # d²C/dx²
    d2C_dx2 = dde.grad.hessian(
        u,
        x,
        i=0,
        j=0
    )

    # Langmuir adsorption:
    #
    # q = q_max*K*C/(1 + K*C)
    #
    # dq/dC = q_max*K/(1 + K*C)^2
    #
    dq_dt = (
        q_max * K /
        (1 + K * C) ** 2
    ) * dC_dt

    # Governing equation:
    #
    # dC/dt - D*d²C/dx² + rho*dq/dt = 0
    #

    return (
        dC_dt
        - D * d2C_dx2
        + rho * dq_dt
    )


# ============================================================
# Geometry
# ============================================================

geom = dde.geometry.Interval(
    0,
    L
)

timedomain = dde.geometry.TimeDomain(
    0,
    T
)

geomtime = dde.geometry.GeometryXTime(
    geom,
    timedomain
)


# ============================================================
# Boundary conditions
# ============================================================

# ------------------------------------------------------------
# Left boundary: x = 0
# Zero flux
# dC/dx = 0
# ------------------------------------------------------------

def boundary_left(x, on_boundary):

    return (
        on_boundary
        and np.isclose(x[0], 0)
    )


bc_left = dde.icbc.NeumannBC(
    geomtime,
    lambda x: 0,
    boundary_left
)


# ------------------------------------------------------------
# Right boundary: x = L
# Fixed concentration
# C = 1
# ------------------------------------------------------------

def boundary_right(x, on_boundary):

    return (
        on_boundary
        and np.isclose(x[0], L)
    )


bc_right = dde.icbc.DirichletBC(
    geomtime,
    lambda x: 1.0,
    boundary_right
)


# ============================================================
# Initial condition
# ============================================================

def init_cond(x):

    return np.clip(
        (x[:, 0:1] - 0.9) / 0.1,
        0,
        1
    )


ic = dde.icbc.IC(
    geomtime,
    init_cond,
    lambda x, on_initial: on_initial
)


# ============================================================
# Training data
# ============================================================

data = dde.data.TimePDE(

    geomtime,

    pde,

    [bc_left, bc_right, ic],

    # Powers of 2 are preferable for Sobol sampling
    num_domain=2048,
    num_boundary=256,
    num_initial=512,

    train_distribution="Sobol"
)


# ============================================================
# Neural network
# ============================================================

net = dde.nn.FNN(
    [2] + [64] * 4 + [1],
    "tanh",
    "Glorot normal"
)


# ============================================================
# Model
# ============================================================

model = dde.Model(
    data,
    net
)


# ============================================================
# Adam
# ============================================================

model.compile(
    "adam",
    lr=1e-3,

    # PDE
    # Left BC
    # Right BC
    # Initial condition
    loss_weights=[
        1,
        10,
        10,
        10
    ]
)


# ============================================================
# PDE point resampling
# ============================================================
#
# IMPORTANT:
#
# bc_points=False
#
# prevents DeepXDE from changing the number of boundary
# points assigned to each BC.
#
# This fixes:
#
# ValueError:
# `num_bcs` changed!
#
# ============================================================

resampler = dde.callbacks.PDEPointResampler(
    period=1000,
    pde_points=True,
    bc_points=False
)


# ============================================================
# Adam training
# ============================================================

losshistory_adam, train_state_adam = model.train(

    iterations=12000,

    display_every=1000,

    disregard_previous_best=True,

    callbacks=[
        resampler
    ]
)


# ============================================================
# L-BFGS fine tuning
# ============================================================

model.compile(
    "L-BFGS"
)

losshistory_lbfgs, train_state_lbfgs = model.train()


# ============================================================
# Combine loss histories
# ============================================================

loss_train_adam = np.array(
    losshistory_adam.loss_train
)

loss_train_lbfgs = np.array(
    losshistory_lbfgs.loss_train
)

loss_train = np.vstack(
    [
        loss_train_adam,
        loss_train_lbfgs
    ]
)


# ============================================================
# Prediction grid
# ============================================================

x = np.linspace(
    0,
    L,
    300
)

t = np.linspace(
    0,
    T,
    200
)

X, T_grid = np.meshgrid(
    x,
    t
)


X_star = np.column_stack(
    [
        X.ravel(),
        T_grid.ravel()
    ]
)


# ============================================================
# Prediction
# ============================================================

u_pred = model.predict(
    X_star
).reshape(
    X.shape
)


# ============================================================
# Plot
# ============================================================

plt.figure(
    figsize=(14, 6)
)


# ============================================================
# Loss history
# ============================================================

plt.subplot(
    1,
    2,
    1
)

num_components = loss_train.shape[1]


if num_components >= 4:

    plt.semilogy(
        loss_train[:, 0],
        label="PDE loss"
    )

    plt.semilogy(
        loss_train[:, 1],
        label="Left BC loss"
    )

    plt.semilogy(
        loss_train[:, 2],
        label="Right BC loss"
    )

    plt.semilogy(
        loss_train[:, 3],
        label="IC loss"
    )

else:

    total_loss = np.sum(
        loss_train,
        axis=1
    )

    plt.semilogy(
        total_loss,
        label="Total loss"
    )


plt.xlabel(
    "Training step"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Loss history"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)


# ============================================================
# Concentration profiles
# ============================================================

plt.subplot(
    1,
    2,
    2
)


plot_times = [
    0,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0
]


for ti in plot_times:

    idx = np.argmin(
        np.abs(t - ti)
    )

    plt.plot(
        x,
        u_pred[idx],
        label=f"t = {t[idx]:.3f}",
        linewidth=2
    )


plt.xlabel(
    "x (particle center → surface)"
)

plt.ylabel(
    "C"
)

plt.title(
    "Diffusion–adsorption concentration profile"
)

plt.legend(
    loc="center left"
)

plt.grid(
    True,
    alpha=0.3
)


# ============================================================
# Display
# ============================================================

plt.tight_layout()

plt.show()
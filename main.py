import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils import generate_grid, predict_on_grid, plot_ns_solution, plot_error_map, l2_relative_error

# ==== Define Context Encoder ====
class ContextEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.Tanh(),
            nn.Linear(32, out_dim)
        )

    def forward(self, x, y):
        return self.encoder(torch.cat([x, y], dim=1))

# ==== Define MCP PINN Model ====
class NSPINN(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.context_encoder = ContextEncoder(2, context_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + context_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3)  # u, v, p
        )

    def forward(self, x, y):
        context = self.context_encoder(x, y)
        return self.net(torch.cat([x, y, context], dim=1))

# ==== Residual Computation ====
def ns_residual(model, x, y, nu=0.01):
    x.requires_grad_(True)
    y.requires_grad_(True)
    out = model(x, y)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    grads = lambda z: torch.autograd.grad(z, [x, y], grad_outputs=torch.ones_like(z), create_graph=True)
    u_x, u_y = grads(u)
    v_x, v_y = grads(v)
    p_x, p_y = grads(p)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    mom_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    mom_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    continuity = u_x + v_y
    return mom_u, mom_v, continuity

# ==== Subdomain training ====
def train_ns_subdomain(model, optimizer, x_in, y_in, x_bc, y_bc, u_bc, v_bc, epochs=1000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        mom_u, mom_v, cont = ns_residual(model, x_in, y_in)
        loss_pde = torch.mean(mom_u**2 + mom_v**2 + cont**2)
        out = model(x_bc, y_bc)
        u_pred, v_pred = out[:, 0:1], out[:, 1:2]
        loss_bc = torch.mean((u_pred - u_bc)**2 + (v_pred - v_bc)**2)
        loss = loss_pde + loss_bc
        loss.backward()
        optimizer.step()

# ==== Subdomain setup ====
def create_subdomains(n=2):
    bounds = []
    step = 1.0 / n
    for i in range(n):
        for j in range(n):
            bounds.append(((i*step, (i+1)*step), (j*step, (j+1)*step)))
    return bounds

def sample_subdomain_points(xrange, yrange, n):
    x = torch.rand(n, 1) * (xrange[1] - xrange[0]) + xrange[0]
    y = torch.rand(n, 1) * (yrange[1] - yrange[0]) + yrange[0]
    return x, y

# ==== Main ====
context_dim = 8
n_sub = 2
models = []
optimizers = []
subdomains = create_subdomains(n_sub)

# Train each subdomain
for ((xr, yr)) in subdomains:
    model = NSPINN(context_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_in, y_in = sample_subdomain_points(xr, yr, 1000)
    x_bc, y_bc = sample_subdomain_points(xr, yr, 200)
    u_bc = torch.zeros_like(x_bc)
    v_bc = torch.zeros_like(x_bc)
    u_bc[y_bc > 0.99] = 1.0  # Top lid
    train_ns_subdomain(model, optimizer, x_in, y_in, x_bc, y_bc, u_bc, v_bc)
    models.append(model)
    optimizers.append(optimizer)

# Inference and plot
X, Y, x_flat, y_flat = generate_grid(100, 100)
u_pred, v_pred, p_pred = predict_on_grid(models, subdomains, x_flat, y_flat)
plot_ns_solution(X, Y, u_pred, v_pred, p_pred, 100, 100)

# Load CFD solution if available
try:
    data = np.load("cfd_solution.npz")
    u_true = data['u']
    v_true = data['v']
    p_true = data['p']
    abs_error_u = np.abs(u_pred.reshape(100, 100) - u_true)
    abs_error_v = np.abs(v_pred.reshape(100, 100) - v_true)
    abs_error_p = np.abs(p_pred.reshape(100, 100) - p_true)
    plot_error_map(X, Y, abs_error_u, "Absolute Error in u")
    plot_error_map(X, Y, abs_error_v, "Absolute Error in v")
    plot_error_map(X, Y, abs_error_p, "Absolute Error in p")
    print("L2 Errors:",
          l2_relative_error(u_pred, u_true),
          l2_relative_error(v_pred, v_true),
          l2_relative_error(p_pred, p_true))
except FileNotFoundError:
    print("CFD ground truth not found. Skipping error visualization.")

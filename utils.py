import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_grid(nx=100, ny=100):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).reshape(-1, 1)
    y_flat = torch.tensor(Y.flatten(), dtype=torch.float32).reshape(-1, 1)
    return X, Y, x_flat, y_flat

def predict_on_grid(models, subdomains, x_flat, y_flat):
    u_grid = np.zeros_like(x_flat.numpy())
    v_grid = np.zeros_like(x_flat.numpy())
    p_grid = np.zeros_like(x_flat.numpy())
    for model, ((xmin, xmax), (ymin, ymax)) in zip(models, subdomains):
        mask = (x_flat >= xmin) & (x_flat <= xmax) & (y_flat >= ymin) & (y_flat <= ymax)
        if mask.any():
            x_sub = x_flat[mask.squeeze()]
            y_sub = y_flat[mask.squeeze()]
            with torch.no_grad():
                output = model(x_sub, y_sub)
            u_grid[mask.squeeze()] = output[:, 0].numpy()
            v_grid[mask.squeeze()] = output[:, 1].numpy()
            p_grid[mask.squeeze()] = output[:, 2].numpy()
    return u_grid, v_grid, p_grid

def plot_ns_solution(X, Y, u, v, p, nx=100, ny=100):
    U = u.reshape(ny, nx)
    V = v.reshape(ny, nx)
    P = p.reshape(ny, nx)
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1)
    plt.quiver(X, Y, U, V, scale=5)
    plt.title('Velocity Field (Quiver)')
    plt.subplot(1, 3, 2)
    speed = np.sqrt(U**2 + V**2)
    plt.streamplot(X, Y, U, V, color=speed, cmap='jet')
    plt.title('Streamlines')
    plt.subplot(1, 3, 3)
    cp = plt.contourf(X, Y, P, levels=50, cmap='coolwarm')
    plt.colorbar(cp)
    plt.title('Pressure Contour')
    plt.tight_layout()
    plt.show()

def plot_error_map(X, Y, error_field, title):
    plt.figure(figsize=(6, 5))
    cp = plt.contourf(X, Y, error_field, levels=50, cmap='viridis')
    plt.colorbar(cp)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()

def l2_relative_error(pred, true):
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-8)

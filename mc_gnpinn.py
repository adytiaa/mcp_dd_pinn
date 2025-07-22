import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import time

# --- Configuration ---
# Use GPU if available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# PINN Hyperparameters
LEARNING_RATE = 1e-3
# Note: For a real problem, epochs should be much higher (e.g., 10,000+)
# We keep it low here for a quick demonstration.
EPOCHS = 2000
N_COLLOCATION = 2500  # Number of points inside the domain for PDE loss
N_BOUNDARY = 400    # Number of points on the boundaries for BC loss
RHO = 1.0           # Density
NU = 0.01           # Kinematic viscosity (1/Reynolds Number)

# --- 1. PINN Architecture ---
class PINN(nn.Module):
    """A simple fully-connected neural network"""
    def __init__(self, num_layers=5, hidden_size=32):
        super().__init__()
        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh()])
        layers.append(nn.Linear(hidden_size, 2)) # Output: [psi, p]
        
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        # Reshape input and run through the network
        points = torch.cat([x, y], dim=1)
        output = self.net(points)
        psi = output[:, 0:1] # Stream function
        p = output[:, 1:2]   # Pressure
        return psi, p

# --- 2. Ground Truth Data (from analytical solution) ---
def get_ground_truth(grid_x, grid_y, U_inf=1.0, R=0.5):
    """Generates the ground truth velocity and pressure fields for potential flow."""
    cylinder_mask = np.sqrt(grid_x**2 + grid_y**2) < R
    r = np.sqrt(grid_x**2 + grid_y**2)
    theta = np.arctan2(grid_y, grid_x)
    r[r == 0] = 1e-6 # Avoid division by zero
    
    u = U_inf * (1 - (R**2 * (grid_x**2 - grid_y**2)) / (r**4))
    v = -U_inf * (2 * R**2 * grid_x * grid_y) / (r**4)
    p = 0.5 * U_inf**2 * (2 * (R**2 / r**2) * np.cos(2*theta) - (R**4 / r**4))
    
    u[cylinder_mask] = 0
    v[cylinder_mask] = 0
    p[cylinder_mask] = 0
    return u, v, p

# --- 3. Domain Decomposition and Data Sampling ---
def decompose_domain(x_domain, y_domain):
    """Decomposes the main domain into a 2x2 grid."""
    x_min, x_max = x_domain
    y_min, y_max = y_domain
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    
    subdomains = [
        {'x_domain': (x_min, x_mid), 'y_domain': (y_min, y_mid), 'id': 'SD1 (Bottom-Left)'},
        {'x_domain': (x_mid, x_max), 'y_domain': (y_min, y_mid), 'id': 'SD2 (Bottom-Right)'},
        {'x_domain': (x_min, x_mid), 'y_domain': (y_mid, y_max), 'id': 'SD3 (Top-Left)'},
        {'x_domain': (x_mid, x_max), 'y_domain': (y_mid, y_max), 'id': 'SD4 (Top-Right)'},
    ]
    return subdomains

def get_training_data_for_subdomain(sd):
    """Generates collocation and boundary points for a given subdomain."""
    x_min, x_max = sd['x_domain']
    y_min, y_max = sd['y_domain']
    
    # Collocation points (for PDE loss)
    x_col = torch.rand(N_COLLOCATION, 1) * (x_max - x_min) + x_min
    y_col = torch.rand(N_COLLOCATION, 1) * (y_max - y_min) + y_min
    
    # Boundary points (for Boundary Condition loss)
    # Using freestream velocity U_inf=1.0 as the boundary condition
    bc_pts = []
    bc_vals = []
    
    # Left/Right boundaries
    y_bc = torch.rand(N_BOUNDARY // 4, 1) * (y_max - y_min) + y_min
    bc_pts.extend([torch.full_like(y_bc, x_min), y_bc]) # Left
    bc_vals.extend([torch.ones_like(y_bc), torch.zeros_like(y_bc)]) # u=1, v=0
    bc_pts.extend([torch.full_like(y_bc, x_max), y_bc]) # Right
    bc_vals.extend([torch.ones_like(y_bc), torch.zeros_like(y_bc)]) # u=1, v=0
    
    # Top/Bottom boundaries
    x_bc = torch.rand(N_BOUNDARY // 4, 1) * (x_max - x_min) + x_min
    bc_pts.extend([x_bc, torch.full_like(x_bc, y_min)]) # Bottom
    bc_vals.extend([torch.ones_like(x_bc), torch.zeros_like(x_bc)]) # u=1, v=0
    bc_pts.extend([x_bc, torch.full_like(x_bc, y_max)]) # Top
    bc_vals.extend([torch.ones_like(x_bc), torch.zeros_like(x_bc)]) # u=1, v=0
    
    return {
        'collocation_x': x_col.to(DEVICE).requires_grad_(),
        'collocation_y': y_col.to(DEVICE).requires_grad_(),
        'boundary_x': torch.cat(bc_pts[::2]).to(DEVICE).requires_grad_(),
        'boundary_y': torch.cat(bc_pts[1::2]).to(DEVICE).requires_grad_(),
        'boundary_u': torch.cat(bc_vals[::2]).to(DEVICE),
        'boundary_v': torch.cat(bc_vals[1::2]).to(DEVICE)
    }

# --- 4. Gradient-Normalized Training ---
def train_pinn(model, data, subdomain_id):
    """Trains a single PINN using gradient-normalized loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    
    lambdas = {'pde': 1.0, 'bc': 1.0} # Initialize weights
    start_time = time.time()

    print(f"--- Training {subdomain_id} ---")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # --- Loss Calculation ---
        # 1. PDE Loss (Navier-Stokes)
        psi, p = model(data['collocation_x'], data['collocation_y'])
        
        u = torch.autograd.grad(psi, data['collocation_y'], grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        v = -torch.autograd.grad(psi, data['collocation_x'], grad_outputs=torch.ones_like(psi), create_graph=True)[0]

        u_x = torch.autograd.grad(u, data['collocation_x'], grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, data['collocation_y'], grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, data['collocation_x'], grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, data['collocation_y'], grad_outputs=torch.ones_like(v), create_graph=True)[0]
        p_x = torch.autograd.grad(p, data['collocation_x'], grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, data['collocation_y'], grad_outputs=torch.ones_like(p), create_graph=True)[0]

        u_xx = torch.autograd.grad(u_x, data['collocation_x'], grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, data['collocation_y'], grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, data['collocation_x'], grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, data['collocation_y'], grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

        f_u = u * u_x + v * u_y + (1/RHO) * p_x - NU * (u_xx + u_yy)
        f_v = u * v_x + v * v_y + (1/RHO) * p_y - NU * (v_xx + v_yy)
        loss_pde = torch.mean(f_u**2) + torch.mean(f_v**2)
        
        # 2. Boundary Condition Loss
        psi_bc, _ = model(data['boundary_x'], data['boundary_y'])
        u_bc = torch.autograd.grad(psi_bc, data['boundary_y'], grad_outputs=torch.ones_like(psi_bc), create_graph=True)[0]
        v_bc = -torch.autograd.grad(psi_bc, data['boundary_x'], grad_outputs=torch.ones_like(psi_bc), create_graph=True)[0]
        loss_bc = torch.mean((u_bc - data['boundary_u'])**2) + torch.mean((v_bc - data['boundary_v'])**2)
        
        # --- Gradient Normalization (at the beginning of training) ---
        if epoch == 0:
            loss_pde.backward(retain_graph=True)
            grad_pde_norm = torch.cat([p.grad.flatten() for p in model.parameters()]).abs().mean()
            optimizer.zero_grad()
            
            loss_bc.backward(retain_graph=True)
            grad_bc_norm = torch.cat([p.grad.flatten() for p in model.parameters()]).abs().mean()
            optimizer.zero_grad()
            
            mean_grad = (grad_pde_norm + grad_bc_norm) / 2
            lambdas['pde'] = float(mean_grad / grad_pde_norm)
            lambdas['bc'] = float(mean_grad / grad_bc_norm)
            print(f"Gradient Norms -> PDE: {grad_pde_norm:.4f}, BC: {grad_bc_norm:.4f}")
            print(f"Calculated Lambdas -> PDE: {lambdas['pde']:.4f}, BC: {lambdas['bc']:.4f}")
        
        # Weighted Total Loss
        total_loss = lambdas['pde'] * loss_pde + lambdas['bc'] * loss_bc
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss.item():.4e}, "
                  f"Loss PDE: {loss_pde.item():.4e}, Loss BC: {loss_bc.item():.4e}")

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.2f} seconds.")
    return model

# --- 5. Mode Connectivity Protocol (Weight Averaging) ---
def mode_connectivity_by_averaging(models):
    """Averages the weights of the trained models to get a unified model."""
    print("\nApplying MCP by averaging model weights...")
    avg_state_dict = OrderedDict()
    
    # Sum up all state dictionaries
    for model in models:
        for key, value in model.state_dict().items():
            if key in avg_state_dict:
                avg_state_dict[key] += value.cpu()
            else:
                avg_state_dict[key] = value.cpu()

    # Divide by the number of models to get the average
    for key in avg_state_dict:
        avg_state_dict[key] /= len(models)
        
    unified_model = PINN().to(DEVICE)
    unified_model.load_state_dict(avg_state_dict)
    print("MCP complete. Unified model created.")
    return unified_model

# --- 6. Prediction, Visualization, and Error ---
def predict_on_full_domain(model, x_domain, y_domain, grid_points):
    """Uses the unified model to predict the flow field on the entire domain."""
    model.eval()
    x_space = np.linspace(x_domain[0], x_domain[1], grid_points)
    y_space = np.linspace(y_domain[0], y_domain[1], grid_points)
    grid_x, grid_y = np.meshgrid(x_space, y_space)
    
    tensor_x = torch.from_numpy(grid_x.flatten()).float().unsqueeze(1).to(DEVICE).requires_grad_()
    tensor_y = torch.from_numpy(grid_y.flatten()).float().unsqueeze(1).to(DEVICE).requires_grad_()
    
    with torch.no_grad():
        psi, p = model(tensor_x, tensor_y)
    
    # Calculate velocities from psi using autograd
    u = torch.autograd.grad(psi, tensor_y, grad_outputs=torch.ones_like(psi), create_graph=False)[0]
    v = -torch.autograd.grad(psi, tensor_x, grad_outputs=torch.ones_like(psi), create_graph=False)[0]
    
    # Reshape back to grid
    u_pred = u.detach().cpu().numpy().reshape(grid_x.shape)
    v_pred = v.detach().cpu().numpy().reshape(grid_y.shape)
    p_pred = p.detach().cpu().numpy().reshape(p.shape).reshape(grid_x.shape)
    
    # Zero out the flow inside the cylinder for visualization
    cylinder_mask = np.sqrt(grid_x**2 + grid_y**2) < 0.5
    u_pred[cylinder_mask] = 0
    v_pred[cylinder_mask] = 0
    p_pred[cylinder_mask] = 0
    
    return u_pred, v_pred, p_pred

def visualize_results(grid_x, grid_y, u, v, p, title_prefix=""):
    """Generates plots for velocity, streamlines, and pressure."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    velocity_mag = np.sqrt(u**2 + v**2)
    
    contour1 = axes[0].contourf(grid_x, grid_y, velocity_mag, levels=50, cmap='viridis')
    fig.colorbar(contour1, ax=axes[0])
    axes[0].set_title(f'{title_prefix} Velocity Magnitude')
    axes[0].set_aspect('equal')

    axes[1].streamplot(grid_x, grid_y, u, v, density=2, color='darkred', linewidth=1)
    axes[1].set_title(f'{title_prefix} Streamlines')
    axes[1].set_aspect('equal')
    
    contour3 = axes[2].contourf(grid_x, grid_y, p, levels=50, cmap='plasma')
    fig.colorbar(contour3, ax=axes[2])
    axes[2].set_title(f'{title_prefix} Pressure Field')
    axes[2].set_aspect('equal')
    
    plt.suptitle(title_prefix, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def compare_with_ground_truth(pred, true):
    """Calculates L2 relative error."""
    error = np.linalg.norm(pred - true) / np.linalg.norm(true)
    return error

# --- Main Execution ---
if __name__ == "__main__":
    DOMAIN_X = (-2.0, 2.0)
    DOMAIN_Y = (-2.0, 2.0)
    GRID_POINTS = 100
    
    # 1. Get Ground Truth
    full_x_space = np.linspace(DOMAIN_X[0], DOMAIN_X[1], GRID_POINTS)
    full_y_space = np.linspace(DOMAIN_Y[0], DOMAIN_Y[1], GRID_POINTS)
    full_grid_x, full_grid_y = np.meshgrid(full_x_space, full_y_space)
    u_true, v_true, p_true = get_ground_truth(full_grid_x, full_grid_y)
    visualize_results(full_grid_x, full_grid_y, u_true, v_true, p_true, "Ground Truth (Analytical Solution)")
    
    # 2. Decompose Domain and Train PINNs
    subdomains = decompose_domain(DOMAIN_X, DOMAIN_Y)
    trained_models = []
    
    for sd in subdomains:
        pinn_model = PINN().to(DEVICE)
        training_data = get_training_data_for_subdomain(sd)
        trained_model = train_pinn(pinn_model, training_data, sd['id'])
        trained_models.append(trained_model)
        
    # 3. Apply MCP (Weight Averaging)
    unified_model = mode_connectivity_by_averaging(trained_models)
    
    # 4. Predict and Visualize with Unified Model
    u_pred, v_pred, p_pred = predict_on_full_domain(unified_model, DOMAIN_X, DOMAIN_Y, GRID_POINTS)
    visualize_results(full_grid_x, full_grid_y, u_pred, v_pred, p_pred, "MCP-Unified PINN Prediction")
    
    # 5. Error Comparison
    error_u = compare_with_ground_truth(u_pred, u_true)
    error_v = compare_with_ground_truth(v_pred, v_true)
    error_p = compare_with_ground_truth(p_pred, p_true)
    
    print("\n--- Final Error Comparison ---")
    print(f"L2 Relative Error in u-velocity: {error_u:.4%}")
    print(f"L2 Relative Error in v-velocity: {error_v:.4%}")
    print(f"L2 Relative Error in pressure (p): {error_p:.4%}")
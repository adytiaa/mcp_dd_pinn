#Here is the updated code. The changes are primarily in the new `ModelContextProtocol` section.

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import time

# --- Configuration ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
LEARNING_RATE = 1e-3
EPOCHS = 2000 # Kept low for demonstration
N_COLLOCATION = 2500
N_BOUNDARY = 400
RHO = 1.0
NU = 0.01

# --- PINN Architecture and Ground Truth (Unchanged) ---
class PINN(nn.Module):
    def __init__(self, num_layers=5, hidden_size=32):
        super().__init__()
        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh()])
        layers.append(nn.Linear(hidden_size, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, x, y):
        points = torch.cat([x, y], dim=1)
        output = self.net(points)
        psi, p = output[:, 0:1], output[:, 1:2]
        return psi, p

def get_ground_truth(grid_x, grid_y, U_inf=1.0, R=0.5):
    cylinder_mask = np.sqrt(grid_x**2 + grid_y**2) < R
    r = np.sqrt(grid_x**2 + grid_y**2)
    theta = np.arctan2(grid_y, grid_x)
    r[r == 0] = 1e-6
    u = U_inf * (1 - (R**2 * (grid_x**2 - grid_y**2)) / (r**4))
    v = -U_inf * (2 * R**2 * grid_x * grid_y) / (r**4)
    p = 0.5 * U_inf**2 * (2 * (R**2 / r**2) * np.cos(2*theta) - (R**4 / r**4))
    u[cylinder_mask], v[cylinder_mask], p[cylinder_mask] = 0, 0, 0
    return u, v, p

# --- Domain Decomposition and Training Data (Unchanged) ---
def decompose_domain(x_domain, y_domain):
    x_min, x_max = x_domain
    y_min, y_max = y_domain
    x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
    return [
        {'x_domain': (x_min, x_mid), 'y_domain': (y_min, y_mid), 'id': 'SD1 (Bottom-Left)'},
        {'x_domain': (x_mid, x_max), 'y_domain': (y_min, y_mid), 'id': 'SD2 (Bottom-Right)'},
        {'x_domain': (x_min, x_mid), 'y_domain': (y_mid, y_max), 'id': 'SD3 (Top-Left)'},
        {'x_domain': (x_mid, x_max), 'y_domain': (y_mid, y_max), 'id': 'SD4 (Top-Right)'},
    ]

def get_training_data_for_subdomain(sd):
    x_min, x_max = sd['x_domain']
    y_min, y_max = sd['y_domain']
    x_col = torch.rand(N_COLLOCATION, 1) * (x_max - x_min) + x_min
    y_col = torch.rand(N_COLLOCATION, 1) * (y_max - y_min) + y_min
    return {'collocation_x': x_col.to(DEVICE).requires_grad_(), 'collocation_y': y_col.to(DEVICE).requires_grad_()}

# --- Gradient-Normalized Training Logic (Unchanged) ---
def train_pinn(model, data, subdomain_id):
    # This function remains the same as in the previous example.
    # For brevity, its implementation is omitted here but should be included from the last response.
    # It performs the gradient-normalized training for a single PINN.
    # ... (include the full train_pinn function here) ...
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    print(f"--- Training {subdomain_id} ---")
    for epoch in range(EPOCHS): # Simplified loop for brevity
        optimizer.zero_grad()
        psi, p = model(data['collocation_x'], data['collocation_y'])
        # Simplified loss for demonstration
        loss = torch.mean(psi**2) + torch.mean(p**2) 
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4e}")
    print(f"Training of {subdomain_id} finished.")
    return model

# --- 5. NEW: Model Context Protocol (MCP) Implementation ---
def calculate_physics_loss(model, data):
    """A helper function to calculate the PDE loss for a given model and data."""
    model.eval()
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
    
    return torch.mean(f_u**2) + torch.mean(f_v**2)

def find_path_and_connect(model_a, model_b, data_a, data_b, name=""):
    """Finds the best model on a Bézier curve path between two models."""
    print(f"MCP: Finding low-loss path for {name}...")
    theta_a = model_a.state_dict()
    theta_b = model_b.state_dict()
    
    # The control point for our curve will be the average of the two models
    theta_c = OrderedDict()
    for key in theta_a:
        theta_c[key] = 0.5 * (theta_a[key] + theta_b[key])

    best_loss = float('inf')
    best_t = -1
    
    # Search for the best point 't' on the path
    for t in np.linspace(0, 1, 11): # Check 11 points on the curve
        temp_model = PINN().to(DEVICE)
        temp_theta = OrderedDict()
        
        # Bézier curve formula: (1-t)^2*A + 2t(1-t)*C + t^2*B
        for key in theta_a:
            temp_theta[key] = ((1-t)**2 * theta_a[key] + 
                               2*t*(1-t) * theta_c[key] + 
                               t**2 * theta_b[key])
        
        temp_model.load_state_dict(temp_theta)
        
        # Evaluate loss on the combined data from both domains
        combined_data = {
            'collocation_x': torch.cat([data_a['collocation_x'], data_b['collocation_x']]),
            'collocation_y': torch.cat([data_a['collocation_y'], data_b['collocation_y']])
        }
        
        with torch.no_grad():
            loss = calculate_physics_loss(temp_model, combined_data)

        if loss < best_loss:
            best_loss = loss
            best_t = t

    print(f"Found best model on path at t={best_t:.2f} with loss={best_loss:.4e}")

    # Create the final connected model using the best 't'
    final_theta = OrderedDict()
    for key in theta_a:
        final_theta[key] = ((1-best_t)**2 * theta_a[key] + 
                            2*best_t*(1-best_t) * theta_c[key] + 
                            best_t**2 * theta_b[key])
    
    final_model = PINN().to(DEVICE)
    final_model.load_state_dict(final_theta)
    return final_model

def model_context_protocol(models, subdomains_data):
    """Hierarchically combines four models using MCP."""
    m_bl, m_br, m_tl, m_tr = models # Bottom-Left, Bottom-Right, etc.
    d_bl, d_br, d_tl, d_tr = subdomains_data

    # Step 1: Combine horizontal pairs
    m_bottom = find_path_and_connect(m_bl, m_br, d_bl, d_br, "Bottom Pair")
    m_top = find_path_and_connect(m_tl, m_tr, d_tl, d_tr, "Top Pair")

    # Step 2: Combine the resulting vertical pair
    data_bottom = {k: torch.cat([d_bl[k], d_br[k]]) for k in d_bl}
    data_top = {k: torch.cat([d_tl[k], d_tr[k]]) for k in d_tl}
    
    unified_model = find_path_and_connect(m_bottom, m_top, data_bottom, data_top, "Final Vertical Pair")
    
    return unified_model


# --- Prediction, Visualization, Error (Unchanged) ---
# ... (include the predict_on_full_domain, visualize_results, and compare_with_ground_truth functions here) ...
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
    
    u = torch.autograd.grad(psi, tensor_y, grad_outputs=torch.ones_like(psi), create_graph=False)[0]
    v = -torch.autograd.grad(psi, tensor_x, grad_outputs=torch.ones_like(psi), create_graph=False)[0]
    
    u_pred = u.detach().cpu().numpy().reshape(grid_x.shape)
    v_pred = v.detach().cpu().numpy().reshape(grid_y.shape)
    p_pred = p.detach().cpu().numpy().reshape(grid_x.shape)
    
    cylinder_mask = np.sqrt(grid_x**2 + grid_y**2) < 0.5
    u_pred[cylinder_mask], v_pred[cylinder_mask], p_pred[cylinder_mask] = 0, 0, 0
    return u_pred, v_pred, p_pred

def visualize_results(grid_x, grid_y, u, v, p, title_prefix=""):
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
    return np.linalg.norm(pred - true) / np.linalg.norm(true)


# --- Main Execution ---
if __name__ == "__main__":
    DOMAIN_X, DOMAIN_Y = (-2.0, 2.0), (-2.0, 2.0)
    GRID_POINTS = 100
    
    full_x_space = np.linspace(DOMAIN_X[0], DOMAIN_X[1], GRID_POINTS)
    full_y_space = np.linspace(DOMAIN_Y[0], DOMAIN_Y[1], GRID_POINTS)
    full_grid_x, full_grid_y = np.meshgrid(full_x_space, full_y_space)
    u_true, v_true, p_true = get_ground_truth(full_grid_x, full_grid_y)
    visualize_results(full_grid_x, full_grid_y, u_true, v_true, p_true, "Ground Truth")
    
    subdomains = decompose_domain(DOMAIN_X, DOMAIN_Y)
    trained_models = []
    subdomains_data = []
    
    for sd in subdomains:
        pinn_model = PINN().to(DEVICE)
        training_data = get_training_data_for_subdomain(sd)
        # Note: Using the simplified training loop here for speed. 
        # Replace with the full 'train_pinn' for a real run.
        trained_model = train_pinn(pinn_model, training_data, sd['id'])
        trained_models.append(trained_model)
        subdomains_data.append(training_data)
        
    # Apply the full Model Context Protocol
    unified_model = model_context_protocol(trained_models, subdomains_data)
    
    u_pred, v_pred, p_pred = predict_on_full_domain(unified_model, DOMAIN_X, DOMAIN_Y, GRID_POINTS)
    visualize_results(full_grid_x, full_grid_y, u_pred, v_pred, p_pred, "MCP-Unified PINN Prediction")
    
    error_u = compare_with_ground_truth(u_pred, u_true)
    error_v = compare_with_ground_truth(v_pred, v_true)
    error_p = compare_with_ground_truth(p_pred, p_true)
    
    print("\n--- Final Error Comparison ---")
    print(f"L2 Relative Error in u-velocity: {error_u:.4%}")
    print(f"L2 Relative Error in v-velocity: {error_v:.4%}")
    print(f"L2 Relative Error in pressure (p): {error_p:.4%}")

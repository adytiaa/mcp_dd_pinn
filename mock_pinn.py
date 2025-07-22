import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# --- 1. Ground Truth Data (Simulated CFD) ---
# Using the analytical solution for potential flow around a cylinder
def get_ground_truth(grid_x, grid_y, U_inf=1.0, R=0.5):
    """
    Generates the ground truth velocity and pressure fields.
    
    Args:
        grid_x (np.ndarray): X-coordinates of the grid.
        grid_y (np.ndarray): Y-coordinates of the grid.
        U_inf (float): Freestream velocity.
        R (float): Cylinder radius.

    Returns:
        tuple: u, v, p velocity and pressure fields.
    """
    # Create a mask for points inside the cylinder
    cylinder_mask = np.sqrt(grid_x**2 + grid_y**2) < R
    
    # Calculate velocity components (u, v)
    r = np.sqrt(grid_x**2 + grid_y**2)
    theta = np.arctan2(grid_y, grid_x)
    
    # Avoid division by zero at the center
    r[r == 0] = 1e-6
    
    u = U_inf * (1 - (R**2 * (grid_x**2 - grid_y**2)) / (r**4))
    v = -U_inf * (2 * R**2 * grid_x * grid_y) / (r**4)
    
    # Set velocity inside the cylinder to zero
    u[cylinder_mask] = 0
    v[cylinder_mask] = 0
    
    # Calculate pressure field (from Bernoulli's equation)
    p = 0.5 * U_inf**2 * (2 * (R**2 / r**2) * np.cos(2*theta) - (R**4 / r**4))
    p[cylinder_mask] = 0 # Set pressure inside the cylinder
    
    return u, v, p

# --- 2. Domain Decomposition (2x2 Grid) ---
def decompose_domain(x_domain, y_domain, grid_points):
    """
    Decomposes the main domain into a 2x2 grid of subdomains.
    
    Returns:
        list: A list of dictionaries, each containing the boundaries and
              grid points for a subdomain.
    """
    x_min, x_max = x_domain
    y_min, y_max = y_domain
    
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    
    subdomains_def = [
        {'x_domain': (x_min, x_mid), 'y_domain': (y_min, y_mid)}, # Bottom-left (SD1)
        {'x_domain': (x_mid, x_max), 'y_domain': (y_min, y_mid)}, # Bottom-right (SD2)
        {'x_domain': (x_min, x_mid), 'y_domain': (y_mid, y_max)}, # Top-left (SD3)
        {'x_domain': (x_mid, x_max), 'y_domain': (y_mid, y_max)}, # Top-right (SD4)
    ]
    
    subdomains = []
    for i, sd in enumerate(subdomains_def):
        nx, ny = grid_points // 2, grid_points // 2
        x_space = np.linspace(sd['x_domain'][0], sd['x_domain'][1], nx)
        y_space = np.linspace(sd['y_domain'][0], sd['y_domain'][1], ny)
        grid_x, grid_y = np.meshgrid(x_space, y_space)
        sd['grid_x'] = grid_x
        sd['grid_y'] = grid_y
        sd['id'] = i + 1
        subdomains.append(sd)
        
    return subdomains

# --- 3. Simulated PINN Models and MCP ---

# A mock/placeholder for a neural network model's weights
class MockPINN:
    def __init__(self, subdomain_id):
        self.id = subdomain_id
        # In a real scenario, these weights would be the result of training.
        # Here, we simulate "good" weights by generating predictions close
        # to the ground truth but with some unique noise for each model.
        self.weights = None # This would be a large tensor/vector in reality

    def train(self, grid_x, grid_y):
        """Simulates the training of a PINN on a subdomain."""
        print(f"Simulating training for PINN on subdomain {self.id}...")
        u_true, v_true, p_true = get_ground_truth(grid_x, grid_y)
        
        # Add some noise to simulate imperfect, specialized models
        noise_level = 0.05
        u_pred = u_true + np.random.normal(0, noise_level * np.abs(u_true).mean(), u_true.shape)
        v_pred = v_true + np.random.normal(0, noise_level * np.abs(v_true).mean(), v_true.shape)
        p_pred = p_true + np.random.normal(0, noise_level * np.abs(p_true).mean(), p_true.shape)
        
        # In a real implementation, `self.weights` would be set by the optimizer.
        # For this simulation, we store the predictions directly.
        self.predictions = {'u': u_pred, 'v': v_pred, 'p': p_pred}
        print(f"Training for PINN {self.id} complete.")

    def predict(self):
        """Returns the simulated predictions."""
        return self.predictions['u'], self.predictions['v'], self.predictions['p']
        
def mode_connectivity_protocol(models):
    """
    Simulates the MCP by averaging the predictions of the models.
    
    A real MCP implementation would find a low-loss path between the model *weights*
    and select a point on that path. Averaging the output is a simplified analogy.
    """
    print("\nApplying Mode Connectivity Protocol (MCP)...")
    
    # We simply average the predictions where they overlap.
    # This is a conceptual simplification of finding a model on a low-loss path.
    all_u = [m.predictions['u'] for m in models]
    all_v = [m.predictions['v'] for m in models]
    all_p = [m.predictions['p'] for m in models]
    
    # Reconstruct the full domain from the 2x2 grid of predictions
    # Top row: SD3 and SD4
    # Bottom row: SD1 and SD2
    u_top = np.concatenate((all_u[2], all_u[3]), axis=1)
    u_bottom = np.concatenate((all_u[0], all_u[1]), axis=1)
    u_final = np.concatenate((u_bottom, u_top), axis=0)
    
    v_top = np.concatenate((all_v[2], all_v[3]), axis=1)
    v_bottom = np.concatenate((all_v[0], all_v[1]), axis=1)
    v_final = np.concatenate((v_bottom, v_top), axis=0)
    
    p_top = np.concatenate((all_p[2], all_p[3]), axis=1)
    p_bottom = np.concatenate((all_p[0], all_p[1]), axis=1)
    p_final = np.concatenate((p_bottom, p_top), axis=0)
    
    print("MCP complete. Unified model created.")
    return u_final, v_final, p_final


# --- 4. Visualization ---
def visualize_results(grid_x, grid_y, u, v, p, title_prefix=""):
    """Generates plots for velocity, streamlines, and pressure."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    
    # Plot Velocity Magnitude
    velocity_mag = np.sqrt(u**2 + v**2)
    contour1 = axes[0].contourf(grid_x, grid_y, velocity_mag, levels=50, cmap='viridis')
    fig.colorbar(contour1, ax=axes[0])
    axes[0].set_title(f'{title_prefix} Velocity Magnitude')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')

    # Plot Streamlines
    axes[1].streamplot(grid_x, grid_y, u, v, density=2, color='darkred')
    axes[1].set_title(f'{title_prefix} Streamlines')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    
    # Plot Pressure
    contour3 = axes[2].contourf(grid_x, grid_y, p, levels=50, cmap='plasma')
    fig.colorbar(contour3, ax=axes[2])
    axes[2].set_title(f'{title_prefix} Pressure Field')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

# --- 5. Error Comparison ---
def compare_with_ground_truth(u_pred, v_pred, p_pred, u_true, v_true, p_true):
    """Calculates and prints the L2 relative error."""
    error_u = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
    error_v = np.linalg.norm(v_pred - v_true) / np.linalg.norm(v_true)
    error_p = np.linalg.norm(p_pred - p_true) / np.linalg.norm(p_true)
    
    print("\n--- Error Comparison ---")
    print(f"L2 Relative Error in u-velocity: {error_u:.4%}")
    print(f"L2 Relative Error in v-velocity: {error_v:.4%}")
    print(f"L2 Relative Error in pressure (p): {error_p:.4%}")
    
# --- Main Execution ---
if __name__ == "__main__":
    # --- Setup ---
    DOMAIN_X = (-2.0, 2.0)
    DOMAIN_Y = (-2.0, 2.0)
    GRID_POINTS = 100 # Resolution for the full domain
    
    # --- 1. Generate Ground Truth Data ---
    print("--- Step 1: Generating Ground Truth Data ---")
    full_x_space = np.linspace(DOMAIN_X[0], DOMAIN_X[1], GRID_POINTS)
    full_y_space = np.linspace(DOMAIN_Y[0], DOMAIN_Y[1], GRID_POINTS)
    full_grid_x, full_grid_y = np.meshgrid(full_x_space, full_y_space)
    u_true, v_true, p_true = get_ground_truth(full_grid_x, full_grid_y)
    
    # Visualize Ground Truth
    visualize_results(full_grid_x, full_grid_y, u_true, v_true, p_true, "Ground Truth")
    
    # --- 2. Decompose Domain ---
    print("\n--- Step 2: Decomposing Domain into 2x2 Grid ---")
    subdomains = decompose_domain(DOMAIN_X, DOMAIN_Y, GRID_POINTS)
    print(f"Created {len(subdomains)} subdomains.")
    
    # --- 3. Simulate PINN Training on each Subdomain ---
    print("\n--- Step 3: Simulating Training of Individual PINNs ---")
    trained_models = []
    for sd in subdomains:
        model = MockPINN(subdomain_id=sd['id'])
        model.train(sd['grid_x'], sd['grid_y'])
        trained_models.append(model)
        
    # --- 4. Apply MCP to get unified model ---
    print("\n--- Step 4: Applying MCP to Unify Models ---")
    u_mcp, v_mcp, p_mcp = mode_connectivity_protocol(trained_models)
    
    # Visualize MCP-unified results
    visualize_results(full_grid_x, full_grid_y, u_mcp, v_mcp, p_mcp, "MCP-Unified PINN")
    
    # --- 5. Compare with Ground Truth ---
    compare_with_ground_truth(u_mcp, v_mcp, p_mcp, u_true, v_true, p_true)
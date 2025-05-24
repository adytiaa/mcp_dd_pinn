**PINNs with Domain Decomposition using the Model Context Protocol (MCP)** 

This is a powerful technique for tackling large or complex PDE problems that might be difficult for a single, monolithic PINN to handle.

**Core Idea:**

Instead of training one massive neural network for the entire problem domain, you:

1.  **Decompose the Domain:** Split the original domain (Ω) into several smaller, possibly overlapping or non-overlapping subdomains (Ω₁, Ω₂, ..., Ωₙ).
2.  **Assign a PINN to Each Subdomain:** Train a separate, smaller PINN for each subdomain. `PINN₁` solves the PDE in Ω₁, `PINN₂` in Ω₂, and so on.
3.  **Ensure Consistency at Interfaces:** The crucial part is making sure the solutions from adjacent PINNs "match up" at their shared interfaces (boundaries). This is where the Model Context Protocol comes in.

**Model Context Protocol (MCP):**

MCP is an iterative method for enforcing continuity and consistency across subdomain interfaces. It works like this:

*   **Alternating Training:** The PINNs for the subdomains are trained in an alternating or sequential fashion.
*   **Context Provision:** When training `PINNᵢ` for subdomain Ωᵢ, the neighboring PINNs (e.g., `PINNⱼ` for an adjacent subdomain Ωⱼ) provide "context" at the shared interface Γᵢⱼ.
*   **Frozen Context:** This context is typically the *output* (and/or its derivatives) of `PINNⱼ` evaluated at the interface Γᵢⱼ, using `PINNⱼ`'s *current (but temporarily frozen)* weights from its last training stage.
*   **Interface Loss:** `PINNᵢ` includes loss terms that penalize discrepancies between its own predictions at Γᵢⱼ and the context provided by `PINNⱼ`. These losses enforce:
    *   **Continuity of the solution:** `uᵢ(x) = uⱼ_context(x)` for `x` on Γᵢⱼ.
    *   **Continuity of fluxes (derivatives):** `∂uᵢ(x)/∂n = ∂uⱼ_context(x)/∂n` for `x` on Γᵢⱼ (where `n` is the normal vector to the interface). For systems like Navier-Stokes, this applies to each variable (u, v, p) and their relevant derivatives.
*   **Iteration:** This process is repeated for several "MCP iterations." In each MCP iteration, all subdomain PINNs are trained once, each using the latest context from its neighbors.

**Flow of an MCP Iteration (e.g., for two subdomains Ω₁ and Ω₂ with interface Γ):**

1.  **Initialize:** Randomly initialize `PINN₁` and `PINN₂`.
2.  **MCP Loop (repeat `M` times):**
    *   **(a) Train `PINN₁`:**
        *   Freeze the weights of `PINN₂`. Create a "context model" `PINN₂_context` which is a copy of `PINN₂`.
        *   Sample collocation points in Ω₁, physical boundary points of Ω₁, and interface points on Γ.
        *   **Loss for `PINN₁`:**
            *   `L_PDE₁`: PDE residual loss in Ω₁.
            *   `L_BC₁`: Physical boundary condition loss on the parts of ∂Ω₁ not touching Γ.
            *   `L_interface₁`:
                *   `|| u₁(x_Γ) - u₂_context(x_Γ) ||²` (value continuity)
                *   `|| ∂u₁/∂n(x_Γ) - ∂u₂_context/∂n(x_Γ) ||²` (flux continuity)
        *   Optimize `PINN₁` using its total loss.
    *   **(b) Train `PINN₂`:**
        *   Freeze the weights of the *newly updated* `PINN₁`. Create `PINN₁_context`.
        *   Sample collocation points in Ω₂, physical boundary points of Ω₂, and interface points on Γ.
        *   **Loss for `PINN₂`:**
            *   `L_PDE₂`: PDE residual loss in Ω₂.
            *   `L_BC₂`: Physical boundary condition loss on the parts of ∂Ω₂ not touching Γ.
            *   `L_interface₂`:
                *   `|| u₂(x_Γ) - u₁_context(x_Γ) ||²`
                *   `|| ∂u₂/∂n(x_Γ) - ∂u₁_context/∂n(x_Γ) ||²`
        *   Optimize `PINN₂` using its total loss.
3.  **Convergence:** The MCP iterations continue until the solutions stabilize, or the interface mismatch losses become sufficiently small.

**Why use PINNs with DDMCP?**

1.  **Scalability for Large Domains:** A single PINN might struggle with very large or high-dimensional domains due to optimization challenges (barren plateaus, vanishing gradients) or memory limitations. Smaller PINNs for subdomains are often easier to train.
2.  **Handling Complex Geometries:** Complex geometries can be decomposed into simpler sub-geometries, each handled by a dedicated PINN.
3.  **Parallelization Potential:** While the MCP scheme described is sequential, variations can allow for more parallel updates, especially if the subdomains don't all share interfaces with each other directly (e.g., a checkerboard decomposition).
4.  **Flexibility in Model Architecture:** Different subdomains might benefit from different PINN architectures or hyperparameters, which is easier to manage with DDM.
5.  **Improved Training Dynamics:** Smaller networks might converge faster or to better local minima for their respective sub-problems. The MCP then stitches these local solutions together.
6.  **Tackling Multiphysics/Multiscale Problems:** Different physics or scales dominant in different regions can be naturally handled by specialized PINNs in those subdomains.

**Key Components in Implementation (like the Navier-Stokes example):**

*   **PINN Models:** Standard feed-forward neural networks for each subdomain.
*   **PDE Residual Calculation:** Using `torch.autograd.grad` for automatic differentiation to compute derivatives needed for PDE residuals and interface fluxes.
*   **Point Sampling Strategies:**
    *   Collocation points within each subdomain.
    *   Points on physical boundaries.
    *   Points specifically on the interfaces between subdomains.
*   **Context Model Handling:**
    *   Creating copies of neighbor models (e.g., `model_1_context.load_state_dict(model_1.state_dict())`).
    *   Setting context models to evaluation mode (`model_1_context.eval()`) to freeze their weights and disable gradient tracking for them during the neighbor's training.
*   **Loss Functions:**
    *   PDE residual loss.
    *   Physical boundary condition loss.
    *   Interface loss (for both solution values and fluxes).
*   **Loss Weights:** Hyperparameters (`lambda_pde`, `lambda_bc`, `lambda_interface_val`, `lambda_interface_flux`) that balance the different loss terms. These are crucial and often require careful tuning.
*   **Optimizers:** Adam is common, sometimes followed by L-BFGS for fine-tuning.
*   **Outer MCP Loop:** Controls the alternating training and context updates.

**Challenges:**

*   **Convergence of MCP:** The iterative process needs to converge to a consistent global solution.
*   **Hyperparameter Tuning:** More hyperparameters are introduced (loss weights for interface conditions, number of MCP iterations, epochs per MCP iteration).
*   **Interface Complexity:** For complex interface geometries, sampling and defining normal vectors can be tricky.
*   **Type of Interface Conditions:** Deciding which variables and which derivatives to enforce continuity for at the interface. Strong enforcement (pointwise matching like in the code) is common.
*   **Computational Cost:** While individual PINNs are smaller, the sequential nature of MCP can increase overall training time if not parallelized effectively.

**In summary, PINNs with DDMCP offer a structured and modular way to apply physics-informed machine learning to complex PDE problems by breaking them down into manageable sub-problems and iteratively enforcing consistency between them.** 


# PINN Navier-Stokes Solver with Domain Decomposition and MCP

This project implements a Physics-Informed Neural Network (PINN) for solving the 2D incompressible Navier-Stokes equations using:

- Domain decomposition (2×2 grid)
- Model Context Protocol (MCP)
- Visualization of velocity, streamlines, and pressure
- Error comparison with CFD ground truth

## Structure

- `main.py`: Main script for training, prediction, and visualization
- `utils.py`: Utility functions for grid generation and plotting
- `cfd_solution.npz`: Placeholder for ground-truth CFD data
- `README.md`: Project instructions

## Usage

1. Place your CFD ground truth data in `cfd_solution.npz` with:
   - `x`, `y`, `u`, `v`, `p`

2. Run training and visualization:
```bash
python main.py
```

Dependencies:
```bash
pip install torch numpy matplotlib
```

## Author
Generated with ❤️ by ChatGPT

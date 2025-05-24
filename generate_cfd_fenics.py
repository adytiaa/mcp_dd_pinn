"""
FEniCS script for generating CFD ground truth for a 2D lid-driven cavity problem.
Saves results as a .npz file: x, y, u, v, p
"""

from fenics import *
import numpy as np

# Mesh and function space
nx, ny = 40, 40
mesh = UnitSquareMesh(nx, ny)
V = VectorElement("P", mesh.ufl_cell(), 2)
Q = FiniteElement("P", mesh.ufl_cell(), 1)
TH = MixedElement([V, Q])
W = FunctionSpace(mesh, TH)

# Boundary conditions
inflow = 'near(x[1], 1)'
walls = 'near(x[1], 0) || near(x[0], 0) || near(x[0], 1)'

lid_velocity = Constant((1.0, 0.0))
noslip = Constant((0.0, 0.0))

bcs = [
    DirichletBC(W.sub(0), lid_velocity, inflow),
    DirichletBC(W.sub(0), noslip, walls)
]

# Define trial and test functions
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Parameters
mu = 0.01  # viscosity
rho = 1.0  # density

# Variational form for Stokes equations
a = mu*inner(grad(u), grad(v))*dx - div(v)*p*dx - q*div(u)*dx
L = dot(Constant((0, 0)), v)*dx

# Solve
w = Function(W)
solve(a == L, w, bcs)

u_sol, p_sol = w.split()

# Interpolate on regular grid
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
U = np.zeros_like(X)
V = np.zeros_like(X)
P = np.zeros_like(X)

for i in range(100):
    for j in range(100):
        point = Point(X[i, j], Y[i, j])
        try:
            u_val = u_sol(point)
            p_val = p_sol(point)
            U[i, j] = u_val[0]
            V[i, j] = u_val[1]
            P[i, j] = p_val
        except:
            U[i, j] = V[i, j] = P[i, j] = 0.0

np.savez("cfd_solution.npz", x=X, y=Y, u=U, v=V, p=P)
print("CFD ground truth saved as 'cfd_solution.npz'")

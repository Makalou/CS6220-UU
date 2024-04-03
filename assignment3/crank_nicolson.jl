using Pkg
Pkg.add("Plots")
Pkg.add("IterativeSolvers")
using Plots
using LinearAlgebra
using IterativeSolvers
using SparseArrays

nu = 0.1

u(x,y,t) = sin(pi*x)cos(pi*y)*exp(-pi*t)

f(x,y,t) = (2* pi * pi * nu - pi)sin(pi*x)cos(pi*y)*exp(-pi*t)

dt = 0.1
grid_size = 100
dx = 1.0/grid_size

rows = [1, 2, 3, 4, 2]
cols = [1, 2, 3, 4, 3]
vals = [1.0, 2.0, 3.0, 4.0, 5.0]

# Create the sparse matrix
A = sparse(rows, cols, vals, 4, 4)

b = [1.0, 2.0, 3.0,4.0]

# Generate some example 3D data
x = range(0, 1, length=1024)
y = range(0, 1, length=1024)

# Define time steps
t_values = range(0, π/2, length=500)

# Plot the time evolution of the heatmap
anim = @animate for t in t_values
    z = u.(x', y, t)  # Evaluate the function for the current time step
    heatmap(x, y, z, xlabel="X", ylabel="Y", title="Time = $t", c=:viridis, clims=(-1, 1))
end

# Save the animation as a GIF
gif(anim, "time_evolution_heatmap.gif", fps = 60)




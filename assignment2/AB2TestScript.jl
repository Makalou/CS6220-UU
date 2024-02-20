# Set simulation parameters
dt = 0.01  # Time-step
T = 10  # Final time
N = Int(T / dt)  # Number of steps to final time

# Set ODE parameters
f(t, y) = -y^2  # The function on the rhs of the ODE
y0 = 1  # Initial value
y1 = 1 / (1 + dt)  # Exact value at t_1

# Run a simulation
include("AB2.jl")
(tvec, yvec) = AB2(0, y0, y1, f, dt, N)

using Plots
plt = plot(tvec, yvec, xlabel="Time", ylabel="y(t)", label="AB2 Approximation")
display(plt)
readline()

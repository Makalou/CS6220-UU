using Pkg
Pkg.add("Plots")
Pkg.add("IterativeSolvers")
using Plots
using LinearAlgebra
using IterativeSolvers
using SparseArrays

# dq/dt + u*dq/dx = 0
u = 0.15
q0(x) = 1 / (1 + (9x - 5)^2)

dt = 0.001
grid_size = 1000
dx = 10.0/grid_size

qn = zeros(grid_size)
qn1 = zeros(grid_size)

# Define time steps
t_values = range(0, 10, length = Int64(10 / dt))

x_domain = range(0,10.0,1000);

function advect_ftcs(q1,q2,dt,dx)
    for i in 2 : 999
        q2[i] = q1[i] - dt * u * (q1[i + 1] - q1[i-1]) / ( 2 * dx)
    end
end

function advect_upwind1(q1,q2,dt,dx)
    for i in 2 : 1000
        q2[i] = q1[i] - dt * u * (q1[i] - q1[i-1]) / (dx)
    end
end

function advect_upwind2(q1,q2,dt,dx)
    for i in 3 : 1000
        q2[i] = q1[i] - dt * u * ( 3 * q1[i] - 4 * q1[i-1] + q1[i-2]) / (2 * dx)
    end
end

qn = q0.(x_domain)
anim_ftcs = @animate for t in t_values
    # Evaluate the function for the current time step
    advect_ftcs(qn,qn1,dt,dx)
    advect_ftcs(qn1,qn,dt,dx)
    plot(x_domain, qn1)
end

qn = q0.(x_domain)
anim_upwind1 = @animate for t in t_values
    # Evaluate the function for the current time step
    advect_upwind1(qn,qn1,dt,dx)
    advect_upwind1(qn1,qn,dt,dx)
    plot(x_domain, qn1)
end

qn = q0.(x_domain)
anim_upwind2 = @animate for t in t_values
    # Evaluate the function for the current time step
    advect_upwind2(qn,qn1,dt,dx)
    advect_upwind2(qn1,qn,dt,dx)
    plot(x_domain, qn1)
end

# Save the animation as a GIF
gif(anim_ftcs, "1d_advection_ftcs.gif", fps = 60)
gif(anim_upwind1, "1d_advection_upwind1.gif", fps = 60)
gif(anim_upwind2, "1d_advection_upwind2.gif", fps = 60)


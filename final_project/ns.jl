using Pkg
Pkg.add("Plots")
Pkg.add("IterativeSolvers")
using Plots
using LinearAlgebra
using IterativeSolvers
using SparseArrays

nu = 1.0

# reference solution
w(t) = 1 + sin(2 * pi * t^2)
dwdt(t) = 4 * pi * t * cos(2 * pi * t^2)
u(x,y,t) = cos(2*pi*(x - w(t))) * (3 * y ^ 2 - 2 * y)
v(x,y,t) = 2*pi*sin(2 * pi * (x - w(t))) * y^2 * (y - 1)
p(x,y,t) = -(dwdt(t)/(2 * pi)) * sin(2*pi*(x-w(t))) * (sin(2 * pi * y) - 2*pi*y + pi) - nu * cos(2*pi*(x - w(t))) * (-2 * sin(2*pi*y) + 2*pi*y - pi)

h = 0.001
dt = 0.5 * h

t1 = 0
t2 = 1.0

N = Int((t2 - t1) / dt)

u_n = zeros(100)
v_n = zeros(100)
p_n = zeros(100)
phi_n = zeros(100)
u_star = zeros(100)
v_star = zeros(100)
p_grad_x_n = zeros(100)
p_grad_y_n = zeros(100)

# Set initial state


for n in 1 : N
    tn = dt * N
    tn1 = tn + dt

end


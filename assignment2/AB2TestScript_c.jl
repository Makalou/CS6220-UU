# Set simulation parameters
exact_dt = 0.0001  # Time-step
T = 10  # Final time
#N = Int(T / dt)  # Number of steps to final time

# Set ODE parameters
f(t, y) = -y^2  # The function on the rhs of the ODE
y0 = 1  # Initial value
y1_Exact = 1 / (1 + exact_dt)  # Exact value at t_1
include("AB2.jl")
# Compute exact solution
(tvec_Exact, yvec_Exact) = AB2(0,y0,y1_Exact, f, exact_dt, Int(T / exact_dt))

yfinal_Exact = yvec_Exact[end]

dt_N = 2000 # test timestep range from 0.001 to dt_N * 0.001
dts = []
error_FE = []
error_RK2_Cls = []
error_RK2_Mid = []
error_RK2_Ralston = []

function RK2Step(f,t0,y0,dt,b1,b2,c2,a21)
    k1 = f(t0,y0)
    k2 = f(t0 + c2 * dt, y0 + dt * a21 * k1)
    return y0 + dt * (b1 * k1 + b2 * k2)
end

relative_l2(x,y) = ((x-y)/x)^2

for n = 1:dt_N
    dt = 0.001 * n
    N = Int(floor(T / dt))
    extra_dt = T - N * dt # For case where T is not multiple of dt
    if extra_dt > 0
        continue
    end
    push!(dts,dt)

    y1_FE = y0 + dt * f(0,y0) # Use Backward Euler
    y1_RK2_Cls = RK2Step(f,0,y0,dt,0.5,0.5,1,1) # Use classical RK2
    y1_RK2_Mid = RK2Step(f,0,y0,dt,0,1,0.5,0.5); # Use MiddlePoint RK2
    y1_RK2_Ralston = RK2Step(f,0,y0,dt,0.25,0.75,(2.0/3.0),(2.0/3.0)) # Use Ralston RK2

    # Run a simulation

    (tvec_FE, yvec_FE) = AB2(0, y0, y1_FE, f, dt,N)
    (tvec_RK2_Cls, yvec_RK2_Cls) = AB2(0, y0, y1_RK2_Cls, f, dt, N)
    (tvec_RK2_Mid, yvec_RK2_Mid) = AB2(0, y0, y1_RK2_Mid, f, dt, N)
    (tvec_RK2_Ralston, yvec_RK2_Ralston) = AB2(0, y0, y1_RK2_Ralston, f, dt, N)

    yfinal_FE = yvec_FE[end]
    yfinal_RK2_Cls = yvec_RK2_Cls[end]
    yfinal_RK2_Mid = yvec_RK2_Mid[end]
    yfinal_RK2_Ralston = yvec_RK2_Ralston[end]

    # For case where T is not multiple of dt
    if extra_dt > 0
        (tvec_FE, yvec_FE_extra) = AB2((N-1)*dt, yvec_FE[end - 1], yvec_FE[end], f, extra_dt,1)
        (tvec_RK2_Cls, yvec_RK2_Cls_extra) = AB2((N-1)*dt, yvec_RK2_Cls[end - 1], yvec_RK2_Cls[end], f, extra_dt,1)
        (tvec_RK2_Mid, yvec_RK2_Mid_extra) = AB2((N-1)*dt, yvec_RK2_Mid[end - 1], yvec_RK2_Mid[end], f, extra_dt,1)
        (tvec_RK2_Ralston, yvec_RK2_Ralston_extra) = AB2((N-1)*dt, yvec_RK2_Ralston[end - 1], yvec_RK2_Ralston[end], f, extra_dt,1)

        yfinal_FE = yvec_FE_extra[end]
        yfinal_RK2_Cls = yvec_RK2_Cls_extra[end]
        yfinal_RK2_Mid = yvec_RK2_Mid_extra[end]
        yfinal_RK2_Ralston = yvec_RK2_Ralston_extra[end]
    end

    push!(error_FE,relative_l2(yfinal_Exact,yfinal_FE))
    push!(error_RK2_Cls,relative_l2(yfinal_Exact,yfinal_RK2_Cls))
    push!(error_RK2_Mid,relative_l2(yfinal_Exact,yfinal_RK2_Mid))
    push!(error_RK2_Ralston,relative_l2(yfinal_Exact,yfinal_RK2_Ralston))
end

using Plots
plt = plot(dts, error_FE, yscale = :log2, xlabel="TimeStep(interval = 0.001)", ylabel="Relative l2 error(log scale)", label="FE-bootstrapped-AB2")
plot!(plt,dts, error_RK2_Cls, label="RK2Classical-bootstrapped-AB2")
plot!(plt,dts, error_RK2_Mid, label="RK2MidPoint-bootstrapped-AB2")
plot!(plt,dts, error_RK2_Ralston, label="RK2Ralston-bootstrapped-AB2")
#display(plt)
savefig(plt,"accuracy2.png")

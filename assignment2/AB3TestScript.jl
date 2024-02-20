# Set simulation parameters
exact_dt = 0.0001  # Time-step
T = 10  # Final time
#N = Int(T / dt)  # Number of steps to final time

# Set ODE parameters
f(t, y) = -y^2  # The function on the rhs of the ODE
y0 = 1  # Initial value
y1_Exact = 1 / (1 + exact_dt)  # Exact value at t_1
include("AB2.jl")
include("AB3.jl")
# Compute exact solution
(tvec_Exact, yvec_Exact) = AB2(0,y0,y1_Exact, f, exact_dt, Int(T / exact_dt))

yfinal_Exact = yvec_Exact[end]

dt_N = 500 # test timestep range from 0.001 to dt_N * 0.001
dts = []
error_AB2= []
error_AB3 = []

relative_l2(x,y) = ((x-y)/x)^2

for n = 1:dt_N
    dt = 0.001 * n
    N = Int(floor(T / dt))
    extra_dt = T - N * dt # For case where T is not multiple of dt
    if extra_dt > 0
        continue
    end
    push!(dts,dt)

    y1 = RK3(f,0,y0,dt)

    # Run a simulation

    (tvec_AB2, yvec_AB2) = AB2(0, y0, y1, f, dt,N)
    (tvec_AB3, yvec_AB3) = AB3(0, y0, y1, f, dt,N)

    yfinal_AB2 = yvec_AB2[end]
    yfinal_AB3 = yvec_AB3[end]

    # For case where T is not multiple of dt
    if extra_dt > 0
        (tvec_AB2, yvec_AB2_extra) = AB2((N-1)*dt, yvec_AB2[end - 1], yvec_AB2[end], f, extra_dt,1)
        (tvec_AB3, yvec_AB3_extra) = AB2((N-1)*dt, yvec_AB3[end - 1], yvec_AB3[end], f, extra_dt,1)

        yfinal_AB2 = yvec_AB2_extra[end]
        yfinal_AB3 = yvec_AB3_extra[end]
    end

    push!(error_AB2,relative_l2(yfinal_Exact,yfinal_AB2))
    push!(error_AB3,relative_l2(yfinal_Exact,yfinal_AB3))
end

using Plots
plt = plot(dts, error_AB2, yscale = :log2, xlabel="TimeStep(interval = 0.001)", ylabel="Relative l2 error(log scale)", label="AB2")
plot!(plt,dts, error_AB3, label="AB3")
#display(plt)
savefig(plt,"accuracy3(1).png")

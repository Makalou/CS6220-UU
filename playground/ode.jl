using Pkg
Pkg.add("Plots")
using Plots
include("solver.jl")
using .MySolver

f(t,y) = sin(y^2)

y_0 = 1.0
c = log(cot(0.5*y_0))

ground_truth(t) = 2 * acot(exp(c - t))

y_ab1 = [y_0]
y_ab2 = [y_0]
y_am1 = [y_0]
y_am2 = [y_0]
y_abam2pc = [y_0]
y_bdf2 = [y_0]
y_rk2 = [y_0]
y_rk4 = [y_0]
y_ground_truth = [y_0]

y_n = y_0
delta_t = 1.0
t = [i * delta_t for i in 0:25]

for n in range(1,length(t)-1)
    push!(y_ground_truth,ground_truth(t[n+1]))
    # we want to know y_n+1
    push!(y_ab1,MySolver.AB1(f,y_ab1[n],t[n],delta_t))
    push!(y_am1,MySolver.AM1(f,y_am1[n],t[n],delta_t))
    push!(y_am2,MySolver.AM2(f,y_am2[n],t[n],delta_t))
    push!(y_rk2,MySolver.RK2(f,y_rk2[n],t[n],delta_t))
    push!(y_rk4,MySolver.RK4(f,y_rk4[n],t[n],delta_t))
    if n==1
        push!(y_ab2,MySolver.AB1(f,y_ab1[n],t[n],delta_t))
        push!(y_abam2pc,MySolver.AB1(f,y_ab1[n],t[n],delta_t))
        push!(y_bdf2,MySolver.AB1(f,y_ab1[n],t[n],delta_t))
    else
        push!(y_ab2,MySolver.AB2(f,y_ab1[n-1],t[n-1],y_ab1[n],t[n],delta_t))
        push!(y_abam2pc,MySolver.ABAM2PC(f,y_abam2pc[n-1],t[n-1],y_abam2pc[n],t[n],t[n+1],delta_t))
        push!(y_bdf2,MySolver.BDF2(f,y_bdf2[n-1],y_bdf2[n],t[n],delta_t))
    end
end

#plt = plot(t, y_ground_truth, label = "Ground Truth",title="Solve dydt = siny(dt = $delta_t)")
plt = plot(label = "Ground Truth",title="Solve dydt = siny(dt = $delta_t)")
plot!(plt,t,y_ab1, label = "AB1(Explicit Euler)", shape = :circle)
plot!(plt,t,y_am1, label = "AM1(Implicit Euler)", shape = :square)
plot!(plt,t,y_ab2, label = "AB2")
plot!(plt,t,y_am2, label = "AM2", shape = :circle)
plot!(plt,t,y_abam2pc, label = "AB-AM2 Predictor Correction",shape = :square)
plot!(plt,t,y_bdf2, label = "BDF2")
plot!(plt,t,y_rk2, label = "Explicit RK2",shape = :circle)
plot!(plt,t,y_rk4, label = "Explicit RK4", shape = :square)

display(plt)

# Prevent process close too soon
# Presse enter or use ctrl + C to exit the process
readline()
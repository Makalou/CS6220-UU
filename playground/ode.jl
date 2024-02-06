using Pkg
Pkg.add("Plots")
using Plots
Pkg.add("Roots")
using Roots

f(t,y) = sin(y)

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

function AB1(y_n,t_n,delta_t)
    return y_n + delta_t * f(t_n,y_n);
end

function AB2(y_n_1,t_n_1,y_n,t_n,delta_t)
    return y_n + delta_t * (1.5 * f(t_n,y_n) - 0.5 * f(t_n_1,y_n_1))
end

function AM1(y_n,t_n,delta_t)
    y_n_p_1_guess = y_n + delta_t * f(t_n,y_n)
    g(y_n_p_1) = y_n + delta_t * f(t_n + delta_t,y_n_p_1) - y_n_p_1
    return find_zero(g,y_n_p_1_guess)
end

function AM2(y_n,t_n,delta_t)
    f_n = f(t_n,y_n)
    y_n_p_1_guess = y_n + delta_t * f_n
    g(y_n_p_1) = y_n + delta_t * (0.5 * f(t_n + delta_t,y_n_p_1) + 0.5 * f_n) - y_n_p_1
    return find_zero(g,y_n_p_1_guess)
end

function ABAM2PC(y_n_1,t_n_1,y_n,t_n, t_n_p_1,delta_t)
    y_n_p_1 = AB2(y_n_1,t_n_1,y_n,t_n,delta_t)
    return y_n + delta_t * (0.5 * f(t_n_p_1,y_n_p_1) + 0.5 * f(t_n,y_n))
end

function BDF2(y_n_1,y_n,t_n,delta_t)
    y_n_p_1_guess = y_n + delta_t * f(t_n,y_n)
    g(y_n_p_1) = (4.0/3.0) * y_n - (1.0/3.0) * y_n_1 + (2.0/3.0) * delta_t * f(t_n + delta_t, y_n_p_1) - y_n_p_1
    return find_zero(g,y_n_p_1_guess)
end

function RK2(y_n,t_n,delta_t)
    k1 = delta_t * f(t_n,y_n)
    k2 = delta_t * f(t_n + delta_t, y_n + k1)
    return y_n + 0.5*(k1 + k2)
end

function RK4(y_n,t_n,delta_t)
    k1 = delta_t * f(t_n,y_n)
    k2 = delta_t * f(t_n + 0.5 * delta_t, y_n + 0.5 * k1)
    k3 = delta_t * f(t_n + 0.5 * delta_t, y_n + 0.5 * k2)
    k4 = delta_t * f(t_n + delta_t, y_n + k3)
    return y_n + (k1 + 2*k2 + 2*k3 + k4) / 6
end

y_n = y_0
delta_t = 2.0
t = [i * delta_t for i in 0:50]

for n in range(1,length(t)-1)
    push!(y_ground_truth,ground_truth(t[n+1]))
    # we want to know y_n+1
    push!(y_ab1,AB1(y_ab1[n],t[n],delta_t))
    push!(y_am1,AM1(y_am1[n],t[n],delta_t))
    push!(y_am2,AM2(y_am2[n],t[n],delta_t))
    push!(y_rk2,RK2(y_rk2[n],t[n],delta_t))
    push!(y_rk4,RK4(y_rk4[n],t[n],delta_t))
    if n==1
        push!(y_ab2,AB1(y_ab1[n],t[n],delta_t))
        push!(y_abam2pc,AB1(y_ab1[n],t[n],delta_t))
        push!(y_bdf2,AB1(y_ab1[n],t[n],delta_t))
    else
        push!(y_ab2,AB2(y_ab1[n-1],t[n-1],y_ab1[n],t[n],delta_t))
        push!(y_abam2pc,ABAM2PC(y_abam2pc[n-1],t[n-1],y_abam2pc[n],t[n],t[n+1],delta_t))
        push!(y_bdf2,BDF2(y_bdf2[n-1],y_bdf2[n],t[n],delta_t))
    end
end

plt = plot(t, y_ground_truth, label = "Ground Truth",title="Solve dydt = siny(dt = 2.0)")
plot!(plt,t,y_ab1, label = "AB1")
plot!(plt,t,y_am1, label = "AM1")
plot!(plt,t,y_ab2, label = "AB2")
plot!(plt,t,y_am2, label = "AM2")
plot!(plt,t,y_abam2pc, label = "AB-AM2 Predictor Correction")
plot!(plt,t,y_bdf2, label = "BDF2")
plot!(plt,t,y_rk2, label = "RK2")
plot!(plt,t,y_rk4, label = "RK4")

display(plt)

# Prevent process close too soon
# Presse enter or use ctrl + C to exit the process
readline()
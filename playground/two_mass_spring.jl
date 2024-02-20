using Pkg
Pkg.add("Plots")
using Plots
include("solver.jl")
using .MySolver

m1 = 10.0
m2 = 10.0
k = 5.0

l_r = 15.0
l_0 = 20.0

x1_0 = -0.5 * l_0
x2_0 = 0.5 * l_0

v1(v,t,y) = v
v2(v,t,y) = v

a1(x1,x2,t,y) = (abs(x1-x2) - l_r) * k / m1 
a2(x1,x2,t,y) = -(abs(x1-x2) - l_r) * k / m2

v1_0 = 0.0; v2_0 = 0.0

x1_ab1 = [x1_0]; x2_ab1 = [x2_0]; v1_ab1 = [v1_0]; v2_ab1 = [v2_0];
x1_ab2 = [x1_0]; x2_ab2 = [x2_0]; v1_ab2 = [v1_0]; v2_ab2 = [v2_0];
x1_am1 = [x1_0]; x2_am1 = [x2_0]; v1_am1 = [v1_0]; v2_am1 = [v2_0];
x1_am2 = [x1_0]; x2_am2 = [x2_0]; v1_am2 = [v1_0]; v2_am2 = [v2_0];
x1_abam2pc = [x1_0]; x2_abam2pc = [x2_0]; v1_abam2pc = [v1_0]; v2_abam2pc = [v2_0];
x1_bdf2 = [x1_0]; x2_bdf2 = [x2_0]; v1_bdf2 = [v1_0]; v2_bdf2 = [v2_0];
x1_rk2 = [x1_0]; x2_rk2 = [x2_0]; v1_rk2 = [v1_0]; v2_rk2 = [v2_0];
x1_rk4 = [x1_0]; x2_rk4 = [x2_0]; v1_rk4 = [v1_0]; v2_rk4 = [v2_0];

kenergy = [0.0]
penergy =[0.0]

delta_t = 0.01
t = [i * delta_t for i in 0:(100/delta_t)]

for n in range(1,length(t)-1)
    v1_n(t,y) = v1(v1_ab1[n],t,y) 
    v2_n(t,y) = v1(v2_ab1[n],t,y) 
    push!(x1_ab1,MySolver.AB1(v1_n,x1_ab1[n],t[n],delta_t))
    push!(x2_ab1,MySolver.AB1(v2_n,x2_ab1[n],t[n],delta_t))

    a1_n(t,y) = a1(x1_ab1[n],x2_ab1[n],t,y)
    a2_n(t,y) = a2(x1_ab1[n],x2_ab1[n],t,y)
    push!(v1_ab1,MySolver.AB1(a1_n,v1_ab1[n],t[n],delta_t))
    push!(v2_ab1,MySolver.AB1(a2_n,v2_ab1[n],t[n],delta_t))

    push!(kenergy,0.5 * m1 * v1_ab1[n]^2 + 0.5 * m2 * v2_ab1[n]^2 )
    push!(penergy,0.5 * (abs(x1_ab1[n] - x2_ab1[n]) - l_r)^2)
end

plt = plot(t, x1_ab1, label = "X1 AB1",title="Solve (dt = $delta_t)")
plot!(plt,t, x2_ab1, label = "X2 AB1",title="Solve (dt = $delta_t)")
plot!(plt,t, kenergy, label = "Kenetic Enerygy AB1",title="Solve (dt = $delta_t)")
plot!(plt,t, penergy, label = "Potential Enerygy AB1",title="Solve (dt = $delta_t)")
display(plt)

readline()


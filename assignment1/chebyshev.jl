using Pkg
Pkg.add("Plots")
using Plots
using LinearAlgebra
include("utilities.jl")
using .MyUtil

# Define the Runge function
runge(x) = 1 / (1 + 25 * (x * x))

measure_points = MyUtil.equispacefbig(10000)
f = runge.(measure_points)

relative_l2(x,y) = norm(x-y)/norm(x)

num_of_nodes = [i * 1 for i in 2:200] 
times = []
errors = []

for num_of_node in num_of_nodes
    start_time = time_ns();
    Vandermonde_chebyshev,samples_chebyshev = MyUtil.generate_vandermonde(num_of_node, runge, MyUtil.chebyshevfbig)
    coefficients_chebyshev = Vandermonde_chebyshev \ samples_chebyshev
    interpol(x)= sum(c * x^n for (c, n) in zip(coefficients_chebyshev, 0:num_of_node - 1))
    p = interpol.(measure_points)
    end_time = time_ns();
    push!(errors,relative_l2(f,p))
    push!(times, (end_time - start_time)/1e9)
end

plt = plot(num_of_nodes, errors, label="errors", xlabel="num of nodes", ylabel="y", title="Accuracy Vs Error")
plot!(plt,num_of_nodes, times, label="times")
display(plt)
readline()
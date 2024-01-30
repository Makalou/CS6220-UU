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

max_nodes = 200
num_of_nodes = [i * 1 for i in 2:max_nodes] 
times1 = []
times2 = []
times3 = []

errors1 = []
errors2 = []
errors3 = []

for num_of_node in num_of_nodes
    start_time = time_ns();
    Vandermonde_chebyshev,samples_chebyshev = MyUtil.generate_vandermonde(num_of_node, runge, MyUtil.chebyshevf32)
    coefficients_chebyshev = Vandermonde_chebyshev \ samples_chebyshev
    interpol(x)= sum(c * x^n for (c, n) in zip(coefficients_chebyshev, 0:num_of_node - 1))
    p = interpol.(measure_points)
    end_time = time_ns();
    time= end_time - start_time;
    push!(errors1,relative_l2(f,p))
    push!(times1, cbrt(time/1e3))
    println("float32",size(Vandermonde_chebyshev), size(samples_chebyshev),size(coefficients_chebyshev))
end

for num_of_node in num_of_nodes
    start_time = time_ns();
    Vandermonde_chebyshev,samples_chebyshev = MyUtil.generate_vandermonde(num_of_node, runge, MyUtil.chebyshevf64)
    coefficients_chebyshev = Vandermonde_chebyshev \ samples_chebyshev
    interpol(x)= sum(c * x^n for (c, n) in zip(coefficients_chebyshev, 0:num_of_node - 1))
    p = interpol.(measure_points)
    end_time = time_ns();
    time= end_time - start_time;
    push!(errors2,relative_l2(f,p))
    push!(times2, cbrt(time/1e3))
    println("float64",size(Vandermonde_chebyshev), size(samples_chebyshev),size(coefficients_chebyshev))
end

for num_of_node in num_of_nodes
    start_time = time_ns();
    Vandermonde_chebyshev,samples_chebyshev = MyUtil.generate_vandermonde(num_of_node, runge, MyUtil.chebyshevfbig)
    coefficients_chebyshev = Vandermonde_chebyshev \ samples_chebyshev
    interpol(x)= sum(c * x^n for (c, n) in zip(coefficients_chebyshev, 0:num_of_node - 1))
    p = interpol.(measure_points)
    end_time = time_ns();
    time= end_time - start_time;
    push!(errors3,relative_l2(f,p))
    push!(times3, cbrt(time/1e3))
    println("big float",size(Vandermonde_chebyshev), size(samples_chebyshev),size(coefficients_chebyshev))
end

plt1 = plot(num_of_nodes, errors1, label="errors float32", xlabel="num of nodes", ylabel="y", title="Accuracy Chebyshev",ylims = (0,10), xlims=(0, max_nodes))
plot!(num_of_nodes, errors2, label="errors float64")
plot!(num_of_nodes, errors3, label="errors bigfloat")

plt2 = plot(num_of_nodes, times1, label="total time float32", xlabel="num of nodes", ylabel="time/ms", title="Total time Chebyshev",xlims=(0, max_nodes),aspect_ratio=:equal)
plot!(num_of_nodes, times2, label="total time float64")
plot!(num_of_nodes, times3, label="total time bigfloat")
# display(plt)
savefig(plt1,"Accuracy.png")
savefig(plt2,"Time.png")

# Prevent process close too soon
# Presse enter or use ctrl + C to exit the process
# readline()
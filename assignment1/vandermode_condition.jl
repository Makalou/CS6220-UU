using Pkg
Pkg.add("Plots")
using Plots
using LinearAlgebra
include("utilities.jl")
using .MyUtil

# Define the Runge function
runge(x) = 1 / (1 + 25 * (x * x))
num_of_nodes = [i * 1 for i in 2:75] 

# write my own condition number because default cond doesn't support BigFloat
cond_num(x) = norm(x,2) * norm(inv(x),2)
# Define get condition number function
getCondNum(num_points,generator) = cond_num(MyUtil.generate_vandermonde(num_points,runge,generator)[1])

plt = plot(num_of_nodes, getCondNum.(num_of_nodes,equispacef32), label="Equispace", xlabel="num of nodes", ylabel="cond", title="Vandermonde Condition Number Float32")
plot!(plt,num_of_nodes, getCondNum.(num_of_nodes,chebyshevf32), label="chebyshev")
display(plt)

# Prevent process close too soon
# Presse enter or use ctrl + C to exit the process
readline()

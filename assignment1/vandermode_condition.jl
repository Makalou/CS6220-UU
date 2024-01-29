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

plt = plot(num_of_nodes, getCondNum.(num_of_nodes,equispacef32), label="Equispace", xlabel="num of nodes", ylabel="cond", title="Vandermonde Condition Number")
plot!(plt,num_of_nodes, getCondNum.(num_of_nodes,chebyshevf32), label="chebyshev")
display(plt)

# Vandermonde_equispace,samples_equispace = generate_vandermonde(20, runge, equispace)
# Vandermonde_chebyshev,samples_chebyshev = generate_vandermonde(20, runge, chebyshev)

# coefficients_equispace = Vandermonde_equispace \ samples_equispace
# coefficients_chebyshev = Vandermonde_chebyshev \ samples_chebyshev

# degree = 19

# poly_interpolant_equispace(x) = sum(c * x^n for (c, n) in zip(coefficients_equispace, 0:degree))
# poly_interpolant_chebyshev(x) = sum(c * x^n for (c, n) in zip(coefficients_chebyshev, 0:degree))

# Test the interpolant at some points
# test_points = range(-2.0, stop=2.0, length=100)

# plt = plot(test_points, runge.(test_points), label="Runge function", xlabel="x", ylabel="y", title="Polynomial Interpolation")
# plot!(plt,test_points, poly_interpolant_equispace.(test_points), label="Polynomial Interpolant Equispace", xlabel="x", ylabel="y", title="Polynomial Interpolation")
# plot!(plt,test_points, poly_interpolant_chebyshev.(test_points), label="Polynomial Interpolant chebyshev", xlabel="x", ylabel="y", title="Polynomial Interpolation")
# ylims!(plt,(-0.25, 1.25)) 
# display(plt)

# Prevent process close too soon
# Presse enter or use ctrl + C to exit the process
readline()

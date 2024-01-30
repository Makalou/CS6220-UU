using Pkg
Pkg.add("Plots")
using Plots
using LinearAlgebra
include("utilities.jl")
using .MyUtil

# Define the Runge function
runge(x) = 1 / (1 + 25 * (x * x))

function get_interpolant(num_points,target_function,generator)
    V, samples = MyUtil.generate_vandermonde(num_points,target_function,generator)
    coefficients = V \ samples
    degree = num_points - 1
    interpo(x) = sum(c * x^n for (c, n) in zip(coefficients, 0:degree))
    return interpo
end

num_of_nodes = 50

poly_interpolant_equispace = get_interpolant(num_of_nodes,runge,MyUtil.equispacef32)
poly_interpolant_chebyshev = get_interpolant(num_of_nodes,runge,MyUtil.chebyshevf32)

# Test the interpolant at some points
test_points = range(-2.0, stop=2.0, length=100)

plt = plot(test_points, runge.(test_points), label="Runge function", xlabel="x", ylabel="y", title="Polynomial Interpolation")
plot!(plt,test_points, poly_interpolant_equispace.(test_points), label="Polynomial Interpolant Equispace", xlabel="x", ylabel="y", title="Polynomial Interpolation")
plot!(plt,test_points, poly_interpolant_chebyshev.(test_points), label="Polynomial Interpolant chebyshev", xlabel="x", ylabel="y", title="Polynomial Interpolation")
ylims!(plt,(-0.25, 1.25)) 
display(plt)

# Prevent process close too soon
# Presse enter or use ctrl + C to exit the process
readline()





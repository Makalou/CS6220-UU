using Pkg
Pkg.add("Plots")
using Plots
using LinearAlgebra
Pkg.add("ChebyshevApprox")
using ChebyshevApprox

#define Chebyshev polynomials
#T(n,x) = (n == 0) ? 1 : ((n==1) ? x : 2x*T(n-1,x) - T(n-2,x))
T(n,x) = cos(n * acos(x))

dTdx(n,x) = (n <= 1) ? n : 2x *dTdx(n-1,x) + 2*T(n-1,x) - dTdx(n-2,x)

#d2Tdx2(n,x) = (n <= 1) ? 0 : 2x * d2Tdx2(n-1,x) + 4 * dTdx(n-1,x) - d2Tdx2(n-2,x)
d2Tdx2(n,x) = -n*n * (cos(n * acos(x))/(1 - x*x)) + ((n * x * sin(n * acos(x)))/sqrt(((1-x*x)^3)))

u(x) = cos(pi/2 * x)
f(x) = pi*pi/4*cos(pi/2 * x)

n = 5  # Number of extrema (degree of the Chebyshev polynomial)
extrema = nodes(n,:chebyshev_extrema).points

Tmat = zeros(n,n)

for j in 1 : n 
    Tmat[1, j] = T(j-1,extrema[1])
    Tmat[n, j] = T(j-1,extrema[n])
end

for i in 2:n-1
    for j in 1:n
        Tmat[i, j] = d2Tdx2(j-1,extrema[i])
    end
end

b = zeros(n)

b[1] = b[n] = 0

for j in 2 : n-1
    b[j] = -f(extrema[j])
end

println("Solve for coefficients...")
c = Tmat \ b
println("Solve done.")

u_1(x) = sum(c[i] * T(i-1,x) for i in 1:length(c))

test_x = range(-1, stop=1, length=1000)

u_ref = u.(test_x)

plt = plot(test_x, u.(test_x))
plot!(plt,test_x, u_1.(test_x))
display(plt)
# Prevent process close too soon
# Presse enter or use ctrl + C to exit the process
readline()
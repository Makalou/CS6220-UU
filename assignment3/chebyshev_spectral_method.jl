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

#u(x) = cos(pi/2 * x)
#f(x) = (pi^2)/4*cos(pi/2 * x)

w1 = pi
w2 = 2pi
w3 = 4pi
w4 = 8pi
w5 = 16pi
w6 = 32pi
w7 = 64pi
w8 = 128pi

u(x) = sin(w1 *x) + sin(w2 * x) + sin(w3 * x) + sin(w4 * x) + sin(w5 * x) + sin(w6 * x) + sin(w7* x) + sin(w8 * x)
f(x) = (w1)^2 * sin(w1*x) + (w2)^2 * sin(w2*x) + (w3)^2 * sin(w3*x) + (w4)^2 * sin(w4*x) + (w5)^2 * sin(w5*x) + (w6)^2 * sin(w6*x) + (w7)^2 * sin(w7*x) + (w8)^2 * sin(w8*x)

frame = 0

for n in 20 : 2000 # Number of extrema (degree of the Chebyshev polynomial)
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
    
    if frame == 0
        plt = plot(test_x, u.(test_x),ylims = (-4,4))
        plot!(plt,test_x, u_1.(test_x))
        savefig(plt,"res/res_$n.png")
    end
    #display(plt)
    # Prevent process close too soon
    # Presse enter or use ctrl + C to exit the process
    #readline()
    println("n = $n")
    u_ref = u.(test_x)
    error = u_ref  - u_1.(test_x)
    println(norm(error,2)/norm(u_ref,2))
    println(norm(error,Inf)/norm(u_ref,Inf))
    global frame = (frame + 1) % 100
end

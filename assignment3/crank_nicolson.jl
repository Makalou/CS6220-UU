using Pkg
Pkg.add("Plots")
Pkg.add("IterativeSolvers")
Pkg.add("Preconditioners")
using Plots
using LinearAlgebra
using IterativeSolvers
using SparseArrays
using Preconditioners

#nu = 1/(2 * pi)

decay = pi
freq = pi
nu = 1.0
u(x,y,t) = sin(freq*x)*cos(freq*y)*exp(-decay*t) + 1.0
f(x,y,t) = (2 * freq^2 * nu - decay)*sin(freq*x)*cos(freq*y)*exp(-decay*t)

grid_size = 101
h = 1.0/(grid_size-1)
dt = 0.001#2 * h^2/nu #0.001

alpha1 = 1.0 + (2.0 * nu * dt)/(h^2)
alpha2 = 1.0 - (2.0 * nu * dt)/(h^2)
beta = (nu * dt)/(2.0*h^2) 

# Construct the L matrix(without boundray)
grid_size_in = grid_size - 2
N_in = (grid_size_in)^2

get_idx(i,j,grid_size_in) = (i - 1) * grid_size_in + (j - 1) + 1

rows = []
cols = []
vals = Float64[]

for i in 1 : grid_size_in
    for j in 1 : grid_size_in
        idx0 = get_idx(i,j,grid_size_in)
        push!(rows,idx0)
        push!(cols,idx0)
        push!(vals,alpha1)

        idx_left = get_idx(i-1,j,grid_size_in)
        idx_right = get_idx(i+1,j,grid_size_in)
        idx_down = get_idx(i,j-1,grid_size_in)
        idx_up = get_idx(i,j+1,grid_size_in)

        if j == 1 
            idx_down = 0
        end

        if j == grid_size_in
            idx_up = 0
        end

        if i == 1
            idx_left = 0
        end

        if i == grid_size_in
            idx_right = 0
        end

        if idx_left != 0
            push!(rows,idx0)
            push!(cols,idx_left)
            push!(vals,-beta)
        end

        if idx_right != 0
            push!(rows,idx0)
            push!(cols,idx_right)
            push!(vals,-beta)
        end

        if idx_down != 0
            push!(rows,idx0)
            push!(cols,idx_down)
            push!(vals,-beta)
        end

        if idx_up != 0
            push!(rows,idx0)
            push!(cols,idx_up)
            push!(vals,-beta)
        end
    end
end

L = sparse(rows, cols, vals, N_in, N_in)
@assert issymmetric(L)
@assert isposdef(L)

# Incomplete Cholesky preconditioner with cut-off level 2
PreConL = CholeskyPreconditioner(L, 2)

u_n = zeros(N_in)
rhs = zeros(N_in)
u_ref = zeros(N_in)

# Set the initial state
for i in 1 : grid_size_in
    for j in 1 : grid_size_in
        u_n[get_idx(i,j,grid_size_in)] = u(i * h, j * h,0)
    end
end

# Start simulation
frame = 0
for n in 0 : Int(.5/dt)
    tn  = dt * n
    tn1 = tn + dt
    println("t : ",tn)
    # Construct the rhs vector
    for i in 1 : grid_size_in
        for j in 1 : grid_size_in
            idx0 = get_idx(i,j,grid_size_in)
            @assert idx0 >= 1 && idx0 <= N_in
            u_n_center = u_n[idx0]
            u_n_left = 0.0
            u_n_right = 0.0
            u_n_down = 0.0
            u_n_up = 0.0

            x = i * h
            y = j * h

            if i == 1 
                @assert x - h == 0.0
                u_n_left = u(0,y,tn) # Use Dirichelet boundary condition
            else
                u_n_left = u_n[get_idx(i - 1,j,grid_size_in)]
            end

            if i == grid_size_in
                @assert x + h == 1.0
                u_n_right = u(1.0,y,tn) # Use Dirichelet boundary condition
            else
                u_n_right = u_n[get_idx(i + 1,j,grid_size_in)]
            end

            if j == 1 
                @assert y - h == 0.0
                u_n_down = u(x, 0 ,tn) # Use Dirichelet boundary condition
            else
                u_n_down = u_n[get_idx(i,j - 1,grid_size_in)]
            end

            if j == grid_size_in 
                @assert y + h == 1.0
                u_n_up = u(x,1.0,tn) # Use Dirichelet boundary condition
            else
                u_n_up = u_n[get_idx(i,j + 1,grid_size_in)]
            end
        
            rhs[idx0] = alpha2 * u_n_center + beta * (u_n_left + u_n_right + u_n_down + u_n_up) + dt * 0.5 * (f(x, y, tn) + f(x, y, tn1))

            if i == 1 
                @assert x - h == 0.0
                rhs[idx0] += beta * u(0, y, tn1)
            end

            if i == grid_size_in
                @assert x + h == 1.0
                rhs[idx0] += beta * u(1.0,y,tn1)
            end

            if j == 1 
                @assert y - h == 0.0
                rhs[idx0] += beta * u(x, 0, tn1)
            end

            if j == grid_size_in 
                @assert y + h == 1.0
                rhs[idx0] += beta * u(x, 1.0,tn1)
            end
        end
    end

    # Use conjugate gradient to solve for u_n+1
    # Set u_n as the inital guess. u_n will be updated in place
    cg!(u_n,L,rhs,Pl = PreConL, reltol = 1e-8)

    # Mesure the relative L2 error with reference solution
    for i in 1 : grid_size_in
        for j in 1 : grid_size_in
            u_ref[get_idx(i,j,grid_size_in)] = u(i * h, j * h, tn1)
        end
    end

    println("relative l2 : ", norm(u_ref - u_n,2)/norm(u_ref,2))
    println("relative l_inf : ", norm(u_ref - u_n,Inf)/norm(u_ref,Inf))

    if frame == 0
        p1 = heatmap(reshape(u_ref,grid_size_in,grid_size_in), aspect_ratio=:equal, c=:viridis, clims=(0, 2), title="Reference")
        p2 = heatmap(reshape(u_n,grid_size_in,grid_size_in), aspect_ratio=:equal,c=:viridis, clims=(0, 2), title="Solution")
        plt = plot(p1, p2, layout=(1,2))
        savefig(plt,"res/$n.png")
        p3 = heatmap(reshape(abs.(u_n - u_ref) ./ abs.(u_ref),grid_size_in,grid_size_in), aspect_ratio=:equal,c=:viridis, clims=(0, 0.01), title="Error")
        plt = plot(p3)
        savefig(plt,"res/erro_$n.png")

        p1 = surface(reshape(u_ref,grid_size_in,grid_size_in), aspect_ratio=:equal, zlims=(0, 2), title="Reference")
        p2 = surface(reshape(u_n,grid_size_in,grid_size_in), aspect_ratio=:equal, zlims=(0, 2), title="Solution")
        plt = plot(p1, p2, layout=(1,2))
        savefig(plt,"res/height_$n.png")
        p3 = surface(reshape(abs.(u_n - u_ref) ./ abs.(u_ref),grid_size_in,grid_size_in), aspect_ratio=:equal, zlims=(0, 0.01), title="Error")
        plt = plot(p3)
        savefig(plt,"res/height_erro_$n.png")
    end
    global frame = (frame + 1) % 100
end





using Pkg
Pkg.add("Plots")
Pkg.add("IterativeSolvers")
using Plots
using LinearAlgebra
using IterativeSolvers
using SparseArrays

nu = 0.001

u(x,y,t) = sin(pi*x)cos(pi*y)*exp(-pi*t) + 1.0
f(x,y,t) = (2 * pi^2 * nu - pi)sin(pi*x)cos(pi*y)*exp(-pi*t)

dt = 0.001
grid_size = 501
h = 1.0/Float64(grid_size-1)

alpha1 = 1 + (2.0 * nu * dt)/(h^2)
alpha2 = 1 - (2.0 * nu * dt)/(h^2)
beta = (nu * dt)/(2.0*h^2) 

# Construct the L matrix(without boundray)
grid_size_in = grid_size - 2
N_in = (grid_size_in)^2

get_idx(i,j,grid_size_in) = (i - 1) * grid_size_in + (j - 1) + 1
get_left_idx(i,j,grid_size_in) = get_idx(i,j,grid_size_in) - grid_size_in
get_right_idx(i,j,grid_size_in) = get_idx(i,j,grid_size_in) + grid_size_in
get_down_idx(i,j,grid_size_in) = get_idx(i,j,grid_size_in) - 1
get_up_idx(i,j,grid_size_in) = get_idx(i,j,grid_size_in) + 1

rows = []
cols = []
vals = Float64[]

for i in 1 : grid_size_in
    for j in 1 : grid_size_in
        idx0 = get_idx(i,j,grid_size_in)
        push!(rows,idx0)
        push!(cols,idx0)
        push!(vals,alpha1)

        idx_left = get_left_idx(i,j,grid_size_in)
        idx_right = get_right_idx(i,j,grid_size_in)
        idx_down = get_down_idx(i,j,grid_size_in)
        idx_up = get_up_idx(i,j,grid_size_in)

        if idx_left >= 1 && idx_left <= N_in
            push!(rows,idx0)
            push!(cols,idx_left)
            push!(vals,-beta)
        end

        if idx_right >= 1 && idx_right <= N_in
            push!(rows,idx0)
            push!(cols,idx_right)
            push!(vals,-beta)
        end

        if idx_down >= 1 && idx_down <= N_in
            push!(rows,idx0)
            push!(cols,idx_down)
            push!(vals,-beta)
        end

        if idx_up >= 1 && idx_up <= N_in
            push!(rows,idx0)
            push!(cols,idx_up)
            push!(vals,-beta)
        end

    end
end

L = sparse(rows, cols, vals, N_in, N_in)
@assert issymmetric(L)

#println("Calculating the condition number of L matrix(Maybe extremely slow)")
#println(cond(Array(L),2))

u_n = zeros(N_in)
rhs = zeros(N_in)

# Set the initial state
for i in 1 : grid_size_in
    for j in 1 : grid_size_in
        u_n[get_idx(i,j,grid_size_in)] = u(i * h, j * h,0)
    end
end

u_ref = zeros(N_in)

# Start simulation
frame = 0
for n in 0 : Int(1.0/dt)
    tn  = dt * n
    tn1 = tn + dt
    println("t : ",tn)
    # Construct the rhs vector
    for i in 1 : grid_size_in
        for j in 1 : grid_size_in
            idx0 = get_idx(i,j,grid_size_in)
            u_n_center = u_n[idx0]
            u_n_left = 0.0
            u_n_right = 0.0
            u_n_up = 0.0
            u_n_down = 0.0

            x = i * h
            y = j * h

            if i == 1 
                @assert x - h == 0.0
                u_n_left = u(0,y,tn) # Use dirichelet boundary condition
            else
                u_n_left = u_n[get_left_idx(i,j,grid_size_in)]#u(x-h,y,tn)#u_n[get_left_idx(i,j,grid_size_in)]
            end

            if i == grid_size_in
                @assert x + h == 1.0
                u_n_right = u(1.0,y,tn) # Use dirichelet boundary condition
            else
                u_n_right = u_n[get_right_idx(i,j,grid_size_in)]#u(x+h,y,tn)#u_n[get_right_idx(i,j,grid_size_in)]
            end

            if j == 1 
                @assert y - h == 0.0
                u_n_down = u(x, 0 ,tn) # Use dirichelet boundary condition
            else
                u_n_down = u_n[get_down_idx(i,j,grid_size_in)]#u(x , y-h , tn)#u_n[get_down_idx(i,j,grid_size_in)]
            end

            if j == grid_size_in 
                @assert y + h == 1.0
                u_n_up = u(x,1.0,tn) # Use dirichelet boundary condition
            else
                u_n_up = u_n[get_up_idx(i,j,grid_size_in)]#u(x , y + h, tn)#u_n[get_up_idx(i,j,grid_size_in)]
            end
            
            global rhs[idx0] = alpha2 * u_n_center + beta * (u_n_left + u_n_right + u_n_down + u_n_up) + dt * 0.5 * (f(x, y, tn) + f(x, y, tn1))

            if i == 1 
                @assert x - h == 0.0
                global rhs[idx0] += beta * u(0, y, tn1)
            end

            if i == grid_size_in
                @assert x + h == 1.0
                global rhs[idx0] += beta * u(1.0,y,tn1)
            end

            if j == 1 
                @assert y - h == 0.0
                global rhs[idx0] += beta * u(x, 0, tn1)
            end

            if j == grid_size_in 
                @assert y + h == 1.0
                global rhs[idx0] += beta * u(x, 1.0,tn1)
            end
        end
    end

    # Use conjugate gradient to solve for u_n+1
    # Set u_n as the inital guess. u_n will be updated in place
    cg!(u_n,L,rhs,reltol = 1e-9)

    # Mesure the relative L2 error with reference solution
    for i in 1 : grid_size_in
        for j in 1 : grid_size_in
            global u_ref[get_idx(i,j,grid_size_in)] = u(i * h, j * h, tn1)
        end
    end

    println("relative l2 : ", norm(u_ref - u_n,2)/norm(u_ref,2))
    println("absolute l2 : ", norm(u_ref - u_n,2))
    println("relative l_inf : ", norm(u_ref - u_n,Inf)/norm(u_ref,Inf))
    println("absolute l_inf : ", norm(u_ref - u_n,Inf))

    if frame == 0
        p1 = heatmap(reshape(u_ref,grid_size_in,grid_size_in), aspect_ratio=:equal, c=:viridis, clims=(0, 2), title="reference")
        p2 = heatmap(reshape(u_n,grid_size_in,grid_size_in), aspect_ratio=:equal,c=:viridis, clims=(0, 2), title="Solution")
        plt = plot(p1, p2, layout=(1,2))
        savefig(plt,"res1/$n.png")
        p3 = heatmap(reshape(abs.(u_n - u_ref) ./ abs.(u_ref),grid_size_in,grid_size_in), aspect_ratio=:equal,c=:viridis, clims=(0, 1), title="Error")
        plt = plot(p3)
        savefig(plt,"res1/erro_$n.png")
    end
    global frame = (frame + 1) % 100
end





using Pkg
#Pkg.add("Plots")
#Pkg.add("LaTeXStrings")
#Pkg.add("IterativeSolvers")
#Pkg.add("Preconditioners")
using Plots
using LinearAlgebra
using IterativeSolvers
using SparseArrays
using LaTeXStrings
using Preconditioners
using Statistics

nu = 1.0

# reference solution (no permeate free-slip)
u(x,y,t) = -sin(x) * cos(y) * exp(-2*t)
v(x,y,t) = cos(x) * sin(y) * exp(-2*t)
p(x,y,t) = 1/4 * (cos(2*x) + cos(2*y)) * exp(-4*t)

function solve_NS2D_projection(grid_size, Tn,T,verbose = false)

    h = pi/(grid_size-1)
    #dt = 0.5 * h
    dt = (T)/Tn

    grid_size_in = grid_size - 2

    global u_n = zeros(grid_size_in,grid_size_in)
    global v_n = zeros(grid_size_in,grid_size_in)

    global u_n_1 = zeros(grid_size_in,grid_size_in) # u^{n-1}
    global v_n_1 = zeros(grid_size_in,grid_size_in) # v^{n-1}
    global u_n_2 = zeros(grid_size_in,grid_size_in) # u^{n+1}
    global v_n_2 = zeros(grid_size_in,grid_size_in) # v^{n+1}

    global u_star = zeros(grid_size_in,grid_size_in) # u^*
    global v_star = zeros(grid_size_in,grid_size_in) # v^*

    global p_n = zeros(grid_size,grid_size)

    global u_ref = zeros(grid_size_in,grid_size_in)
    global v_ref = zeros(grid_size_in,grid_size_in)
    global p_ref = zeros(grid_size,grid_size)

    # Construct the L matrix(without boundary)
    N_in = (grid_size_in)^2

    get_idx(i,j,grid_size_in) = (i - 1) * grid_size_in + (j - 1) + 1

    rows = []
    cols = []
    vals = Float64[]

    alpha1 = 1.0 + (2.0 * nu * dt)/(h^2)
    alpha2 = 1.0 - (2.0 * nu * dt)/(h^2)
    beta = (nu * dt)/(2.0*h^2) 

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
    PreConL = CholeskyPreconditioner(L, 2)

    rows2 = []
    cols2 = []
    vals2 = Float64[]

    for i in 1 : grid_size
        for j in 1 : grid_size
            idx0 = get_idx(i,j,grid_size)

            idx_left = get_idx(i-1,j,grid_size)
            idx_right = get_idx(i+1,j,grid_size)
            idx_down = get_idx(i,j-1,grid_size)
            idx_up = get_idx(i,j+1,grid_size)

            s = 0
            sl = sr = sd = su = 1

            if j == 1 
                idx_down = 0
                su += 1
                s += 1
            end

            if j == grid_size
                idx_up = 0
                sd += 1
                s += 1
            end

            if i == 1
                idx_left = 0
                sr += 1
                s += 1
            end

            if i == grid_size
                idx_right = 0
                sl += 1
                s += 1
            end

            s = 2^(s)

            if idx_left != 0
                push!(rows2,idx0)
                push!(cols2,idx_left)
                push!(vals2,-sl/(s * h^2))
            end

            if idx_right != 0
                push!(rows2,idx0)
                push!(cols2,idx_right)
                push!(vals2,-sr/(s * h^2))
            end

            if idx_down != 0
                push!(rows2,idx0)
                push!(cols2,idx_down)
                push!(vals2,-sd/(s * h^2))
            end

            if idx_up != 0
                push!(rows2,idx0)
                push!(cols2,idx_up)
                push!(vals2,-su/(s * h^2))
            end

            push!(rows2,idx0)
            push!(cols2,idx0)
            push!(vals2,4/(s * h^2))
        end
    end

    Lp = sparse(rows2, cols2, vals2, grid_size * grid_size, grid_size * grid_size)
    @assert issymmetric(Lp)
    # Lp is symmetric but semi-definite
    PreConLp = CholeskyPreconditioner(Lp, 2)

    # Set initial state
    for i in 1 : grid_size_in
        for j in 1 : grid_size_in
            x,y = i*h, j * h 
            u_n_1[i,j] = u(x,y,0)
            v_n_1[i,j] = v(x,y,0)
            u_n[i,j] = u(x,y,dt)
            v_n[i,j] = v(x,y,dt)
        end
    end

    for i in 1 : grid_size
        for j in 1 : grid_size
            x,y = (i-1)*h, (j-1)*h 
            p_n[i,j] = p(x,y,dt*0.5)
        end
    end

    ul2e = 0
    ulinfe = 0
    vl2e = 0
    vlinfe =0
    pl2e =0
    plinfe =0

    # Time marching
    for n in 1 : Tn
        tn = dt * n

        u_rhs = zeros(grid_size_in,grid_size_in)
        v_rhs = zeros(grid_size_in,grid_size_in)

        u_dbc_n(x,y) = u(x,y,tn)
        u_dbc_n_1(x,y) = u(x,y,tn + dt)
        v_dbc_n(x,y) = v(x,y,tn)
        v_dbc_n_1(x,y) = v(x,y,tn + dt)

        for i in 1 : grid_size_in
            for j in 1 : grid_size_in
                x,y = i*h, j *h 
                u_n_center = u_n[i,j]
                u_n_left = u_n_right = u_n_down = u_n_up = 0.0
                u_n_1_left = u_n_1_right = u_n_1_down = u_n_1_up = 0.0

                v_n_center = v_n[i,j]
                v_n_left = v_n_right = v_n_down = v_n_up = 0.0
                v_n_1_left = v_n_1_right = v_n_1_down = v_n_1_up = 0.0

                @inbounds u_rhs[i,j] = 0
                @inbounds v_rhs[i,j] = 0

                if i == 1 
                    u_n_left = u(x - h,y,tn) # Use Dirichelet boundary condition
                    v_n_left = v(x - h,y,tn) 
                    u_n_1_left = u(x - h, y, tn - dt)
                    v_n_1_left = v(x - h, y, tn - dt)
                    @inbounds u_rhs[i,j] += beta * u(x - h, y, tn + dt) 
                    @inbounds v_rhs[i,j] += beta * v(x - h, y, tn + dt)
                else
                    u_n_left = @inbounds u_n[i - 1,j]
                    v_n_left = @inbounds v_n[i - 1,j]
                    u_n_1_left = @inbounds u_n_1[i-1,j]
                    v_n_1_left = @inbounds v_n_1[i-1,j]
                end

                if i == grid_size_in
                    u_n_right = u(x + h,y,tn) # Use Dirichelet boundary condition
                    v_n_right = v(x + h,y,tn)
                    u_n_1_right = u(x + h, y, tn - dt)
                    v_n_1_right = v(x + h, y, tn - dt)
                    @inbounds u_rhs[i,j] += beta * u(x + h, y, tn + dt)
                    @inbounds v_rhs[i,j] += beta * v(x + h, y, tn + dt)
                else
                    u_n_right = @inbounds u_n[i + 1,j]
                    v_n_right = @inbounds v_n[i + 1,j]
                    u_n_1_right = @inbounds u_n_1[i + 1,j]
                    v_n_1_right = @inbounds v_n_1[i + 1,j]
                end

                if j == 1 
                    u_n_down = u(x, y - h, tn) # Use Dirichelet boundary condition
                    v_n_down = v(x, y - h, tn)
                    u_n_1_down = u(x, y - h, tn - dt)
                    v_n_1_down = v(x, y - h, tn - dt)
                    @inbounds u_rhs[i,j] += beta * u(x, y - h, tn + dt)
                    @inbounds v_rhs[i,j] += beta * v(x, y - h, tn + dt)
                else
                    u_n_down = @inbounds u_n[i,j - 1]
                    v_n_down = @inbounds v_n[i,j - 1]
                    u_n_1_down = @inbounds u_n_1[i,j - 1]
                    v_n_1_down = @inbounds v_n_1[i,j - 1]
                end

                if j == grid_size_in 
                    u_n_up = u(x, y + h, tn) # Use Dirichelet boundary condition
                    v_n_up = v(x, y + h, tn) 
                    u_n_1_up = u(x, y + h, tn - dt) 
                    v_n_1_up = v(x, y + h, tn - dt) 
                    @inbounds u_rhs[i,j] += beta * u(x, y + h, tn + dt)
                    @inbounds v_rhs[i,j] += beta * v(x, y + h, tn + dt)
                else
                    u_n_up = @inbounds u_n[i,j + 1]
                    v_n_up = @inbounds v_n[i,j + 1]
                    u_n_1_up = @inbounds u_n_1[i,j + 1]
                    v_n_1_up = @inbounds v_n_1[i,j + 1]
                end

                # diffuse
                @inbounds u_rhs[i,j] += alpha2 * u_n_center + beta *(u_n_left + u_n_right + u_n_down + u_n_up)
                @inbounds v_rhs[i,j] += alpha2 * v_n_center + beta *(v_n_left + v_n_right + v_n_down + v_n_up)

                # convection
                dudx_n_1 = (u_n_1_right - u_n_1_left)/(2*h)#(u(x + h,y,tn - dt) - u(x - h,y,tn - dt))/(2*h)
                dudx_n = (u_n_right - u_n_left)/(2*h)#(u(x + h,y,tn) - u(x - h,y,tn))/(2*h)

                dudy_n_1 = (u_n_1_up - u_n_1_down)/(2*h)#(u(x,y + h,tn - dt) - u(x,y - h,tn - dt))/(2*h)
                dudy_n = (u_n_up - u_n_down)/(2*h)#(u(x,y + h,tn) - u(x,y - h,tn))/(2*h)

                dvdx_n_1 = (v_n_1_right - v_n_1_left)/(2*h)#(v(x + h,y,tn - dt) - v(x - h,y,tn - dt))/(2*h)
                dvdx_n = (v_n_right - v_n_left)/(2*h)#(v(x + h,y,tn) - v(x - h,y,tn))/(2*h)

                dvdy_n_1 = (v_n_1_up - v_n_1_down)/(2*h)#(v(x,y + h,tn - dt) - v(x,y - h,tn - dt))/(2*h)
                dvdy_n = (v_n_up - v_n_down)/(2*h)#(v(x,y + h,tn) - v(x,y - h,tn))/(2*h)
                
                convect_u_n_1 = u_n_1[i,j] * dudx_n_1 + v_n_1[i,j] * dudy_n_1
                convect_u_n = u_n[i,j] * dudx_n + v_n[i,j] * dudy_n

                convect_v_n_1 = u_n_1[i,j] * dvdx_n_1 + v_n_1[i,j] * dvdy_n_1
                convect_v_n = u_n[i,j] * dvdx_n + v_n[i,j] * dvdy_n
               
                @inbounds u_rhs[i,j] -=  dt * (1.5 * convect_u_n - 0.5 * convect_u_n_1)
                @inbounds v_rhs[i,j] -=  dt * (1.5 * convect_v_n - 0.5 * convect_v_n_1)

                # lag pressure gradient
                u_rhs[i,j] -= dt * (p_n[i+2,j+1] - p_n[i,j+1]) / (2*h)
                v_rhs[i,j] -= dt * (p_n[i+1,j+2] - p_n[i+1,j]) / (2*h)
                #u_rhs[i,j] -= dt * (p(x+h,y,tn - 0.5*dt) - p(x-h,y,tn - 0.5 *dt)) / (2*h)
                #v_rhs[i,j] -= dt * (p(x,y + h,tn - 0.5*dt) - p(x,y - h,tn - 0.5*dt)) / (2*h)
            end
        end

        # solve for u_star
        u_star = reshape(cg(L,vec(u_rhs),Pl = PreConL, reltol = 1e-9),grid_size_in,grid_size_in)
        v_star = reshape(cg(L,vec(v_rhs),Pl = PreConL, reltol = 1e-9),grid_size_in,grid_size_in)

        #projection
        #solve for phi
        phi_rhs = zeros(grid_size,grid_size)
        #corners
        x = 0; y = 0
        dudx = (u_dbc_n_1(x + 3*h,y) - u_dbc_n_1(x + h,y))/(2*h) - 2*(u_dbc_n_1(x,y) - 2 * u_dbc_n_1(x + h,y) + u_dbc_n_1(x + 2 * h,y))/h
        dvdy = (v_dbc_n_1(x,y + 3*h) - v_dbc_n_1(x,y + h))/(2*h) - 2*(v_dbc_n_1(x,y) - 2 * v_dbc_n_1(x,y + h) + v_dbc_n_1(x, y + 2 * h))/h
        phi_rhs[1,1] =  -(dudx + dvdy)/4 #divide by four

        x = 0; y = (grid_size_in + 1) * h
        dudx = (u_dbc_n_1(x + 3*h,y) - u_dbc_n_1(x + h,y))/(2*h) - 2*(u_dbc_n_1(x,y) - 2 * u_dbc_n_1(x + h,y) + u_dbc_n_1(x + 2 * h,y))/h
        dvdy = (v_dbc_n_1(x,y-h) - v_dbc_n_1(x,y-3*h))/(2*h) - 2*(v_dbc_n_1(x,y) - 2 * v_dbc_n_1(x,y-h) + v_dbc_n_1(x,y - 2*h))/h
        phi_rhs[1,grid_size] =  -(dudx + dvdy)/4 #divide by four

        x = (grid_size_in + 1) * h; y = 0
        dudx = (u_dbc_n_1(x + 3*h,y) - u_dbc_n_1(x + h, y))/(2*h) - 2*(u_dbc_n_1(x,y) - 2 * u_dbc_n_1(x + h,y) + u_dbc_n_1(x + 2*h,y))/h
        dvdy = (v_dbc_n_1(x,y + 3*h) - v_dbc_n_1(x,y + h))/(2*h) - 2*(v_dbc_n_1(x,y) - 2 * v_dbc_n_1(x,y + h) + v_dbc_n_1(x, y + 2 * h))/h
        phi_rhs[grid_size,1] =  -(dudx + dvdy)/4 #divide by four

        x = (grid_size_in + 1) * h; y = (grid_size_in + 1) * h
        dudx = (u_dbc_n_1(x + 3*h,y) - u_dbc_n_1(x + h, y))/(2*h) - 2*(u_dbc_n_1(x,y) - 2 * u_dbc_n_1(x + h,y) + u_dbc_n_1(x + 2*h,y))/h
        dvdy = (v_dbc_n_1(x,y-h) - v_dbc_n_1(x,y-3*h))/(2*h) - 2*(v_dbc_n_1(x,y) - 2 * v_dbc_n_1(x,y-h) + v_dbc_n_1(x,y - 2*h))/h
        phi_rhs[grid_size,grid_size] = -(dudx + dvdy)/4 #divide by four

        # boundaries
        for i in 2 : grid_size - 1
            x = (i-1) * h; y = 0
            dudx = (u_dbc_n_1(x+h,y) - u_dbc_n_1(x-h,y))/(2*h)
            dvdy = (v_star[i - 1,3] - v_star[i - 1,1])/(2*h) - 2*(v_dbc_n_1(x,y) - 2*v_star[i-1,1] + v_star[i-1,2])/h
            phi_rhs[i,1] = -(dudx + dvdy)/2 #divide by two

            y = (grid_size-1) * h
            dudx = (u_dbc_n_1(x+h,y) - u_dbc_n_1(x-h,y))/(2*h)
            dvdy = (v_star[i - 1,grid_size - 3] - v_star[i-1,grid_size - 5])/(2*h) + 2*(v(x,y,tn + dt) - 2*v_star[i-1,grid_size - 2] + v_star[i-1,grid_size - 3])/h
            phi_rhs[i,grid_size] = -(dudx + dvdy)/2 #divide by two
        end
        for j in 2 : grid_size - 1
            x = 0; y = (j-1) * h
            dvdy = (v_dbc_n_1(x,y+h) - v_dbc_n_1(x,y-h))/(2*h)
            dudx = (u_star[3,j - 1] - u_star[1,j - 1])/(2*h) - 2*(u(x,y,tn + dt) - 2*u_star[1,j - 1] + u_star[2,j - 1])/h
            phi_rhs[1,j] = -(dudx + dvdy)/2 # divide by two

            x = (grid_size-1) * h
            dvdy = (v_dbc_n_1(x,y+h) - v_dbc_n_1(x,y-h))/(2*h)
            dudx = (u_star[grid_size - 3,j - 1] - u_star[grid_size - 5,j - 1])/(2*h) + 2*(u_dbc_n_1(x,y) - 2*u_star[grid_size - 2,j - 1] + u_star[grid_size - 3,j - 1])/h
            phi_rhs[grid_size,j] = -(dudx + dvdy)/2 #divide by two
        end
        # interior points
        for i in 2 : grid_size - 1
            for j in 2 : grid_size - 1
                x,y = (i-1) * h, (j-1) * h
                u_i = i - 1; u_j = j - 1
                u_left = u_right = 0
                v_up = v_down = 0
                if i == 2
                    u_left = u_dbc_n_1(x - h,y)
                else
                    u_left = u_star[u_i - 1,u_j]
                end
                if i == grid_size - 1
                    u_right = u_dbc_n_1(x + h, y)
                else
                    u_right = u_star[u_i + 1, u_j]
                end
                if j == 2
                    v_down = v_dbc_n_1(x,y-h)
                else
                    v_down = v_star[u_i, u_j - 1]
                end
                if j == grid_size - 1
                    v_up = v_dbc_n_1(x,y+h)
                else
                    v_up = v_star[u_i, u_j + 1]
                end

                dudx = (u_right - u_left)/(2*h)
                dvdy = (v_up - v_down)/(2*h)
                phi_rhs[i,j] = -(dudx + dvdy)
            end
        end

        phi_rhs_vec = vec(phi_rhs)
        phi_rhs_mean = mean(phi_rhs_vec)
        phi_rhs_vec = phi_rhs_vec .- phi_rhs_mean
        phi = reshape(cg(Lp,phi_rhs_vec,Pl = PreConLp, reltol = 1e-9),grid_size,grid_size)

        # correct velocity field
        for i in 1 : grid_size_in
            for j in 1 : grid_size_in
                x = i * h; y = j * h
                # k1 = 0.5; k2 = 0.5
                # dpdx_old = (p(x + h,y,tn - k1 * dt) - p(x - h,y,tn - k1 * dt)) / (2*h)
                # dpdx_new = (p(x + h,y,tn + k2 * dt) - p(x - h,y,tn + k2 * dt)) / (2*h)
                # dpdy_old = (p(x,y + h,tn - k1 * dt) - p(x,y - h,tn - k1 * dt)) / (2*h)
                # dpdy_new = (p(x,y + h,tn + k2 * dt) - p(x,y - h,tn + k2 * dt)) / (2*h)
                dphidx = (phi[i+2,j+1] - phi[i,j+1])/(2*h)
                dphidy = (phi[i+1,j+2] - phi[i+1,j])/(2*h)
                u_star[i,j] -= dt * dphidx
                v_star[i,j] -= dt * dphidy
            end
        end

        u_n_2 = u_star
        v_n_2 = v_star

        # update the pressure
        for i in 1 : grid_size
            for j in 1 : grid_size
                phi_left = phi_right = phi_up = phi_down = 0
                if i == 1
                    phi_left = phi[i + 1,j]
                else
                    phi_left = phi[i - 1,j]
                end
                if i == grid_size
                    phi_right = phi[i - 1,j]
                else
                    phi_right = phi[i + 1,j]
                end
                if j == 1
                    phi_down = phi[i, j + 1]
                else
                    phi_down = phi[i, j - 1]
                end
                if j == grid_size
                    phi_up = phi[i, j - 1]
                else
                    phi_up = phi[i, j + 1]
                end

                p_n[i,j] += phi[i,j] - nu * dt/2 * (-4 * phi[i,j] + phi_left + phi_right + phi_up + phi_down)/(h^2)
            end
        end

        # update old velocity
        global u_n_1 = u_n
        global v_n_1 = v_n

        global u_n = u_n_2
        global v_n = v_n_2
        
        for i in 1 : grid_size_in
            for j in 1 : grid_size_in
                x,y = i*h, j *h 
                @inbounds u_ref[i,j] = u(x,y,tn + dt)
                @inbounds v_ref[i,j] = v(x,y,tn + dt)   
            end
        end

        for i in 1 : grid_size
            for j in 1 : grid_size
                x,y = (i-1)*h, (j-1) *h 
                p_ref[i,j] = p(x,y,tn + dt)
            end
        end

        global uerror = u_n - u_ref
        ul2e = norm(uerror,2)/norm(u_ref,2)
        ulinfe = norm(uerror,Inf)/norm(u_ref,Inf)
        global verror = v_n - v_ref
        vl2e = norm(verror,2)/norm(v_ref,2)
        vlinfe = norm(verror,Inf)/norm(v_ref,Inf)
        global perror = p_n - p_ref
        pl2e = norm(perror,2)/norm(p_ref,2)
        plinfe = norm(perror,Inf)/norm(p_ref,Inf)

        if verbose
            println("t : ",tn)
            println("relative l2 for u: ", ul2e)
            println("relative l_inf for u: ", ulinfe)
            println("relative l2 for v: ", vl2e)
            println("relative l_inf for v: ", vlinfe)
            println("relative l2 for p: ", pl2e)
            println("relative l_inf for p ", plinfe)
        end
    end
    
    return uerror, ul2e, ulinfe, verror, vl2e, vlinfe, perror, pl2e, plinfe
end

dts = []
ul2s = []
ul_infs = []
vl2s = []
vl_infs = []
pl2s = []
pl_infs = []

T = 1.0

for Tn in 10 : 100
    dt = Float64(T) / Float64(Tn)
    println("dt : ",dt)
    ue, ul2error, ulinf_error, ve, vl2error, vlinf_error, pe, pl2error, plinf_error = solve_NS2D_projection(101,Tn,T,false)
    println("relative l2 for u: ", ul2error)
    println("relative l_inf for u: ", ulinf_error)
    println("relative l2 for v: ", vl2error)
    println("relative l_inf for v: ", vlinf_error)
    println("relative l2 for p: ", pl2error)
    println("relative l_inf for p ", plinf_error)
    push!(dts,dt)
    push!(ul2s,ul2error)
    push!(ul_infs,ulinf_error)
    push!(vl2s,vl2error)
    push!(vl_infs,vlinf_error)
    push!(pl2s,pl2error)
    push!(pl_infs,plinf_error)

    plt = heatmap(abs.(pe), aspect_ratio=:equal, c=:viridis, title="pointwise error of pressure")
    savefig(plt,"res3/pointwise_error/ppwe$Tn.png")

    plt = plot(dts,ul2s,title = L"dt vs $L_2$ Error", ylabel = L"relative $L_2$ error", xlabel = "dt",label= L"$l_2$ error",legend=:bottomright)
    savefig(plt,"res3/udtvsl2.png")
    plt = plot(dts,ul_infs,title = L"dt vs $L_{\infty}$ Error", ylabel = L"relative $L_{\infty}$ error", xlabel = "dt",label= L"$l_{\infty}$ error",legend=:bottomright)
    savefig(plt,"res3/udtvslinfty.png")

    line(k,x0,y0,x) = 2 .^(log2(y0) + k * (log2(x) - log2(x0)))
    plt = plot(dts,[ul2s,line.(1,dts[end],ul2s[end],dts),line.(2,dts[end],ul2s[end],dts), line.(3,dts[end],ul2s[end],dts)],title = L"dt vs $L_2$ Error log-log", ylabel = L"log(relative $L_2$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_2$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
    savefig(plt,"res3/udtvsl2loglog.png")
    plt = plot(dts,[ul_infs,line.(1,dts[end],ul_infs[end],dts),line.(2,dts[end],ul_infs[end],dts), line.(3,dts[end],ul_infs[end],dts)],title = L"dt vs $L_{\infty}$ Error log-log", ylabel = L"log(relative $L_{\infty}$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_{\infty}$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
    savefig(plt,"res3/udtvslinftyloglog.png")

    plt = plot(dts,vl2s,title = L"dt vs $L_2$ Error", ylabel = L"relative $L_2$ error", xlabel = "dt",label= L"$l_2$ error",legend=:bottomright)
    savefig(plt,"res3/vdtvsl2.png")
    plt = plot(dts,vl_infs,title = L"dt vs $L_{\infty}$ Error", ylabel = L"relative $L_{\infty}$ error", xlabel = "dt",label= L"$l_{\infty}$ error",legend=:bottomright)
    savefig(plt,"res3/vdtvslinfty.png")
    plt = plot(dts,[vl2s,line.(1,dts[end],vl2s[end],dts),line.(2,dts[end],vl2s[end],dts), line.(3,dts[end],vl2s[end],dts)],title = L"dt vs $L_2$ Error log-log", ylabel = L"log(relative $L_2$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_2$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
    savefig(plt,"res3/vdtvsl2loglog.png")
    plt = plot(dts,[vl_infs,line.(1,dts[end],vl_infs[end],dts),line.(2,dts[end],vl_infs[end],dts), line.(3,dts[end],vl_infs[end],dts)],title = L"dt vs $L_{\infty}$ Error log-log", ylabel = L"log(relative $L_{\infty}$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_{\infty}$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
    savefig(plt,"res3/vdtvslinftyloglog.png")

    plt = plot(dts,pl2s,title = L"dt vs $L_2$ Error", ylabel = L"relative $L_2$ error", xlabel = "dt",label= L"$l_2$ error",legend=:bottomright)
    savefig(plt,"res3/pdtvsl2.png")
    plt = plot(dts,pl_infs,title = L"dt vs $L_{\infty}$ Error", ylabel = L"relative $L_{\infty}$ error", xlabel = "dt",label= L"$l_{\infty}$ error",legend=:bottomright)
    savefig(plt,"res3/pdtvslinfty.png")
    plt = plot(dts,[pl2s,line.(1,dts[end],pl2s[end],dts),line.(2,dts[end],pl2s[end],dts), line.(3,dts[end],pl2s[end],dts)],title = L"dt vs $L_2$ Error log-log", ylabel = L"log(relative $L_2$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_2$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
    savefig(plt,"res3/pdtvsl2loglog.png")
    plt = plot(dts,[pl_infs,line.(1,dts[end],pl_infs[end],dts),line.(2,dts[end],pl_infs[end],dts), line.(3,dts[end],pl_infs[end],dts)],title = L"dt vs $L_{\infty}$ Error log-log", ylabel = L"log(relative $L_{\infty}$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_{\infty}$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
    savefig(plt,"res3/pdtvslinftyloglog.png")
end
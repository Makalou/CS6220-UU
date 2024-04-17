using Pkg
Pkg.add("Plots")
Pkg.add("LaTeXStrings")
Pkg.add("IterativeSolvers")
Pkg.add("Preconditioners")
using Plots
using LinearAlgebra
using IterativeSolvers
using SparseArrays
using LaTeXStrings
using Preconditioners

nu = 1.0

# reference solution
u(x,y,t) = -cos(x) * sin(y) * exp(-2*t)
v(x,y,t) = sin(x) * cos(y) * exp(-2*t)
p(x,y,t) = -1/4 * (cos(2*x) + cos(2*y)) * exp(-4*t)

function solve_NS2D(grid_size, Tn,T, max_iter = 10, verbose = false)

    h = 1.0/(grid_size-1)
    #dt = 0.5 * h
    dt = (T)/Tn

    grid_size_in = grid_size - 2

    global u_n = zeros(grid_size_in,grid_size_in)
    global v_n = zeros(grid_size_in,grid_size_in)

    global u_n_1 = zeros(grid_size_in,grid_size_in) # u^{n-1}
    global v_n_1 = zeros(grid_size_in,grid_size_in) # v^{n-1}
    global u_n_2 = zeros(grid_size_in,grid_size_in) # u^{n+1}
    global v_n_2 = zeros(grid_size_in,grid_size_in) # v^{n+1}

    global p_n = zeros(grid_size_in,grid_size_in)
    global p_n_2 = zeros(grid_size_in,grid_size_in)

    global u_ref = zeros(grid_size_in,grid_size_in)
    global v_ref = zeros(grid_size_in,grid_size_in)
    global p_ref = zeros(grid_size_in,grid_size_in)

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

    for i in 1 : grid_size_in
        for j in 1 : grid_size_in
            idx0 = get_idx(i,j,grid_size_in)
            push!(rows2,idx0)
            push!(cols2,idx0)
            push!(vals2,4/(h^2))

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
                push!(rows2,idx0)
                push!(cols2,idx_left)
                push!(vals2,-1/(h^2))
            end

            if idx_right != 0
                push!(rows2,idx0)
                push!(cols2,idx_right)
                push!(vals2,-1/(h^2))
            end

            if idx_down != 0
                push!(rows2,idx0)
                push!(cols2,idx_down)
                push!(vals2,-1/(h^2))
            end

            if idx_up != 0
                push!(rows2,idx0)
                push!(cols2,idx_up)
                push!(vals2,-1/(h^2))
            end
        end
    end

    Lp = sparse(rows2, cols2, vals2, N_in, N_in)
    @assert issymmetric(Lp)
    @assert isposdef(Lp)
    PreConLp = CholeskyPreconditioner(Lp, 2)

    # Set initial state
    for i in 1 : grid_size_in
        for j in 1 : grid_size_in
            x,y = i*h, j *h 
            u_n_1[i,j] = u(x,y,0)
            v_n_1[i,j] = v(x,y,0)
            u_n[i,j] = u(x,y,dt)
            v_n[i,j] = v(x,y,dt)
            p_n[i,j] = p(x,y,dt)
            p_n_2[i,j] = p(x,y,dt)
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
            end
        end

        u_rhs1 = zeros(grid_size_in,grid_size_in)
        v_rhs1 = zeros(grid_size_in,grid_size_in)
        p_rhs = zeros(grid_size_in,grid_size_in)
        div = zeros(grid_size_in,grid_size_in)
        for it in 1 : max_iter
            if verbose
                println("iteration : ",it)
            end
            for i in 1 : grid_size_in
                for j in 1 : grid_size_in
                    x,y = i*h, j *h 
                    p_n_left = p_n_right = p_n_down = p_n_up = 0.0
                    p_n_2_left = p_n_2_right = p_n_2_down = p_n_2_up = 0.0

                    if i == 1
                        p_n_left = p(x - h,y,tn)
                        p_n_2_left = p(x - h,y,tn+dt)
                    else
                        p_n_left = p_n[i - 1,j]
                        p_n_2_left = p_n_2[i - 1,j]
                    end

                    if i == grid_size_in
                        p_n_right = p(x + h,y,tn)
                        p_n_2_right = p(x + h,y,tn+dt)
                    else
                        p_n_right = p_n[i + 1,j]
                        p_n_2_right = p_n_2[i + 1,j]
                    end

                    if j == 1
                        p_n_down = p(x, y - h, tn)
                        p_n_2_down = p(x, y - h, tn+dt)
                    else
                        p_n_down = p_n[i, j - 1]
                        p_n_2_down = p_n_2[i, j - 1]
                    end

                    if j == grid_size_in
                        p_n_up = p(x, y + h, tn)
                        p_n_2_up = p(x, y + h, tn+dt)
                    else
                        p_n_up = p_n[i, j + 1]
                        p_n_2_up = p_n_2[i, j + 1]
                    end

                    p_n_half_left = (p_n_left + p_n_2_left)/2
                    p_n_half_right = (p_n_right + p_n_2_right)/2
                    p_n_half_down = (p_n_down + p_n_2_down)/2
                    p_n_half_up = (p_n_up + p_n_2_up)/2

                    dpdx_n_half = (p_n_half_right - p_n_half_left)/(2*h)
                    dpdy_n_half = (p_n_half_up - p_n_half_down)/(2*h)

                    @inbounds u_rhs1[i,j] = u_rhs[i,j] - dt * dpdx_n_half
                    @inbounds v_rhs1[i,j] = v_rhs[i,j] - dt * dpdy_n_half
                end
            end
            #solve velocity with updated pressure
            global u_n_2 = reshape(cg(L,vec(u_rhs1),Pl = PreConL, reltol = 1e-9),grid_size_in,grid_size_in)
            global v_n_2 = reshape(cg(L,vec(v_rhs1),Pl = PreConL, reltol = 1e-9),grid_size_in,grid_size_in)
                    
            # use updated velocity to build pressure poisson equation
            for i in 1 : grid_size_in
                for j in 1 : grid_size_in
                    x,y = i*h, j *h 
                    u_left = u_right = u_down = u_up = 0.0
                    v_left = v_right = v_down = v_up = 0.0
                    p_rhs[i,j] = 0
                    if i == 1 
                        u_left = u(x - h,y,tn + dt) # Use Dirichelet boundary condition
                        v_left = v(x - h,y,tn + dt) 
                        p_rhs[i,j] += p(x - h, y , tn + dt)/(h^2)
                    else
                        u_left = u_n_2[i - 1,j]
                        v_left = v_n_2[i - 1,j]
                    end
    
                    if i == grid_size_in
                        u_right = u(x + h,y,tn + dt) # Use Dirichelet boundary condition
                        v_right = v(x + h,y,tn + dt)
                        p_rhs[i,j] += p(x + h, y , tn + dt)/(h^2)
                    else
                        u_right = u_n_2[i + 1,j]
                        v_right = v_n_2[i + 1,j]
                    end
    
                    if j == 1 
                        u_down = u(x, y - h, tn + dt) # Use Dirichelet boundary condition
                        v_down = v(x, y - h, tn + dt)
                        p_rhs[i,j] += p(x, y - h, tn + dt)/(h^2)
                    else
                        u_down = u_n_2[i,j - 1]
                        v_down = v_n_2[i,j - 1]
                    end
    
                    if j == grid_size_in 
                        u_up = u(x, y + h, tn + dt) # Use Dirichelet boundary condition
                        v_up = v(x, y + h, tn + dt) 
                        p_rhs[i,j] += p(x, y + h, tn + dt)/(h^2)
                    else
                        u_up = u_n_2[i,j + 1]
                        v_up = v_n_2[i,j + 1]
                    end

                    dudx = (u_right - u_left)/(2*h)
                    dudy = (u_up - u_down)/(2*h)
                    dvdx = (v_right - v_left)/(2*h)
                    dvdy = (v_up - v_down)/(2*h)
                    p_rhs[i,j] += 2.0 * (dvdx * dudy - dudx * dvdy)
                    div[i,j] = dudx + dvdy;
                end
            end
            if verbose
                println("div : ",norm(vec(div),Inf))
            end
            #solve pressure with updated velocity
            global p_n_2 = reshape(cg(Lp,vec(p_rhs),Pl = PreConLp, reltol = 1e-9),grid_size_in,grid_size_in)
        end

        # plt = surface(div, aspect_ratio=:equal, title="Divergence")
        # display(plt)

        global u_n_1 = u_n
        global v_n_1 = v_n

        global u_n = u_n_2
        global v_n = v_n_2

        global p_n = p_n_2
        
        for i in 1 : grid_size_in
            for j in 1 : grid_size_in
                x,y = i*h, j *h 
                @inbounds u_ref[i,j] = u(x,y,tn + dt)
                @inbounds v_ref[i,j] = v(x,y,tn + dt)
                @inbounds p_ref[i,j] = p(x,y,tn + dt)
            end
        end

        uerror = vec(u_n) - vec(u_ref)
        ul2e = norm(uerror,2)/norm(u_ref,2)
        ulinfe = norm(uerror,Inf)/norm(u_ref,Inf)
        verror = vec(v_n) - vec(v_ref)
        vl2e = norm(verror,2)/norm(v_ref,2)
        vlinfe = norm(verror,Inf)/norm(v_ref,Inf)
        perror = vec(p_n) - vec(p_ref)
        pl2e = norm(perror,2)/norm(p_ref,2)
        plinfe = norm(perror,Inf)/norm(p_ref,Inf)

        if verbose
            println("t : ",tn)
            println("relative l2 for u: ", ul2e)
            println("relative l_inf for u: ", ulinfe)
            println("relative l2 for v: ", vl2e)
            println("relative l_inf for v: ", vlinfe)
            println("relative l2 for p: ", pl2e)
            println("relative l_inf for vp ", plinfe)
        end
    end

    return ul2e, ulinfe, vl2e, vlinfe, pl2e, plinfe
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
    ul2error, ulinf_error, vl2error, vlinf_error, pl2error, plinf_error = solve_NS2D(101,Tn,T,10,false)
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

    if true
        plt = plot(dts,ul2s,title = L"dt vs $L_2$ Error", ylabel = L"relative $L_2$ error", xlabel = "dt",label= L"$l_2$ error",legend=:bottomright)
        savefig(plt,"res2/udtvsl2.png")
        plt = plot(dts,ul_infs,title = L"dt vs $L_{\infty}$ Error", ylabel = L"relative $L_{\infty}$ error", xlabel = "dt",label= L"$l_{\infty}$ error",legend=:bottomright)
        savefig(plt,"res2/udtvslinfty.png")
        plt = plot(dts,[ul2s,(1e-1 * dts),(1e-1 * dts) .^ 2, (1e-1 * dts) .^ 3],title = L"dt vs $L_2$ Error log-log", ylabel = L"log(relative $L_2$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_2$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
        savefig(plt,"res2/udtvsl2loglog.png")
        plt = plot(dts,[ul_infs,(1e-1 * dts),(1e-1 * dts) .^ 2, (1e-1 * dts) .^ 3],title = L"dt vs $L_{\infty}$ Error log-log", ylabel = L"log(relative $L_{\infty}$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_{\infty}$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
        savefig(plt,"res2/udtvslinftyloglog.png")

        plt = plot(dts,vl2s,title = L"dt vs $L_2$ Error", ylabel = L"relative $L_2$ error", xlabel = "dt",label= L"$l_2$ error",legend=:bottomright)
        savefig(plt,"res2/vdtvsl2.png")
        plt = plot(dts,vl_infs,title = L"dt vs $L_{\infty}$ Error", ylabel = L"relative $L_{\infty}$ error", xlabel = "dt",label= L"$l_{\infty}$ error",legend=:bottomright)
        savefig(plt,"res2/vdtvslinfty.png")
        plt = plot(dts,[vl2s,(1e-1 * dts),(1e-1 * dts) .^ 2, (1e-1 * dts) .^ 3],title = L"dt vs $L_2$ Error log-log", ylabel = L"log(relative $L_2$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_2$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
        savefig(plt,"res2/vdtvsl2loglog.png")
        plt = plot(dts,[vl_infs,(1e-1 * dts),(1e-1 * dts) .^ 2, (1e-1 * dts) .^ 3],title = L"dt vs $L_{\infty}$ Error log-log", ylabel = L"log(relative $L_{\infty}$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_{\infty}$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
        savefig(plt,"res2/vdtvslinftyloglog.png")

        plt = plot(dts,pl2s,title = L"dt vs $L_2$ Error", ylabel = L"relative $L_2$ error", xlabel = "dt",label= L"$l_2$ error",legend=:bottomright)
        savefig(plt,"res2/pdtvsl2.png")
        plt = plot(dts,pl_infs,title = L"dt vs $L_{\infty}$ Error", ylabel = L"relative $L_{\infty}$ error", xlabel = "dt",label= L"$l_{\infty}$ error",legend=:bottomright)
        savefig(plt,"res2/pdtvslinfty.png")
        plt = plot(dts,[pl2s,(1e-1 * dts),(1e-1 * dts) .^ 2, (1e-1 * dts) .^ 3],title = L"dt vs $L_2$ Error log-log", ylabel = L"log(relative $L_2$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_2$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
        savefig(plt,"res2/pdtvsl2loglog.png")
        plt = plot(dts,[pl_infs,(1e-1 * dts),(1e-1 * dts) .^ 2, (1e-1 * dts) .^ 3],title = L"dt vs $L_{\infty}$ Error log-log", ylabel = L"log(relative $L_{\infty}$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_{\infty}$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
        savefig(plt,"res2/pdtvslinftyloglog.png")
    end
end
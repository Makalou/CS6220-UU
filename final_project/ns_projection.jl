using Pkg
#Pkg.add("Plots")
#Pkg.add("LaTeXStrings")
#Pkg.add("IterativeSolvers")
#Pkg.add("Preconditioners")
#Pkg.add("IncompleteLU")
using Plots
using LinearAlgebra
using IterativeSolvers
using SparseArrays
using LaTeXStrings
using Preconditioners
using BenchmarkTools
using IncompleteLU
include("utilities.jl")
using .MyUtil

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
    p_grid_size = grid_size - 1

    global u_n = zeros(grid_size_in,grid_size_in)
    global v_n = zeros(grid_size_in,grid_size_in)

    global u_n_1 = zeros(grid_size_in,grid_size_in) # u^{n-1}
    global v_n_1 = zeros(grid_size_in,grid_size_in) # v^{n-1}
    global u_n_2 = zeros(grid_size_in,grid_size_in) # u^{n+1}
    global v_n_2 = zeros(grid_size_in,grid_size_in) # v^{n+1}

    global u_star = zeros(grid_size_in,grid_size_in) # u^*
    global v_star = zeros(grid_size_in,grid_size_in) # v^*

    global p_n = zeros(p_grid_size,p_grid_size)

    global u_ref = zeros(grid_size_in,grid_size_in)
    global v_ref = zeros(grid_size_in,grid_size_in)
    global p_ref = zeros(p_grid_size,p_grid_size)

    # Construct the L matrix(without boundary)

    alpha1 = 1.0 + (2.0 * nu * dt)/(h^2)
    alpha2 = 1.0 - (2.0 * nu * dt)/(h^2)
    beta = (nu * dt)/(2.0*h^2) 

    L = Laplacian(grid_size_in,alpha1,-beta,Dirichlet)
    @assert issymmetric(L)
    @assert isposdef(L)
    PreConL = CholeskyPreconditioner(L, 2)

    Lp, Sp = Laplacian(p_grid_size,4/(h^2),-1/(h^2),Neumann)
    #display(spy(Lp))
    #readline()
    Lp = Sp * Lp
    Lp = [Lp ones(p_grid_size*p_grid_size); ones(p_grid_size * p_grid_size)' 0]
    @assert issymmetric(Lp)
    #print(eigen(Matrix(Lp)).values)
    #display(cond(Matrix(Lp)))
    PreConLp = ilu(Lp,τ = 0.1)

    # Set initial state
    u_n_1 = [u(i*h,j*h,0) for i in 1 : grid_size_in, j in 1 : grid_size_in]
    v_n_1 = [v(i*h,j*h,0) for i in 1 : grid_size_in, j in 1 : grid_size_in]
    u_n = [u(i*h,j*h,dt) for i in 1 : grid_size_in, j in 1 : grid_size_in]
    v_n = [v(i*h,j*h,dt) for i in 1 : grid_size_in, j in 1 : grid_size_in]

    p_n = [p(i * h - 0.5 * h, j * h - 0.5 * h ,dt) for i in 1 : p_grid_size, j in 1 : p_grid_size]

    ul2e = ulinfe = vl2e = vlinfe = pl2e = plinfe =0

    # Time marching
    for n in 1 : Tn
        tn = dt * n

        u_rhs = zeros(grid_size_in,grid_size_in)
        v_rhs = zeros(grid_size_in,grid_size_in)

        u_dbc_n(x,y) = u(x,y,tn)
        u_dbc_n_1(x,y) = u(x,y,tn + dt)
        v_dbc_n(x,y) = v(x,y,tn)
        v_dbc_n_1(x,y) = v(x,y,tn + dt)

        # Padding Dirichlet Boundary
        x1 = 0; x2 = (grid_size - 1) * h; y1 = 0; y2 = (grid_size - 1) * h
        u_n_1_full = [[u(x1,j*h,tn - dt) for j in 0 : grid_size - 1]';  [u(i*h,y1,tn - dt) for i in 1 : grid_size - 2] u_n_1 [u(i*h,y2,tn - dt) for i in 1 : grid_size - 2]; [u(x2,j*h,tn - dt) for j in 0 : grid_size - 1]']
        v_n_1_full = [[v(x1,j*h,tn - dt) for j in 0 : grid_size - 1]';  [v(i*h,y1,tn - dt) for i in 1 : grid_size - 2] v_n_1 [v(i*h,y2,tn - dt) for i in 1 : grid_size - 2]; [v(x2,j*h,tn - dt) for j in 0 : grid_size - 1]']
        u_n_full = [[u(x1,j*h,tn) for j in 0 : grid_size - 1]';  [u(i*h,y1,tn) for i in 1 : grid_size - 2] u_n [u(i*h,y2,tn) for i in 1 : grid_size - 2]; [u(x2,j*h,tn) for j in 0 : grid_size - 1]']
        v_n_full = [[v(x1,j*h,tn) for j in 0 : grid_size - 1]';  [v(i*h,y1,tn) for i in 1 : grid_size - 2] v_n [v(i*h,y2,tn) for i in 1 : grid_size - 2]; [v(x2,j*h,tn) for j in 0 : grid_size - 1]']

        # construct rhs for velocity
        for i in 2 : grid_size - 1
            for j in 2 : grid_size - 1
                x = (i-1) * h; y = (j-1) * h
                if i == 2
                    u_rhs[i - 1,j - 1] += beta * u(x - h, y, tn + dt) 
                    v_rhs[i - 1,j - 1] += beta * v(x - h, y, tn + dt)
                end

                if i == grid_size - 1
                    u_rhs[i - 1,j - 1] += beta * u(x + h, y, tn + dt)
                    v_rhs[i - 1,j - 1] += beta * v(x + h, y, tn + dt)
                end

                if j == 2
                    u_rhs[i - 1,j - 1] += beta * u(x, y - h, tn + dt)
                    v_rhs[i - 1,j - 1] += beta * v(x, y - h, tn + dt)
                end

                if j == grid_size - 1
                    u_rhs[i - 1,j - 1] += beta * u(x, y + h, tn + dt)
                    v_rhs[i - 1,j - 1] += beta * v(x, y + h, tn + dt)
                end

                u_n_center = u_n_full[i,j]; u_n_left = u_n_full[i-1,j];u_n_right = u_n_full[i+1,j];u_n_down = u_n_full[i,j-1];u_n_up = u_n_full[i,j+1]
                u_n_1_left = u_n_1_full[i-1,j]; u_n_1_right = u_n_1_full[i+1,j]; u_n_1_down = u_n_1_full[i,j-1]; u_n_1_up = u_n_1_full[i,j+1]

                v_n_center = v_n_full[i,j]; v_n_left = v_n_full[i-1,j]; v_n_right = v_n_full[i+1,j]; v_n_down = v_n_full[i,j-1]; v_n_up = v_n_full[i,j+1]
                v_n_1_left = v_n_1_full[i-1,j]; v_n_1_right = v_n_1_full[i+1,j]; v_n_1_down = v_n_1_full[i,j-1]; v_n_1_up = v_n_1_full[i,j+1]
                # diffuse
                @inbounds u_rhs[i - 1,j - 1] += alpha2 * u_n_center + beta *(u_n_left + u_n_right + u_n_down + u_n_up)
                @inbounds v_rhs[i - 1,j - 1] += alpha2 * v_n_center + beta *(v_n_left + v_n_right + v_n_down + v_n_up)

                # convection
                dudx_n_1 = (u_n_1_right - u_n_1_left)/(2*h)#(u(x + h,y,tn - dt) - u(x - h,y,tn - dt))/(2*h)
                dudx_n = (u_n_right - u_n_left)/(2*h)#(u(x + h,y,tn) - u(x - h,y,tn))/(2*h)

                dudy_n_1 = (u_n_1_up - u_n_1_down)/(2*h)#(u(x,y + h,tn - dt) - u(x,y - h,tn - dt))/(2*h)
                dudy_n = (u_n_up - u_n_down)/(2*h)#(u(x,y + h,tn) - u(x,y - h,tn))/(2*h)

                dvdx_n_1 = (v_n_1_right - v_n_1_left)/(2*h)#(v(x + h,y,tn - dt) - v(x - h,y,tn - dt))/(2*h)
                dvdx_n = (v_n_right - v_n_left)/(2*h)#(v(x + h,y,tn) - v(x - h,y,tn))/(2*h)

                dvdy_n_1 = (v_n_1_up - v_n_1_down)/(2*h)#(v(x,y + h,tn - dt) - v(x,y - h,tn - dt))/(2*h)
                dvdy_n = (v_n_up - v_n_down)/(2*h)#(v(x,y + h,tn) - v(x,y - h,tn))/(2*h)
                
                convect_u_n_1 = u_n_1[i - 1,j - 1] * dudx_n_1 + v_n_1[i - 1,j - 1] * dudy_n_1
                convect_u_n = u_n[i - 1,j - 1] * dudx_n + v_n[i - 1,j - 1] * dudy_n

                convect_v_n_1 = u_n_1[i - 1,j - 1] * dvdx_n_1 + v_n_1[i - 1,j - 1] * dvdy_n_1
                convect_v_n = u_n[i - 1,j - 1] * dvdx_n + v_n[i - 1,j - 1] * dvdy_n
               
                @inbounds u_rhs[i - 1,j - 1] -=  dt * (1.5 * convect_u_n - 0.5 * convect_u_n_1)
                @inbounds v_rhs[i - 1,j - 1] -=  dt * (1.5 * convect_v_n - 0.5 * convect_v_n_1)


                p_left = (p_n[i - 1,j - 1] + p_n[i - 1,j])/2
                p_right = (p_n[i,j - 1] + p_n[i,j])/2
                p_down = (p_n[i - 1,j - 1] + p_n[i,j - 1])/2
                p_up = (p_n[i - 1,j] + p_n[i,j])/2

                dpdx = (p_right - p_left) / h
                dpdy = (p_up - p_down) / h
                u_rhs[i - 1,j - 1] -= dt * dpdx
                v_rhs[i - 1,j - 1] -= dt * dpdy
            end
        end
        # solve for u_star
        u_star = reshape(cg(L,vec(u_rhs),Pl = PreConL, reltol = 1e-9),grid_size_in,grid_size_in)
        v_star = reshape(cg(L,vec(v_rhs),Pl = PreConL, reltol = 1e-9),grid_size_in,grid_size_in)

        u_star_full = [[u(x1,j*h,tn + dt) for j in 0 : grid_size - 1]';  [u(i*h,y1,tn + dt) for i in 1 : grid_size - 2] u_star [u(i*h,y2,tn + dt) for i in 1 : grid_size - 2]; [u(x2,j*h,tn + dt) for j in 0 : grid_size - 1]']
        v_star_full = [[v(x1,j*h,tn + dt) for j in 0 : grid_size - 1]';  [v(i*h,y1,tn + dt) for i in 1 : grid_size - 2] v_star [v(i*h,y2,tn + dt) for i in 1 : grid_size - 2]; [v(x2,j*h,tn + dt) for j in 0 : grid_size - 1]']

        #projection

        #construct rhs for phi
        phi_rhs = zeros(p_grid_size,p_grid_size)

        for i in 1 : p_grid_size
            for j in 1 : p_grid_size
                u_star_left = (u_star_full[i,j] + u_star_full[i,j+1])/2
                u_star_right = (u_star_full[i+1,j] + u_star_full[i+1,j+1])/2
                v_star_down = (v_star_full[i,j] + v_star_full[i+1,j])/2
                v_star_up = (v_star_full[i,j+1] + v_star_full[i+1,j+1])/2

                dudx = (u_star_right - u_star_left) / h
                dvdy = (v_star_up - v_star_down) / h
                phi_rhs[i,j] = -(dudx + dvdy) / dt
            end
        end

        #solve for phi
        phi_rhs_vec = Sp * vec(phi_rhs)
        phi = cg(Lp,[phi_rhs_vec; 0],Pl = PreConLp, reltol = 1e-9)
        #lambda = phi[end]
        #println("λ : $lambda")
        phi = reshape(phi[1 : end - 1],p_grid_size,p_grid_size)

        # correct velocity field
        for i in 1 : grid_size_in
            for j in 1 : grid_size_in
                phi_left = (phi[i,j] + phi[i,j+1])/2
                phi_right = (phi[i+1,j] + phi[i+1,j+1])/2

                phi_down = (phi[i,j] + phi[i+1,j])/2
                phi_up = (phi[i,j+1] + phi[i+1,j+1])/2

                dphidx = (phi_right - phi_left)/h
                dphidy = (phi_up - phi_down)/h
                
                u_star[i,j] -= dt * dphidx
                v_star[i,j] -= dt * dphidy
            end
        end

        u_n_2 = u_star
        v_n_2 = v_star

        # update the pressure
        for i in 1 : p_grid_size
            for j in 1 : p_grid_size
                phi_left = (i == 1) ? phi[i + 1,j] : phi[i - 1,j]
                phi_right = (i == p_grid_size) ? phi[i - 1,j] : phi[i + 1,j]
                phi_down = (j == 1) ? phi[i, j + 1] : phi[i, j - 1]
                phi_up = (j == p_grid_size) ? phi[i, j - 1] : phi[i, j + 1]
                p_n[i,j] += phi[i,j] - nu * dt/2 * (-4 * phi[i,j] + phi_left + phi_right + phi_up + phi_down)/(h^2)
            end
        end

        # update old velocity
        global u_n_1 = u_n
        global v_n_1 = v_n

        global u_n = u_n_2
        global v_n = v_n_2

        u_ref = [u(i*h,j*h,tn + dt) for i in 1 : grid_size_in, j in 1 : grid_size_in]
        v_ref = [v(i*h,j*h,tn + dt) for i in 1 : grid_size_in, j in 1 : grid_size_in]
        p_ref = [p(i * h - 0.5 * h, j * h - 0.5 * h,tn + dt) for i in 1 : p_grid_size, j in 1 : p_grid_size]

        global uerror = u_n - u_ref
        plt = surface(abs.(uerror), aspect_ratio=:equal, c=:viridis, title="pointwise error of u")
        savefig(plt,"res4/pointwise_error/upwe$Tn.png")
        ul2e = norm(uerror,2)/norm(u_ref,2)
        ulinfe = norm(uerror,Inf)/norm(u_ref,Inf)
        global verror = v_n - v_ref
        plt = surface(abs.(verror), aspect_ratio=:equal, c=:viridis, title="pointwise error of v")
        savefig(plt,"res4/pointwise_error/vpwe$Tn.png")
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
@time begin
for Tn in 10 : 100
    dt = Float64(T) / Float64(Tn)
    println("dt : ",dt)
    @time begin
    ue, ul2error, ulinf_error, ve, vl2error, vlinf_error, pe, pl2error, plinf_error = solve_NS2D_projection(201,Tn,T,false)
    end
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

    plt = surface(abs.(pe), aspect_ratio=:equal, c=:viridis, title="pointwise error of pressure")
    savefig(plt,"res4/pointwise_error/ppwe$Tn.png")

    plt = plot(dts,ul2s,title = L"dt vs $L_2$ Error", ylabel = L"relative $L_2$ error", xlabel = "dt",label= L"$l_2$ error",legend=:bottomright)
    savefig(plt,"res4/udtvsl2.png")
    plt = plot(dts,ul_infs,title = L"dt vs $L_{\infty}$ Error", ylabel = L"relative $L_{\infty}$ error", xlabel = "dt",label= L"$l_{\infty}$ error",legend=:bottomright)
    savefig(plt,"res4/udtvslinfty.png")

    line(k,x0,y0,x) = 2 .^(log2(y0) + k * (log2(x) - log2(x0)))
    plt = plot(dts,[ul2s,line.(1,dts[end],ul2s[end],dts),line.(2,dts[end],ul2s[end],dts), line.(3,dts[end],ul2s[end],dts)],title = L"dt vs $L_2$ Error log-log", ylabel = L"log(relative $L_2$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_2$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
    savefig(plt,"res4/udtvsl2loglog.png")
    plt = plot(dts,[ul_infs,line.(1,dts[end],ul_infs[end],dts),line.(2,dts[end],ul_infs[end],dts), line.(3,dts[end],ul_infs[end],dts)],title = L"dt vs $L_{\infty}$ Error log-log", ylabel = L"log(relative $L_{\infty}$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_{\infty}$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
    savefig(plt,"res4/udtvslinftyloglog.png")

    plt = plot(dts,vl2s,title = L"dt vs $L_2$ Error", ylabel = L"relative $L_2$ error", xlabel = "dt",label= L"$l_2$ error",legend=:bottomright)
    savefig(plt,"res4/vdtvsl2.png")
    plt = plot(dts,vl_infs,title = L"dt vs $L_{\infty}$ Error", ylabel = L"relative $L_{\infty}$ error", xlabel = "dt",label= L"$l_{\infty}$ error",legend=:bottomright)
    savefig(plt,"res4/vdtvslinfty.png")
    plt = plot(dts,[vl2s,line.(1,dts[end],vl2s[end],dts),line.(2,dts[end],vl2s[end],dts), line.(3,dts[end],vl2s[end],dts)],title = L"dt vs $L_2$ Error log-log", ylabel = L"log(relative $L_2$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_2$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
    savefig(plt,"res4/vdtvsl2loglog.png")
    plt = plot(dts,[vl_infs,line.(1,dts[end],vl_infs[end],dts),line.(2,dts[end],vl_infs[end],dts), line.(3,dts[end],vl_infs[end],dts)],title = L"dt vs $L_{\infty}$ Error log-log", ylabel = L"log(relative $L_{\infty}$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_{\infty}$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
    savefig(plt,"res4/vdtvslinftyloglog.png")

    plt = plot(dts,pl2s,title = L"dt vs $L_2$ Error", ylabel = L"relative $L_2$ error", xlabel = "dt",label= L"$l_2$ error",legend=:bottomright)
    savefig(plt,"res4/pdtvsl2.png")
    plt = plot(dts,pl_infs,title = L"dt vs $L_{\infty}$ Error", ylabel = L"relative $L_{\infty}$ error", xlabel = "dt",label= L"$l_{\infty}$ error",legend=:bottomright)
    savefig(plt,"res4/pdtvslinfty.png")
    plt = plot(dts,[pl2s,line.(1,dts[end],pl2s[end],dts),line.(2,dts[end],pl2s[end],dts), line.(3,dts[end],pl2s[end],dts)],title = L"dt vs $L_2$ Error log-log", ylabel = L"log(relative $L_2$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_2$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
    savefig(plt,"res4/pdtvsl2loglog.png")
    plt = plot(dts,[pl_infs,line.(1,dts[end],pl_infs[end],dts),line.(2,dts[end],pl_infs[end],dts), line.(3,dts[end],pl_infs[end],dts)],title = L"dt vs $L_{\infty}$ Error log-log", ylabel = L"log(relative $L_{\infty}$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_{\infty}$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
    savefig(plt,"res4/pdtvslinftyloglog.png")
end
end 
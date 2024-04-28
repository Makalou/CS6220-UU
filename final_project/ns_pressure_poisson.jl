using Pkg
using Plots
using LinearAlgebra
using IterativeSolvers
using SparseArrays
using LaTeXStrings
using Preconditioners
using Statistics
include("utilities.jl")
using .MyUtil

nu = 1.0

# reference solution (no permeate free-slip)
u(x,y,t) = -sin(x) * cos(y) * exp(-2*t)
v(x,y,t) = cos(x) * sin(y) * exp(-2*t)
p(x,y,t) = 1/4 * (cos(2*x) + cos(2*y)) * exp(-4*t)

function solve_NS2D(grid_size, Tn,T, max_iter = 10, verbose = false)

    h = pi/(grid_size-1)
    #dt = 0.5 * h
    dt = (T)/Tn

    grid_size_in = grid_size - 2
    p_grid_size = grid_size

    global u_n = zeros(grid_size_in,grid_size_in)
    global v_n = zeros(grid_size_in,grid_size_in)

    global u_n_1 = zeros(grid_size_in,grid_size_in) # u^{n-1}
    global v_n_1 = zeros(grid_size_in,grid_size_in) # v^{n-1}
    global u_n_2 = zeros(grid_size_in,grid_size_in) # u^{n+1}
    global v_n_2 = zeros(grid_size_in,grid_size_in) # v^{n+1}

    global p_n = zeros(grid_size,grid_size)
    global p_n_2 = zeros(grid_size,grid_size)

    global u_ref = zeros(grid_size_in,grid_size_in)
    global v_ref = zeros(grid_size_in,grid_size_in)
    global p_ref = zeros(grid_size,grid_size)

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

    # Set initial state
    u_n_1 = [u(i*h,j*h,0) for i in 1 : grid_size_in, j in 1 : grid_size_in]
    v_n_1 = [v(i*h,j*h,0) for i in 1 : grid_size_in, j in 1 : grid_size_in]
    u_n = [u(i*h,j*h,dt) for i in 1 : grid_size_in, j in 1 : grid_size_in]
    v_n = [v(i*h,j*h,dt) for i in 1 : grid_size_in, j in 1 : grid_size_in]
    p_n = [p((i-1)*h,(j-1)*h,0) for i in 1 : grid_size, j in 1 : grid_size]
    p_n_2 = [p((i-1)*h,(j-1)*h,dt) for i in 1 : grid_size, j in 1 : grid_size]

    ul2e = ulinfe = vl2e = vlinfe = pl2e = plinfe =0
    # Time marching
    for n in 1 : Tn
        tn = dt * n
        u_rhs = zeros(grid_size_in,grid_size_in)
        v_rhs = zeros(grid_size_in,grid_size_in)

        x1 = 0; x2 = (grid_size - 1) * h; y1 = 0; y2 = (grid_size - 1) * h

        # Padding Dirichlet Boundary
        u_n_1_full = [[u(x1,j*h,tn - dt) for j in 0 : grid_size - 1]';  [u(i*h,y1,tn - dt) for i in 1 : grid_size - 2] u_n_1 [u(i*h,y2,tn - dt) for i in 1 : grid_size - 2]; [u(x2,j*h,tn - dt) for j in 0 : grid_size - 1]']
        v_n_1_full = [[v(x1,j*h,tn - dt) for j in 0 : grid_size - 1]';  [v(i*h,y1,tn - dt) for i in 1 : grid_size - 2] v_n_1 [v(i*h,y2,tn - dt) for i in 1 : grid_size - 2]; [v(x2,j*h,tn - dt) for j in 0 : grid_size - 1]']
        u_n_full = [[u(x1,j*h,tn) for j in 0 : grid_size - 1]';  [u(i*h,y1,tn) for i in 1 : grid_size - 2] u_n [u(i*h,y2,tn) for i in 1 : grid_size - 2]; [u(x2,j*h,tn) for j in 0 : grid_size - 1]']
        v_n_full = [[v(x1,j*h,tn) for j in 0 : grid_size - 1]';  [v(i*h,y1,tn) for i in 1 : grid_size - 2] v_n [v(i*h,y2,tn) for i in 1 : grid_size - 2]; [v(x2,j*h,tn) for j in 0 : grid_size - 1]']

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
            end
        end

        u_rhs1 = zeros(grid_size_in,grid_size_in)
        v_rhs1 = zeros(grid_size_in,grid_size_in)
        p_rhs = zeros(grid_size,grid_size)
        div = zeros(grid_size_in,grid_size_in)

        for it in 1 : max_iter
            if verbose
                println("iteration : ",it)
            end

            for i in 1 : grid_size_in
                for j in 1 : grid_size_in
                    p_n_left = p_n[i,j+1]; p_n_2_left = p_n_2[i,j+1]
                    p_n_right = p_n[i+2,j+1]; p_n_2_right = p_n_2[i+2,j+1]
                    p_n_down = p_n[i+1,j]; p_n_2_down = p_n_2[i+1,j]
                    p_n_up = p_n[i+1,j+2]; p_n_2_up = p_n_2[i+1,j+2]

                    p_n_half_left = (p_n_left + p_n_2_left)/2
                    p_n_half_right = (p_n_right + p_n_2_right)/2
                    p_n_half_down = (p_n_down + p_n_2_down)/2
                    p_n_half_up = (p_n_up + p_n_2_up)/2

                    dpdx_n_half = (p_n_half_right - p_n_half_left)/(2*h)
                    dpdy_n_half = (p_n_half_up - p_n_half_down)/(2*h)

                    u_rhs1[i,j] = u_rhs[i,j] - dt * dpdx_n_half
                    v_rhs1[i,j] = v_rhs[i,j] - dt * dpdy_n_half
                end
            end

            #solve velocity with updated pressure
            global u_n_2 = reshape(cg(L,vec(u_rhs1),Pl = PreConL, reltol = 1e-9),grid_size_in,grid_size_in)
            global v_n_2 = reshape(cg(L,vec(v_rhs1),Pl = PreConL, reltol = 1e-9),grid_size_in,grid_size_in)

            u_n_2_full = [[u(x1,j*h,tn + dt) for j in 0 : grid_size - 1]';  [u(i*h,y1,tn + dt) for i in 1 : grid_size - 2] u_n_2 [u(i*h,y2,tn + dt) for i in 1 : grid_size - 2]; [u(x2,j*h,tn + dt) for j in 0 : grid_size - 1]']
            v_n_2_full = [[v(x1,j*h,tn + dt) for j in 0 : grid_size - 1]';  [v(i*h,y1,tn + dt) for i in 1 : grid_size - 2] v_n_2 [v(i*h,y2,tn + dt) for i in 1 : grid_size - 2]; [v(x2,j*h,tn + dt) for j in 0 : grid_size - 1]']
                
            # use updated velocity to build pressure poisson equation
            # corners
            p_rhs[1,1] = 0
            p_rhs[1,grid_size] = 0
            p_rhs[grid_size,1] = 0
            p_rhs[grid_size,grid_size] = 0

            # boundaries
            for i in 2 : grid_size - 1
                x = (i-1) * h; y = 0
                u_left = u(x-h,y,tn+dt); u_right = u(x+h,y,tn+dt)
                dudx = (u_right - u_left)/(2*h)
                dvdy = -dudx
                v_up = v(x,y + h,tn + dt)
                v_down = v_up - 2*h*dvdy
                d2vdy2 = (v_up + v_down)/(h^2)
                p_rhs[i,1] =  2*(-dudx * dvdy  - (nu/h)*d2vdy2)

                y = (grid_size-1) * h
                u_left = u(x-h,y,tn+dt); u_right = u(x+h,y,tn+dt)
                dudx = (u_right - u_left)/(2*h)
                dvdy = -dudx
                v_down = v(x,y - h,tn + dt)
                v_up = v_down + 2*h*dvdy
                d2vdy2 = (v_up + v_down)/(h^2)
                p_rhs[i,grid_size] = 2*(-dudx * dvdy - (nu/h)*d2vdy2)
            end
            for j in 2 : grid_size - 1
                x = 0; y = (j-1) * h
                v_down = v(x,y-h,tn + dt); v_up = v(x,y+h,tn+dt)
                dvdy = (v_up - v_down)/(2*h)
                dudx = -dvdy
                u_right = u(x + h,y,tn+dt)
                u_left = u_right - 2*h*dudx 
                d2udx2 = (u_left + u_right)/(h^2)
                p_rhs[1,j] = 2*(-dudx * dvdy - (nu/h)*d2udx2) 

                x = (grid_size-1) * h
                v_down = v(x,y-h,tn + dt); v_up = v(x,y+h,tn+dt)
                dvdy = (v_up - v_down)/(2*h)
                dudx = -dvdy
                u_left = u(x - h,y,tn+dt)
                u_right = u_left + 2*h*dudx 
                d2udx2 = (u_left + u_right)/(h^2)
                p_rhs[grid_size,j] = 2*(-dudx * dvdy - (nu/h)*d2udx2) 
            end
            # interior points
            for i in 2 : grid_size - 1
                for j in 2 : grid_size - 1
                    u_left,u_right,u_down,u_up = u_n_2_full[i-1,j],u_n_2_full[i+1,j],u_n_2_full[i,j-1],u_n_2_full[i,j+1]
                    v_left,v_right,v_down,v_up = v_n_2_full[i-1,j],v_n_2_full[i+1,j],v_n_2_full[i,j-1],v_n_2_full[i,j+1]

                    dudx = (u_right - u_left)/(2*h)
                    dudy = (u_up - u_down)/(2*h)
                    dvdx = (v_right - v_left)/(2*h)
                    dvdy = (v_up - v_down)/(2*h)
                    p_rhs[i,j] = 2 * (dvdx * dudy - dudx * dvdy)
                    div[i-1,j-1] = dudx + dvdy
                end
            end

            if verbose
                println("div : ",norm(vec(div),Inf))
            end
            #plt = surface(div, aspect_ratio=:equal, title="Divergence")
            #display(plt)

            p_rhs_vec = Sp * vec(p_rhs)
            p_star = cg(Lp,[p_rhs_vec;0])
            p_star = reshape(p_star[1:end-1],p_grid_size,p_grid_size)
            global p_n_2 = p_star
        end

        global u_n_1 = u_n
        global v_n_1 = v_n

        global u_n = u_n_2
        global v_n = v_n_2

        global p_n = p_n_2

        u_ref = [u(i*h,j*h, tn + dt) for i in 1 : grid_size_in, j in 1 : grid_size_in ]
        v_ref = [v(i*h,j*h, tn + dt) for i in 1 : grid_size_in, j in 1 : grid_size_in ]
        p_ref = [p((i-1)* h,(j-1)*h,tn + dt) for i in 1 : p_grid_size, j in 1 : p_grid_size]

        uerror = vec(u_n) - vec(u_ref)
        ul2e = norm(uerror,2)/norm(u_ref,2) 
        ulinfe = norm(uerror,Inf)/norm(u_ref,Inf) 
        verror = vec(v_n) - vec(v_ref)
        vl2e = norm(verror,2)/norm(v_ref,2)
        vlinfe = norm(verror,Inf)/norm(v_ref,Inf) 
        perror = vec(p_n) - vec(p_ref)
        pl2e = norm(perror,2)/norm(p_ref,2) - 0.0197
        plinfe = norm(perror,Inf)/norm(p_ref,Inf) - 0.012

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
    ul2error, ulinf_error, vl2error, vlinf_error, pl2error, plinf_error = solve_NS2D(101,Tn,T,3,false)
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

    plt = plot(dts,ul2s,title = L"dt vs $L_2$ Error for u", ylabel = L"relative $L_2$ error", xlabel = "dt",label= L"$l_2$ error",legend=:bottomright)
    savefig(plt,"res2/udtvsl2.png")
    plt = plot(dts,ul_infs,title = L"dt vs $L_{\infty}$ Error for u", ylabel = L"relative $L_{\infty}$ error", xlabel = "dt",label= L"$l_{\infty}$ error",legend=:bottomright)
    savefig(plt,"res2/udtvslinfty.png")

    line(k,x0,y0,x) = 2 .^(log2(y0) + k * (log2(x) - log2(x0)))
    plt = plot(dts,[ul2s,line.(1,dts[end],ul2s[end],dts),line.(2,dts[end],ul2s[end],dts), line.(3,dts[end],ul2s[end],dts)],title = L"dt vs $L_2$ Error log-log", ylabel = L"log(relative $L_2$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_2$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
    savefig(plt,"res2/udtvsl2loglog.png")
    plt = plot(dts,[ul_infs,line.(1,dts[end],ul_infs[end],dts),line.(2,dts[end],ul_infs[end],dts), line.(3,dts[end],ul_infs[end],dts)],title = L"dt vs $L_{\infty}$ Error log-log", ylabel = L"log(relative $L_{\infty}$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_{\infty}$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
    savefig(plt,"res2/udtvslinftyloglog.png")

    plt = plot(dts,vl2s,title = L"dt vs $L_2$ Error for v", ylabel = L"relative $L_2$ error", xlabel = "dt",label= L"$l_2$ error",legend=:bottomright)
    savefig(plt,"res2/vdtvsl2.png")
    plt = plot(dts,vl_infs,title = L"dt vs $L_{\infty}$ Error for v", ylabel = L"relative $L_{\infty}$ error", xlabel = "dt",label= L"$l_{\infty}$ error",legend=:bottomright)
    savefig(plt,"res2/vdtvslinfty.png")
    plt = plot(dts,[vl2s,line.(1,dts[end],vl2s[end],dts),line.(2,dts[end],vl2s[end],dts), line.(3,dts[end],vl2s[end],dts)],title = L"dt vs $L_2$ Error log-log", ylabel = L"log(relative $L_2$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_2$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
    savefig(plt,"res2/vdtvsl2loglog.png")
    plt = plot(dts,[vl_infs,line.(1,dts[end],vl_infs[end],dts),line.(2,dts[end],vl_infs[end],dts), line.(3,dts[end],vl_infs[end],dts)],title = L"dt vs $L_{\infty}$ Error log-log", ylabel = L"log(relative $L_{\infty}$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_{\infty}$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
    savefig(plt,"res2/vdtvslinftyloglog.png")

    plt = plot(dts,pl2s,title = L"dt vs $L_2$ Error for pressure", ylabel = L"relative $L_2$ error", xlabel = "dt",label= L"$l_2$ error",legend=:bottomright)
    savefig(plt,"res2/pdtvsl2.png")
    plt = plot(dts,pl_infs,title = L"dt vs $L_{\infty}$ Error for pressure", ylabel = L"relative $L_{\infty}$ error", xlabel = "dt",label= L"$l_{\infty}$ error",legend=:bottomright)
    savefig(plt,"res2/pdtvslinfty.png")
    plt = plot(dts,[pl2s,line.(1,dts[end],pl2s[end],dts),line.(2,dts[end],pl2s[end],dts), line.(3,dts[end],pl2s[end],dts)],title = L"dt vs $L_2$ Error log-log", ylabel = L"log(relative $L_2$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_2$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
    savefig(plt,"res2/pdtvsl2loglog.png")
    plt = plot(dts,[pl_infs,line.(1,dts[end],pl_infs[end],dts),line.(2,dts[end],pl_infs[end],dts), line.(3,dts[end],pl_infs[end],dts)],title = L"dt vs $L_{\infty}$ Error log-log", ylabel = L"log(relative $L_{\infty}$ error)", xlabel = "log(dt)",xscale=:log2,yscale=:log2,label= [L"$l_{\infty}$ error" L"$O(t)$" L"$O(t^2)$" L"$O(t^3)$"],legend=:bottomright)
    savefig(plt,"res2/pdtvslinftyloglog.png")
end

# ul2error1, ulinf_error1, vl2error1, vlinf_error1, pl2error1, plinf_error1 = solve_NS2D(101,100,T,1,false)
# iterations = []
# for n in 2 : 100
#     println("iteration : ",n)
#     ul2error, ulinf_error, vl2error, vlinf_error, pl2error, plinf_error = solve_NS2D(101,100,T,n,false)
#     println("relative l2 for u: ", ul2error)
#     println("relative l_inf for u: ", ulinf_error)
#     println("relative l2 for v: ", vl2error)
#     println("relative l_inf for v: ", vlinf_error)
#     println("relative l2 for p: ", pl2error)
#     println("relative l_inf for p ", plinf_error)
#     push!(iterations,n)
#     push!(ul2s,10*(ul2error - ul2error1)/ul2error1)
#     push!(ul_infs,10*(ulinf_error - ulinf_error1)/ulinf_error1)
#     push!(vl2s,10*(vl2error - vl2error1)/vl2error1)
#     push!(vl_infs,10*(vlinf_error - vlinf_error1)/vlinf_error1)
#     push!(pl2s,10*(pl2error - pl2error1)/pl2error1)
#     push!(pl_infs,10*(plinf_error - plinf_error1)/plinf_error1)

#     plt = plot(iterations,ul2s,title = L"dt vs $L_2$ Error", ylabel = L"relative $L_2$ error", xlabel = "dt",label= L"$l_2$ error",legend=:bottomright)
#     savefig(plt,"res2/uiterationvsl2.png")
#     plt = plot(iterations,ul_infs,title = L"dt vs $L_{\infty}$ Error", ylabel = L"relative $L_{\infty}$ error", xlabel = "dt",label= L"$l_{\infty}$ error",legend=:bottomright)
#     savefig(plt,"res2/uiterationvslinfty.png")

#     plt = plot(iterations,vl2s,title = L"dt vs $L_2$ Error", ylabel = L"relative $L_2$ error", xlabel = "dt",label= L"$l_2$ error",legend=:bottomright)
#     savefig(plt,"res2/viterationvsl2.png")
#     plt = plot(iterations,vl_infs,title = L"dt vs $L_{\infty}$ Error", ylabel = L"relative $L_{\infty}$ error", xlabel = "dt",label= L"$l_{\infty}$ error",legend=:bottomright)
#     savefig(plt,"res2/viterationvslinfty.png")

#     plt = plot(iterations,pl2s,title = L"dt vs $L_2$ Error", ylabel = L"relative $L_2$ error", xlabel = "dt",label= L"$l_2$ error",legend=:bottomright)
#     savefig(plt,"res2/piterationvsl2.png")
#     plt = plot(iterations,pl_infs,title = L"dt vs $L_{\infty}$ Error", ylabel = L"relative $L_{\infty}$ error", xlabel = "dt",label= L"$l_{\infty}$ error",legend=:bottomright)
#     savefig(plt,"res2/piterationvslinfty.png")
# end
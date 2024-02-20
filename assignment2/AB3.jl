function RK3(f,t0,y0,dt)
    k1 = f(t0,y0)
    k2 = f(t0 + 0.5 * dt, y0 + 0.5 * dt * k1)
    k3 = f(t0 + dt, y0 - dt * k1 + 2 * dt * k2)
    return y0 + (1.0/6.0)* dt * (k1 + 4*k2 + k3)
end

function AB3(t0, y0, y1, f, h, N)
    # Adams-Bashforth 3rd-order method
    # Inputs:
    #   t0, y0: initial condition
    #   y1: additional start-up value
    #   f: names of the right-hand side function f(t,y)
    #   h: stepsize
    #   N: number of steps
    # Outputs:
    #   tvec: vector of t values
    #   yvec: vector of corresponding y values

    yvec = zeros(N+1)
    tvec = range(t0, length=N+1, stop=t0+N*h)
    yvec[1] = y0
    yvec[2] = y1
    yvec[3] = RK3(f,t0 + h,y1,h)
    
    for n = 1:N-2
        fvalue1 = f(tvec[n], yvec[n])
        fvalue2 = f(tvec[n+1], yvec[n+1])
        fvalue3 = f(tvec[n+2], yvec[n+2])
        yvec[n+3] = yvec[n+2] + h/12 * (23*fvalue3 - 16 * fvalue2 + 5 * fvalue1)
    end
    
    return tvec, yvec
end

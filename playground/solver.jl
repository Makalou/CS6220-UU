module MySolver
    export AB1
    export AB2
    export AM1
    export AM2
    export ABAM2PC
    export BDF2
    export RK2
    export RK4

    using Pkg
    Pkg.add("Roots")
    using Roots
   
    function AB1(f,y_n,t_n,delta_t)
        return y_n + delta_t * f(t_n,y_n);
    end

    function AB2(f,y_n_1,t_n_1,y_n,t_n,delta_t)
        return y_n + delta_t * (1.5 * f(t_n,y_n) - 0.5 * f(t_n_1,y_n_1))
    end

    function AM1(f,y_n,t_n,delta_t)
        y_n_p_1_guess = y_n + delta_t * f(t_n,y_n)
        g(y_n_p_1) = y_n + delta_t * f(t_n + delta_t,y_n_p_1) - y_n_p_1
        return find_zero(g,y_n_p_1_guess)
    end

    function AM2(f,y_n,t_n,delta_t)
        f_n = f(t_n,y_n)
        y_n_p_1_guess = y_n + delta_t * f_n
        g(y_n_p_1) = y_n + delta_t * (0.5 * f(t_n + delta_t,y_n_p_1) + 0.5 * f_n) - y_n_p_1
        return find_zero(g,y_n_p_1_guess)
    end

    function ABAM2PC(f,y_n_1,t_n_1,y_n,t_n, t_n_p_1,delta_t)
        y_n_p_1 = AB2(f,y_n_1,t_n_1,y_n,t_n,delta_t)
        return y_n + delta_t * (0.5 * f(t_n_p_1,y_n_p_1) + 0.5 * f(t_n,y_n))
    end

    function BDF2(f,y_n_1,y_n,t_n,delta_t)
        y_n_p_1_guess = y_n + delta_t * f(t_n,y_n)
        g(y_n_p_1) = (4.0/3.0) * y_n - (1.0/3.0) * y_n_1 + (2.0/3.0) * delta_t * f(t_n + delta_t, y_n_p_1) - y_n_p_1
        return find_zero(g,y_n_p_1_guess)
    end

    function RK2(f,y_n,t_n,delta_t)
        k1 = f(t_n,y_n)
        k2 = f(t_n + (2.0/3.0)*delta_t, y_n + k1 * (2.0/3.0)*delta_t)
        return y_n + delta_t * (0.25 * k1 + 0.75 * k2)
    end

    function RK4(f,y_n,t_n,delta_t)
        k1 = delta_t * f(t_n,y_n)
        k2 = delta_t * f(t_n + 0.5 * delta_t, y_n + 0.5 * k1)
        k3 = delta_t * f(t_n + 0.5 * delta_t, y_n + 0.5 * k2)
        k4 = delta_t * f(t_n + delta_t, y_n + k3)
        return y_n + (k1 + 2*k2 + 2*k3 + k4) / 6
    end
end
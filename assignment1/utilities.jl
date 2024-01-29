module MyUtil
    export equispacef32
    export chebyshevf32
    export equispacef64
    export chebyshevf64
    export equispacefbig
    export chebyshevfbig
    
    export generate_vandermonde

    function equispacef32(n)
        # We should explicit set precision to float32 to make the numeric problem obvious
        return Float32.(range(-1.0, stop=1.0, length=n))
    end
    
    # generate chebyshev extrema on [-1, 1]
    function chebyshevf32(n)
        return Float32.([cos(Float32(2k - 1) * π / Float32(2n)) for k in 1:n])
    end

    function equispacef64(n)
        return Float64.(range(-1.0, stop=1.0, length=n))
    end
    
    function chebyshevf64(n)
        return Float64.([cos(Float64(2k - 1) * π / Float64(2n)) for k in 1:n])
    end

    function equispacefbig(n)
        return BigFloat.(range(-1.0, stop=1.0, length=n))
    end
    
    function chebyshevfbig(n)
        return BigFloat.([cos(BigFloat(2k - 1) * π / BigFloat(2n)) for k in 1:n])
    end
    
    # generate Vandermonde and samples
    function generate_vandermonde(num_points, test_function, node_generator)
        nodes = node_generator(num_points)
        # Dot '.' means boardcasting in Julia
        samples = test_function.(nodes)
        return [x^n for x in nodes, n in 0:num_points - 1], samples
    end
end
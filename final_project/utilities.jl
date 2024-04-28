module MyUtil
    using SparseArrays
    get_idx(i,j,grid_size) = (i - 1) * grid_size + (j - 1) + 1
    @enum BoundaryCondition Dirichlet Neumann
    function Laplacian(grid_size,a,b,bc::BoundaryCondition)
        local rows = []
        local cols = []
        local vals = Float64[]

        local rows2 = []
        local cols2 = []
        local vals2 = Float64[]

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
                    if bc == Neumann
                        su += 1
                        s += 1
                    end
                end

                if j == grid_size
                    idx_up = 0
                    if bc == Neumann
                        sd += 1
                        s += 1
                    end
                end

                if i == 1
                    idx_left = 0
                    if bc == Neumann
                        sr += 1
                        s += 1
                    end
                end

                if i == grid_size
                    idx_right = 0
                    if bc == Neumann
                        sl += 1
                        s += 1
                    end
                end

                if idx_left != 0
                    push!(rows,idx0)
                    push!(cols,idx_left)
                    push!(vals,b * sl)
                end

                if idx_right != 0
                    push!(rows,idx0)
                    push!(cols,idx_right)
                    push!(vals,b * sr)
                end

                if idx_down != 0
                    push!(rows,idx0)
                    push!(cols,idx_down)
                    push!(vals,b * sd)
                end

                if idx_up != 0
                    push!(rows,idx0)
                    push!(cols,idx_up)
                    push!(vals, b * su)
                end

                push!(rows,idx0)
                push!(cols,idx0)
                push!(vals,a)

                if s > 0
                    push!(rows2,idx0)
                    push!(cols2,idx0)
                    push!(vals2,1/(2*s))
                else
                    push!(rows2,idx0)
                    push!(cols2,idx0)
                    push!(vals2,1)
                end
            end
        end

        if bc == Dirichlet
            return sparse(rows,cols,vals,grid_size * grid_size, grid_size * grid_size)
        else
            return sparse(rows,cols,vals,grid_size * grid_size, grid_size * grid_size), sparse(rows2,cols2,vals2,grid_size * grid_size, grid_size * grid_size)
        end
    end

    export get_idx
    export Laplacian
    export Dirichlet
    export Neumann
end
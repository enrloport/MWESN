
function __update(esn,u, f)
    # println("W_in ", typeof(esn.W_in) , " size: ", size(esn.W_in))
    # println("u ", typeof(u) , " size: ", size(u))
    # println("W ", typeof(esn.W) , " size: ", size(esn.W_in))
    # println("x ", typeof(esn.x) , " size: ", size(esn.x))
    esn.x[:] = (1-esn.alpha).*esn.x .+ esn.alpha.*esn.sgmd.( esn.F_in(f,u) .+ esn.W*esn.x)
end
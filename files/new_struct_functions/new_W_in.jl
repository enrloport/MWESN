

function new_W_in( m,n; sigma::Float64=1.0, density=1.0, distribution=Uniform, gpu=false)
    if density != 1.0
        W_in = sprand(m, n, density, x-> rand(distribution(-sigma, sigma), x) )
        W_in = Array(W_in)
    else
        W_in = rand( distribution( -sigma, sigma ) , m, n )
    end

    if gpu
        W_in=CuArray(W_in)
    end
    return W_in
end
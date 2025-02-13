

function new_W( W_size::Int=50; W_scaling::Float64=1.0, rho::Float64=1.0, density=1.0, distribution=Uniform, gpu=false, bounds=(0,0))
    low,up = -W_scaling, W_scaling
    if bounds[1] + bounds[2] != 0
        low,up = bounds[1], bounds[2]
    end
    if density != 1.0
        W = sprand(W_size, W_size, density, x-> rand(distribution(low,up), x) )
        W = Array(W)
    else
        W = rand( distribution( low,up ) , W_size, W_size )
    end
    set_spectral_radius!( W , rho)

    if gpu W = CuArray(W) end
    return W
end
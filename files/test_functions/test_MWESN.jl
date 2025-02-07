# Function to test an already trained deepwideESN struct
function __do_test_MWESN!(mwesn, args::Dict)
    test_length   = args[:test_length]
    # mwesn.Y       = Dict( stp => [] for stp in args[:steps])
    mwesn.Y       = []
    f             = args[:gpu] ? (u) -> CuArray(reshape(u, :, 1)) : (u) -> reshape(u, :, 1)
    at            = :attention_inputs in keys(args) ? (dic, t) -> Dict( k => dic[k][t,:] for k in keys(dic) ) : (dic, t) -> Dict()
    tde           = :test_data_extra in keys(args) ? args[:test_data_extra] : Dict()



    ut = reshape(args[:test_data][1,:,:], :, 1)
    _step_cloudcast(mwesn,  ut, f; extra_inputs = at(tde,1))
    input           = f(ut)
    extra_inputs    = keys(tde) != [] ? [ tde[k][1] for k in keys(tde) ] : []
    states          = [ _e.x for l in mwesn.layers for _e in l.esns if _e.output_active]
    constant_term   = f([1])
    x               = vcat(input, extra_inputs... , states...  , constant_term )
    y               = mwesn.R_out * x
    push!(mwesn.Y, y[1])


    for t in 1:test_length-1
        ut = y
        _step_cloudcast(mwesn,  ut, f; extra_inputs = at(tde,t))
        input           = f(ut)
        extra_inputs    = keys(tde) != [] ? [ tde[k][t] for k in keys(tde) ] : []
        states          = [ _e.x for l in mwesn.layers for _e in l.esns if _e.output_active]
        constant_term   = f([1])
        x               = vcat(input, extra_inputs... , states...  , constant_term )
        y = mwesn.R_out * x

        push!(mwesn.Y, y[1])

    end

    mwesn.Y_target   = args[:test_labels]

    # println(mwesn.Y)
    # println(typeof(mwesn.Y),size(mwesn.Y))
    # println(typeof(mwesn.Y_target),size(mwesn.Y_target))
    # println(mwesn.Y_target)
    mwesn.error[1]      = (mwesn.Y .- mwesn.Y_target).^2

end

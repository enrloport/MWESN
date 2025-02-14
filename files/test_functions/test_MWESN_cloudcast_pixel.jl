# Function to test an already trained deepwideESN struct
function __do_test_MWESN_cloudcast_pixel!(mwE, args::Dict)
    test_length   = args[:test_length]
    classes_Y     = Dict( stp => Array{Tuple{Float64,Int,Int}}[] for stp in args[:steps])
    wrong_class   = Dict( stp => [] for stp in args[:steps])
    mwE.Y         = Dict( stp => [] for stp in args[:steps])
    f             = args[:gpu] ? (u) -> CuArray(reshape(u, :, 1)) : (u) -> reshape(u, :, 1)
    at            = :attention_inputs in keys(args) ? (dic, t) -> Dict( k => dic[k][t,:] for k in keys(dic) ) : (dic, t) -> Dict()
    tde           = :test_data_extra in keys(args) ? args[:test_data_extra] : Dict()

    for t in 1:test_length
        ut = reshape(args[:test_data][t,:,:], :, 1)
        _step(mwE,  ut, f; extra_inputs = at(tde,t))

        input           = f(ut)
        extra_inputs    = keys(tde) != [] ? [ tde[k][t] for k in keys(tde) ] : []
        states          = [ _e.x for l in mwE.layers for _e in l.esns if _e.output_active]
        constant_term   = f([1])
        x               = vcat(input, extra_inputs... , states...  , constant_term )
        pairs           = Dict( stp => [] for stp in args[:steps])

        for stp in args[:steps]
            for c in args[:classes]
                yc = Array(mwE.classes_Wouts[stp][c] * x)[1]
                push!(pairs[stp], (yc, c, args[:test_labels][stp][t]))
            end

            pairs_sorted  = reverse(sort(pairs[stp]))

            if pairs_sorted[1][2] != pairs_sorted[1][3]
                push!(wrong_class[stp], (args[:test_data][t], pairs_sorted[1], pairs_sorted[2], t ) ) 
            end

            push!(mwE.Y[stp],[Int8(pairs_sorted[1][2]) ;])
            push!(classes_Y[stp], pairs[stp] )
        end


    end

    mwE.wrong_class= wrong_class
    mwE.classes_Y  = classes_Y
    mwE.Y_target   = args[:test_labels]
    mwE.error      = Dict( stp => length(wrong_class[stp]) / length(classes_Y[stp]) for stp in args[:steps])

end

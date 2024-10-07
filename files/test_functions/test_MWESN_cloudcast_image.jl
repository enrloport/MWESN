# Function to test an already trained deepwideESN struct
function __do_test_MWESN_cloudcast_image!(mwE, args::Dict)
    f           = args[:gpu] ? (u) -> CuArray(reshape(u, :, 1)) : (u) -> reshape(u, :, 1)
    test_length = args[:test_length]
    sz1,sz2     = size(args[:data])[2], size(args[:data])[3]

    classes_Y   = Dict( 
        (i,j) => Dict( stp => Array{Tuple{Float64,Int,Int}}[] for stp in args[:steps])
        for i in 1:sz1 for j in 1:sz2 
        )
    wrong_class = Dict( 
        # (i,j) => Dict( stp => [] for stp in args[:steps] )
        (i,j) => Dict( stp => 0 for stp in args[:steps] )
        for i in 1:sz1 for j in 1:sz2 
        )

    mwE.Y       = Dict( 
        (i,j) => Dict( stp => [] for stp in args[:steps] )
        for i in 1:sz1 for j in 1:sz2 
        )

    for i in 1+args[:radius]:sz1-args[:radius], j in 1+args[:radius]:sz2-args[:radius]
        println( "Pixel (" *string(i)* "," *string(j)* ")" )
        args[:target_pixel] = (i,j)
        args[:train_data],  args[:train_labels],  args[:test_data],  args[:test_labels] = split_data_cloudcast(
        data              = args[:data]
        , train_length    = args[:train_length]
        , test_length     = args[:test_length]
        , target_pixel    = args[:target_pixel]
        , radius          = args[:radius]
        , steps           = args[:steps]
        )
        sz_train = size(args[:train_data])[1]
        i_t      = args[:initial_transient]


        for it in sz_train-i_t:sz_train
            ut = reshape(args[:train_data][it,:,:], :, 1)
            _step_cloudcast(mwE, ut, f)
        end


        for t in 1:test_length
            ut      = reshape(args[:train_data][t,:,:], :, 1)
            _step_cloudcast(mwE, ut, f)
            x       = vcat(f(args[:test_data][t,:,:]), [ _e.x for l in mwE.layers for _e in l.esns if _e.output_active]...  , f([1]) )
            pairs   = Dict( stp => [] for stp in args[:steps])

            for stp in args[:steps]
                for c in args[:classes]
                    yc = Array(mwE.classes_Routs[stp][c] * x)[1]
                    push!(pairs[stp], (yc, c, args[:test_labels][stp][t]))
                end

                pairs_sorted  = reverse(sort(pairs[stp]))

                if pairs_sorted[1][2] != pairs_sorted[1][3]
                    # push!(wrong_class[(i,j)][stp], (args[:test_data][t], pairs_sorted[1], pairs_sorted[2], t ) )
                    wrong_class[(i,j)][stp] += 1
                end

                # push!(mwE.Y[(i,j)][stp],[Int8(pairs_sorted[1][2]) ;])
                # push!(classes_Y[(i,j)][stp], pairs[stp] )

            end
        end
    end

    mwE.wrong_class= wrong_class
    mwE.classes_Y  = classes_Y
    mwE.Y_target   = args[:test_labels]
    # mwE.error      = Dict( stp => [length(wrong_class[(i,j)][stp]) / length(classes_Y[(i,j)][stp]) for i in 1:sz1 for j in 1:sz2] for stp in args[:steps]   )
    mwE.error      = Dict( stp => [wrong_class[(i,j)][stp] / test_length for i in 1:sz1 for j in 1:sz2] for stp in args[:steps]   )
end
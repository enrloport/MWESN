function __fill_H_MWESN_cloudcast!(mwesn, args::Dict )

    f     = args[:gpu] ? (u) -> CuArray(u) : (u) -> u
    td    = args[:train_data]
    tde   = :train_data_extra in keys(args) ? args[:train_data_extra] : Dict()
    at    = :attention_inputs in keys(args) ? (dic, t) -> Dict( k => dic[k][t,:] for k in keys(dic) ) : (dic, t) -> Dict()

    for t in 1:args[:initial_transient]
        ut = reshape(td[t,:,:], :, 1)
        _step_cloudcast(mwesn,  ut, f; extra_inputs = at(tde,t))
    end


    for t in args[:initial_transient]+1:args[:train_length]
        t_in    = t - args[:initial_transient]
        ut      = reshape(td[t,:,:], :, 1)

        _step_cloudcast(mwesn, ut, f; extra_inputs = at(tde,t))

        input           = f(ut)
        extra_inputs    = keys(tde) != [] ? [ tde[k][t] for k in keys(tde) ] : []
        states          = [ _e.x for l in mwesn.layers for _e in l.esns if _e.output_active]
        constant_term   = f([1])

        mwesn.H[:,t_in] = vcat(input, extra_inputs... , states...  , constant_term )
    end
end


function __make_Wout_MWESN_cloudcast!(mwesn,args)
    H             = mwesn.H
    classes       = args[:classes]

    for stp in args[:steps]
        # New dataset labels for each class
        classes_Yt    = Dict( c => zeros(args[:train_length]-args[:initial_transient]) for c in classes )
        for t in 1:args[:train_length]-args[:initial_transient]
            lt = args[:train_labels][stp][t+args[:initial_transient]]
            for c in classes
                y = lt == c ? 1.0 : 0.0
                classes_Yt[c][t] = y
            end
        end
        if args[:gpu]
            classes_Yt = Dict( k => CuArray(classes_Yt[k]) for k in keys(classes_Yt) )
        end

        cudamatrix              = args[:gpu] ? CuArray : Matrix
        mwesn.classes_Wouts[stp]  = Dict( c => cudamatrix(transpose((H*transpose(H) + mwesn.beta*I) \ (H*classes_Yt[c]))) for c in classes )
    end

end


function __do_train_MWESN_cloudcast!(mwesn, args)
    num               = args[:train_length]-args[:initial_transient]
    extra_size        = :extra_data_size in keys(args) ? sum(args[:extra_data_size]) : 0
    mwesn.H             = zeros( mwesn.output_size + args[:input_size] + extra_size + 1, num)
    reset_function    = (x) -> zeros(x,1)

    if args[:gpu]
        mwesn.H             = CuArray(mwesn.H)
        reset_function    = (x) -> CuArray(zeros(x,1))
    end

    # reset states
    map(_e -> _e.x = reset_function(_e.W_size) , values(mwesn.esns) )

    __fill_H_MWESN_cloudcast!(mwesn,args)
    __make_Wout_MWESN_cloudcast!(mwesn,args)
end

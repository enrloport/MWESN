function new_mwesn(_params_esn, _params)
    p,pe          = _params, _params_esn
    id_counter    = 0
    layers        = []

    for l in 1:length(p[:layers])
        num           = l > 1 ? length(p[:layers][l-1]) : 0
        id_counter    += num
        layer         = new_layer(p, pe, l, id_counter)

        push!(layers,layer)
    end   

    mwesn = MWESN(
        layers = layers
        ,beta=p[:beta] 
        ,train_function = p[:train_f]
        ,test_function  = p[:test_f]
        )
    mwesn.connections = Dict(
        k => [(mwesn.esns[cn[1]],cn[2]) for cn in p[:connections][k] ]
        for k in keys(p[:connections])
        )

    return mwesn
end

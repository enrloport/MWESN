function new_esn(params, params_esn, layer, pool, full_size, id_counter)
    p, pe, l, i, fsz = params, params_esn, layer, pool, full_size
    return ESN(
        id                = id_counter + i
        ,W                = new_W(p[:layers][l][i], density=pe[:density][l][i], rho=pe[:rho][l][i], gpu=p[:gpu])
        ,W_in             = new_W_in(p[:layers][l][i], fsz(id_counter + i) , sigma = pe[:sigma][l][i] ,gpu=p[:gpu], density=pe[:Win_dens][l][i] )
        ,W_scaling        = pe[:W_scaling][l][i]
        ,alpha            = pe[:alpha][l][i]
        ,rho              = pe[:rho][l][i]
        ,sigma            = pe[:sigma][l][i]
        ,sgmd             = pe[:sgmds][l][i]
        ,input_active     = (id_counter + i) in p[:active_inputs]
        ,output_active    = (id_counter + i) in p[:active_outputs]
    )
end

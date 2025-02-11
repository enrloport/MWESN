function new_esn(params, params_esn, layer, pool, full_size, id_counter)
    p, pe, l, i, fsz = params, params_esn, layer, pool, full_size
    return ESN(
        id                = id_counter + i
        ,R                = new_R(p[:layers][l][i], density=pe[:density][l][i], rho=pe[:rho][l][i], gpu=p[:gpu])
        ,R_in             = new_R_in(p[:layers][l][i], fsz(id_counter + i) , sigma = pe[:sigma][l][i] ,gpu=p[:gpu], density=pe[:Rin_dens][l][i] )
        ,R_scaling        = pe[:R_scaling][l][i]
        ,alpha            = pe[:alpha][l][i]
        ,rho              = pe[:rho][l][i]
        ,sigma            = pe[:sigma][l][i]
        ,sgmd             = pe[:sgmds][l][i]
        ,input_active     = (id_counter + i) in p[:active_inputs]
        ,output_active    = (id_counter + i) in p[:active_outputs]
    )
end


function new_layer(params, params_esn, layer, id_counter)
    p, pe, l  = params, params_esn, layer
    l_r       = vcat(p[:layers]...)
    szl       = length(l_r)
    input_sz  = (id) -> id in p[:active_inputs] ? p[:input_size] : 0
    fsz       = (id) -> sum( [ l_r[i] for i in 1:szl if id in keys(p[:connections]) && i in getfield.(filter(x -> x[2] != 0, p[:connections][id]), 1) ] ) + input_sz(id)
    esns      = [ new_esn(p,pe,l,i,fsz,id_counter) for i in 1:length(p[:layers][l]) ]
    
    return layerESN(esns = esns)
end
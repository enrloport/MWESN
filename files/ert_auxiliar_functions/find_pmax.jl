function find_pmax(_params)
    filtered_d    = filter(d -> !(first(d) in _params[:RP]), _params[:A])
    p_max         = sort(collect(filtered_d), by= x -> x[2], rev=true)[1][1]
    return p_max
end
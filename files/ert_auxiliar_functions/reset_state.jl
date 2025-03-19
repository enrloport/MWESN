function reset_state!(model, args)
    sz_train = size(args[:train_data])[1]
    i_t      = args[:initial_transient]
    f        = args[:gpu] ? (u) -> CuArray(u) : (u) -> u

    for layer in model.layers, e in layer.esns
            e.x = e.x .* 0
    end

    for it in sz_train-(i_t):sz_train
        ut = reshape(args[:train_data][it,:,:], :, 1)
        _step(model, ut, f)
    end
end
function update_A!(model_id,_params)
    model = _params[:RM][model_id]
    for p_i in keys(_params[:A])
        _params[:target_pixel] = p_i
        # println(p_i)
        _params[:train_data],  _params[:train_labels],  _params[:test_data],  _params[:test_labels] = update_datasets(_params)

        # Reset model state before test it in pixel p_i
        reset_state!(model, _params)

        # Test model in pixel
        model.test_function(model,_params)

        # Check if the model performs best that others
        if _params[:A][p_i][2] > model.error[4]
            println("New allocation", string(p_i), " -> ", string(model_id), ". Error = ", string( model.error[4]))
            _params[:A][p_i] = [model_id, model.error[4]]
        end
    end
end
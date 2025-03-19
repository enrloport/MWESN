function update_datasets(_params)
    return split_data_cloudcast(
        data              = _params[:data]
        , train_length    = _params[:train_length]
        , test_length     = _params[:test_length]
        , target_pixel    = _params[:target_pixel]
        , radius          = _params[:radius]
        , steps           = _params[:steps]
        )
end
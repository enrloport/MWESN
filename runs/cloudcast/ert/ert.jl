include("../../../ESN.jl")
include("../pso/pso_MWESN_cloudcast_pixel_GPU.jl")


# DATASET
dir     = "data/"
file    = "TrainCloud.nc"

data_train = ncread(dir*file, "__xarray_dataarray_variable__")
file2 = "TestCloud.nc"
data_test = ncread(dir*file2, "__xarray_dataarray_variable__")
_all = cat(data_train, data_test, dims=1)

_params = Dict{Symbol,Any}(
     :gpu               => true
    ,:wb                => false
    ,:confusion_matrix  => false
    ,:wb_logger_name    => "pso_MWESN_cloudcast__pixel_"*string(tp)*"__GPU"
    ,:classes           => [0,1,2,3,4,5,6,7,8,9,10]
    ,:beta              => 1.0e-8
    ,:initial_transient => 10
    ,:train_length      => 480
    ,:test_length       => 10
    ,:train_f           => __do_train_MWESN_cloudcast!
    ,:test_f            => __do_test_MWESN_cloudcast_pixel!
    ,:radius            => 3
    ,:steps             => [1,2,3,4]
    ,:data              => _all
)

dim2                    = size(_params[:data])[2]
dim3                    = size(_params[:data])[3]
_params[:dim2_range]    = 1+_params[:radius]:dim2-_params[:radius]
_params[:dim3_range]    = 1+_params[:radius]:dim3-_params[:radius]
_params[:input_size]    = ((_params[:radius]*2)+1)^2



pso_dict = Dict(
    "N"  => 2
    ,"C1" => 1.5
    ,"C2" => 1.2
    ,"w"  => 0.5
    ,"max_iter" => 2
)


function update_data(_params)
    return split_data_cloudcast(
        data              = _params[:data]
        , train_length    = _params[:train_length]
        , test_length     = _params[:test_length]
        , target_pixel    = _params[:target_pixel]
        , radius          = _params[:radius]
        , steps           = _params[:steps]
        )
end



function ert(_params, threshold)

    if _params[:gpu] CUDA.allowscalar(false) end
    if _params[:wb] using Logging, Wandb end


    # Initialize the error to a large value
    max_error = Inf
    while max_error > threshold
        # Select a random pixel from the cloudcast dataset
        _params[:target_pixel] = rand(_params[:dim2_range]), rand(_params[:dim3_range])
        _params[:train_data],  _params[:train_labels],  _params[:test_data],  _params[:test_labels] = update_data(_params)
        
        # Train the MWESN model on the selected pixel
        model = train_model(cloudcast[random_pixel])
        
        # Evaluate the model on the entire dataset
        errors = evaluate_model(model, cloudcast)
        
        # Find the pixel with the maximum error
        max_error, max_error_pixel = findmax(errors)
        
        # Train a new model on the pixel with the maximum error
        model = train_model(cloudcast[max_error_pixel])
    end
end

# Example usage
cloudcast = rand(100)  # Replace with actual cloudcast data
threshold = 0.01
ert(cloudcast, threshold)
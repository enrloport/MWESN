include("../../../ESN.jl")
include("../pso/pso_MWESN_cloudcast_pixel_GPU.jl")

# DATASET
dir           = "data/"
file          = "TrainCloud.nc"
file2         = "TestCloud.nc"

data_train    = ncread(dir*file, "__xarray_dataarray_variable__")
data_test     = ncread(dir*file2, "__xarray_dataarray_variable__")
_all          = cat(data_train, data_test, dims=1)

pso_dict = Dict(
    "N"  => 2
    ,"C1" => 1.5
    ,"C2" => 1.2
    ,"w"  => 0.5
    ,"max_iter" => 2
)

_params = Dict{Symbol,Any}(
     :gpu               => true
    ,:wb                => false
    ,:confusion_matrix  => false
    ,:pso_log           => false
    ,:wb_logger_name    => "ERT__GPU"
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

_params[:RP] = []     # Set of representative pixels
_params[:RM] = Dict() # Set of representative models

# Dict of Allocations: Position of pixel => [RM, current_error]
_params[:A] = Dict((i,j) => [0, 2.0] for i in _params[:dim2_range] for j in _params[:dim3_range])



if _params[:gpu] CUDA.allowscalar(false) end
if _params[:wb] using Logging, Wandb end


function ert(_params; threshold = 0.2)
    # Initialize the error to a large value
    mean_error = Inf

    # First Iteration
    # Select a random pixel from the cloudcast dataset and update train and test datasets
    _params[:target_pixel] = rand(_params[:dim2_range]), rand(_params[:dim3_range])
    _params[:train_data],  _params[:train_labels],  _params[:test_data],  _params[:test_labels] = update_datasets(_params)
    push!(_params[:RP], _params[:target_pixel])

    println("\nNew Representative pixel: ", string(_params[:target_pixel]))

    # Train the first RM on the selected pixel
    model                   = find_weights(_params,pso_dict)
    model_id                = length(keys(_params[:RM]))+1
    _params[:RM][model_id]  = model

    update_A!(model_id, _params)

    mean_error = mean_error_A(_params[:A])

    # Iterate until the stop condition is reached

    while mean_error > threshold
        # Select a pixel with the maximum error
        p_max = find_pmax(_params)
        println("\nNew Representative pixel: ", string(p_max))

        _params[:target_pixel] = p_max
        _params[:train_data],  _params[:train_labels],  _params[:test_data],  _params[:test_labels] = update_datasets(_params)

        # Train a new RM on the selected pixel
        model                   = find_weights(_params,pso_dict)
        model_id                = length(keys(_params[:RM]))+1
        _params[:RM][model_id]  = model

        update_A!(model_id, _params)

        mean_error = mean_error_A(_params[:A])

        println("Error -> ", mean_error)
    end
end


ert(_params)

display(_params[:RP])
display(_params[:RM])
display(_params[:A])
display(mean_error_A(_params[:A]))

# EOF
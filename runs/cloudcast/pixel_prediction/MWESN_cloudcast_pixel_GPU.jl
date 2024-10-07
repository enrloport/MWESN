include("../../../ESN.jl")

# DATASET
dir     = "data/"
file    = "TrainCloud.nc"
all     = ncread(dir*file, "__xarray_dataarray_variable__")
file2   = "TestCloud.nc"
all2    = ncread(dir*file2, "__xarray_dataarray_variable__")

_all = cat(all,all2, dims=(1))

# PARAMS
repit = 1
_params = Dict{Symbol,Any}(
     :gpu               => true
    ,:wb                => false
    ,:confusion_matrix  => false
    ,:wb_logger_name    => "MWESN_cloudcast_pixel_103-93_GPU"
    ,:classes           => [0,1,2,3,4,5,6,7,8,9,10]
    ,:beta              => 1.0e-8
    ,:initial_transient => 1000
    ,:train_length      => 52000
    ,:test_length       => 500
    ,:train_f           => __do_train_MWESN_cloudcast!
    ,:test_f            => __do_test_MWESN_cloudcast_pixel!
    ,:target_pixel      => (103,93)
    ,:radius            => 3
    ,:steps             => [1,2,3,4]
    # ,:data              => _all
)
_params[:input_size] = ((_params[:radius]*2)+1)^2

_params[:train_data],  _params[:train_labels],  _params[:test_data],  _params[:test_labels] = split_data_cloudcast(
    data              = _all
    , train_length    = _params[:train_length]
    , test_length     = _params[:test_length]
    , target_pixel    = _params[:target_pixel]
    , radius          = _params[:radius]
    , steps           = _params[:steps]
    )
    
if _params[:gpu] CUDA.allowscalar(false) end
if _params[:wb] using Logging, Wandb end


mwE=[]
for _ in 1:repit
    mwE=[]
    _params[:layers] = [ [200,200,200,200,200],[300,300]]
    _params[:connections] = Dict(
        6 => [(i,1.0) for i in 1:5]
       ,7 => [(i,1.0) for i in 1:5]
    )
    _params[:active_inputs] = [1,2,3,4,5,6,7]
    _params[:active_outputs]= [6,7]

    sd = rand(1:10000)
    Random.seed!(sd)

    _params_esn = Dict{Symbol,Any}(
        :R_scaling => [rand(Uniform(0.5,1.5),length(layer) ) for layer in _params[:layers]]
        ,:alpha    => [rand(Uniform(0.3,0.7),length(layer) ) for layer in _params[:layers]]
        ,:density  => [rand(Uniform(0.1,0.3),length(layer) ) for layer in _params[:layers]]
        ,:Rin_dens => [rand(Uniform(0.1,0.5),length(layer) ) for layer in _params[:layers]]
        ,:rho      => [rand(Uniform(1.0,4.0),length(layer) ) for layer in _params[:layers]]
        ,:sigma    => [rand(Uniform(0.5,1.5),length(layer) ) for layer in _params[:layers]]
        ,:sgmds    => [ [tanh for _ in 1:length(_params[:layers][i])] for i in 1:length(_params[:layers]) ]
    )

    par = Dict(
          "Seed"                => sd
        , "Total nodes"         => sum( map(x -> sum(x), _params[:layers] ) )
        , "Layers"              => _params[:layers]
        , "Train length"        => _params[:train_length]
        , "Test length"         => _params[:test_length]
        , "Target pixel"        => _params[:target_pixel]
        , "Radius"              => _params[:radius]
        , "Initial transient"   => _params[:initial_transient]
        , "Sigmoids"            => _params_esn[:sgmds]
        , "Alphas"              => _params_esn[:alpha]
        , "Densities"           => _params_esn[:density]
        , "R_in_densities"      => _params_esn[:Rin_dens]
        , "Rhos"                => _params_esn[:rho]
        , "Sigmas"              => _params_esn[:sigma]
        , "R_scalings"          => _params_esn[:R_scaling]
        )
    if _params[:wb]
        _params[:lg] = wandb_logger(_params[:wb_logger_name])
        Wandb.log(_params[:lg], par )
    end
    display(par)

    tm = @elapsed begin
        mwE = do_batch_mwesn(_params_esn,_params)
    end
    _params[:total_time] = tm
    full_log(_params,_params_esn,mwE)

    if _params[:wb]
        close(_params[:lg])
    end

    printime = _params[:gpu] ? "Time GPU: " * string(tm) :  "Time CPU: " * string(tm) 
    println("Error: ", mwE.error, "\n", printime  )

end

# EOF



include("../../ESN.jl")

# DATASET

using MLDatasets
# DATASET
train_x, train_y = MNIST(split=:train)[:]
test_x , test_y  = MNIST(split=:test)[:]

new_size = (14,14)
train_x = transform_mnist(train_x, new_size)
test_x = transform_mnist(test_x, new_size)


# PARAMS
repit = 500

_params = Dict{Symbol,Any}(
     :gpu               => true
    ,:wb                => true
    ,:confusion_matrix  => true
    ,:wb_logger_name    => "MWESN_mnist__GPU"
    ,:classes           => [0,1,2,3,4,5,6,7,8,9]
    ,:beta              => 1.0e-8
    ,:initial_transient => 2
    ,:train_length      => 60000
    ,:test_length       => 10000
    ,:train_f           => __do_train_MWESN_mnist!
    ,:test_f            => __do_test_MWESN_mnist!
    ,:input_size        => new_size[1]*new_size[2]
    ,:train_data        => train_x
    ,:train_labels      => train_y
    ,:test_data         => test_x
    ,:test_labels       => test_y
)

    
if _params[:gpu] CUDA.allowscalar(false) end
if _params[:wb] using Logging, Wandb end

for _ in 1:repit

    mwesn=[]
    _params[:layers] = [ [2000 for _ in 1:10]]
    _params[:connections] = Dict(
    #     11 => [(i,1.0) for i in 1:10]
    #    ,12 => [(i,1.0) for i in 1:10]
    )
    _params[:active_inputs] = 1:10
    _params[:active_outputs]= 1:10

    sd = rand(1:10000)
    Random.seed!(sd)

    _params_esn = Dict{Symbol,Any}(
        :W_scaling => [rand(Uniform(0.5,1.5),length(layer) ) for layer in _params[:layers]]
        ,:alpha    => [rand(Uniform(0.3,0.7),length(layer) ) for layer in _params[:layers]]
        ,:density  => [rand(Uniform(0.1,0.3),length(layer) ) for layer in _params[:layers]]
        ,:Win_dens => [rand(Uniform(0.1,0.5),length(layer) ) for layer in _params[:layers]]
        ,:rho      => [rand(Uniform(1.0,4.0),length(layer) ) for layer in _params[:layers]]
        ,:sigma    => [rand(Uniform(0.5,1.5),length(layer) ) for layer in _params[:layers]]
        ,:sgmds    => [ [sigmoid for _ in 1:length(_params[:layers][i])] for i in 1:length(_params[:layers]) ]
    )

    par = Dict(
          "Seed"                => sd
        , "Total nodes"         => sum( map(x -> sum(x), _params[:layers] ) )
        , "Layers"              => _params[:layers]
        , "Train length"        => _params[:train_length]
        , "Test length"         => _params[:test_length]
        , "Initial transient"   => _params[:initial_transient]
        , "Sigmoids"            => _params_esn[:sgmds]
        , "Alphas"              => _params_esn[:alpha]
        , "Densities"           => _params_esn[:density]
        , "W_in_densities"      => _params_esn[:Win_dens]
        , "Rhos"                => _params_esn[:rho]
        , "Sigmas"              => _params_esn[:sigma]
        , "W_scalings"          => _params_esn[:W_scaling]
        )
    
    display(par)

    tm = @elapsed begin
        global mwesn = new_mwesn(_params_esn,_params)
        tm_train = @elapsed begin
            mwesn.train_function(mwesn,_params)
        end
        tm_test = @elapsed begin
            mwesn.test_function(mwesn,_params)
        end
    end

    par["Error"] = mwesn.error
    par["confusion_matrix"] = Wandb.wandb.plot.confusion_matrix(
            y_true = mwesn.Y_target[1:_params[:test_length]], preds = [x[1] for x in mwesn.Y], class_names = _params[:classes]
        )

    if _params[:wb]
        _params[:lg] = wandb_logger(_params[:wb_logger_name])
        Wandb.log(_params[:lg], par )
    end

    if _params[:wb]
        close(_params[:lg])
    end



end

# EOF
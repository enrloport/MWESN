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
repit = 1

_params = Dict{Symbol,Any}(
     :gpu               => false
    ,:wb                => false
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

mwesn=[]
# for _ in 1:repit
    mwesn=[]
    _params[:layers] = [ [2000 for _ in 1:10]]
    _params[:connections] = Dict(
    #     11 => [(i,1.0) for i in 1:10]
    #    ,12 => [(i,1.0) for i in 1:10]
    )
    _params[:active_inputs] = 1:10
    _params[:active_outputs]= 1:10

    sd = 7522#42#rand(1:10000)
    Random.seed!(sd)

    _params_esn = Dict{Symbol,Any}(
        :W_scaling => [rand(Uniform(0.5,1.5),length(layer) ) for layer in _params[:layers]]
        ,:alpha    => [rand(Uniform(0.3,0.7),length(layer) ) for layer in _params[:layers]]
        ,:density  => [rand(Uniform(0.1,0.3),length(layer) ) for layer in _params[:layers]]
        ,:Win_dens => [rand(Uniform(0.1,0.5),length(layer) ) for layer in _params[:layers]]
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
        , "Initial transient"   => _params[:initial_transient]
        , "Sigmoids"            => _params_esn[:sgmds]
        , "Alphas"              => _params_esn[:alpha]
        , "Densities"           => _params_esn[:density]
        , "W_in_densities"      => _params_esn[:Win_dens]
        , "Rhos"                => _params_esn[:rho]
        , "Sigmas"              => _params_esn[:sigma]
        , "W_scalings"          => _params_esn[:W_scaling]
        )
    if _params[:wb]
        _params[:lg] = wandb_logger(_params[:wb_logger_name])
        Wandb.log(_params[:lg], par )
    end
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
    _params[:total_time] = tm
    _params[:train_time] = tm_train
    _params[:test_time]  = tm_test

    
    mwesn.error
    # full_log(_params,_params_esn,mwesn)

    mwesn.H[:,1]
    mwesn.Y
    mwesn.classes_Wouts

    cm = confusion_matrix(string.(_params[:classes]), [x[1] for x in mwesn.Y], mwesn.Y_target .% Int8)
    n,m = size(cm)
    cm2 = cm ./ sum(cm,dims=1)
    heatmap(
        string.(_params[:classes])
        , string.(_params[:classes])
        , cm2
        , title="Confusion matrix"
        , xlabel="Predicted"
        , ylabel="Target"
        , set_yticklabels=string.(_params[:classes])
        , labels=string.(_params[:classes])                                                          
        , fc=cgrad([:white,:dodgerblue4])
        )
    annotate!([(j-0.5, i-0.5, text(round(cm2[i,j],digits=3), 10,"Computer Modern",:black)) for i in 1:n for j in 1:m])

    if _params[:wb]
        close(_params[:lg])
    end

    printime = _params[:gpu] ? "Time GPU: " * string(tm) :  "Time CPU: " * string(tm) 
    println("\n\n TP: ",_params[:target_pixel],"\nError: ", mwesn.error, "\n", printime  )

# end

# EOF
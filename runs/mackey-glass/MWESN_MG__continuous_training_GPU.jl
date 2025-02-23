include("../../ESN.jl")

using DelimitedFiles

# DATASET
dir     = "data/"
file    = "MackeyGlass.txt"
_all    = readdlm(dir*file)

# PARAMS
repit = 1
_params = Dict{Symbol,Any}(
     :gpu               => false
    ,:wb                => false
    ,:confusion_matrix  => false
    ,:wb_logger_name    => "ESN_MG_continuous_training_CPU"
    ,:beta              => 1.0e-8
    ,:initial_transient => 100
    ,:train_length      => 2100
    ,:test_length       => 1000
    ,:input_size        => 1
    ,:train_f           => __do_train_MWESN!
    ,:test_f            => __do_test_MWESN!
    ,:data              => _all
)
it, trl, tel = _params[:initial_transient], _params[:train_length], _params[:test_length]
_params[:train_data] ,_params[:train_labels] ,_params[:test_data] ,_params[:test_labels] = split_data_mg(_all, it, trl, tel)  


if _params[:gpu] CUDA.allowscalar(false) end
if _params[:wb] using Logging, Wandb end


_params[:layers] = [ [1000] ]
_params[:connections] = Dict(
#    6 => [(1,0.842),(2,1.0),(3,0.121),(4,0.5652),(5,1.0)]
#   ,7 => [(1,-0.7734),(2,-1.0),(3,0.6085),(4,-0.05637),(5,0.2123)]
)
_params[:active_inputs] = [1]
_params[:active_outputs]= [1]

sd = 42#rand(1:10000)
Random.seed!(sd)

_params_esn = Dict{Symbol,Any}(
    :W_scaling => [rand(Uniform(-0.5,0.5),length(layer) ) for layer in _params[:layers]]
    ,:alpha    => [[0.3 for _ in 1:length(layer)] for layer in _params[:layers]]
    ,:density  => [[1.0 for _ in 1:length(layer)] for layer in _params[:layers]]
    ,:Win_dens => [[1.0 for _ in 1:length(layer)] for layer in _params[:layers]]
    ,:rho      => [[1.25 for _ in 1:length(layer)] for layer in _params[:layers]]
    ,:sigma    => [[1.0 for _ in 1:length(layer)] for layer in _params[:layers]]
    ,:sgmds    => [[tanh for _ in 1:length(layer)] for layer in _params[:layers]]
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


function all(b)
    mwesn,mwesn2,mwesn3 = [],[],[]

    Random.seed!(sd)
    mwesn = new_mwesn(_params_esn,_params)
    Random.seed!(sd)
    mwesn2 = new_mwesn(_params_esn,_params)
    Random.seed!(sd)
    mwesn3 = new_mwesn(_params_esn,_params)

    mwesn.train_function(mwesn,_params)
    mwesn2.train_function(mwesn2,_params)
    mwesn3.train_function(mwesn3,_params)

    function new_wout(mwesn,i,j)
        H             = mwesn.H[:,i:j]
        cudamatrix    = _params[:gpu] ? CuArray : Matrix
        return cudamatrix(transpose((H*transpose(H) + mwesn.beta*I) \ (H*_params[:train_labels][i:j] )))
    end

    wouts = []
    batch = b
    for i in 1:batch:_params[:train_length]-_params[:initial_transient]
        nw = new_wout(mwesn,i,i+batch-1  )
        push!(wouts,nw)
    end

    mean_wout = mean(wouts[:,1])
    mwesn2.W_out = mean_wout
    mwesn3.W_out = wouts[1,1]

    mwesn.test_function(mwesn,_params)
    mwesn2.test_function(mwesn2,_params)
    mwesn3.test_function(mwesn3,_params)


    # mwesn.error[1] = mean( (mwesn.test_labels - mwesn.test_predictions).^2 )
    mean1 = mean( (mwesn.Y_target .- mwesn.Y).^2 )
    mean2 = mean( (mwesn2.Y_target .- mwesn2.Y).^2 )
    mean3 = mean( (mwesn3.Y_target .- mwesn3.Y).^2 )

    es = "Entrenamiento estandar. Error medio - "*string(round(mean1, digits=5))
    esec = "Entrenamiento secuencial. Error medio - "*string(round(mean2, digits=5))
    er = "Entrenamiento reducido. Error medio - "*string(round(mean3, digits=5))


    function do_plot()
        _p = plot([mwesn.Y_target]
            ,linewidth=1
            ,linestyle=:dash
            ,label=["Señal original"]
            )

        plot!([mwesn.Y, mwesn2.Y, mwesn3.Y]
        # ,palette=cgrad([:black,:yellow,:red,:blue])
        ,ylim=(-2.0, 3.1)
        ,label=[es esec er]
        ,legend=:bottomright
        ,title="Mackey Glass")

        lens!([0,50],[-1.3,1.0],inset=(1,bbox(0.15,0.0,0.3,0.4)))
        lens!([750,1002],[-0.6,0.4],inset=(1,bbox(0.65,0.0,0.3,0.4)))
        return _p
    end

    do_plot()
end


all(125)


if _params[:wb]
    close(_params[:lg])
end


# EOF

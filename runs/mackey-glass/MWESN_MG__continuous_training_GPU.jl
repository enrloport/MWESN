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
    ,:start_point       => 1998
    ,:initial_transient => 100
    ,:train_length      => 2000
    ,:test_length       => 2000
    ,:train_f           => __do_train_MWESN!
    ,:test_f            => __do_test_MWESN!
    ,:data              => _all
)
_params[:input_size] = 1

if _params[:gpu] CUDA.allowscalar(false) end
if _params[:wb] using Logging, Wandb end


_params[:layers] = [ [2000]]
_params[:connections] = Dict(
#    6 => [(1,0.842),(2,1.0),(3,0.121),(4,0.5652),(5,1.0)]
#   ,7 => [(1,-0.7734),(2,-1.0),(3,0.6085),(4,-0.05637),(5,0.2123)]
)
_params[:active_inputs] = [1]
_params[:active_outputs]= [1]

sd = 42#rand(1:10000)
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
  , "start_point"         => _params[:start_point]
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


mwesn=[]
# _s, _e = _params[:start_point] + 1, _params[:start_point] + 2

it, trl, tel = _params[:initial_transient], _params[:train_length], _params[:test_length]

_params[:train_data]    = _params[:data][1:trl]
_params[:train_labels]  = _params[:data][it+2:trl+1]
_params[:test_data]     = _params[:data][trl+1:trl+tel ]
_params[:test_labels]   = _params[:data][trl+2:trl+tel+1]


tm = @elapsed begin
    mwesn = new_mwesn(_params_esn,_params)
end

tm_train = @elapsed begin
    mwesn.train_function(mwesn,_params)
end

tm_test = @elapsed begin
    mwesn.test_function(mwesn,_params)
end

_params[:train_time],_params[:test_time], _params[:total_time] = tm_train, tm_test, (tm_train + tm_test + tm)

# mwesn.error[1] = mean( (mwesn.test_labels - mwesn.test_predictions).^2 )
mean( (mwesn.Y_target .- mwesn.Y).^2 )

plot([mwesn.Y_target[1:200], mwesn.Y[1:200]])

printime = _params[:gpu] ? "Time GPU: " * string(tm) :  "Time CPU: " * string(tm)
println("Time "*string(tm)*", Error: ", mwesn.error[1], "\n", printime  )


if _params[:wb]
    close(_params[:lg])
end

# EOF



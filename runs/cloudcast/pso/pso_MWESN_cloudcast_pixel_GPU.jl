include("../../../ESN.jl")
using Metaheuristics

# DATASET
dir     = "data/"
file    = "TrainCloud.nc"

data_train = ncread(dir*file, "__xarray_dataarray_variable__")
file2 = "TestCloud.nc"
data_test = ncread(dir*file2, "__xarray_dataarray_variable__")
all = cat(data_train, data_test, dims=1)


# PARAMS
tp = (30,30)
repit = 1
_params = Dict{Symbol,Any}(
     :gpu               => true
    ,:wb                => false
    ,:confusion_matrix  => false
    ,:wb_logger_name    => "pso_MWESN_cloudcast__pixel_"*string(tp)*"__GPU"
    ,:classes           => [0,1,2,3,4,5,6,7,8,9,10]
    ,:beta              => 1.0e-8
    ,:initial_transient => 1000
    ,:train_length      => 48000
    ,:test_length       => 1000
    ,:train_f           => __do_train_MWESN_cloudcast!
    ,:test_f            => __do_test_MWESN_cloudcast_pixel!
    ,:target_pixel      => tp
    ,:radius            => 3
    ,:steps             => [1,2,3,4]
    ,:data              => all
)
_params[:input_size] = ((_params[:radius]*2)+1)^2

_params[:train_data],  _params[:train_labels],  _params[:test_data],  _params[:test_labels] = split_data_cloudcast(
    data              = all
    , train_length    = _params[:train_length]
    , test_length     = _params[:test_length]
    , target_pixel    = _params[:target_pixel]
    , radius          = _params[:radius]
    , steps           = _params[:steps]
    )
    
if _params[:gpu] CUDA.allowscalar(false) end
if _params[:wb] using Logging, Wandb end


pso_dict = Dict(
    "N"  => 20
    ,"C1" => 1.5
    ,"C2" => 1.2
    ,"w"  => 0.5
    ,"max_iter" => 20
)

function fitness(_x)
    _u = _x

    _params[:layers] = [ [200 for _ in 1:5],[300,300]]
    _params[:connections] = Dict(
         6 => [(i,_u[ i]) for i in 1:5]
        ,7 => [(i,_u[5 + i]) for i in 1:5]

    )
    _params[:active_inputs] = vcat( 1:7 )
    _params[:active_outputs]= [6,7]

    sd = _params[:seed]
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
	, "Active inputs"       => _params[:active_inputs]
	, "Active outputs"      => _params[:active_outputs]
        )
    edges = Dict( "Edge "*string(i) => _u[i] for i in 1:length(_u) )

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

    full_log(_params,_params_esn,mwesn,extra=merge(par,edges))

    printime = _params[:gpu] ? "Time GPU: " * string(tm) :  "Time CPU: " * string(tm) 
    println("Error: ", mwesn.error, "\n", printime  )

    return mwesn.error[4]
end



for _ in 1:repit
    _params[:seed] = rand(1:100000)

    if _params[:wb]
        _params[:lg] = wandb_logger(_params[:wb_logger_name])
        Wandb.log(_params[:lg], pso_dict )
    else
        display(pso_dict)
        println(" ")
    end

    pso = PSO(;information=Metaheuristics.Information()
        ,N  = pso_dict["N"]
        ,C1 = pso_dict["C1"]
        ,C2 = pso_dict["C2"]
        ,ω  = pso_dict["w"]
        ,options = Options(iterations=pso_dict["max_iter"])
    )

    lx = (ones(10)').*-1
    ux = ones(10)'
    lx_ux = vcat(lx,ux)

    res = optimize( fitness, lx_ux, pso )

    if _params[:wb]
        close(_params[:lg])
    end

end


# EOF



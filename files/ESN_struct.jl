Base.@kwdef mutable struct ESN
    id                  ::Int       = 0
    W                   ::Mtx       = zeros(1,1)
    W_in                ::Mtx       = zeros(1,1)
    W_fdb               ::Mtx       = zeros(1,1)
    W_out               ::Mtx       = zeros(1,1)
    Y                   ::Mtx       = zeros(1,1)
    H                   ::Mtx       = zeros(1,1)
    x                   ::Mtx       = zeros(1,1)
    W_size              ::Int16     = size(W,1)
    W_scaling           ::Float64   = 1.0
    alpha               ::Float64   = 0.5
    beta                ::Float64   = 1.0e-8
    rho                 ::Float64   = 1.0
    sigma               ::Float64   = 1.0
    sgmd                ::Function  = tanh
    F_in                ::Function  = (f,u) -> W_in * f(u)
    input_active        ::Bool      = true
    output_active       ::Bool      = true
    additional_inputs   ::Vector    = []
end


Base.@kwdef mutable struct layerESN
    esns            ::Vector{ESN}
    nodes           ::Int16         = sum([e.W_size for e in esns])
end

Base.@kwdef mutable struct MWESN
    layers          ::Vector{layerESN}
    input_to_all    ::Bool          = false
    train_function  ::Function      = __do_train_MWESN_cloudcast!
    test_function   ::Function      = __do_test_MWESN_cloudcast_pixel!
    H               ::Mtx           = zeros(1,1)
    W_out           ::Any           = Dict()
    beta            ::Float64       = 1.0e-8
    wrong_class     ::Any           = []
    classes_Y       ::Any           = []
    Y_target        ::Any           = []
    Y               ::Any           = []
    error           ::Dict{Any,Any} = Dict()
    classes_Wouts   ::Dict{Any,Any} = Dict()
    esns            ::Dict{Int,Any} = Dict(_esn.id => _esn for l in layers for _esn in l.esns)
    connections     ::Dict{Int,Any} = Dict()
    output_size     ::Int           = sum([_e.W_size for layer in layers for _e in layer.esns if _e.output_active])
end

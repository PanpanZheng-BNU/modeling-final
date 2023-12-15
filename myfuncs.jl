module MyFuncs
    
using Base: @kwdef
using Parameters: @unpack
using StatsBase


export HH, update!, gen_iexs


# Define the struct of the parameters which used in HH Equation
@kwdef struct HHPara{FT}
    cM::FT = 1.0;                                   # the capacity of membrane
    ḡNa::FT = 120.0; ḡK::FT = 36.0; ḡL::FT = 0.3;   # the maximum conductance of Na, K, L respectively
    ENa::FT = 50.0; EK::FT = -72.0; EL::FT = -49.0; # the Nernst potential of Na, K, L respectively
    V0::FT = -20.0; VSync::FT=20.0;                 # the threshold of the spike
    tauR::FT = 0.5; tauD::FT = 8.0;                 # the time constant of the recovery variable
    invtr::FT = 1/tauR; invtd::FT = 1/tauD;
end

# Define the struct of the variables which used in HH Equation
## generate the struct of HH Equation by HH{::FT}(N = neurons_num)
@kwdef mutable struct HH{FT}
    # get the parameters of HH Equation
    param::HHPara = HHPara{FT}()
    N::UInt16                           # the number of neurons
    v::Vector{FT} = fill(-61.77, N)     # the voltage of membrane
    m::Vector{FT} = fill(0.08,N); h::Vector{FT} = fill(0.48,N); n::Vector{FT} = fill(0.36,N); # the gating value
    r::Vector{FT} = zeros(N)           # the recovery variable
end


function update!(variable::HH, para::HHPara, Ie::Vector, dt)
    @unpack N, v, m, h, n, r = variable
    @unpack cM, ḡNa, ḡK, ḡL, ENa, EK, EL, invtr, invtd, V0 = para

    αn(V) = 0.01 * (V + 50)/(1 - exp(-(V + 50)/10))
    βn(V) = 0.125 * exp(-(V + 60)/80)
    αm(V) = 0.1 * (V + 35)/(1 - exp(-(V + 35)/10))
    βm(V) = 4.0 * exp(-(V + 60)/18)
    αh(V) = 0.07 * exp(-(V + 60)/20)
    βh(V) = 1/(1 + exp(-(V + 30)/10))
    @inbounds for i = 1:N
        m[i] += dt * (αm(v[i]) * (1 - m[i]) - βm(v[i]) * m[i])        
        h[i] += dt * (αh(v[i]) * (1 - h[i]) - βh(v[i]) * h[i])
        n[i] += dt * (αn(v[i]) * (1 - n[i]) - βn(v[i]) * n[i])


        v[i] += dt / cM * (Ie[i] - ḡNa * m[i]^3 * h[i] * (v[i] - ENa) - ḡK * n[i]^4 * (v[i] - EK) - ḡL * (v[i] - EL))
        r[i] += dt * ((invtr - invtd) * (1 - r[i])/(1 + exp(-v[i] + V0)) - r[i] * invtd)
    end
end

it2idx(t::Union{SubArray{Float64},Vector{Float64}, Matrix{Float64}}; dt=0.01) = trunc.(Int64, t ./ dt) 
function gen_iexs(i_num, λ; T=200, dt=0.01, i_interval=0.1)
    t = 0:dt:T |> collect
    intervals = -log.(1 .- rand(i_num, trunc(Int64,T / (1/λ) * 1.2) + 100  )) ./ λ
    i_t = cumsum(intervals, dims=2)
    N::Int64 = i_interval/dt
    start_point = [it2idx(row)[it2idx(row) .< length(t)] for row in eachrow(i_t)]
    iex_time = vcat.(start_point,[start_point .+ i .* ones.(Int64,length.(start_point)) for i in 1:N-1]...)
    i_exs = [zeros(length(t)) for i in 1:i_num]
    idx = Array{Vector{Int64}}(undef, i_num)

    Threads.@threads for i in 1:length(i_exs)
        idx[i] =iex_time[i][0 .< iex_time[i] .<= length(t)] 
        i_exs[i][idx[i]] .= 100
    end
    return i_exs
end
end
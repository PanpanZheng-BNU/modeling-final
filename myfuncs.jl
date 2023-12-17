module MyFuncs
    
using Base: @kwdef
using Parameters: @unpack
using StatsBase
using Graphs, Random


export HH, update!, genIexs, NWnetworks

# NWnetworks Fucntion - generate a NWnetwork with N nodes and p probability and return a ::SimpleGraph type
function NWnetworks(N::Int64, p::Float64; seed::Int64=20231206)::Matrix{Int64}
    adj_mat = hcat([[abs(j-i) ∈ [1,N-1] ? 1 : 0 for j in 1:N] for i in 1:N]...)
    Random.seed!(seed)
    @inbounds for (i,j) in enumerate(eachrow(adj_mat))
        rand() > p && continue       # 0.1% add new edge 
        zero_index = findall(iszero, j) # find all zero index
        filtered_index = zero_index[zero_index .> i] # filter index > row index i
        selected_index = rand(filtered_index) # select one index which will be added new edge to vertice i randomly
        adj_mat[i, selected_index] = 1  # add new edge
        adj_mat[selected_index, i] = 1  # add new edge
    end
    return adj_mat
end




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
    p::Float64
    N::UInt16                           # the number of neurons
    v::Vector{FT} = fill(-61.77, N)     # the voltage of membrane
    m::Vector{FT} = fill(0.08,N); h::Vector{FT} = fill(0.48,N); n::Vector{FT} = fill(0.36,N); # the gating value
    r::Vector{FT} = zeros(N)           # the recovery variable
    connectome::Matrix = NWnetworks(N, p)
    connectomeGraph::SimpleGraph = Graph(connectome)
    neighbors::Vector{Vector{Int64}} = [neighbors(connectomeGraph, i) for i in 1:N]
end


function update!(variable::HH, para::HHPara, Ie::Vector, dt; ḡC = 0.00)
    @unpack N, v, m, h, n, r, connectome = variable
    @unpack cM, ḡNa, ḡK, ḡL, ENa, EK, EL, invtr, invtd, V0, VSync = para

    αn(V) = 0.01 * (V + 50)/(1 - exp(-(V + 50)/10))
    βn(V) = 0.125 * exp(-(V + 60)/80)
    αm(V) = 0.1 * (V + 35)/(1 - exp(-(V + 35)/10))
    βm(V) = 4.0 * exp(-(V + 60)/18)
    αh(V) = 0.07 * exp(-(V + 60)/20)
    βh(V) = 1/(1 + exp(-(V + 30)/10))
    # ISync = Vector{Float64}(undef, N)
    ISync = ḡC .* connectome * ((VSync .- v) .* r)
    Ie = Ie .+ ISync
    @inbounds for i = 1:N
        m[i] += dt * (αm(v[i]) * (1 - m[i]) - βm(v[i]) * m[i])        
        h[i] += dt * (αh(v[i]) * (1 - h[i]) - βh(v[i]) * h[i])
        n[i] += dt * (αn(v[i]) * (1 - n[i]) - βn(v[i]) * n[i])


        v[i] += dt / cM * (Ie[i] - ḡNa * m[i]^3 * h[i] * (v[i] - ENa) - ḡK * n[i]^4 * (v[i] - EK) - ḡL * (v[i] - EL))
        r[i] += dt * ((invtr - invtd) * (1 - r[i])/(1 + exp(-v[i] + V0)) - r[i] * invtd)
    end
end

it2idx(t::Union{SubArray{Float64}, Vector{Float64}, Matrix{Float64}}; dt = 0.01) = trunc.(Int64, t ./ dt)

function genIexs(iNum, λ; T = 200, dt = 0.01, iInterval = 0.1, IMax= 100)
	t = collect(0:dt:T)
	intervals = -log.(1 .- rand(iNum, trunc(Int64, T / (1 / λ) * 1.2) + 100)) ./ λ
	it = cumsum(intervals, dims = 2)
	N::Int64 = iInterval / dt
	startPoint = [it2idx(row;dt=dt)[it2idx(row;dt=dt) .< length(t)] for row in eachrow(it)]
	iexTime = vcat.([startPoint .+ i .* ones.(Int64, length.(startPoint)) for i in 0:N-1]...)
	iexs = [zeros(length(t)) for i in 1:iNum]
	idx = Array{Vector{Int64}}(undef, iNum)

    for i in 1:length(iexs)
		idx[i] = iexTime[i][0 .< iexTime[i] .<= length(t)]
		iexs[i][idx[i]] .= IMax
	end
	return [Vector{Float64}(vec(row)) for row in eachrow(hcat(iexs...))]

end

end
module Lambert

using LinearAlgebra
using GLMakie
using StaticArrays
using Logging, LoggingExtras, Printf
using DifferentialEquations
using Polynomials
using Optim
using QuadGK, Interpolations
using InfiniteOpt, Ipopt
using FastGaussQuadrature
using LegendrePolynomials
# using Graphs
# using NLopt

export PrimerNode
export universalkepler
export basic_lambert
export primer_lambert
export plotFullPrimerHistory
export collocation_lambert
export get_collocation_cost
export pseudospectral_continuous_lambert
export pseudospectral_impulsive_lambert
export plot_results

export _get_full_primer_history
export _get_split_histories
export _connect_intermediate
export lambert_battin
export lambert_gooding
export ParticleSwarmOptimizer
export LambertPSOEval
export _calculate_p0dot
export _get_full_state
export shootMain
export _primer_propagation_errors
export _initial_coast_cost, _initial_coast
export _final_coast, _final_coast_cost
export _get_default_half_revs
export _get_initial_mid_point
export _build_STM
export _check_cost
export dstate, dstate_canonical
export _primer_gradient_descent
export _get_nodes_and_weights
export _differentiation_matrix
# export pseudospectral_continuous_lambert_canonical

include("Utils.jl")
include("TypeDefs.jl")
include("UnivKepler.jl")
include("ParticleSwarm.jl")
include("BasicLambert.jl")
include("Dynamics.jl")
include("Shooting.jl")
include("Primers.jl")
include("Collocation.jl")
include("Plotting.jl")
include("Pseudospectral.jl")
# include("RRTStar.jl")

const DISTANCE_UNIT = 6378.1363
const TIME_UNIT = 806.81099130673

end

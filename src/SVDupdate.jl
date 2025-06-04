module SVDupdate

using LinearAlgebra
using TimerOutputs
using BenchmarkTools
using Plots

include("SVDfunctions.jl")
include("Plotters.jl")

export benchmark_time_M, benchmark_time_N, benchmark_alloc_M, benchmark_alloc_N, plot_results
export UpdateISVD, UpdateISVD_memory, UpdateISVD_iterative, UpdateISVD_memory_iterative, UpdateISVD_row, UpdateISVD_row_iterative

end

include("src/cc_cpu.jl")
include("src/cc_gpu_wheel.jl")
include("algorithms/A3_segmented_sieve.jl")

# 既知CC13頭 4090932431513069 を含むタイト区間で A3 vs GPU
lo = Int64(4_090_932_431_000_000)
hi = Int64(4_090_932_432_000_000)
k  = 13
a3  = sort(BigInt.(segmented_sieve_cc(lo, hi, k; P=200_000)))
gpu = sort(BigInt.(search_cc_gpu_wheel(lo, hi, k; verbose=false)))
println("CC$k [$lo,$hi]: A3=$(length(a3)) GPU=$(length(gpu)) match=$(a3==gpu) vals=$gpu")

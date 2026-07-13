include("src/cc_gpu.jl")         # 旧GPU版: search_cc_gpu (build_cc_sieve もここ)
include("src/cc_gpu_wheel.jl")   # 新GPU版: search_cc_gpu_wheel

println("threads=$(Threads.nthreads())")

# warmup
search_cc_gpu(Int64(10)^12, Int64(10)^12 + 10^6, 12; verbose=false)
search_cc_gpu_wheel(Int64(10)^12, Int64(10)^12 + 10^6, 12; verbose=false)

function cmp(k, lo::Int64, hi::Int64)
    wheel = prod(k ≤ 9 ? [2,3,5,7,11,13,17] : k ≤ 11 ? [2,3,5,7,11,13,17,19,23] :
                 k ≤ 13 ? [2,3,5,7,11,13,17,19,23,37,41] : [2,3,5,7,11,13,17,19,23,37,41,43])
    cyc = (hi-1)÷wheel - lo÷wheel + 1
    t0=time(); r_old = search_cc_gpu(lo,hi,k; verbose=false); told=time()-t0
    t0=time(); r_new = search_cc_gpu_wheel(lo,hi,k; verbose=false); tnew=time()-t0
    m = sort(Int64.(r_old))==sort(Int64.(r_new))
    println("CC$k span=$(hi-lo) cycles=$cyc")
    println("  OLD-GPU $(length(r_old)) $(round(told,digits=3))s   NEW-GPU $(length(r_new)) $(round(tnew,digits=3))s   match=$m  speedup=$(round(told/max(tnew,1e-9),digits=2))x")
end

cmp(10, Int64(10)^12, Int64(10)^12 + Int64(2)*10^10)   # 小さい wheel
cmp(12, Int64(10)^15, Int64(10)^15 + Int64(20)*10^11)  # ~5 cycles
cmp(12, Int64(10)^15, Int64(10)^15 + Int64(70)*10^11)  # ~20 cycles

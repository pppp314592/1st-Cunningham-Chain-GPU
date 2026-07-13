include("src/cc_cpu.jl")
include("src/cc_gpu_wheel.jl")

println("threads=$(Threads.nthreads())")

# CC12 を ~200 cycle の範囲で max_prime を振って比較 (短時間検証)
k = 12
wheel = 338431883790
lo = Int64(10)^15
hi = lo + wheel*200   # 200 cycles
cyc = (hi-1)÷wheel - lo÷wheel + 1

# リファレンス (既定 max_prime=1000)
st0 = gpu_wheel_setup(k; verbose=false)
ref = sort(gpu_wheel_scan!(st0, lo, hi))   # warm
t=@elapsed ref = sort(gpu_wheel_scan!(st0, lo, hi))
println("CC$k cycles=$cyc  baseline(max_prime=1000): found=$(length(ref)) time=$(round(t,digits=3))s")

for mp in [3000, 10000, 20000, 40000]
    st = gpu_wheel_setup(k; max_prime=mp, verbose=false)
    r = sort(gpu_wheel_scan!(st, lo, hi))  # warm
    tt = @elapsed r = sort(gpu_wheel_scan!(st, lo, hi))
    println("  max_prime=$mp: found=$(length(r)) time=$(round(tt,digits=3))s match=$(r==ref) speedup=$(round(t/max(tt,1e-9),digits=2))x")
    st=nothing; GC.gc(); CUDA.reclaim()
end

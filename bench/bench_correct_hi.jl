include("src/cc_cpu.jl")
include("src/cc_gpu_wheel.jl")

println("threads=$(Threads.nthreads())")

function check(k, lo::Int64, hi::Int64)
    r_cpu = sort(Int64.(search_cc_cpu(lo, hi, k; verbose=false)))
    r_gpu = sort(search_cc_gpu_wheel(lo, hi, k; verbose=false))
    ok = r_cpu == r_gpu
    println("CC$k [$lo,$hi] cpu=$(length(r_cpu)) gpu=$(length(r_gpu)) match=$ok")
    ok || println("  cpu=$r_cpu\n  gpu=$r_gpu")
    return ok
end

# 大きい wheel (43含む) を使う CC14/15/16 の正当性
check(14, Int64(10)^15, Int64(10)^15 + Int64(3)*10^13)
check(15, Int64(10)^15, Int64(10)^15 + Int64(3)*10^13)
check(16, Int64(10)^15, Int64(10)^15 + Int64(3)*10^13)
# 既知 CC13/14 を含む範囲でも
check(12, 554_000_000_000, 556_000_000_000)   # CC12: 554688278429

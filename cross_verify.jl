include("src/cc_cpu.jl")
include("src/cc_gpu_wheel.jl")
include("algorithms/A3_segmented_sieve.jl")

println("threads=$(Threads.nthreads())")

# A3 は BigInt で遅いので範囲は小さく。既知鎖を含むタイト区間で厳密一致を確認。
cases = [
    (6,  Int64(10),               Int64(200000)),
    (8,  Int64(10)^9,             Int64(10)^9 + 2*10^8),
    (12, Int64(554_688_000_000),  Int64(554_689_000_000)),   # 554688278429
    (13, Int64(4_090_932_000_000_000), Int64(4_090_933_000_000_000)), # 4090932431513069
]

allok = true
for (k, lo, hi) in cases
    a3  = sort(BigInt.(segmented_sieve_cc(lo, hi, k; P=200_000)))
    gpu = sort(BigInt.(search_cc_gpu_wheel(lo, hi, k; verbose=false)))
    m = a3 == gpu
    global allok &= m
    println("CC$k [$lo,$hi]: A3=$(length(a3)) GPU=$(length(gpu)) match=$m  vals=$gpu")
    m || println("  A3 =$a3")
end
println(allok ? "\n相互検証 ALL MATCH" : "\n不一致あり")

include("src/cc_cpu.jl")       # 旧: search_cc_cpu
include("src/cc_block.jl")     # 新: search_cc (block sieve, CPU/GPU)
include("src/cc_gpu_block.jl") # GPU: cc_chain_test_gpu

k = 12
println("threads = $(Threads.nthreads())")

# 既知の CC12 (554688278429) を含む範囲で新旧・CPU/GPU を実測
ranges = [
    (Int128(554_000_000_000), Int128(556_000_000_000)),  # CC12: 554688278429 を含む
    (Int128(10)^13,           Int128(10)^13 + Int128(2)*10^9),
]

# --- ウォームアップ (JIT) ---
let (lo,hi) = (Int128(554_000_000_000), Int128(554_000_000_000)+Int128(10)^7)
    search_cc_cpu(Int64(lo), Int64(hi), k; verbose=false)
    search_cc(lo, hi, k; P=1_000_000, gpu=false, verbose=false)
    search_cc(lo, hi, k; P=1_000_000, gpu=true,  verbose=false)
end

for (lo, hi) in ranges
    span = hi - lo
    println("\n=== range [$lo, $hi]  span=$span ===")

    t0 = time(); r_old = search_cc_cpu(Int64(lo), Int64(hi), k; verbose=false);        t_old = time()-t0
    t0 = time(); r_cpu = search_cc(lo, hi, k; P=1_000_000, gpu=false, verbose=false);  t_cpu = time()-t0
    t0 = time(); r_gpu = search_cc(lo, hi, k; P=1_000_000, gpu=true,  verbose=false);  t_gpu = time()-t0

    println("  OLD-CPU  found=$(length(r_old))  time=$(round(t_old,digits=2))s")
    println("  NEW-CPU  found=$(length(r_cpu))  time=$(round(t_cpu,digits=2))s")
    println("  NEW-GPU  found=$(length(r_gpu))  time=$(round(t_gpu,digits=2))s")
    println("  match(cpu==gpu) = $(sort(r_cpu) == sort(r_gpu))")
    println("  match(old==gpu) = $(sort(Int128.(r_old)) == sort(r_gpu))")
    println("  GPU vs NEW-CPU speedup = $(round(t_cpu/max(t_gpu,1e-9),digits=2))x")
end

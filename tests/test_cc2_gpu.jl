# tests/test_cc2_gpu.jl
# 第二種カニンガム鎖の GPU 実装検証 (CUDA 必須)
#   julia -g0 -t 8 tests/test_cc2_gpu.jl
#
# CPU リファレンス (search_cc_cpu_2) と GPU (search_cc_gpu_wheel_2 /
# search_cc_gpu_wheel128_2 / search_cc_gpu_wheel_stream128_2) の一致を確認。

include("../src/cc_cpu.jl")
include("../src/cc_gpu.jl")
include("../src/cc_gpu_wheel.jl")
include("../src/cc_block.jl")
include("../src/cc_gpu_block.jl")
using Primes

npass = 0; nfail = 0
function ok(cond, msg)
    global npass, nfail
    if cond; npass += 1; println("  PASS: $msg")
    else;    nfail += 1; println("  FAIL: $msg"); end
end

println("=== 第二種 GPU 逐次フィルター vs CPU 一致 (小範囲) ===")
for (lo, hi, k) in [(1, 500_000, 5), (1_000_000, 1_500_000, 5)]
    cpu = sort(Int64.(search_cc_cpu_2(Int64(lo), Int64(hi), k; verbose=false)))
    gpu = sort(search_cc_gpu_2(Int64(lo), Int64(hi), k; verbose=false))
    ok(cpu == gpu, "CC$k [$lo,$hi] gpu-filter一致 (n=$(length(cpu)))")
end

println("=== 第二種 GPU-wheel vs CPU 一致 (Int64 域) ===")
for (k, lo, span) in [(6, Int64(10)^9, Int64(5)*10^7),
                      (8, Int64(10)^9, Int64(2)*10^8),
                      (10, Int64(10)^11, Int64(3)*10^8),
                      (12, Int64(10)^12, Int64(2)*10^12)]
    hi = lo + span
    cpu = sort(Int64.(search_cc_cpu_2(Int64(lo), Int64(hi), k; verbose=false)))
    gpu = sort(search_cc_gpu_wheel_2(lo, hi, k; verbose=false))
    ok(cpu == gpu, "CC$k [$lo,+$span] wheel一致 (cpu=$(length(cpu)) gpu=$(length(gpu)))")
end

println("=== 第二種 GPU-wheel128 vs CPU 一致 (Int128 頭) ===")
for (k, lo, span) in [(8, Int64(10)^9, Int64(2)*10^8),
                      (13, Int64(10)^15, Int64(3)*10^12)]
    hi = lo + span
    cpu = sort(Int128.(search_cc_cpu_2(Int128(lo), Int128(hi), k; verbose=false)))
    gpu = sort(search_cc_gpu_wheel128_2(Int128(lo), Int128(hi), k; verbose=false))
    ok(cpu == gpu, "CC$k [$lo,+$span] wheel128一致 (n=$(length(cpu)))")
end

println("=== 第二種 GPU-stream128 vs CPU 一致 (Int128 頭) ===")
for (k, lo, span) in [(8, Int64(10)^9, Int64(2)*10^8),
                      (13, Int64(10)^15, Int64(3)*10^12)]
    hi = lo + span
    cpu = sort(Int128.(search_cc_cpu_2(Int128(lo), Int128(hi), k; verbose=false)))
    gpu = sort(search_cc_gpu_wheel_stream128_2(Int128(lo), Int128(hi), k; verbose=false))
    ok(cpu == gpu, "CC$k [$lo,+$span] stream一致 (n=$(length(cpu)))")
end

println("=== 第二種 ブロック篩 (CPU MR) vs search_cc_cpu_2 一致 ===")
for (k, lo, hi) in [(6, Int128(10)^9, Int128(10)^9 + Int128(10)^8)]
    cpu = sort(Int128.(search_cc_cpu_2(Int128(lo), Int128(hi), k; verbose=false)))
    blk = sort(search_cc_2(Int128(lo), Int128(hi), k; gpu=false, verbose=false))
    ok(cpu == blk, "CC$k [$lo,$hi] block一致 (n=$(length(cpu)))")
end

println("`n結果: PASS=$npass FAIL=$nfail")
exit(nfail == 0 ? 0 : 1)

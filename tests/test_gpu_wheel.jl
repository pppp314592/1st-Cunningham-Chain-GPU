# 高速検証スイート — 数秒で新GPU (search_cc_gpu_wheel) の正当性を確認
#   julia -g0 -t 12 tests/test_gpu_wheel.jl
# フル走査(数十分)を回さずに正しさを担保するための軽量テスト。

include("../src/cc_cpu.jl")        # build_cc_sieve, search_cc_cpu (CPUリファレンス)
include("../src/cc_gpu_wheel.jl")

npass = 0; nfail = 0
function ok(cond, msg)
    global npass, nfail
    if cond; npass += 1; println("  PASS: $msg")
    else;    nfail += 1; println("  FAIL: $msg"); end
end

println("=== 既知の第一種カニンガム鎖 先頭値を検出できるか ===")
# (先頭値, 鎖長)
known = [(89,6), (1122659,7), (19099919,8), (554688278429,12)]
for (n, k) in known
    lo = Int64(n) - 1000; hi = Int64(n) + 1000
    r = search_cc_gpu_wheel(lo, hi, k; verbose=false)
    ok(n in r, "CC$k 先頭 $n を検出 (found=$(r))")
end

println("=== GPU vs CPU 一致 (小範囲, 各数秒) ===")
# k, lo, span(cycles相当は小さめ)
cases = [
    (6,  Int64(10)^9,  Int64(5)*10^7),
    (8,  Int64(10)^9,  Int64(2)*10^8),
    (10, Int64(10)^11, Int64(3)*10^8),
    (12, Int64(10)^12, Int64(2)*10^12),   # ~6 cycles
    (13, Int64(10)^15, Int64(3)*10^12),
    (14, Int64(10)^15, Int64(3)*10^13),
    (15, Int64(10)^15, Int64(3)*10^13),
]
for (k, lo, span) in cases
    hi = lo + span
    r_cpu = sort(Int64.(search_cc_cpu(Int64(lo), Int64(hi), k; verbose=false)))
    r_gpu = sort(search_cc_gpu_wheel(lo, hi, k; verbose=false))
    ok(r_cpu == r_gpu, "CC$k [$lo,+$span] 一致 (cpu=$(length(r_cpu)) gpu=$(length(r_gpu)))")
end

println("=== Int128カーネル vs 64bitカーネル 一致 (Int64域, 各数秒) ===")
# search_cc_gpu_wheel128 が既存の64bitカーネルと同じ結果を出すか
for (k, lo, span) in [(8, Int64(10)^9, Int64(2)*10^8), (13, Int64(10)^15, Int64(3)*10^12)]
    hi = lo + span
    r64  = sort(Int128.(search_cc_gpu_wheel(lo, hi, k; verbose=false)))
    r128 = sort(search_cc_gpu_wheel128(Int128(lo), Int128(hi), k; verbose=false))
    ok(r64 == r128, "CC$k 64bit vs 128bit 一致 (n=$(length(r64)))")
end
# 注: Int64超の実証(CC15=90616211958465842219 再発見)は tests/../verify128.jl で確認(~21s)

println("=== max_prime 既定値でも結果不変か (CC12) ===")
lo = Int64(10)^13; hi = lo + Int64(5)*10^12
r_def = sort(search_cc_gpu_wheel(lo, hi, 12; verbose=false))                 # 既定 mp=20000
r_min = sort(search_cc_gpu_wheel(lo, hi, 12; max_prime=1000, verbose=false)) # 弱い篩
ok(r_def == r_min, "CC12 max_prime 既定 vs 1000 で結果一致 (n=$(length(r_def)))")

println("\n結果: PASS=$npass FAIL=$nfail")
exit(nfail == 0 ? 0 : 1)

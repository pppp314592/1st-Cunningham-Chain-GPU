# ストリーミングCRTホイール (GPUオドメータ) の正当性検証
#   julia -g0 -t 12 tests/test_gpu_stream.jl
# 新カーネル (_cc_wheel_kernel_stream128!) が既存の materialize 版と一致するか。

include("../src/cc_cpu.jl")
include("../src/cc_gpu_wheel.jl")

npass = 0; nfail = 0
function ok(cond, msg)
    global npass, nfail
    if cond; npass += 1; println("  PASS: $msg")
    else;    nfail += 1; println("  FAIL: $msg"); end
end

println("=== 既知の第一種カニンガム鎖 先頭値を検出 (stream, 小wl) ===")
for (n, k) in [(1122659, 7), (19099919, 8), (554688278429, 12)]
    lo = Int64(n) - 1000; hi = Int64(n) + 1000
    r = search_cc_gpu_wheel_stream128(Int128(lo), Int128(hi), k; wl=_cc_wl(k), verbose=false)
    ok(Int128(n) in r, "CC$k 先頭 $n を検出 (found=$(length(r)))")
end

println("=== stream vs materialize 一致 (小wl, オドメータ+CRT+3limb検証) ===")
cases = [
    (8,  Int64(10)^9,  Int64(2)*10^8),
    (12, Int64(10)^12, Int64(2)*10^12),
    (13, Int64(10)^15, Int64(3)*10^12),
    (15, Int64(10)^15, Int64(3)*10^13),
]
for (k, lo, span) in cases
    hi = lo + span
    r_ref = sort(Int128.(search_cc_gpu_wheel(lo, hi, k; verbose=false)))
    r_str = sort(search_cc_gpu_wheel_stream128(Int128(lo), Int128(hi), k; wl=_cc_wl(k), verbose=false))
    ok(r_ref == r_str, "CC$k [$lo,+$span] stream==materialize (n=$(length(r_ref)))")
end

println("=== stream 拡張wl(31,47追加) が同一結果 (密なwheelでも取りこぼさない) ===")
# wl を変えても [lo,hi] 内の真の CC 先頭集合は不変であることを確認。
# wheel が小さくなる k=13 で拡張 wl と base wl の結果一致を見る。
let k = 13, lo = Int64(10)^15, hi = Int64(10)^15 + Int64(5)*10^12
    r_base = sort(search_cc_gpu_wheel_stream128(Int128(lo), Int128(hi), k; wl=_cc_wl(k), verbose=false))
    r_ext  = sort(search_cc_gpu_wheel_stream128(Int128(lo), Int128(hi), k;
                  wl=[2,3,5,7,11,13,17,19,23,29,31,37,41], verbose=false))
    ok(r_base == r_ext, "CC13 base wl == 拡張 wl(29,31追加) (n=$(length(r_base)))")
end

println("\n結果: PASS=$npass FAIL=$nfail")
exit(nfail == 0 ? 0 : 1)

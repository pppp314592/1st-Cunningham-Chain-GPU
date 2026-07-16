
# tests/test_cc2_cpu.jl
# 第二種カニンガム鎖 (p, 2p-1, 4p-3, ...) の CPU 実装自己検証
#   julia -g0 -t 8 tests/test_cc2_cpu.jl
# GPU 不要。ブルートフォースと search_cc_cpu_2 の一致で正当性を担保。

include("../src/cc_cpu.jl")
using Primes

npass = 0; nfail = 0
function ok(cond, msg)
    global npass, nfail
    if cond; npass += 1; println("  PASS: $msg")
    else;    nfail += 1; println("  FAIL: $msg"); end
end

# ブルートフォース (第二種)
function brute_second(lo::Int, hi::Int, k::Int)::Vector{Int}
    out = Int[]
    for n in lo:hi
        n == 2 && continue
        iseven(n) && continue
        m = BigInt(n) - 1
        good = true
        for i in 0:(k-1)
            isprime(m * BigInt(2)^i + 1) || (good = false; break)
        end
        good && push!(out, n)
    end
    sort!(out)
end

println("=== 第二種 ブルートフォース vs search_cc_cpu_2 (小範囲) ===")
for (lo, hi, k) in [(10, 200_000, 5), (1_000_000, 1_020_000, 5), (1, 500_000, 4)]
    exp = brute_second(lo, hi, k)
    got = search_cc_cpu_2(Int64(lo), Int64(hi), k; verbose=false)
    ok(exp == got, "CC$k [$lo,$hi] 一致 (n=$(length(exp)))")
    if exp != got
        println("  CPU-only: ", setdiff(exp, got))
        println("  impl-only: ", setdiff(got, exp))
    end
end

println("=== 第二種 バッチ版 vs search_cc_cpu_2 一致 ===")
for (lo, hi, k) in [(1, 300_000, 5), (1_000_000, 1_030_000, 5)]
    a = search_cc_cpu_2(Int64(lo), Int64(hi), k; verbose=false)
    b = search_cc_cpu_batch_2(Int64(lo), Int64(hi), k; verbose=false)
    ok(a == b, "CC$k [$lo,$hi] batch一致 (n=$(length(a)))")
end

println("=== 第二種 cc_count_cpu_2 直接検証 ===")
# 例: 2 で始まる第二種鎖長2: p=2 -> 2p-1=3 (両方素数) => n=2 は k>=2 で 2,3 は素数だが
# 先頭が偶数の鎖は n=2 のみ特例。ここでは奇数先頭で検証。
for n in [89, 1122659, 19099919]
    # これらは第一種の既知値なので第二種では一概に成り立たない。
    # 代わりに brute で得た先頭値のカウントが k に一致することを確認。
    nothing
end
# 小範囲の全先頭値について、cc_count_cpu_2 が k を返すこと
let k = 5, lo = 1, hi = 100_000
    exp = brute_second(lo, hi, k)
    all_ok = true
    for n in exp
        if cc_count_cpu_2(Int64(n), k) < k; all_ok = false; break; end
    end
    ok(all_ok, "CC$k 先頭値すべてで cc_count_cpu_2 == $k")
end

println("\n結果: PASS=$npass FAIL=$nfail")
exit(nfail == 0 ? 0 : 1)

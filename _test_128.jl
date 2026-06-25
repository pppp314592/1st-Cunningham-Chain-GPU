include("cc_gpu.jl")
using Primes

println("=" ^ 60)
println("テスト1: 既知 CC15 候補の _cc_count128 確認")
println("=" ^ 60)
n = Int128(90616211958465842219)
x_lo = UInt64(n & 0xFFFFFFFFFFFFFFFF)
x_hi = UInt64((n >> 64) & 0xFFFFFFFFFFFFFFFF)
cnt = _cc_count128(x_lo, x_hi, 15)
println("  _cc_count128 = $cnt => $(cnt >= 15 ? "PASS ✅" : "FAIL ❌")")

println("\n" ^ 2)
println("=" ^ 60)
println("テスト2: search_cc_gpu128  CC15候補を含む小範囲")
println("=" ^ 60)
cc15 = Int128(90616211958465842219)
lo = cc15 - 1000
hi = cc15 + 10000
@time res = search_cc_gpu128(lo, hi, 15, verbose=true)
found = cc15 in res
println("  CC15 発見: $found ($(length(res))件) => $(found ? "PASS ✅" : "FAIL ❌")")

println("\n" ^ 2)
println("=" ^ 60)
println("テスト3: 64-bit path リグレッション (件数確認)")
println("=" ^ 60)
@time res64 = search_cc_gpu(1, 100000, 5, verbose=false)
expected_count = 7
ok = length(res64) == expected_count
println("  CC5 [1,100000): $(length(res64))件 (期待 $expected_count) => $(ok ? "PASS ✅" : "FAIL ❌ (got $(res64))")")
println("  実際の結果: $res64")

println("\n" ^ 2)
println("=" ^ 60)
println("テスト4: CPU vs GPU Int128 比較 (CC5 小規模)")
println("=" ^ 60)
function cpu_cc5(lo::Int128, hi::Int128)
    r = Int128[]
    s = lo | 1
    for n in s:2:hi
        x = n
        ok = true
        for _ in 1:5
            isprime(x) || (ok = false; break)
            x = 2x + 1
        end
        ok && push!(r, n)
    end
    return r
end
cpu_res = cpu_cc5(Int128(10)^18, Int128(10)^18 + 50000)
@time gpu_res = search_cc_gpu128(Int128(10)^18, Int128(10)^18 + 50000, 5, verbose=false)
match = cpu_res == gpu_res
println("  CPU: $(length(cpu_res))件 -> $cpu_res")
println("  GPU: $(length(gpu_res))件 -> $gpu_res")
println("  => $(match ? "PASS ✅" : "FAIL ❌")")

println("\n" ^ 2)
println("=" ^ 60)
println("全テスト完了")
println("=" ^ 60)

include("../src/cc_gpu.jl")
using Primes

println("=" ^ 60)
println("繝・せ繝・: 譌｢遏･ CC15 蛟呵｣懊・ _cc_count128 遒ｺ隱・)
println("=" ^ 60)
n = Int128(90616211958465842219)
x_lo = UInt64(n & 0xFFFFFFFFFFFFFFFF)
x_hi = UInt64((n >> 64) & 0xFFFFFFFFFFFFFFFF)
cnt = _cc_count128(x_lo, x_hi, 15)
println("  _cc_count128 = $cnt => $(cnt >= 15 ? "PASS 笨・ : "FAIL 笶・)")

println("\n" ^ 2)
println("=" ^ 60)
println("繝・せ繝・: search_cc_gpu128  CC15蛟呵｣懊ｒ蜷ｫ繧蟆冗ｯ・峇")
println("=" ^ 60)
cc15 = Int128(90616211958465842219)
lo = cc15 - 1000
hi = cc15 + 10000
@time res = search_cc_gpu128(lo, hi, 15, verbose=true)
found = cc15 in res
println("  CC15 逋ｺ隕・ $found ($(length(res))莉ｶ) => $(found ? "PASS 笨・ : "FAIL 笶・)")

println("\n" ^ 2)
println("=" ^ 60)
println("繝・せ繝・: 64-bit path 繝ｪ繧ｰ繝ｬ繝・す繝ｧ繝ｳ (莉ｶ謨ｰ遒ｺ隱・")
println("=" ^ 60)
@time res64 = search_cc_gpu(1, 100000, 5, verbose=false)
expected_count = 7
ok = length(res64) == expected_count
println("  CC5 [1,100000): $(length(res64))莉ｶ (譛溷ｾ・$expected_count) => $(ok ? "PASS 笨・ : "FAIL 笶・(got $(res64))")")
println("  螳滄圀縺ｮ邨先棡: $res64")

println("\n" ^ 2)
println("=" ^ 60)
println("繝・せ繝・: CPU vs GPU Int128 豈碑ｼ・(CC5 蟆剰ｦ乗ｨ｡)")
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
println("  CPU: $(length(cpu_res))莉ｶ -> $cpu_res")
println("  GPU: $(length(gpu_res))莉ｶ -> $gpu_res")
println("  => $(match ? "PASS 笨・ : "FAIL 笶・)")

println("\n" ^ 2)
println("=" ^ 60)
println("蜈ｨ繝・せ繝亥ｮ御ｺ・)
println("=" ^ 60)

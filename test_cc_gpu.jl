include("cc_gpu.jl")
using Primes

println("=" ^ 60)
println("テスト1: 既知のカニンガム鎖の確認 (ハイブリッド GPU+CPU)")
println("=" ^ 60)

function assert_cc(n, target_cc, label)
    r = filter_cc_gpu([n], target_cc)
    ok = length(r) > 0
    println("  $label = $n -> $(ok ? "PASS ✅" : "FAIL ❌")")
    return ok
end

assert_cc(554688278429, 12, "CC12")
assert_cc(4090932431513069, 13, "CC13")
assert_cc(95405042230542329, 14, "CC14")

# 合成数は reject
r = filter_cc_gpu([554688278429 + 2], 1)
println("  合成数 reject: $(length(r) == 0 ? "PASS ✅" : "FAIL ❌")")

println("\n" ^ 2)
println("=" ^ 60)
println("テスト2: CPU (Primes.jl) と GPU の結果比較 (小規模 CC5)")
println("=" ^ 60)

function cpu_search_cc(lo::Int, hi::Int, target_cc::Int)::Vector{Int}
    results = Int[]
    start = (lo | 1)
    for n in start:2:hi
        x = Int128(n)
        ok = true
        for _ in 1:target_cc
            if !isprime(x)
                ok = false
                break
            end
            x = 2x + 1
        end
        ok && push!(results, n)
    end
    return results
end

for (lo, hi, cc) in [(1, 100_000, 5), (1_000_000, 1_010_000, 5)]
    println("\n範囲: [$lo, $hi), CC$cc")
    cpu_res = cpu_search_cc(lo, hi, cc)
    gpu_res = search_cc_gpu(lo, hi, cc, verbose=false)
    match = (cpu_res == gpu_res)
    println("  CPU: $(length(cpu_res)) 件, GPU: $(length(gpu_res)) 件, 一致: $(match ? "✅" : "❌")")
    if !match
        println("  CPU only: $(setdiff(cpu_res, gpu_res))")
        println("  GPU only: $(setdiff(gpu_res, cpu_res))")
    end
end

println("\n" ^ 2)
println("=" ^ 60)
println("テスト3: GPU フィルター性能 (中規模 CC5)")
println("=" ^ 60)

lo, hi, cc = 1, 500_000, 5
println("範囲: [$lo, $hi), CC$cc")

t = @elapsed r = search_cc_gpu(lo, hi, cc, verbose=false)
println("  結果: $(length(r)) 件, $(round(t, digits=2))s")

println("\n" ^ 2)
println("=" ^ 60)
println("テスト4: CPU のみ vs 篩+GPU (中規模)")
println("=" ^ 60)

lo, hi, cc = 1, 1_000_000, 5
t_cpu = @elapsed cpu_res = cpu_search_cc(lo, hi, cc)
t_gpu = @elapsed gpu_res = search_cc_gpu(lo, hi, cc, verbose=false)
println("  CPU: $(length(cpu_res)) 件, $(round(t_cpu, digits=2))s")
println("  GPU: $(length(gpu_res)) 件, $(round(t_gpu, digits=2))s")
println("  加速比: $(round(t_cpu / t_gpu, digits=1))x")
println("  一致: $(cpu_res == gpu_res ? "✅" : "❌")")

include("src/cc_cpu.jl")

# 正しさ確認
println("=== 正しさ確認 ===")
for (lo, hi, cc) in [(Int64(1), Int64(100000), 5), (Int64(1), Int64(500000), 5)]
    r1 = search_cc_cpu(lo, hi, cc; verbose=false)
    r2 = search_cc_cpu_fast(lo, hi, cc; verbose=false)
    ok = r1 == r2
    println("CC$cc [$lo, $hi): $(length(r1)) items, match=$(ok)")
end

# 速度比較
println("\n=== 速度比較 CC13 ===")
lo, hi, cc = Int64(10)^13, Int64(10)^13 + Int64(10)^12, 13

# warmup
search_cc_cpu(lo, hi, cc; verbose=false)
search_cc_cpu_fast(lo, hi, cc; verbose=false)

for N in [10, 20, 30, 40, 50, 0]
    t = @elapsed search_cc_cpu_fast(lo, hi, cc; N=N, verbose=false)
    println("  fast N=$N: $(round(t, digits=3))s")
end

t = @elapsed search_cc_cpu(lo, hi, cc; verbose=false)
println("  usual:    $(round(t, digits=3))s")

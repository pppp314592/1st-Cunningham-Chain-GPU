include("cc_gpu.jl")

println("=" ^ 60)
println("GPU vs CPU (notebook) カニンガム鎖探索 比較")
println("対象: CC13, 10^13 〜 10^15")
println("=" ^ 60)

ranges = [
    (10^13, 10^14, "10^13 〜 10^14 (14桁)"),
    (10^14, 10^15, "10^14 〜 10^15 (15桁)"),
    (4*10^15, 5*10^15, "4e15 〜 5e15 (CC13発見範囲)"),
]

global all_results = Int[]
global total_time = 0.0

for (lo, hi, label) in ranges
    println("\n--- $label ---")
    t = @elapsed res = search_cc_gpu(lo, hi, 13, verbose=false)
    global all_results = vcat(all_results, res)
    global total_time += t
    println("  GPU: $(length(res)) 件, ⏱ $(round(t, digits=2))s")
end

println("\n" ^ 2)
println("=" ^ 60)
println("結果サマリ")
println("=" ^ 60)
println("対象: CC13, 10^13 〜 5×10^15")
println("総GPU時間: $(round(total_time, digits=2))s ($(round(total_time/60, digits=1))分)")
println("発見 CC13+: $(length(all_results)) 件")
if !isempty(all_results)
    for r in sort(all_results)
        println("  CC13: $r")
    end
end
println()
println("参考: notebook (CPU i5-3330, 4T)")
println("  10^13〜10^14:  14.59秒")
println("  10^14〜10^15: 148.91秒")
println("  4e15〜5e15:   163.19秒 → CC13: 4090932431513069")
println("  (上記合計: 約 327秒)")
println("=" ^ 60)

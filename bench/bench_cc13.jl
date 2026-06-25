include("../src/cc_gpu.jl")

println("=" ^ 60)
println("GPU vs CPU (notebook) カニンガム鎖探索 比輁E)
println("対象: CC13, 10^13 、E10^15")
println("=" ^ 60)

ranges = [
    (10^13, 10^14, "10^13 、E10^14 (14桁E"),
    (10^14, 10^15, "10^14 、E10^15 (15桁E"),
    (4*10^15, 5*10^15, "4e15 、E5e15 (CC13発見篁E��)"),
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
println("対象: CC13, 10^13 、E5ÁE0^15")
println("総GPU時間: $(round(total_time, digits=2))s ($(round(total_time/60, digits=1))刁E")
println("発要ECC13+: $(length(all_results)) 件")
if !isempty(all_results)
    for r in sort(all_results)
        println("  CC13: $r")
    end
end
println()
println("参老E notebook (CPU i5-3330, 4T)")
println("  10^13、E0^14:  14.59私E)
println("  10^14、E0^15: 148.91私E)
println("  4e15、Ee15:   163.19私EↁECC13: 4090932431513069")
println("  (上記合訁E 紁E327私E")
println("=" ^ 60)

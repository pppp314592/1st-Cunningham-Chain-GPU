include("src/cc_gpu.jl")        # 旧: search_cc_gpu
include("src/cc_gpu_wheel.jl")  # 新: search_cc_gpu_wheel

lo = Int64(10); hi = Int64(10)^16; k = 15

# warmup（コンパイル除外）
search_cc_gpu(lo, lo+10^12, k; verbose=false)
search_cc_gpu_wheel(lo, lo+10^12, k; verbose=false)

println(">>> 旧 cc_gpu.jl (search_cc_gpu) CC$k [10,1e16]")
t_old = @elapsed r_old = search_cc_gpu(lo, hi, k; verbose=true)
println(">>> 新 cc_gpu_wheel.jl (search_cc_gpu_wheel) CC$k [10,1e16]")
t_new = @elapsed r_new = search_cc_gpu_wheel(lo, hi, k; verbose=true)

println("\n===== 結果 =====")
println("旧 search_cc_gpu      : $(round(t_old,digits=1))s  found=$(length(r_old))")
println("新 search_cc_gpu_wheel: $(round(t_new,digits=1))s  found=$(length(r_new))")
println("速度比 (旧/新) = $(round(t_old/t_new,digits=2))x  (>1なら新が速い)")

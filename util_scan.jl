include("src/cc_gpu.jl")
include("src/cc_gpu_wheel.jl")
# 30〜60秒程度で終わる範囲を1回スキャン（この間のGPU使用率を外部から観測）
search_cc_gpu_wheel(Int64(10), Int64(10)^12, 15; verbose=false)  # warmup
open("logs/util_scan_ready.txt","w") do io; println(io,"ready"); end
t = @elapsed r = search_cc_gpu_wheel(Int64(10), Int64(10)^16, 15; verbose=true)
println("SCAN_DONE t=$(round(t,digits=1))s found=$(length(r))")

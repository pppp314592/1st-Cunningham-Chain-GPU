include("src/cc_gpu.jl"); include("src/cc_gpu_wheel.jl")
# setup一回、scanのみ計測(64bit vs 128bit)で純オーバヘッドを測る
k=15
st = gpu_wheel_setup(k; verbose=false)
base = Int64(10)^17
span = Int64(3)*10^15  # ~206 cycles
lo64=base; hi64=base+span
# warmup
gpu_wheel_scan!(st, lo64, lo64+Int64(10)^13)
gpu_wheel_scan128!(st, Int128(lo64), Int128(lo64)+Int128(10)^13)
t64 = @elapsed r64 = gpu_wheel_scan!(st, lo64, hi64)
t128 = @elapsed r128 = gpu_wheel_scan128!(st, Int128(lo64), Int128(hi64))
ncyc = span ÷ st.wheel
println("span=$span cycles≈$ncyc  R=$(st.R)")
println("64bit  scan: $(round(t64,digits=3))s  ($(round(t64/ncyc*1000,digits=2)) ms/cycle) found=$(length(r64))")
println("128bit scan: $(round(t128,digits=3))s  ($(round(t128/ncyc*1000,digits=2)) ms/cycle) found=$(length(r128))")
println("128bit overhead = $(round(t128/t64,digits=2))x")

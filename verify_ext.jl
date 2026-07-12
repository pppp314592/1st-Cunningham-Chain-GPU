include("src/cc_gpu.jl"); include("src/cc_gpu_wheel.jl")
target = parse(Int128,"810433818265726529159")
lo = target - Int128(3)*10^15; hi = target + Int128(3)*10^15
tsetup = @elapsed st = gpu_wheel_setup(16; wl=_cc_wl_ext(16), verbose=true)
println("setup(拡張wl 29追加) = $(round(tsetup,digits=1))s  R=$(st.R) wheel=$(st.wheel)")
r = gpu_wheel_scan128!(st, lo, hi)
println("found=$r")
println(target in r ? "✔ 拡張wheelでも最小CC16を正しく検出 (正当性OK)" : "�’EFAIL")
CUDA.unsafe_free!(st.d_out)

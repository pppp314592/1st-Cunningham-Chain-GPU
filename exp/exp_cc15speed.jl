include("src/cc_gpu.jl"); include("src/cc_gpu_wheel.jl")
k=15; base=Int128(10)^20
st = gpu_wheel_setup(k; wl=_cc_wl_ext(k), verbose=false)
gpu_wheel_scan128!(st, base, base+Int128(st.wheel))   # warmup
ncy=12
t = @elapsed gpu_wheel_scan128!(st, base, base+Int128(st.wheel)*ncy)
spr = t/ncy/st.wheel                     # s per unit range
println("CC15 拡張wheel: wheel=$(st.wheel) R=$(st.R)")
println("  $(round(t/ncy*1000,digits=1)) ms/cycle,  $(round(spr*1e18,digits=0)) s/1e18range")
for h in [1,2]  # 1h,2h でスキャンできる幅
    w = h*3600/spr
    println("  $(h)時間で走査可能な幅 ≈ $(round(w/1e18,digits=2)) e18")
end
CUDA.unsafe_free!(st.d_out)

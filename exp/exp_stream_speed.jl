include("src/cc_cpu.jl")
include("src/cc_gpu_wheel.jl")
using CUDA

# 同一レンジで materialize拡張wheel(29) vs stream(31,47) の実測比較 (CC16スケール)
k = 16
base = Int128(9) * Int128(10)^20
rng  = Int128(5) * Int128(10)^16          # 固定レンジ幅
lo = base; hi = base + rng
println("=== CC$k throughput: range width=$rng near $base ===")

# --- warmup (コンパイル) ---
let l = Int128(10)^12, h = l + Int128(10)^11
    search_cc_gpu_wheel_stream128(l, h, 13; verbose=false)
    st0 = gpu_wheel_setup(13; wl=_cc_wl_ext(13), verbose=false)
    gpu_wheel_scan128!(st0, l, h); CUDA.unsafe_free!(st0.d_out)
end

# --- materialize 拡張wheel (29追加) ---
ste = gpu_wheel_setup(k; wl=_cc_wl_ext(k), verbose=true)
t1 = time(); re = gpu_wheel_scan128!(ste, lo, hi); te = time()-t1
CUDA.unsafe_free!(ste.d_out)
println("  materialize-ext(29): $(round(te,digits=2))s  found=$(length(re))")

# --- stream (31,47追加) ---
sts = gpu_wheel_stream_setup(k; verbose=true)
t2 = time(); rs = gpu_wheel_scan_stream128!(sts, lo, hi); ts = time()-t2
println("  stream(31,47):       $(round(ts,digits=2))s  found=$(length(rs))")

println("  speedup stream/ext = $(round(te/ts, digits=3))x")
println("  ext  rate = $(round(Float64(rng)/te/1e12, digits=3)) e12/s")
println("  strm rate = $(round(Float64(rng)/ts/1e12, digits=3)) e12/s")

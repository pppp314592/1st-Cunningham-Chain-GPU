# 新規CC16探索 (拡張wheel[29追加] + Int128頭 + 進捗ログ)
#   julia -g0 -t 12 run_cc16_scan.jl <start> <width> [k]
# 例: julia -g0 -t 12 run_cc16_scan.jl 820000000000000000000 100000000000000000000
#   → [8.2e20, 9.2e20) を走査 (~31時間, CC16期待値≈0.9)。
# 見つかった鎖先頭は logs/cc16_found.log に追記保存。
include("src/cc_gpu.jl"); include("src/cc_gpu_wheel.jl")

start = length(ARGS) >= 1 ? parse(Int128, ARGS[1]) : parse(Int128, "820000000000000000000")
width = length(ARGS) >= 2 ? parse(Int128, ARGS[2]) : parse(Int128, "10000000000000000000")  # 1e19
k     = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 16
lo = start; hi = start + width

println("=== CC$k 新規探索 ===")
println("range = [$lo, $hi)  width=$width")
st = gpu_wheel_setup(k; wl = _cc_wl_ext(k), verbose = true)
println("wheel=$(st.wheel) R=$(st.R) primes=$(st.nprimes)  cycles≈$(width ÷ st.wheel)")

t0 = time()
r = gpu_wheel_scan128!(st, lo, hi; progress = true)
dt = time() - t0

open("logs/cc16_found.log", "a") do io
    println(io, "# scan [$lo,$hi) k=$k  $(round(dt,digits=0))s  found=$(length(r))  $(Libc.strftime(time()))")
    for x in r
        println(io, x)
    end
end
println("=== 完了: $(length(r)) 本 in $(round(dt/3600,digits=2))h ===")
for x in r; println("  CC$k head = $x"); end
CUDA.unsafe_free!(st.d_out)

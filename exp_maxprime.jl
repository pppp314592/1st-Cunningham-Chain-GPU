include("src/cc_gpu.jl")
include("src/cc_gpu_wheel.jl")

# 固定区間でのピュアscan時間をmax_primeについて掃引（setup除外）。
# 1プロセスで全計測 → 起動コスト1回。
function sweep(k::Int, lo::Int64, width::Int64, mps::Vector{Int})
    hi = lo + width
    println("\n### CC$k  range=[$lo, +$width]  cycles≈$(width ÷ Int64(first(mp_wheel(k))))")
    println(rpad("max_prime",12), rpad("scan_s",10), rpad("found",8), "R(residues)")
    base = nothing
    for mp in mps
        st = gpu_wheel_setup(k; max_prime=mp, verbose=false)
        gpu_wheel_scan!(st, lo, lo + width÷20)          # warmup（小）
        t = @elapsed r = gpu_wheel_scan!(st, lo, hi)
        base === nothing && (base = t)
        println(rpad(mp,12), rpad(round(t,digits=3),10), rpad(length(r),8), st.R)
        CUDA.unsafe_free!(st.d_out)
    end
end
mp_wheel(k) = [prod(_cc_wl(k))]  # wheel値（cycles概算用）

# ~2-4sで終わる区間を選定
sweep(13, Int64(10)^15, Int64(3)*10^14, [2000,5000,10000,20000,40000])
sweep(15, Int64(10)^16, Int64(2)*10^15, [1000,2000,5000,10000,20000])
println("\nDONE")

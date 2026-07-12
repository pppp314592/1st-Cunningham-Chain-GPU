# exp_occupancy.jl — 起動構成 (threads/block) スイープ
# 篩支配レジーム(CC16拡張wheel, 大レンジ)で threads/block を変えて実走査カーネルを
# 直接起動し、篩時間を比較する。ソースは変更せず state を再利用。
#   julia -g0 -t 12 exp_occupancy.jl
include("src/cc_cpu.jl")
include("src/cc_gpu_wheel.jl")
using CUDA

k = 16
st = gpu_wheel_setup(k; wl=_cc_wl_ext(k), verbose=true)
w = Int128(st.wheel)

# 既知最小CC16 8.1e20 付近, レンジ 2e16 (篩支配)
center = Int128(810433818265726529159)
lo = center - Int128(10_000_000_000_000_000)
hi = center + Int128(10_000_000_000_000_000)
k_start = Int64(lo ÷ w)
k_end   = Int64((hi - Int128(1)) ÷ w)
total_cyc = k_end - k_start + 1
println("range cyc=$total_cyc R=$(st.R) total_work=$(total_cyc*st.R)")

cap = 1<<12
d_out = CUDA.zeros(Int128, cap)
d_cnt = CUDA.zeros(Int32, 1)

function run_scan(threads::Int)
    fill!(d_cnt, Int32(0))
    work_tile = 1<<25
    cyc_per = max(1, work_tile ÷ Int(st.R))
    cyc = k_start
    while cyc <= k_end
        ncyc = min(cyc_per, k_end - cyc + 1)
        work = ncyc * st.R
        blocks = cld(work, threads)
        @cuda threads=threads blocks=blocks _cc_wheel_kernel128!(
            d_out, d_cnt, Int32(cap), st.d_wheel_n, st.R, st.wheel, cyc, ncyc,
            lo, hi, st.d_primes, st.d_wheel_mod, st.d_pow20, st.d_mu,
            st.d_bad_off, st.d_bad, st.nprimes, Int32(st.k))
        cyc += ncyc
    end
    CUDA.synchronize()
    return Array(d_cnt)[1]
end

# ウォームアップ (JIT)
run_scan(256)

for th in (32, 48, 64, 80, 96, 128)
    # 2回計測して安定値
    run_scan(th)
    t = CUDA.@elapsed run_scan(th)
    cnt = Array(d_cnt)[1]
    println("threads=$(lpad(th,4))  time=$(round(t,digits=3))s  found=$cnt")
end

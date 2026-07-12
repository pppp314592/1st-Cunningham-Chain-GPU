# exp_split.jl — 篩カーネルとMRカーネルを分離 (hot/cold path split)
# 仮説: MR判定(128bit, is_prime_mr*)コードが全スレッドのレジスタを占有し、篩(98.4%)の
#   占有率を下げている。篩のみの軽量カーネルにすれば占有率↑で篩が加速するはず。
# 方式:
#   Kernel A (sieve): 生存候補 n を d_surv に atomic 追記 (MRコードを含まない)。
#   Kernel B (mr):    d_surv[1..cnt] のみ鎖MR → d_out。生存~1.4e-5%なので極小。
#   julia -g0 -t 12 exp_split.jl
include("src/cc_cpu.jl")
include("src/cc_gpu_wheel.jl")
using CUDA

# --- 篩のみカーネル (生存 n を d_surv に追記) ---
function kern_sieve!(d_surv::CuDeviceVector{Int128}, scnt::CuDeviceVector{Int32}, scap::Int32,
        d_wheel_n, R::Int64, wheel::Int64, k_base::Int64, ncyc::Int64,
        lo::Int128, hi::Int128, d_primes, d_wheel_mod, d_pow20, d_mu,
        d_bad_off, d_bad, nprimes::Int32)
    g = (blockIdx().x - Int64(1)) * blockDim().x + threadIdx().x
    total = ncyc * R
    if g <= total
        cyc = k_base + (g - Int64(1)) ÷ R
        ridx = (g - Int64(1)) % R + Int64(1)
        @inbounds r = d_wheel_n[ridx]
        n = Int128(cyc) * Int128(wheel) + Int128(r)
        if n > lo && n < hi
            r_hi = UInt32((r >> 20) & 0x7FFFFFFF)
            r_lo = UInt32(r & 0x00000000000FFFFF)
            cyc32 = UInt32(cyc & 0x7FFFFFFF)
            ok = true
            @inbounds for pi in 1:nprimes
                p = UInt32(d_primes[pi]); mu = d_mu[pi]
                rhp = bmod(r_hi, p, mu)
                rp = bmod(rhp * UInt32(d_pow20[pi]) + r_lo, p, mu)
                cp = bmod(cyc32, p, mu)
                np = bmod(cp * UInt32(d_wheel_mod[pi]) + rp, p, mu)
                if d_bad[d_bad_off[pi] + Int64(np) + Int64(1)] != 0x00
                    ok = false; break
                end
            end
            if ok
                idx = CUDA.atomic_add!(pointer(scnt, 1), Int32(1)) + Int32(1)
                idx <= scap && (@inbounds d_surv[idx] = n)
            end
        end
    end
    return nothing
end

# --- MRカーネル (生存候補のみ, nsurv をデバイスから読む: ホスト同期不要) ---
function kern_mr!(d_surv::CuDeviceVector{Int128}, d_scnt::CuDeviceVector{Int32},
        out::CuDeviceVector{Int128}, counter::CuDeviceVector{Int32}, cap::Int32, kk::Int32)
    i = (blockIdx().x - Int64(1)) * blockDim().x + threadIdx().x
    @inbounds nsurv = d_scnt[1]
    if i <= nsurv
        @inbounds n = d_surv[i]
        x_lo = UInt64(n & Int128(0xFFFFFFFFFFFFFFFF))
        x_hi = UInt64((n >> 64) & Int128(0xFFFFFFFFFFFFFFFF))
        good = true
        @inbounds for _ in 1:kk
            isp = if x_hi == UInt64(0) && x_lo <= 0x7FFFFFFFFFFFFFFF
                is_prime_mr(Int64(x_lo))
            else
                (x_hi > 0x7FFFFFFFFFFFFFFF) ? false : is_prime_mr128(x_lo, x_hi)
            end
            isp || (good = false; break)
            carry = (x_lo > 0x7FFFFFFFFFFFFFFF) ? UInt64(1) : UInt64(0)
            x_lo = (x_lo << 1) | UInt64(1)
            x_hi = (x_hi << 1) | carry
        end
        if good
            idx = CUDA.atomic_add!(pointer(counter, 1), Int32(1)) + Int32(1)
            idx <= cap && (@inbounds out[idx] = n)
        end
    end
    return nothing
end

k = 16
st = gpu_wheel_setup(k; wl=_cc_wl_ext(k), verbose=true)
w = Int128(st.wheel)
center = Int128(810433818265726529159)
lo = center - Int128(10_000_000_000_000_000)
hi = center + Int128(10_000_000_000_000_000)
k_start = Int64(lo ÷ w); k_end = Int64((hi - Int128(1)) ÷ w)

cap = 1<<12
d_out = CUDA.zeros(Int128, cap); d_cnt = CUDA.zeros(Int32, 1)
scap = 1<<18
d_surv = CUDA.zeros(Int128, scap); d_scnt = CUDA.zeros(Int32, 1)

# レジスタ数比較
kf = @cuda launch=false _cc_wheel_kernel128!(d_out,d_cnt,Int32(cap),st.d_wheel_n,st.R,st.wheel,k_start,Int64(1),lo,hi,st.d_primes,st.d_wheel_mod,st.d_pow20,st.d_mu,st.d_bad_off,st.d_bad,st.nprimes,Int32(k))
ks = @cuda launch=false kern_sieve!(d_surv,d_scnt,Int32(scap),st.d_wheel_n,st.R,st.wheel,k_start,Int64(1),lo,hi,st.d_primes,st.d_wheel_mod,st.d_pow20,st.d_mu,st.d_bad_off,st.d_bad,st.nprimes)
println("registers: fused=$(CUDA.registers(kf))  sieve-only=$(CUDA.registers(ks))")

function run_fused(threads)
    fill!(d_cnt, Int32(0)); wt=1<<25; cp=max(1,wt÷Int(st.R)); cyc=k_start
    while cyc <= k_end
        ncyc=min(cp,k_end-cyc+1); work=ncyc*st.R; blocks=cld(work,threads)
        @cuda threads=threads blocks=blocks _cc_wheel_kernel128!(
            d_out,d_cnt,Int32(cap),st.d_wheel_n,st.R,st.wheel,cyc,ncyc,lo,hi,
            st.d_primes,st.d_wheel_mod,st.d_pow20,st.d_mu,st.d_bad_off,st.d_bad,st.nprimes,Int32(st.k))
        cyc+=ncyc
    end
    CUDA.synchronize(); Array(d_cnt)[1]
end
function run_split(sth, mth; drain_every=1_000_000)   # 蓄積 → 稀に MR ドレイン
    fill!(d_cnt, Int32(0)); fill!(d_scnt, Int32(0))
    mr_blocks = cld(scap, mth)
    wt=1<<25; cp=max(1,wt÷Int(st.R)); cyc=k_start; tile=0
    while cyc <= k_end
        ncyc=min(cp,k_end-cyc+1); work=ncyc*st.R; blocks=cld(work,sth)
        @cuda threads=sth blocks=blocks kern_sieve!(
            d_surv,d_scnt,Int32(scap),st.d_wheel_n,st.R,st.wheel,cyc,ncyc,lo,hi,
            st.d_primes,st.d_wheel_mod,st.d_pow20,st.d_mu,st.d_bad_off,st.d_bad,st.nprimes)
        tile+=1
        if tile % drain_every == 0
            @cuda threads=mth blocks=mr_blocks kern_mr!(d_surv, d_scnt, d_out, d_cnt, Int32(cap), Int32(st.k))
            fill!(d_scnt, Int32(0))
        end
        cyc+=ncyc
    end
    # 末尾ドレイン
    @cuda threads=mth blocks=mr_blocks kern_mr!(d_surv, d_scnt, d_out, d_cnt, Int32(cap), Int32(st.k))
    CUDA.synchronize(); (Array(d_cnt)[1], -1)
end

run_fused(64); run_split(64,64)  # warm
for th in (64, 96, 128)
    tf = CUDA.@elapsed (cf = run_fused(th))
    ts = CUDA.@elapsed (cs = run_split(th, 64))  # 末尾ドレインのみ
    println("threads=$th  fused=$(round(tf,digits=3))s(f=$cf)  split=$(round(ts,digits=3))s(f=$(cs[1]))  speedup=$(round(tf/ts,digits=3))x")
end

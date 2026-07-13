# 周期(cyc)ごとに (cyc*wheel) % p を事前計算し R 残基で共有する篩を、
# 現行 Barrett 篩と比較する。cyc 依存項(cp, cp*wheel_mod)を候補ループから除去できるか検証。
include("src/cc_cpu.jl")
include("src/cc_gpu_wheel.jl")
using CUDA

@inline function bmod(a::UInt32, p::UInt32, mu::UInt32)::UInt32
    q = UInt32((UInt64(a) * UInt64(mu)) >> 32)
    r = a - q * p
    r >= p && (r -= p)
    r >= p && (r -= p)
    return r
end

# 事前計算: cycbase[ci*nprimes + pi] = (cyc*wheel) % p
function _precompute_cycbase!(cycbase::CuDeviceVector{UInt32}, k_base::Int64, ncyc::Int64,
        d_primes::CuDeviceVector{Int32}, d_wheel_mod::CuDeviceVector{Int32},
        d_mu::CuDeviceVector{UInt32}, nprimes::Int32)
    idx = (blockIdx().x - Int64(1)) * blockDim().x + threadIdx().x
    total = ncyc * Int64(nprimes)
    if idx <= total
        ci = (idx - Int64(1)) ÷ Int64(nprimes)
        pi = (idx - Int64(1)) % Int64(nprimes) + Int64(1)
        cyc = k_base + ci
        @inbounds begin
            p = UInt32(d_primes[pi]); mu = d_mu[pi]
            cp = bmod(UInt32(cyc & 0x7FFFFFFF), p, mu)
            cycbase[ci * Int64(nprimes) + pi] = bmod(cp * UInt32(d_wheel_mod[pi]), p, mu)
        end
    end
    return nothing
end

# 事前計算 cycbase を使う篩 (候補ループは rp と最終 bmod のみ)
function _sieve_kernel_cycbase!(sieve_cnt::CuDeviceVector{Int64},
        d_wheel_n::CuDeviceVector{Int64}, R::Int64,
        wheel::Int64, k_base::Int64, ncyc::Int64, lo::Int128, hi::Int128,
        d_primes::CuDeviceVector{Int32}, d_pow20::CuDeviceVector{Int32},
        d_mu::CuDeviceVector{UInt32}, cycbase::CuDeviceVector{UInt32},
        d_bad_off::CuDeviceVector{Int32}, d_bad::CuDeviceVector{UInt8}, nprimes::Int32)
    g = (blockIdx().x - Int64(1)) * blockDim().x + threadIdx().x
    total = ncyc * R
    if g <= total
        ci = (g - Int64(1)) ÷ R
        cyc = k_base + ci
        ridx = (g - Int64(1)) % R + Int64(1)
        @inbounds r = d_wheel_n[ridx]
        n = Int128(cyc) * Int128(wheel) + Int128(r)
        if n > lo && n < hi
            r_hi = UInt32((r >> 20) & 0x7FFFFFFF)
            r_lo = UInt32(r & 0x00000000000FFFFF)
            base = ci * Int64(nprimes)
            ok = true
            @inbounds for pi in 1:nprimes
                p = UInt32(d_primes[pi]); mu = d_mu[pi]
                rhp = bmod(r_hi, p, mu)
                rp = bmod(rhp * UInt32(d_pow20[pi]) + r_lo, p, mu)
                np = bmod(cycbase[base + pi] + rp, p, mu)
                if d_bad[d_bad_off[pi] + Int64(np) + Int64(1)] != 0x00
                    ok = false; break
                end
            end
            ok && CUDA.atomic_add!(pointer(sieve_cnt, 1), Int64(1))
        end
    end
    return nothing
end

# 現行 Barrett 篩 (基準)
function _sieve_kernel_barrett!(sieve_cnt::CuDeviceVector{Int64},
        d_wheel_n::CuDeviceVector{Int64}, R::Int64,
        wheel::Int64, k_base::Int64, ncyc::Int64, lo::Int128, hi::Int128,
        d_primes::CuDeviceVector{Int32}, d_wheel_mod::CuDeviceVector{Int32},
        d_pow20::CuDeviceVector{Int32}, d_mu::CuDeviceVector{UInt32},
        d_bad_off::CuDeviceVector{Int32}, d_bad::CuDeviceVector{UInt8}, nprimes::Int32)
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
            ok && CUDA.atomic_add!(pointer(sieve_cnt, 1), Int64(1))
        end
    end
    return nothing
end

function run_barrett(st, lo, hi, d_mu; work_tile=1<<25)
    w=Int128(st.wheel); k_start=Int64(lo÷w); k_end=Int64((hi-Int128(1))÷w)
    d_sv=CUDA.zeros(Int64,1); cyc_per=max(1,work_tile÷Int(st.R)); threads=256; cyc=k_start
    while cyc<=k_end
        ncyc=min(cyc_per,k_end-cyc+1); work=ncyc*st.R
        @cuda threads=threads blocks=cld(work,threads) _sieve_kernel_barrett!(
            d_sv, st.d_wheel_n, st.R, st.wheel, cyc, ncyc, lo, hi,
            st.d_primes, st.d_wheel_mod, st.d_pow20, d_mu, st.d_bad_off, st.d_bad, st.nprimes)
        cyc+=ncyc
    end
    CUDA.synchronize(); sv=Array(d_sv)[1]; CUDA.unsafe_free!(d_sv); return sv
end

function run_cycbase(st, lo, hi, d_mu; work_tile=1<<25)
    w=Int128(st.wheel); k_start=Int64(lo÷w); k_end=Int64((hi-Int128(1))÷w)
    d_sv=CUDA.zeros(Int64,1); cyc_per=max(1,work_tile÷Int(st.R)); threads=256; cyc=k_start
    np=Int(st.nprimes)
    d_cb=CuArray{UInt32}(undef, cyc_per*np)
    while cyc<=k_end
        ncyc=min(cyc_per,k_end-cyc+1); work=ncyc*st.R
        pc=ncyc*Int64(np)
        @cuda threads=threads blocks=cld(pc,threads) _precompute_cycbase!(
            d_cb, cyc, ncyc, st.d_primes, st.d_wheel_mod, d_mu, st.nprimes)
        @cuda threads=threads blocks=cld(work,threads) _sieve_kernel_cycbase!(
            d_sv, st.d_wheel_n, st.R, st.wheel, cyc, ncyc, lo, hi,
            st.d_primes, st.d_pow20, d_mu, d_cb, st.d_bad_off, st.d_bad, st.nprimes)
        cyc+=ncyc
    end
    CUDA.synchronize(); sv=Array(d_sv)[1]; CUDA.unsafe_free!(d_sv); CUDA.unsafe_free!(d_cb); return sv
end

k=16
st=gpu_wheel_setup(k; wl=_cc_wl_ext(k), verbose=true)
d_mu=CuArray(UInt32.(fld.(UInt64(2)^32, UInt64.(Array(st.d_primes)))))
lo=Int128(9)*Int128(10)^20; hi=lo+Int128(2)*Int128(10)^16
# warmup
run_barrett(st, Int128(10)^12, Int128(10)^12+Int128(10)^11, d_mu)
run_cycbase(st, Int128(10)^12, Int128(10)^12+Int128(10)^11, d_mu)

println("=== CC$k sieve: Barrett版 vs cyc事前計算版 (range=2e16) ===")
t=time(); s1=run_barrett(st, lo, hi, d_mu); tb=time()-t
t=time(); s2=run_cycbase(st, lo, hi, d_mu); tc=time()-t
println("  Barrett版     : $(round(tb,digits=2))s  survivors=$s1")
println("  cyc事前計算版 : $(round(tc,digits=2))s  survivors=$s2")
println("  一致: $(s1==s2)   speedup = $(round(tb/tc,digits=3))x")
CUDA.unsafe_free!(st.d_out)

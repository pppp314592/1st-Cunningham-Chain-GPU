# Barrett還元で篩の % p を除算レス化し、篩時間を比較する
include("src/cc_cpu.jl")
include("src/cc_gpu_wheel.jl")
using CUDA

# a % p を Barrett還元で (a < 2^31, p < 46340, mu = floor(2^32/p))
@inline function bmod(a::UInt32, p::UInt32, mu::UInt32)::UInt32
    q = UInt32((UInt64(a) * UInt64(mu)) >> 32)
    r = a - q * p
    r >= p && (r -= p)
    r >= p && (r -= p)
    return r
end

# Barrett版 篩カーネル (MRは付けない=篩時間の純比較)。sieve生存数のみ数える。
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

# 元(除算)版 篩カーネル (比較基準)
function _sieve_kernel_div!(sieve_cnt::CuDeviceVector{Int64},
        d_wheel_n::CuDeviceVector{Int64}, R::Int64,
        wheel::Int64, k_base::Int64, ncyc::Int64, lo::Int128, hi::Int128,
        d_primes::CuDeviceVector{Int32}, d_wheel_mod::CuDeviceVector{Int32},
        d_pow20::CuDeviceVector{Int32}, d_bad_off::CuDeviceVector{Int32},
        d_bad::CuDeviceVector{UInt8}, nprimes::Int32)
    g = (blockIdx().x - Int64(1)) * blockDim().x + threadIdx().x
    total = ncyc * R
    if g <= total
        cyc = k_base + (g - Int64(1)) ÷ R
        ridx = (g - Int64(1)) % R + Int64(1)
        @inbounds r = d_wheel_n[ridx]
        n = Int128(cyc) * Int128(wheel) + Int128(r)
        if n > lo && n < hi
            r_hi = Int32((r >> 20) & 0x7FFFFFFF)
            r_lo = Int32(r & 0x00000000000FFFFF)
            cyc32 = Int32(cyc)
            ok = true
            @inbounds for pi in 1:nprimes
                p = d_primes[pi]
                rp = (((r_hi % p) * d_pow20[pi]) % p + r_lo) % p
                cp = cyc32 % p
                np = (cp * d_wheel_mod[pi] + rp) % p
                if d_bad[d_bad_off[pi] + Int64(np) + Int64(1)] != 0x00
                    ok = false; break
                end
            end
            ok && CUDA.atomic_add!(pointer(sieve_cnt, 1), Int64(1))
        end
    end
    return nothing
end

function run_scan(kern, st, lo, hi, extra...; work_tile=1<<25)
    w = Int128(st.wheel); k_start=Int64(lo÷w); k_end=Int64((hi-Int128(1))÷w)
    d_sv = CUDA.zeros(Int64,1); cyc_per=max(1,work_tile÷Int(st.R)); threads=256
    cyc=k_start
    while cyc <= k_end
        ncyc=min(cyc_per,k_end-cyc+1); work=ncyc*st.R
        @cuda threads=threads blocks=cld(work,threads) kern(
            d_sv, st.d_wheel_n, st.R, st.wheel, cyc, ncyc, lo, hi,
            st.d_primes, st.d_wheel_mod, st.d_pow20, extra...,
            st.d_bad_off, st.d_bad, st.nprimes)
        cyc += ncyc
    end
    CUDA.synchronize(); sv=Array(d_sv)[1]; CUDA.unsafe_free!(d_sv); return sv
end

k = 16
st = gpu_wheel_setup(k; wl=_cc_wl_ext(k), verbose=true)
mu = UInt32.(fld.(UInt64(2)^32, UInt64.(Array(st.d_primes))))
d_mu = CuArray(mu)

lo = Int128(9)*Int128(10)^20; hi = lo + Int128(2)*Int128(10)^16
# warmup
run_scan(_sieve_kernel_div!, st, Int128(10)^12, Int128(10)^12+Int128(10)^11)
run_scan(_sieve_kernel_barrett!, st, Int128(10)^12, Int128(10)^12+Int128(10)^11, d_mu)

println("=== CC$k sieve: 除算版 vs Barrett版 (range=2e16) ===")
t=time(); s1=run_scan(_sieve_kernel_div!, st, lo, hi); td=time()-t
t=time(); s2=run_scan(_sieve_kernel_barrett!, st, lo, hi, d_mu); tb=time()-t
println("  除算版    : $(round(td,digits=2))s  survivors=$s1")
println("  Barrett版 : $(round(tb,digits=2))s  survivors=$s2")
println("  一致: $(s1==s2)   speedup = $(round(td/tb,digits=3))x")
CUDA.unsafe_free!(st.d_out)

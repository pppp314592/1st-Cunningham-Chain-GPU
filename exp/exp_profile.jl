# 篩 vs MR128 の時間内訳と篩生存率を実測 (materialize拡張wheelを使用)
include("src/cc_cpu.jl")
include("src/cc_gpu_wheel.jl")
using CUDA

# プロファイル用カーネル: _cc_wheel_kernel128! と同一 + 篩生存数カウント + MR ON/OFF
function _prof_kernel!(out::CuDeviceVector{Int128},
        counter::CuDeviceVector{Int32}, sieve_cnt::CuDeviceVector{Int64}, cap::Int32,
        d_wheel_n::CuDeviceVector{Int64}, R::Int64,
        wheel::Int64, k_base::Int64, ncyc::Int64, lo::Int128, hi::Int128,
        d_primes::CuDeviceVector{Int32}, d_wheel_mod::CuDeviceVector{Int32},
        d_pow20::CuDeviceVector{Int32}, d_bad_off::CuDeviceVector{Int32},
        d_bad::CuDeviceVector{UInt8}, nprimes::Int32, kk::Int32, do_mr::Int32)
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
            if ok
                CUDA.atomic_add!(pointer(sieve_cnt, 1), Int64(1))
                if do_mr != Int32(0)
                    x_lo = UInt64(n & Int128(0xFFFFFFFFFFFFFFFF))
                    x_hi = UInt64((n >> 64) & Int128(0xFFFFFFFFFFFFFFFF))
                    good = true
                    @inbounds for _ in 1:kk
                        isp = if x_hi == UInt64(0) && x_lo <= 0x7FFFFFFFFFFFFFFF
                            is_prime_mr(Int64(x_lo))
                        else
                            (x_hi > 0x7FFFFFFFFFFFFFFF) ? false : is_prime_mr128(x_lo, x_hi)
                        end
                        if !isp; good = false; break; end
                        carry = (x_lo > 0x7FFFFFFFFFFFFFFF) ? UInt64(1) : UInt64(0)
                        x_lo = (x_lo << 1) | UInt64(1)
                        x_hi = (x_hi << 1) | carry
                    end
                    if good
                        idx = CUDA.atomic_add!(pointer(counter, 1), Int32(1)) + Int32(1)
                        idx <= cap && (@inbounds out[idx] = n)
                    end
                end
            end
        end
    end
    return nothing
end

function prof_scan(st, lo::Int128, hi::Int128, do_mr::Int32; work_tile=1<<25)
    w = Int128(st.wheel)
    k_start = Int64(lo ÷ w); k_end = Int64((hi - Int128(1)) ÷ w)
    cap = 1<<12
    d_out = CUDA.zeros(Int128, cap); d_cnt = CUDA.zeros(Int32,1); d_sv = CUDA.zeros(Int64,1)
    cyc_per = max(1, work_tile ÷ Int(st.R)); threads = 256
    cyc = k_start
    while cyc <= k_end
        ncyc = min(cyc_per, k_end - cyc + 1); work = ncyc * st.R
        @cuda threads=threads blocks=cld(work,threads) _prof_kernel!(
            d_out, d_cnt, d_sv, Int32(cap), st.d_wheel_n, st.R, st.wheel, cyc, ncyc,
            lo, hi, st.d_primes, st.d_wheel_mod, st.d_pow20, st.d_bad_off, st.d_bad,
            st.nprimes, Int32(st.k), do_mr)
        cyc += ncyc
    end
    CUDA.synchronize()
    sv = Array(d_sv)[1]; cnt = Array(d_cnt)[1]
    CUDA.unsafe_free!(d_out); CUDA.unsafe_free!(d_cnt); CUDA.unsafe_free!(d_sv)
    return sv, cnt
end

k = 16
base = Int128(9)*Int128(10)^20
rng  = Int128(2)*Int128(10)^16
lo = base; hi = base + rng
st = gpu_wheel_setup(k; wl=_cc_wl_ext(k), verbose=true)

# warmup
prof_scan(st, Int128(10)^12, Int128(10)^12+Int128(10)^11, Int32(1))

println("=== CC$k profile range=$rng ===")
t = time(); sv0, _ = prof_scan(st, lo, hi, Int32(0)); t_sieve = time()-t
t = time(); sv1, c1 = prof_scan(st, lo, hi, Int32(1)); t_full = time()-t
cand = Float64(rng) * (st.R / st.wheel)
println("  candidates(wheel通過) ≈ $(round(cand/1e9,digits=2))e9")
println("  篩生存(MR到達) = $sv0  (生存率 $(round(sv0/cand*100,digits=4))%)")
println("  found CC = $c1")
println("  sieve-only 時間 = $(round(t_sieve,digits=2))s")
println("  full(+MR)  時間 = $(round(t_full,digits=2))s")
println("  MR 寄与 = $(round((t_full-t_sieve),digits=2))s = $(round((t_full-t_sieve)/t_full*100,digits=1))%")
CUDA.unsafe_free!(st.d_out)

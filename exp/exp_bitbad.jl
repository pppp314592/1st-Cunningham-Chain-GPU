# exp_bitbad.jl — d_bad をビットパックしてキャッシュ効率を上げる実験
# 現行: d_bad は素数ごとに p バイト(UInt8) の残基表 → 合計~44MB, L2(1.5MB)を大幅超過。
#   ランダムアクセス(np)で DRAM トラフィック支配の疑い。
# 案: 1残基=1ビットに圧縮(~5.5MB)。ワード=UInt32, off は「ワード単位」。
#   bit test: (d_badbits[off + (np>>5)] >> (np&31)) & 1
# ext/128 経路(scan.jl 本命)で現行バイト版 vs ビット版を同一レンジで比較。
#   julia -g0 -t 12 exp_bitbad.jl
include("src/cc_cpu.jl")
include("src/cc_gpu_wheel.jl")
using CUDA

# --- ビットパック flatten (mdllist から) ---
function flatten_bad_bits(mdllist)
    primes32 = Int32[Int32(p) for (p, _) in mdllist]
    word_off = Int32[]        # ワード(UInt32)単位のオフセット
    words = UInt32[]
    off = 0
    for (p, s) in mdllist
        push!(word_off, Int32(off))
        nw = cld(p, 32)
        w = zeros(UInt32, nw)
        for rr in s
            w[(rr >> 5) + 1] |= (UInt32(1) << (rr & 31))
        end
        append!(words, w)
        off += nw
    end
    return primes32, word_off, words
end

# --- ビット版 Int128 カーネル (bmod は現行と同一, d_bad アクセスのみビット) ---
function kern_bit128!(out, counter, cap::Int32,
        d_wheel_n, R::Int64, wheel::Int64, k_base::Int64, ncyc::Int64,
        lo::Int128, hi::Int128, d_primes, d_wheel_mod, d_pow20, d_mu,
        d_bad_woff, d_badbits, nprimes::Int32, kk::Int32)
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
                word = d_badbits[d_bad_woff[pi] + Int64(np >> 5) + Int64(1)]
                if ((word >> (np & UInt32(31))) & UInt32(1)) != UInt32(0)
                    ok = false
                    break
                end
            end
            if ok
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
        end
    end
    return nothing
end

k = 16
st = gpu_wheel_setup(k; wl=_cc_wl_ext(k), verbose=true)
w = Int128(st.wheel)

# ビットテーブル構築 (同じ mdllist を再現)
mp = _default_max_prime(k)
mdllist = gpu_build_extra(k, mp; wl=_cc_wl_ext(k))
primes32, woff, words = flatten_bad_bits(mdllist)
d_woff = CuArray(woff); d_words = CuArray(words)
println("byte d_bad = $(round(length(st.d_bad)/1e6,digits=1))MB  bit d_badbits = $(round(length(words)*4/1e6,digits=1))MB")

center = Int128(810433818265726529159)
lo = center - Int128(10_000_000_000_000_000)
hi = center + Int128(10_000_000_000_000_000)
k_start = Int64(lo ÷ w); k_end = Int64((hi - Int128(1)) ÷ w)

cap = 1<<12
d_out = CUDA.zeros(Int128, cap); d_cnt = CUDA.zeros(Int32, 1)

function run_byte(threads)
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
function run_bit(threads)
    fill!(d_cnt, Int32(0)); wt=1<<25; cp=max(1,wt÷Int(st.R)); cyc=k_start
    while cyc <= k_end
        ncyc=min(cp,k_end-cyc+1); work=ncyc*st.R; blocks=cld(work,threads)
        @cuda threads=threads blocks=blocks kern_bit128!(
            d_out,d_cnt,Int32(cap),st.d_wheel_n,st.R,st.wheel,cyc,ncyc,lo,hi,
            st.d_primes,st.d_wheel_mod,st.d_pow20,st.d_mu,d_woff,d_words,st.nprimes,Int32(st.k))
        cyc+=ncyc
    end
    CUDA.synchronize(); Array(d_cnt)[1]
end

run_byte(64); run_bit(64)  # warm
for th in (64, 96)
    tb = CUDA.@elapsed (cb = run_byte(th))
    ti = CUDA.@elapsed (ci = run_bit(th))
    println("threads=$th  byte=$(round(tb,digits=3))s(found=$cb)  bit=$(round(ti,digits=3))s(found=$ci)  speedup=$(round(tb/ti,digits=3))x")
end

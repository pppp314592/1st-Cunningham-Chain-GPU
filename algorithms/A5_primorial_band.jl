# algorithms/A5_primorial_band.jl
# プリモリアル・バンド探索: 巨大数 CC 探索の実用ドライバ。
# M = 最初の w 個の奇素数の積。A4 で生成した生存残基 r について
# m = r + t*M ∈ [mlo, mhi] を候補とし、追加篩(P まで)＋MR で CC 先頭 p を返す。

using Primes: primes, isprime

include("A4_wheel_crt.jl")

function primorial_band_search(k::Int; w::Int, P::Int,
                                mlo::Integer, mhi::Integer)
    wheel = wheel_primes_for(w)           # 2 を除く最初の w 個
    M = prod(wheel)
    residues = wheel_residues(k, wheel)   # A4 の DFS で生成

    # 追加篩素数（wheel より大きく P 以下）。各残基 r に例外 m_exc を添える。
    extra = primes(P)
    filter!(p -> p > last(wheel), extra)
    extrabad = Dict{Int,Vector{Tuple{Int,Int}}}()
    for p in extra
        seen = Dict{Int,Int}()
        for i in 0:(k-1)
            inva = invmod(powermod(2, i, p), p)   # (2^i)^{-1} mod p
            num  = p + 1
            exc  = (num % (powermod(2, i, p))) == 0 ? Int(num ÷ powermod(2, i, p)) : -1
            if !haskey(seen, inva); seen[inva] = exc; end
        end
        extrabad[p] = [(r, seen[r]) for r in keys(seen)]
    end

    results = BigInt[]
    for r in residues
        m = BigInt(r)
        if m < mlo
            t = cld(mlo - r, M)
            m = r + t * M
        end
        while m <= mhi
            ok = true
            for (p, ent) in extrabad
                mm = mod(m, p)
                for (bad, exc) in ent
                    if mm == bad && m != exc
                        ok = false; break
                    end
                end
                ok || break
            end
            if ok
                good = true
                for i in 0:(k-1)
                    isprime(m * BigInt(2)^i - 1) || (good = false; break)
                end
                good && push!(results, m - 1)
            end
            m += M
        end
    end
    return sort!(results)
end

# 自己検証: 狭いバンドで全探索と一致
function selftest_band(; k=6, w=5, P=300, lo=10, hi=200_000)
    function brute(lo, hi, k)
        out = BigInt[]
        for n in lo:hi
            n == 2 && continue; iseven(n) && continue
            m = BigInt(n) + 1
            ok = true
            for i in 0:(k-1)
                isprime(m * BigInt(2)^i - 1) || (ok = false; break)
            end
            ok && push!(out, BigInt(n))
        end
        sort!(out)
    end
    got = primorial_band_search(k; w=w, P=P, mlo=lo, mhi=hi)
    exp = brute(lo, hi, k)
    return got == exp, (got, exp)
end

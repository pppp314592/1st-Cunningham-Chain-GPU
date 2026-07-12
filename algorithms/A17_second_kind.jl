# algorithms/A17_second_kind.jl
# 第二種カニンガム鎖探索: p, 2p-1, 4p-3, ... （第一種から悪残基の符号を反転するだけ）。
# 自己検証付き。詳細は A17_second_kind.md。

using Primes: primes, isprime, invmod, powermod

# 第二種の悪残基（m' = n-1 形式）: BadSet = { -(2^i)^{-1} mod q }
function bad_residues_m2(q::Int, k::Int)
    res = Int[]; seen = Set{Int}()
    for i in 0:(k-1)
        inva = invmod(powermod(2, i, q), q)
        r = mod(-inva, q)
        if r ∉ seen; push!(seen, r); push!(res, r); end
    end
    return res
end

function primorial_band_search2(k::Int; w::Int, P::Int, mlo::Integer, mhi::Integer)
    ps = primes(1000); filter!(p -> p > 2, ps)
    wheel = ps[1:w]
    M = prod(wheel)
    residues = Int[]
    badsets = [Set(bad_residues_m2(q, k)) for q in wheel]
    function dfs(idx, res, prod)
        if idx > length(wheel); push!(residues, res); return; end
        q = wheel[idx]; bad = badsets[idx]; inv = invmod(prod % q, q)
        for a in 0:(q-1)
            a in bad && continue
            delta = mod(a - mod(res, q), q)
            dfs(idx+1, mod(res + prod*delta*inv, prod*q), prod*q)
        end
    end
    dfs(1, 0, 1)

    extra = primes(P); filter!(p -> p > last(wheel), extra)
    extrabad = Dict{Int,Vector{Tuple{Int,Int}}}()
    for p in extra
        seen = Dict{Int,Int}()
        for i in 0:(k-1)
            inva = invmod(powermod(2, i, p), p)
            r = mod(-inva, p)
            num = p - 1
            pw = powermod(2, i, p)
            exc = (num % pw == 0) ? Int(num ÷ pw) : -1
            if !haskey(seen, r); seen[r] = exc; end
        end
        extrabad[p] = [(r, seen[r]) for r in keys(seen)]
    end

    results = BigInt[]
    for r in residues
        m = BigInt(r)
        if m < mlo; t = cld(mlo - r, M); m = r + t*M; end
        while m <= mhi
            ok = true
            for (p, ent) in extrabad
                mm = mod(m, p)
                for (bad, exc) in ent
                    if mm == bad && m != exc; ok = false; break; end
                end
                ok || break
            end
            if ok
                good = true
                for i in 0:(k-1)
                    isprime(m * BigInt(2)^i + 1) || (good = false; break)
                end
                good && push!(results, m + 1)
            end
            m += M
        end
    end
    return sort!(results)
end

# 自己検証: 狭いバンドで全探索と一致
function selftest_band2(; k=5, w=5, P=300, lo=10, hi=200_000)
    function brute(lo, hi, k)
        out = BigInt[]
        for n in lo:hi
            n == 2 && continue; iseven(n) && continue
            m = BigInt(n) - 1
            ok = true
            for i in 0:(k-1)
                isprime(m * BigInt(2)^i + 1) || (ok = false; break)
            end
            ok && push!(out, BigInt(n))
        end
        sort!(out)
    end
    got = primorial_band_search2(k; w=w, P=P, mlo=lo, mhi=hi)
    exp = brute(lo, hi, k)
    return got == exp, (got, exp)
end

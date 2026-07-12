# algorithms/A15_two_stage_wheel.jl
# 2 段 wheel: 小さい素数は列挙（高速）し、大きい素数はその残基ごとに DFS で延伸。
# 純 DFS（A4）と全く同じ集合を出すが、w が大きいとき DFS の分岐爆発を
# 「粗い残基」で先に絞ることで実用的になる。verify で A4 と一致を確認。

using Primes: primes, invmod
include("A2_bad_residues.jl")

# wheel 素数を「小（列挙）」と「大（DFS）」に分けるためのヘルパ
function wheel_primes_for(w::Int)
    ps = primes(1000)
    filter!(p -> p > 2, ps)
    return ps[1:w]
end

# 残基 base (mod prod) から、残りの素数について DFS して伸ばす
function _dfs_extend!(out, base, prod, primes_rest, k; emit!)
    if isempty(primes_rest)
        emit!(base); return
    end
    q = primes_rest[1]
    bad = Set(bad_residues_m(q, k))
    inv = invmod(prod % q, q)
    for a in 0:(q-1)
        a in bad && continue
        delta = mod(a - mod(base, q), q)
        r2 = base + prod * delta * inv
        _dfs_extend!(out, mod(r2, prod * q), prod * q, primes_rest[2:end], k; emit! = emit!)
    end
end

# 2 段構成: 小 wheel を全列挙 → 各残基から大 wheel を DFS 延伸
function wheel_residues_two_stage(k::Int, w::Int; s::Int=min(4, w))
    allp = wheel_primes_for(w)
    small = allp[1:s]
    large = allp[s+1:w]
    # 小 wheel を全列挙
    small_res = Int[0]
    for (j, q) in enumerate(small)
        bad = Set(bad_residues_m(q, k))
        tmp = Int[]
        for r in small_res, a in 0:(q-1)
            a in bad && continue
            push!(tmp, r + (j > 1 ? prod(small[1:j-1]) : 1) * a)
        end
        small_res = tmp
    end
    Msmall = prod(small)
    out = Int[]
    for r0 in small_res
        _dfs_extend!(out, r0, Msmall, large, k; emit! = x -> push!(out, x))
    end
    return sort(out)
end

# 自己検証: A4 の純 DFS と一致
function selftest_two_stage(; k=6, w=6, s=3)
    isdefined(Main, :wheel_residues) || include("A4_wheel_crt.jl")
    full = wheel_residues(k, wheel_primes_for(w))
    two  = wheel_residues_two_stage(k, w; s=s)
    return full == two, (length(full), length(two))
end

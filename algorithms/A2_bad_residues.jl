# algorithms/A2_bad_residues.jl
# 第一種カニンガム鎖の「悪残基」計算。n 形式と m 形式の両方を提供し、
# 両者が m = n+1 で一致することの自己検証を含む。

using Primes: primes, invmod, powermod

# n (= p) についての悪残基。f_i(n)=2^i n + (2^i-1) ≡ 0 (mod q) となる n mod q。
function bad_residues_n(q::Int, k::Int)
    res = Int[]; seen = Set{Int}()
    for i in 0:(k-1)
        a   = powermod(2, i, q)        # 2^i mod q
        inva = invmod(a, q)
        r   = mod(-(a - 1) * inva, q)  # -(2^i-1)*(2^i)^{-1} mod q
        if r ∉ seen; push!(seen, r); push!(res, r); end
    end
    return res
end

# m = n+1 についての悪残基。m*2^i - 1 ≡ 0 (mod q)  ⇔  m ≡ (2^i)^{-1} (mod q)。
function bad_residues_m(q::Int, k::Int)
    res = Int[]; seen = Set{Int}()
    for i in 0:(k-1)
        inva = invmod(powermod(2, i, q), q)   # (2^i)^{-1} mod q
        if inva ∉ seen; push!(seen, inva); push!(res, inva); end
    end
    return res
end

# m 形式の悪残基集合から {r-1 mod q} を取ると n 形式と一致するはず。
function selftest_bad_residues(; kmax=12, qmax=200)
    for k in 2:kmax, q in primes(qmax)
        q == 2 && continue   # 2 は偶奇で処理（形が mod 2 で正則でない）
        rn = Set(bad_residues_n(q, k))
        rm = Set(bad_residues_m(q, k))
        shifted = Set(mod(r - 1, q) for r in rm)
        if rn != shifted
            return false, (k, q, rn, shifted)
        end
    end
    return true, nothing
end

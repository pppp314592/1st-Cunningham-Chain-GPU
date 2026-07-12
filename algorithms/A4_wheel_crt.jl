# algorithms/A4_wheel_crt.jl
# 再帰 CRT ウォークで wheel 生存残基をストリーミング生成（非 materialize）。
# メモリ O(w)。巨大な primorial でも全候補を漏れなく生成できる。

using Primes: primes, invmod

include("A2_bad_residues.jl")

# 最初の w 個の奇素数（2 を除く）を wheel 素数として返す。
function wheel_primes_for(w::Int)
    ps = primes(1000)          # 1000 までに奇素数は 167 個（w≤100 まで余裕）
    filter!(p -> p > 2, ps)
    return ps[1:w]
end

# 各 wheel 素数について許容残基だけを DFS し、生存する残基（mod M）を emit!(res) する。
function wheel_crt_stream(k::Int, wheel_primes::Vector{Int}; emit!::Function)
    badsets = [Set(bad_residues_m(q, k)) for q in wheel_primes]
    function dfs(idx, res, prod)
        if idx > length(wheel_primes)
            emit!(res)
            return
        end
        q   = wheel_primes[idx]
        bad = badsets[idx]
        inv = invmod(prod % q, q)          # prod の mod q での逆元
        for a in 0:(q-1)
            a in bad && continue
            delta = mod(a - mod(res, q), q)
            r2 = res + prod * delta * inv  # CRT 合成
            dfs(idx + 1, mod(r2, prod * q), prod * q)
        end
    end
    dfs(1, 0, 1)
end

# DFS で得た生存残基を収集して返す（小 w 用・検証用）。
function wheel_residues(k::Int, wheel_primes::Vector{Int})
    out = Int[]
    wheel_crt_stream(k, wheel_primes; emit! = r -> push!(out, r))
    return sort(out)
end

# 検証用の正解: [0, M) を全走査し、全 wheel 素数で悪残基に入らないものを残す。
function wheel_residues_naive(k::Int, wheel_primes::Vector{Int})
    M = prod(wheel_primes)
    badsets = [Set(bad_residues_m(q, k)) for q in wheel_primes]
    out = Int[]
    for m in 0:(M-1)
        ok = true
        for (j, q) in enumerate(wheel_primes)
            if mod(m, q) in badsets[j]; ok = false; break; end
        end
        ok && push!(out, m)
    end
    return sort(out)
end

# 自己検証: DFS と素直な列挙が一致
function selftest_wheel_crt(; k=6, w=5)
    wheel = wheel_primes_for(w)
    dfs   = wheel_residues(k, wheel)
    naive = wheel_residues_naive(k, wheel)
    return dfs == naive, (length(dfs), length(naive))
end

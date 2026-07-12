# algorithms/A3_segmented_sieve.jl
# 多倍長・インクリメンタル segmented sieve。巨大数領域（start が 10^20〜）でも動く。
# 核心: ブロック base だけが BigInt、篩本体は全て Int64。ブロック間で
# (base mod p) を足し算で更新し、BigInt 剰余を素数ごとに 1 回だけにする。

using Base.Threads
using Primes: primes, isprime

include("A2_bad_residues.jl")

# q ≤ P の奇素数と、その悪残基集合（n 形式）のペア。
# 各残基 r について、f_i(n) = q ちょうど（n = その値）となる例外 n_exc も保持。
# 小さい n では「項が篩素数 q そのもの」になるため、そのときは抹消してはいけない。
function build_badtable(k::Int, P::Int)
    plist = primes(P)
    filter!(p -> p > 2, plist)               # 2 は偶奇で処理
    entries = Vector{Vector{Tuple{Int,Int}}}()
    for p in plist
        seen = Dict{Int,Int}()               # 残基 -> 例外 n
        for i in 0:(k-1)
            a   = powermod(2, i, p)
            inva = invmod(a, p)
            r   = mod(-(a - 1) * inva, p)
            num = p - (a - 1)
            exc = (num % a == 0) ? Int(num ÷ a) : -1
            if !haskey(seen, r); seen[r] = exc; end
        end
        push!(entries, [(r, seen[r]) for r in keys(seen)])
    end
    return plist, entries
end

# [nlo, nhi] 内で長さ k の第一種カニンガム鎖を始める素数 p を返す。
function segmented_sieve_cc(nlo::Integer, nhi::Integer, k::Int;
                            P::Int=1_000_000, B::Int=1<<22)
    nloB, nhiB = BigInt(nlo), BigInt(nhi)
    plist, badsets = build_badtable(k, P)
    Bmod = [mod(B, p) for p in plist]                 # B mod p（Int）
    nblocks = Int(cld(nhiB - nloB + 1, B))

    results = BigInt[]
    lockres = ReentrantLock()

    # 候補 n の本判定（多倍長）
    function test(n::BigInt)
        m = n + 1
        for i in 0:(k-1)
            isprime(m * BigInt(2)^i - 1) || return false
        end
        return true
    end

    # 1 スレッドが担当する連続ブロック列を処理（bmod は局所で更新）
    function process_chunk!(local_bmod, chunk, local_res)
        for b in chunk
            base = nloB + (b - 1) * B
            L = Int(min(B, nhiB - base + 1)); L <= 0 && continue
            sieve = fill(true, L)
            # 偶奇: n 偶数は p=2 以外合成
            b0 = mod(base, 2)
            for i in 0:(L-1)
                if mod(b0 + i, 2) == 0 && (base + i) != 2
                    sieve[i+1] = false
                end
            end
            for t in 1:length(plist)
                p   = plist[t]; ent = badsets[t]; bm = local_bmod[t]
                for (r, exc) in ent
                    idx = mod(r - bm, p)
                    while idx < L
                        n = base + idx
                        if n != exc           # 項が q そのものなら素数なので残す
                            sieve[idx+1] = false
                        end
                        idx += p
                    end
                end
                local_bmod[t] = mod(bm + Bmod[t], p)   # 次ブロックへ更新
            end
            for i in 0:(L-1)
                sieve[i+1] || continue
                n = base + i; n == 2 && continue
                test(n) && push!(local_res, n)
            end
        end
    end

    nt = max(1, Threads.nthreads())
    chunks = collect(Iterators.partition(1:nblocks, cld(nblocks, nt)))
    if nt > 1
        Threads.@threads for ch in chunks
            local_bmod = [mod(nloB + (first(ch) - 1) * B, p) % Int for p in plist]
            local_res  = BigInt[]
            process_chunk!(local_bmod, ch, local_res)
            if !isempty(local_res)
                lock(lockres) do; append!(results, local_res); end
            end
        end
    else
        local_bmod = [mod(nloB, p) % Int for p in plist]
        process_chunk!(local_bmod, 1:nblocks, results)
    end
    return sort!(results)
end

# 自己検証: 狭い区間で全探索と一致するか
function selftest_segmented(; nlo=2, nhi=5000, k=6, P=2000)
    function brute(nlo, nhi, k)
        out = BigInt[]
        for n in nlo:nhi
            n == 2 && continue
            iseven(n) && continue
            m = BigInt(n) + 1
            ok = true
            for i in 0:(k-1)
                isprime(m * BigInt(2)^i - 1) || (ok = false; break)
            end
            ok && push!(out, BigInt(n))
        end
        return sort(out)
    end
    got = segmented_sieve_cc(nlo, nhi, k; P=P, B=1024)
    exp = brute(nlo, nhi, k)
    return got == exp, (got, exp)
end

# cc_block.jl — 第一カニンガム鎖探索（ブロック・ビット篩 + 任意長MR）
#
# 従来の GPU 版 (cc_gpu.jl) は「CPU で逐次 filter! する弱い篩」＋「GPU の最終 MR」
# だった。本実装は:
#   1. ブロック・ビット篩: 区間 [n0, n0+L) を 1bit/候補 のビット配列で持ち、
#      全素数 p ≦ P について、各鎖位置 j で (2^j n + 2^j-1) ≡ 0 (mod p) となる
#      残基 n ≡ r_j (mod p) を抹消する。これで CCk に対して k 個の残基級が
#      各素数で落とせ、P を大きくするほど生存者が激減する。
#   2. 生存者に対してのみ 128bit Miller-Rabin で本当に鎖が続くかを確認。
#      GPU があればこの MR 段を GPU にオフロードする。
#
# 注意: 128bit に収まる範囲 (x_hi ≦ 2^127-1) のみ扱える。CC20 などは開始 n が
#       2^108 程度以下に収まる場合に限り 128bit 内で検証可能。それ以上の桁数は
#       多倍長演算が必要で、単PCブルートフォースでは到達困難。

include("cc_cpu.jl")         # CPU 用: is_prime_mr_cpu, is_prime_mr128_cpu, Primes
# GPU 版 (cc_chain_test_gpu) は src/cc_gpu_block.jl で定義。
# GPU を使う場合は先にそちらを include すること。

using Primes: primes
using Base.Threads

# ============================================================
# 各素数 p について、鎖位置 j=0..k-1 で「n が bad」となる残基 r_j を計算
#   位置 j の値 v_j(n) = 2^j n + (2^j - 1)
#   v_j ≡ 0 (mod p)  ⇔  n ≡ -(2^j - 1) · (2^j)^{-1} (mod p)
# また、v_j がちょうど p そのもの (n が小さいとき) の場合は素数なので
# 抹消してはいけない。その例外 n_exc も返す。
# ============================================================
struct BadInfo
    p::Int
    residues::Vector{Int}      # r_j (重複除去済)
    exc::Vector{Int128}        # 各 residues に対応する「v_j == p となる n」(範囲内のみ)
end

function bad_info(p::Int, k::Int)
    residues = Int[]
    exc = Int128[]
    seen = Set{Int}()
    for j in 0:(k-1)
        a = powermod(2, j, p)             # 2^j mod p (p は奇素数なので正則)
        inva = invmod(a, p)
        r = mod(-(a - 1) * inva, p)
        if r ∉ seen
            push!(seen, r)
            push!(residues, r)
            # v_j(n) = p  ⇔  n = (p - (a-1)) / a  (整数なら)
            num = p - (a - 1)
            if num % a == 0
                push!(exc, Int128(num) ÷ a)
            else
                push!(exc, Int128(-1))    # 該当なし
            end
        end
    end
    return BadInfo(p, residues, exc)
end

# ============================================================
# ブロック篩: [n0, n0+L) の生存ビットを sieve に書き込む (true = 生存)
#   n が偶数の場合は n=2 を除き全て抹消 (奇素数のみを p リストに含めるため)
# ============================================================
function block_sieve!(sieve::BitVector, n0::Int128, L::Int,
                      badlist::Vector{BadInfo}; skip_even::Bool=true)
    @assert length(sieve) == L
    # 初期化: 全生存
    sieve .= true
    # 偶数抹消 (n=2 は特別に残す)
    if skip_even
        for i in 0:(L-1)
            n = n0 + i
            if iseven(n) && n != 2
                sieve[i+1] = false
            end
        end
    end
    # 各素数で bad 残基を抹消
    @inbounds for bi in 1:length(badlist)
        info = badlist[bi]
        p = info.p
        res = info.residues
        exc = info.exc
        for t in 1:length(res)
            r = res[t]
            n_exc = exc[t]
            # 最初の i ≧ 0 で (n0 + i) ≡ r (mod p)
            start = mod(r - n0, p)
            if start < 0; start += p; end
            i = Int(start)   # 0 ≦ start < p なので Int に収まる
            while i < L
                n = n0 + i
                # v_j == p ちょうどのときは素数なので抹消しない
                if n != n_exc
                    sieve[i+1] = false
                end
                i += p
            end
        end
    end
    return sieve
end

# マルチスレッド版: 素数をスレッドに分割し、部分篩を bitwise OR で合成
function block_sieve_mt(n0::Int128, L::Int, badlist::Vector{BadInfo};
                       skip_even::Bool=true, nthreads::Int=Threads.nthreads())
    nt = max(1, min(nthreads, length(badlist)))
    partials = Vector{BitVector}(undef, nt)
    chunk = cld(length(badlist), nt)
    Threads.@threads for t in 1:nt
        lo = (t-1)*chunk + 1
        hi = min(t*chunk, length(badlist))
        if lo > hi
            partials[t] = falses(L)
        else
            local_sieve = trues(L)
            block_sieve!(local_sieve, n0, L, badlist[lo:hi]; skip_even=false)
            partials[t] = local_sieve
        end
    end
    sieve = trues(L)
    if skip_even
        for i in 0:(L-1)
            n = n0 + i
            if iseven(n) && n != 2
                sieve[i+1] = false
            end
        end
    end
    for t in 1:nt
        sieve .&= partials[t]
    end
    return sieve
end

# ============================================================
# 鎖の本判定 (128bit 内)。すべての位置が素数なら true
# ============================================================
function is_cc128(n::Int128, k::Int)::Bool
    x_lo = UInt64(n & 0xFFFFFFFFFFFFFFFF)
    x_hi = UInt64((n >> 64) & 0xFFFFFFFFFFFFFFFF)
    @inbounds for j in 1:k
        if x_hi == 0 && x_lo <= 0x7FFFFFFFFFFFFFFF
            is_prime_mr_cpu(Int64(x_lo)) || return false
        else
            x_hi > 0x7FFFFFFFFFFFFFFF && return false   # 128bit オーバーフロー
            is_prime_mr128_cpu(x_lo, x_hi) || return false
        end
        carry = (x_lo > 0x7FFFFFFFFFFFFFFF) ? UInt64(1) : UInt64(0)
        x_lo = (x_lo << 1) | UInt64(1)
        x_hi = (x_hi << 1) | carry
    end
    return true
end

# ============================================================
# 探索本体
#   (GPU 版 cc_chain_test_gpu は src/cc_gpu_block.jl で定義)
# ============================================================
function search_cc(n0::Int128, n1::Int128, k::Int;
                   P::Int=1_000_000,
                   blocksize::Int=1<<26,
                   gpu::Bool=false,
                   verbose::Bool=true)
    verbose && println("=== CC$k search [$n0, $n1]  P=$P  blocksize=$blocksize  gpu=$gpu ===")
    t0 = time()

    plist = primes(P)
    filter!(p -> p > 2, plist)            # 奇数のみ (偶数は skip_even で処理)
    badlist = [bad_info(p, k) for p in plist]
    verbose && println("  primes=$(length(badlist))")

    results = Int128[]
    nblocks = cld(Int(n1 - n0 + 1), blocksize)
    done = Threads.Atomic{Int}(0)
    alock = ReentrantLock()

    # ブロック探索 (各ブロックは独立なので並列可)
    ranges = collect(Iterators.partition(n0:blocksize:n1, max(1, nblocks ÷ Threads.nthreads())))

    Threads.@threads for blk in ranges
        local_res = Int128[]
        for nb in blk
            L = Int(min(blocksize, n1 - nb + 1))
            L <= 0 && continue
            sieve = block_sieve_mt(nb, L, badlist)
            # 生存者収集
            cands = Int128[]
            sizehint!(cands, 1024)
            for i in 0:(L-1)
                sieve[i+1] || continue
                n = nb + i
                n == 2 && continue
                push!(cands, n)
            end
            if !isempty(cands)
                if gpu
                    append!(local_res, cc_chain_test_gpu(cands, k))
                else
                    for n in cands
                        is_cc128(n, k) && push!(local_res, n)
                    end
                end
            end
        end
        if !isempty(local_res)
            lock(alock) do; append!(results, local_res); end
        end
        nd = Threads.atomic_add!(done, 1) + 1
        if verbose && (nd % max(1, length(ranges)÷10) == 0 || nd == length(ranges))
            elapsed = round(time() - t0, digits=1)
            pct = round(100 * nd / length(ranges), digits=1)
            @info "  $(pct)% | found=$(length(results)) | $(elapsed)s"
        end
    end

    sort!(results)
    verbose && println("=== Done: $(length(results)) CC$k in $(round(time()-t0, digits=2))s ===")
    return results
end

# ============================================================
# 第二種カニンガム鎖 (p, 2p-1, 4p-3, ...) 対応
#   v_j(n) = 2^j n - (2^j - 1)。bad 残基: n ≡ (2^j - 1)·(2^j)^{-1} (mod p)
#   鎖の進行: x -> 2x - 1
# ============================================================

function bad_info_2(p::Int, k::Int)
    residues = Int[]
    exc = Int128[]
    seen = Set{Int}()
    for j in 0:(k-1)
        a = powermod(2, j, p)
        inva = invmod(a, p)
        r = mod((a - 1) * inva, p)
        if r ∉ seen
            push!(seen, r)
            push!(residues, r)
            # v_j(n) = p  ⇔  n = (p + (a-1)) / a  (整数なら)
            num = p + (a - 1)
            if num % a == 0
                push!(exc, Int128(num) ÷ a)
            else
                push!(exc, Int128(-1))
            end
        end
    end
    return BadInfo(p, residues, exc)
end

# 第二種鎖の本判定 (128bit 内)
function is_cc128_2(n::Int128, k::Int)::Bool
    x_lo = UInt64(n & 0xFFFFFFFFFFFFFFFF)
    x_hi = UInt64((n >> 64) & 0xFFFFFFFFFFFFFFFF)
    @inbounds for j in 1:k
        if x_hi == 0 && x_lo <= 0x7FFFFFFFFFFFFFFF
            is_prime_mr_cpu(Int64(x_lo)) || return false
        else
            x_hi > 0x7FFFFFFFFFFFFFFF && return false
            is_prime_mr128_cpu(x_lo, x_hi) || return false
        end
        carry = (x_lo > 0x7FFFFFFFFFFFFFFF) ? UInt64(1) : UInt64(0)
        x_lo = x_lo << 1
        x_hi = (x_hi << 1) | carry
        if x_lo == 0
            x_lo = 0xFFFFFFFFFFFFFFFF
            x_hi -= UInt64(1)
        else
            x_lo -= UInt64(1)
        end
    end
    return true
end

# 第二種探索本体
function search_cc_2(n0::Int128, n1::Int128, k::Int;
                    P::Int=1_000_000,
                    blocksize::Int=1<<26,
                    gpu::Bool=false,
                    verbose::Bool=true)
    verbose && println("=== CC$k (2nd kind) search [$n0, $n1]  P=$P  blocksize=$blocksize  gpu=$gpu ===")
    t0 = time()

    plist = primes(P)
    filter!(p -> p > 2, plist)
    badlist = [bad_info_2(p, k) for p in plist]
    verbose && println("  primes=$(length(badlist))")

    results = Int128[]
    nblocks = cld(Int(n1 - n0 + 1), blocksize)
    done = Threads.Atomic{Int}(0)
    alock = ReentrantLock()

    ranges = collect(Iterators.partition(n0:blocksize:n1, max(1, nblocks ÷ Threads.nthreads())))

    Threads.@threads for blk in ranges
        local_res = Int128[]
        for nb in blk
            L = Int(min(blocksize, n1 - nb + 1))
            L <= 0 && continue
            sieve = block_sieve_mt(nb, L, badlist)
            cands = Int128[]
            sizehint!(cands, 1024)
            for i in 0:(L-1)
                sieve[i+1] || continue
                n = nb + i
                n == 2 && continue
                push!(cands, n)
            end
            if !isempty(cands)
                if gpu
                    append!(local_res, cc_chain_test_gpu_2(cands, k))
                else
                    for n in cands
                        is_cc128_2(n, k) && push!(local_res, n)
                    end
                end
            end
        end
        if !isempty(local_res)
            lock(alock) do; append!(results, local_res); end
        end
        nd = Threads.atomic_add!(done, 1) + 1
        if verbose && (nd % max(1, length(ranges)÷10) == 0 || nd == length(ranges))
            elapsed = round(time() - t0, digits=1)
            pct = round(100 * nd / length(ranges), digits=1)
            @info "  $(pct)% | found=$(length(results)) | $(elapsed)s"
        end
    end

    sort!(results)
    verbose && println("=== Done: $(length(results)) CC$k (2nd) in $(round(time()-t0, digits=2))s ===")
    return results
end

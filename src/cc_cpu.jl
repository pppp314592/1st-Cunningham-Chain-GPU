# cc_cpu.jl — CPU-only 第一カニンガム鎖探索
# マルチスレッド wheel 篩 + Miller-Rabin
# Int64 / Int128 両対応

using Primes: primes
using Base.Threads

# ============================================================
# 64-bit Miller-Rabin (CPU, 7基底 決定的)
# ============================================================

function mulmod_cpu(a::Int64, b::Int64, m::Int64)::Int64
    r = Int64(0)
    aa = a % m
    bb = b % m
    while bb > 0
        if bb & 1 == 1
            r += aa
            if r >= m; r -= m; end
        end
        aa += aa
        if aa >= m; aa -= m; end
        bb >>= 1
    end
    return r
end

function powermod_cpu(base::Int64, exp::Int64, mod::Int64)::Int64
    r = Int64(1)
    b = base % mod
    e = exp
    while e > 0
        if e & 1 == 1; r = mulmod_cpu(r, b, mod); end
        b = mulmod_cpu(b, b, mod)
        e >>= 1
    end
    return r
end

function is_prime_mr_cpu(n::Int64)::Bool
    n <= 1 && return false
    n <= 3 && return true
    n & 1 == 0 && return false

    d, s = n - 1, 0
    while d & 1 == 0; d >>= 1; s += 1; end

    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
        a % n == 0 && continue
        x = powermod_cpu(a % n, d, n)
        x == 1 && continue
        x == n - 1 && continue
        composite = true
        for _ in 1:(s - 1)
            x = mulmod_cpu(x, x, n)
            if x == n - 1
                composite = false; break
            end
        end
        composite && return false
    end
    return true
end

# ============================================================
# 128-bit ソフトウェアエミュレーション (pure Julia, CUDA 非依存)
# ============================================================

@inline function add128_cpu(lo1::UInt64, hi1::UInt64, lo2::UInt64, hi2::UInt64)::Tuple{UInt64, UInt64, Bool}
    lo = lo1 + lo2
    carry = (lo < lo1) ? UInt64(1) : UInt64(0)
    hi = hi1 + hi2 + carry
    overflow = (hi1 > typemax(UInt64) - hi2 - carry)
    return (lo, hi, overflow)
end

@inline function add128_safe_cpu(lo1::UInt64, hi1::UInt64, lo2::UInt64, hi2::UInt64)::Tuple{UInt64, UInt64}
    lo = lo1 + lo2
    carry = (lo < lo1) ? UInt64(1) : UInt64(0)
    hi = hi1 + hi2 + carry
    return (lo, hi)
end

@inline function sub128_cpu(lo1::UInt64, hi1::UInt64, lo2::UInt64, hi2::UInt64)::Tuple{UInt64, UInt64}
    lo = lo1 - lo2
    borrow = (lo > lo1) ? UInt64(1) : UInt64(0)
    hi = hi1 - hi2 - borrow
    return (lo, hi)
end

@inline function ge128_cpu(lo1::UInt64, hi1::UInt64, lo2::UInt64, hi2::UInt64)::Bool
    return hi1 > hi2 || (hi1 == hi2 && lo1 >= lo2)
end

@inline function is_zero128_cpu(lo::UInt64, hi::UInt64)::Bool
    return lo == 0 && hi == 0
end

@inline function mod_add128_cpu(r_lo::UInt64, r_hi::UInt64,
                                 a_lo::UInt64, a_hi::UInt64,
                                 m_lo::UInt64, m_hi::UInt64)::Tuple{UInt64, UInt64}
    r_lo, r_hi, overflow = add128_cpu(r_lo, r_hi, a_lo, a_hi)
    if overflow
        adj_lo, adj_hi = sub128_cpu(UInt64(0), UInt64(0), m_lo, m_hi)
        r_lo, r_hi = add128_safe_cpu(r_lo, r_hi, adj_lo, adj_hi)
    else
        if ge128_cpu(r_lo, r_hi, m_lo, m_hi)
            r_lo, r_hi = sub128_cpu(r_lo, r_hi, m_lo, m_hi)
        end
        if ge128_cpu(r_lo, r_hi, m_lo, m_hi)
            r_lo, r_hi = sub128_cpu(r_lo, r_hi, m_lo, m_hi)
        end
    end
    return (r_lo, r_hi)
end

@inline function mod_double128_cpu(a_lo::UInt64, a_hi::UInt64,
                                    m_lo::UInt64, m_hi::UInt64)::Tuple{UInt64, UInt64}
    a_lo, a_hi, overflow = add128_cpu(a_lo, a_hi, a_lo, a_hi)
    if overflow
        adj_lo, adj_hi = sub128_cpu(UInt64(0), UInt64(0), m_lo, m_hi)
        a_lo, a_hi = add128_safe_cpu(a_lo, a_hi, adj_lo, adj_hi)
    else
        if ge128_cpu(a_lo, a_hi, m_lo, m_hi)
            a_lo, a_hi = sub128_cpu(a_lo, a_hi, m_lo, m_hi)
        end
        if ge128_cpu(a_lo, a_hi, m_lo, m_hi)
            a_lo, a_hi = sub128_cpu(a_lo, a_hi, m_lo, m_hi)
        end
    end
    return (a_lo, a_hi)
end

@inline function mulmod128_cpu(a_lo::UInt64, a_hi::UInt64,
                                b_lo::UInt64, b_hi::UInt64,
                                m_lo::UInt64, m_hi::UInt64)::Tuple{UInt64, UInt64}
    r_lo, r_hi = UInt64(0), UInt64(0)
    while ge128_cpu(a_lo, a_hi, m_lo, m_hi)
        a_lo, a_hi = sub128_cpu(a_lo, a_hi, m_lo, m_hi)
    end
    while ge128_cpu(b_lo, b_hi, m_lo, m_hi)
        b_lo, b_hi = sub128_cpu(b_lo, b_hi, m_lo, m_hi)
    end
    while !is_zero128_cpu(b_lo, b_hi)
        if (b_lo & 1) == 1
            r_lo, r_hi = mod_add128_cpu(r_lo, r_hi, a_lo, a_hi, m_lo, m_hi)
        end
        a_lo, a_hi = mod_double128_cpu(a_lo, a_hi, m_lo, m_hi)
        b_lo = (b_lo >> 1) | ((b_hi & UInt64(1)) << 63)
        b_hi >>= 1
    end
    return (r_lo, r_hi)
end

@inline function powermod128_cpu(base_lo::UInt64, base_hi::UInt64,
                                  exp_lo::UInt64, exp_hi::UInt64,
                                  mod_lo::UInt64, mod_hi::UInt64)::Tuple{UInt64, UInt64}
    r_lo, r_hi = UInt64(1), UInt64(0)
    while ge128_cpu(base_lo, base_hi, mod_lo, mod_hi)
        base_lo, base_hi = sub128_cpu(base_lo, base_hi, mod_lo, mod_hi)
    end
    e_lo, e_hi = exp_lo, exp_hi
    while !is_zero128_cpu(e_lo, e_hi)
        if (e_lo & 1) == 1
            r_lo, r_hi = mulmod128_cpu(r_lo, r_hi, base_lo, base_hi, mod_lo, mod_hi)
        end
        base_lo, base_hi = mulmod128_cpu(base_lo, base_hi, base_lo, base_hi, mod_lo, mod_hi)
        e_lo = (e_lo >> 1) | ((e_hi & UInt64(1)) << 63)
        e_hi >>= 1
    end
    return (r_lo, r_hi)
end

@inline function _is_one128(x_lo::UInt64, x_hi::UInt64)::Bool
    return x_lo == 1 && x_hi == 0
end

@inline function _is_neg1_128(x_lo::UInt64, x_hi::UInt64, n_lo::UInt64, n_hi::UInt64)::Bool
    if n_lo == 0
        return x_lo == 0xFFFFFFFFFFFFFFFF && x_hi == n_hi - 1
    else
        return x_lo == n_lo - 1 && x_hi == n_hi
    end
end

function is_prime_mr128_cpu(n_lo::UInt64, n_hi::UInt64)::Bool
    is_zero128_cpu(n_lo, n_hi) && return false
    n_lo == 1 && n_hi == 0 && return false
    n_lo == 2 && n_hi == 0 && return true
    n_lo == 3 && n_hi == 0 && return true
    (n_lo & 1) == 0 && return false

    d_lo, d_hi = sub128_cpu(n_lo, n_hi, UInt64(1), UInt64(0))
    s = UInt64(0)
    while (d_lo & 1) == 0
        d_lo = (d_lo >> 1) | ((d_hi & UInt64(1)) << 63)
        d_hi >>= 1; s += 1
    end

    for (b_lo, b_hi) in ((UInt64(2), UInt64(0)),
                          (UInt64(325), UInt64(0)),
                          (UInt64(9375), UInt64(0)),
                          (UInt64(28178), UInt64(0)),
                          (UInt64(450775), UInt64(0)),
                          (UInt64(9780504), UInt64(0)),
                          (UInt64(1795265022), UInt64(0)))
        # b % n
        ba_lo, ba_hi = b_lo, b_hi
        while ge128_cpu(ba_lo, ba_hi, n_lo, n_hi)
            ba_lo, ba_hi = sub128_cpu(ba_lo, ba_hi, n_lo, n_hi)
        end
        is_zero128_cpu(ba_lo, ba_hi) && continue

        x_lo, x_hi = powermod128_cpu(ba_lo, ba_hi, d_lo, d_hi, n_lo, n_hi)
        if _is_one128(x_lo, x_hi)
            continue
        end
        if _is_neg1_128(x_lo, x_hi, n_lo, n_hi)
            continue
        end

        local composite = true
        for _ in 1:(s - 1)
            x_lo, x_hi = mulmod128_cpu(x_lo, x_hi, x_lo, x_hi, n_lo, n_hi)
            if _is_neg1_128(x_lo, x_hi, n_lo, n_hi)
                composite = false; break
            end
        end
        composite && return false
    end
    return true
end

# ============================================================
# チェイン長カウント
# ============================================================

function cc_count_cpu(n::Int64, target_cc::Int)::Int
    x = n
    @inbounds for i in 1:target_cc
        is_prime_mr_cpu(x) || return i - 1
        i == target_cc && return target_cc
        x = 2x + 1
        x <= n && return i  # overflow
    end
    return target_cc
end

function cc_count128_cpu(lo::UInt64, hi::UInt64, target_cc::Int)::Int
    x_lo, x_hi = lo, hi
    @inbounds for i in 1:target_cc
        if x_hi > 0x7FFFFFFFFFFFFFFF
            return i - 1
        end
        if x_hi == 0
            is_prime_mr_cpu(Int64(x_lo)) || return i - 1
        else
            is_prime_mr128_cpu(x_lo, x_hi) || return i - 1
        end
        i == target_cc && return target_cc
        carry = (x_lo > 0x7FFFFFFFFFFFFFFF) ? UInt64(1) : UInt64(0)
        x_lo = (x_lo << 1) | UInt64(1)
        x_hi = (x_hi << 1) | carry
    end
    return target_cc
end

# ============================================================
# 篩構築 (cc_gpu.jl と同アルゴリズム)
# ============================================================

function build_cc_sieve(target_cc::Int)
    wl = if target_cc ≤ 9
        [2,3,5,7,11,13,17]
    elseif target_cc ≤ 11
        [2,3,5,7,11,13,17,19,23]
    elseif target_cc ≤ 13
        [2,3,5,7,11,13,17,19,23,37,41]
    else
        [2,3,5,7,11,13,17,19,23,37,41,43]
    end
    wheel = prod(wl)

    function cc_len_mod(x::Int, p::Int, target_cc::Int)::Int
        n = 0
        y = x
        while n < target_cc && y % p != 0
            y = (2y + 1) % p
            n += 1
        end
        return n
    end

    modsieve_wl = [(p, Set(i for i in 0:p-1 if cc_len_mod(i, p, target_cc) < target_cc)) for p in wl]

    wheel_n = Int[1]
    pprod = 1
    for (p, badset) in modsieve_wl
        tmp = copy(wheel_n)
        wheel_n = Int[]
        for i in 0:p-1
            append!(wheel_n, filter(x -> x % p ∉ badset, tmp .+ i * pprod))
        end
        pprod *= p
    end
    eff = wheel ÷ length(wheel_n)
    println("CC$(target_cc) sieve: wheel=$(wheel) residues=$(length(wheel_n)) efficiency=$(eff)")

    max_prime = if target_cc ≤ 12
        1000
    elseif target_cc ≤ 13
        3000
    else
        5000
    end
    extra_primes = filter(p -> !(p in wl) && p ≤ max_prime, primes(max_prime))
    mdllist = [(p, Set(i for i in 0:p-1 if cc_len_mod(i, p, target_cc) < target_cc)) for p in extra_primes]

    return wheel, wheel_n, mdllist
end

# ============================================================
# 第6章 ビットベクトルOR篩 — バッチ単位の事前計算データ
# ============================================================

struct _BatchData
    starts::Vector{Int}     # wheel_n[start_idx] for each batch
    n_per_batch::Int        # B
    n_batches::Int
    n_total::Int            # length(original wheel_n)
    batch_sizes::Vector{Int} # actual size per batch (last may be smaller)
    # deltas[bi, b] = wheel_n[start_bi+b] - wheel_n[start_bi]
    deltas::Matrix{Int}
end

function _build_batch_data(wheel_n::Vector{Int}, B::Int=64)
    n = length(wheel_n)
    n_batches = cld(n, B)
    deltas = zeros(Int, n_batches, B)
    starts = zeros(Int, n_batches)
    batch_sizes = zeros(Int, n_batches)
    for bi in 0:n_batches-1
        s = bi * B + 1
        e = min(s + B - 1, n)
        sz = e - s + 1
        batch_sizes[bi+1] = sz
        s0 = wheel_n[s]
        starts[bi+1] = s0
        for b in 0:(sz - 1)
            deltas[bi+1, b+1] = wheel_n[s + b] - s0
        end
    end
    return _BatchData(starts, B, n_batches, n, batch_sizes, deltas)
end

# ============================================================
# 第6章 ビットベクトル篩ワーカー (Int64)
#  各バッチ B=64 個の候補を UInt64 ビットマスクで一括処理
# ============================================================

function _sieve_worker64_bv(k_start::Int64, k_end::Int64,
                             wheel::Int64, bd::_BatchData,
                             lo::Int64, hi::Int64,
                             primes_list::Vector{Int},
                             bad_flags::Matrix{UInt8})::Vector{Int}
    result = Int[]
    sizehint!(result, 10000)
    np = length(primes_list)
    B = bd.n_per_batch
    full_mask = typemax(UInt64)

    for k in k_start:k_end
        base = k * wheel
        for bi in 1:bd.n_batches
            base_n = base + bd.starts[bi]
            n_in_batch = bd.batch_sizes[bi]
            batch_full = (n_in_batch == B) ? full_mask : ((UInt64(1) << n_in_batch) - 1)

            last_in_batch = base_n + bd.deltas[bi, n_in_batch]
            (last_in_batch <= lo || base_n >= hi) && continue

            mask = UInt64(0)
            for pi in 1:np
                p = primes_list[pi]
                r0 = base_n % p
                for b in 0:n_in_batch-1
                    n_val = base_n + bd.deltas[bi, b+1]
                    (lo < n_val < hi) || continue
                    n_val <= p && continue
                    r = r0 + bd.deltas[bi, b+1]
                    if r >= p
                        r %= p
                    end
                    if bad_flags[pi, r + 1] != 0
                        mask |= UInt64(1) << b
                        if mask == batch_full
                            @goto next_batch
                        end
                    end
                end
            end

            if mask != batch_full
                for b in 0:n_in_batch-1
                    if (mask & (UInt64(1) << b)) == 0
                        n_val = base_n + bd.deltas[bi, b+1]
                        if lo < n_val < hi
                            push!(result, n_val)
                        end
                    end
                end
            end
            @label next_batch
        end
    end
    return result
end

# ============================================================
# 第6章 ビットベクトル篩ワーカー (Int128)
# ============================================================

function _sieve_worker128_bv(k_start::Int128, k_end::Int128,
                              wheel::Int64, bd::_BatchData,
                              lo::Int128, hi::Int128,
                              primes_list::Vector{Int},
                              bad_flags::Matrix{UInt8})::Vector{Int128}
    w128 = Int128(wheel)
    result = Int128[]
    sizehint!(result, 10000)
    np = length(primes_list)
    B = bd.n_per_batch
    full_mask = typemax(UInt64)

    for k in k_start:k_end
        base = k * w128
        for bi in 1:bd.n_batches
            base_n = base + bd.starts[bi]
            n_in_batch = bd.batch_sizes[bi]
            batch_full = (n_in_batch == B) ? full_mask : ((UInt64(1) << n_in_batch) - 1)

            last_in_batch = base_n + bd.deltas[bi, n_in_batch]
            (last_in_batch <= lo || base_n >= hi) && continue

            mask = UInt64(0)
            for pi in 1:np
                p = primes_list[pi]
                r0 = Int(base_n % p)
                for b in 0:n_in_batch-1
                    n_val = base_n + bd.deltas[bi, b+1]
                    (lo < n_val < hi) || continue
                    n_val <= p && continue
                    r = r0 + bd.deltas[bi, b+1]
                    if r >= p
                        r %= p
                    end
                    if bad_flags[pi, r + 1] != 0
                        mask |= UInt64(1) << b
                        if mask == batch_full
                            @goto next_batch
                        end
                    end
                end
            end

            if mask != batch_full
                for b in 0:n_in_batch-1
                    if (mask & (UInt64(1) << b)) == 0
                        n_val = base_n + bd.deltas[bi, b+1]
                        if lo < n_val < hi
                            push!(result, n_val)
                        end
                    end
                end
            end
            @label next_batch
        end
    end
    return result
end

# ============================================================
# 高速篩: 事前計算 + 素数-外側ループ (prime-outer)
#   wheel_n[i] % p を UInt16 で事前計算 → 内周の % を除去
#   N_pre = 使用する素数数 (デフォルト 30)
# ============================================================

struct _FastSieveData
    primes::Vector{Int}          # 事前計算対象の素数 (先頭N個)
    n_pre::Int
    mods::Matrix{UInt16}         # mods[pi, i] = wheel_n[i] % primes[pi]
end

function _build_fast_sieve(wheel_n::Vector{Int}, primes_list::Vector{Int}, N::Int=30)
    np = min(N, length(primes_list))
    sel = primes_list[1:np]
    mods = zeros(UInt16, np, length(wheel_n))
    for pi in 1:np
        p = sel[pi]
        for i in 1:length(wheel_n)
            mods[pi, i] = UInt16(wheel_n[i] % p)
        end
    end
    return _FastSieveData(sel, np, mods)
end

function _sieve_worker64_fast(k_start::Int64, k_end::Int64,
                               wheel::Int64, wheel_n::Vector{Int},
                               lo::Int64, hi::Int64,
                               primes_list::Vector{Int},
                               bad_flags::Matrix{UInt8},
                               fast::_FastSieveData)::Vector{Int}
    result = Int[]
    sizehint!(result, 10000)
    np_all = length(primes_list)
    n_wn = length(wheel_n)
    n_pre = fast.n_pre
    pre_end = n_pre == 0 ? 0 : fast.primes[end]

    # Phase 1 用に生存者リストを管理 (インデックスのみ追跡)
    alive = collect(1:n_wn)       # 生存者の wheel_n インデックス
    buf   = Vector{Int}(undef, n_wn)  # 事前確保バッファ
    nalive = n_wn

    for k in k_start:k_end
        base = k * wheel
        min_n = base + wheel_n[1]

        # Phase 1: prime-outer with precomputed mods
        @inbounds for pi in 1:n_pre
            nalive == 0 && break
            p = fast.primes[pi]
            r0 = base % p
            nbuf = 0
            if min_n > p
                for j in 1:nalive
                    idx = alive[j]
                    t = r0 + fast.mods[pi, idx]
                    r = t >= p ? t - p : t
                    if bad_flags[pi, r + 1] == 0
                        nbuf += 1; buf[nbuf] = idx
                    end
                end
            else
                for j in 1:nalive
                    idx = alive[j]
                    n_val = base + wheel_n[idx]
                    if n_val <= p
                        nbuf += 1; buf[nbuf] = idx
                    else
                        t = r0 + fast.mods[pi, idx]
                        r = t >= p ? t - p : t
                        if bad_flags[pi, r + 1] == 0
                            nbuf += 1; buf[nbuf] = idx
                        end
                    end
                end
            end
            alive, buf = buf, alive
            nalive = nbuf
        end

        # Phase 2: remaining primes (candidate-outer)
        if nalive > 0
            for j in 1:nalive
                n_val = base + wheel_n[alive[j]]
                (lo < n_val < hi) || continue
                ok = true
                @inbounds for pi in (n_pre + 1):np_all
                    p = primes_list[pi]
                    n_val <= p && continue
                    if bad_flags[pi, (n_val % p) + 1] != 0
                        ok = false; break
                    end
                end
                ok && push!(result, n_val)
            end
        end

        # 次のサイクルへ: alive をリセット
        if k < k_end
            for i in 1:n_wn; alive[i] = i; end
            nalive = n_wn
            # buf は次回 swapp 時に上書きされるのでリセット不要
        end
    end
    return result
end

# ============================================================
# バッチ処理篩ワーカー (Int64) — prime-outer を L1 キャッシュ内で完結
# ============================================================

function _build_mods_matrix(wheel_n::Vector{Int}, primes_list::Vector{Int}, N::Int)::Matrix{UInt16}
    n_wn = length(wheel_n)
    mods = zeros(UInt16, N, n_wn)
    for pi in 1:N
        p = primes_list[pi]
        row = @view mods[pi, :]
        for i in 1:n_wn
            row[i] = UInt16(wheel_n[i] % p)
        end
    end
    return mods
end

@inline function _check_residue(r0::Int, mod_val::UInt16, p::Int, bad_row::AbstractVector{UInt8})::Bool
    t = r0 + mod_val
    r = t >= p ? t - p : t
    return bad_row[r + 1] != 0
end

function _sieve_worker64_batch(k_start::Int64, k_end::Int64,
                                wheel::Int64, wheel_n::Vector{Int},
                                lo::Int64, hi::Int64,
                                primes_list::Vector{Int},
                                bad_flags::Matrix{UInt8};
                                N_pre::Int=10, B::Int=2048,
                                mods::Matrix{UInt16}=Matrix{UInt16}(undef,0,0))::Vector{Int}
    result = Int[]
    sizehint!(result, 10000)
    np_all = length(primes_list)
    n_wn = length(wheel_n)

    N = min(N_pre, np_all)
    local_mods = N > 0 && size(mods, 1) == 0 ? _build_mods_matrix(wheel_n, primes_list, N) : mods

    # Preallocate batch survivor arrays
    max_bs = min(B, n_wn)
    buf1 = Vector{Int}(undef, max_bs)
    buf2 = Vector{Int}(undef, max_bs)

    for k in k_start:k_end
        base = k * wheel
        # Precompute base remainders for Phase 1 primes
        r0s = Vector{Int}(undef, N)
        for pi in 1:N
            r0s[pi] = Int(base % primes_list[pi])
        end

        for batch_start in 1:B:n_wn
            batch_end = min(batch_start + B - 1, n_wn)
            n_cand = batch_end - batch_start + 1

            # Initialize survivors for this batch
            for i in 1:n_cand
                buf1[i] = batch_start + i - 1
            end
            alive = buf1
            tmp   = buf2
            nalive = n_cand

            # Phase 1: precomputed primes (prime-outer, no integer division)
            # primes_list[N] < min candidate の時のみ安全
            pi_start = 1
            use_p1 = N > 0 && primes_list[N] < base + wheel_n[batch_start]
            if use_p1
                @inbounds for pi in 1:N
                    nalive == 0 && break
                    p = primes_list[pi]
                    r0 = r0s[pi]
                    bad_row = @view bad_flags[pi, :]
                    nbuf = 0
                    for j in 1:nalive
                        idx = alive[j]
                        if _check_residue(r0, local_mods[pi, idx], p, bad_row)
                            alive[j] = 0
                        else
                            nbuf += 1; tmp[nbuf] = idx
                        end
                    end
                    alive, tmp = tmp, alive
                    nalive = nbuf
                end
                pi_start = N + 1
            end

            # Phase 2: remaining primes (candidate-outer, integer division)
            if nalive > 0
                @inbounds for j in 1:nalive
                    n_val = base + wheel_n[alive[j]]
                    (lo < n_val < hi) || continue
                    ok = true
                    for pi in pi_start:np_all
                        p = primes_list[pi]
                        n_val <= p && continue
                        if bad_flags[pi, (n_val % p) + 1] != 0
                            ok = false; break
                        end
                    end
                    ok && push!(result, n_val)
                end
            end
        end
    end
    return result
end

# ============================================================
# バッチ処理篩ワーカー (Int128)
# ============================================================

function _sieve_worker128_batch(k_start::Int128, k_end::Int128,
                                 wheel::Int64, wheel_n::Vector{Int},
                                 lo::Int128, hi::Int128,
                                 primes_list::Vector{Int},
                                 bad_flags::Matrix{UInt8};
                                 N_pre::Int=10, B::Int=2048,
                                 mods::Matrix{UInt16}=Matrix{UInt16}(undef,0,0))::Vector{Int128}
    w128 = Int128(wheel)
    result = Int128[]
    sizehint!(result, 10000)
    np_all = length(primes_list)
    n_wn = length(wheel_n)

    N = min(N_pre, np_all)
    local_mods = N > 0 && size(mods, 1) == 0 ? _build_mods_matrix(wheel_n, primes_list, N) : mods

    max_bs = min(B, n_wn)
    buf1 = Vector{Int}(undef, max_bs)
    buf2 = Vector{Int}(undef, max_bs)

    for k in k_start:k_end
        base = k * w128
        r0s = Vector{Int}(undef, N)
        for pi in 1:N
            r0s[pi] = Int(base % primes_list[pi])
        end

        for batch_start in 1:B:n_wn
            batch_end = min(batch_start + B - 1, n_wn)
            n_cand = batch_end - batch_start + 1

            for i in 1:n_cand
                buf1[i] = batch_start + i - 1
            end
            alive = buf1
            tmp   = buf2
            nalive = n_cand

            pi_start = 1
            use_p1 = N > 0 && primes_list[N] < (base + wheel_n[batch_start])
            if use_p1
                @inbounds for pi in 1:N
                    nalive == 0 && break
                    p = primes_list[pi]
                    r0 = r0s[pi]
                    bad_row = @view bad_flags[pi, :]
                    nbuf = 0
                    for j in 1:nalive
                        idx = alive[j]
                        if _check_residue(r0, local_mods[pi, idx], p, bad_row)
                            alive[j] = 0
                        else
                            nbuf += 1; tmp[nbuf] = idx
                        end
                    end
                    alive, tmp = tmp, alive
                    nalive = nbuf
                end
                pi_start = N + 1
            end

            if nalive > 0
                @inbounds for j in 1:nalive
                    n_val = base + wheel_n[alive[j]]
                    (lo < n_val < hi) || continue
                    ok = true
                    for pi in pi_start:np_all
                        p = primes_list[pi]
                        n_val <= p && continue
                        if bad_flags[pi, Int(n_val % p) + 1] != 0
                            ok = false; break
                        end
                    end
                    ok && push!(result, n_val)
                end
            end
        end
    end
    return result
end

# ============================================================
# 元の篩ワーカー (Int64) — 比較用
# ============================================================

function _sieve_worker64(k_start::Int64, k_end::Int64,
                          wheel::Int64, wheel_n::Vector{Int},
                          lo::Int64, hi::Int64,
                          primes_list::Vector{Int},
                          bad_flags::Matrix{UInt8})::Vector{Int}
    result = Int[]
    sizehint!(result, 10000)
    np = length(primes_list)

    for k in k_start:k_end
        base = k * wheel
        for r in wheel_n
            n = base + r
            (lo < n < hi) || continue
            ok = true
            @inbounds for pi in 1:np
                p = primes_list[pi]
                n <= p && continue
                r_mod = (n % p) + 1
                if bad_flags[pi, r_mod] != 0
                    ok = false; break
                end
            end
            ok && push!(result, n)
        end
    end
    return result
end

# ============================================================
# 元の篩ワーカー (Int128) — 比較用
# ============================================================

function _sieve_worker128(k_start::Int128, k_end::Int128,
                           wheel::Int64, wheel_n::Vector{Int},
                           lo::Int128, hi::Int128,
                           primes_list::Vector{Int},
                           bad_flags::Matrix{UInt8})::Vector{Int128}
    w = Int128(wheel)
    result = Int128[]
    sizehint!(result, 10000)
    np = length(primes_list)

    for k in k_start:k_end
        base = k * w
        for r in wheel_n
            n = base + r
            (lo < n < hi) || continue
            ok = true
            @inbounds for pi in 1:np
                p = primes_list[pi]
                n <= p && continue
                r_mod = Int(n % p) + 1
                if bad_flags[pi, r_mod] != 0
                    ok = false; break
                end
            end
            ok && push!(result, n)
        end
    end
    return result
end

# ============================================================
# 高速篩ワーカー (Int128) — prime-outer
# ============================================================

function _sieve_worker128_fast(k_start::Int128, k_end::Int128,
                                wheel::Int64, wheel_n::Vector{Int},
                                lo::Int128, hi::Int128,
                                primes_list::Vector{Int},
                                bad_flags::Matrix{UInt8},
                                fast::_FastSieveData)::Vector{Int128}
    w128 = Int128(wheel)
    result = Int128[]
    sizehint!(result, 10000)
    np_all = length(primes_list)
    n_wn = length(wheel_n)
    n_pre = fast.n_pre
    pre_end = n_pre == 0 ? 0 : fast.primes[end]

    alive = collect(1:n_wn)
    buf   = Vector{Int}(undef, n_wn)
    nalive = n_wn

    for k in k_start:k_end
        base = k * w128
        min_n = min(base + wheel_n[1], typemax(Int128))

        @inbounds for pi in 1:n_pre
            nalive == 0 && break
            p = fast.primes[pi]
            r0 = Int(base % p)
            nbuf = 0
            if min_n > p
                for j in 1:nalive
                    idx = alive[j]
                    t = r0 + fast.mods[pi, idx]
                    r = t >= p ? t - p : t
                    if bad_flags[pi, r + 1] == 0
                        nbuf += 1; buf[nbuf] = idx
                    end
                end
            else
                for j in 1:nalive
                    idx = alive[j]
                    n_val = base + wheel_n[idx]
                    if n_val <= p
                        nbuf += 1; buf[nbuf] = idx
                    else
                        t = r0 + fast.mods[pi, idx]
                        r = t >= p ? t - p : t
                        if bad_flags[pi, r + 1] == 0
                            nbuf += 1; buf[nbuf] = idx
                        end
                    end
                end
            end
            alive, buf = buf, alive
            nalive = nbuf
        end

        if nalive > 0
            for j in 1:nalive
                n_val = base + wheel_n[alive[j]]
                (lo < n_val < hi) || continue
                ok = true
                @inbounds for pi in (n_pre + 1):np_all
                    p = primes_list[pi]
                    n_val <= p && continue
                    if bad_flags[pi, Int(n_val % p) + 1] != 0
                        ok = false; break
                    end
                end
                ok && push!(result, n_val)
            end
        end

        if k < k_end
            for i in 1:n_wn; alive[i] = i; end
            nalive = n_wn
        end
    end
    return result
end

# ============================================================
# 生存者フィルター (マルチスレッド Miller-Rabin, Int64)
# ============================================================

function _filter_cc64(candidates::Vector{Int}, target_cc::Int)::Vector{Int}
    isempty(candidates) && return Int[]
    n = length(candidates)
    results = Int[]
    rlock = ReentrantLock()

    nt = min(Threads.nthreads(), n)
    chunk = cld(n, nt)

    Threads.@threads for tid in 1:nt
        s = (tid - 1) * chunk + 1
        e = min(tid * chunk, n)
        local_res = Int[]
        for i in s:e
            if cc_count_cpu(candidates[i], target_cc) == target_cc
                push!(local_res, candidates[i])
            end
        end
        if !isempty(local_res)
            lock(rlock) do; append!(results, local_res); end
        end
    end

    return sort!(results)
end

# ============================================================
# 生存者フィルター (マルチスレッド Miller-Rabin, Int128)
# ============================================================

function _filter_cc128(candidates::Vector{Int128}, target_cc::Int)::Vector{Int128}
    isempty(candidates) && return Int128[]
    n = length(candidates)
    results = Int128[]
    rlock = ReentrantLock()

    nt = min(Threads.nthreads(), n)
    chunk = cld(n, nt)

    Threads.@threads for tid in 1:nt
        s = (tid - 1) * chunk + 1
        e = min(tid * chunk, n)
        local_res = Int128[]
        for i in s:e
            val = candidates[i]
            if val <= typemax(Int64)
                if cc_count_cpu(Int64(val), target_cc) == target_cc
                    push!(local_res, val)
                end
            else
                lo = UInt64(val & 0xFFFFFFFFFFFFFFFF)
                hi = UInt64((val >> 64) & 0xFFFFFFFFFFFFFFFF)
                if cc_count128_cpu(lo, hi, target_cc) == target_cc
                    push!(local_res, val)
                end
            end
        end
        if !isempty(local_res)
            lock(rlock) do; append!(results, local_res); end
        end
    end

    return sort!(results)
end

# ============================================================
# 篩データ最適化: Set → Matrix{UInt8} (キャッシュ効率向上)
# ============================================================

function _flatten_badflags(mdllist::Vector{Tuple{Int, Set{Int}}})
    np = length(mdllist)
    primes_list = Int[]
    # bad_flags[pi, r+1] = 1 if r is bad
    max_p = maximum(p for (p, _) in mdllist)
    bad_flags = zeros(UInt8, np, max_p + 1)

    for (pi, (p, badset)) in enumerate(mdllist)
        push!(primes_list, p)
        for r in 0:p-1
            bad_flags[pi, r + 1] = r in badset ? UInt8(1) : UInt8(0)
        end
    end

    return primes_list, bad_flags
end

# ============================================================
# 本体探索 (Int64)
# ============================================================

function search_cc_cpu(lo::Int64, hi::Int64, target_cc::Int; verbose::Bool=true)
    verbose && println("=== CC$(target_cc) CPU search [$(lo), $(hi)] ===")
    t_start = time()

    wheel, wheel_n, mdllist = build_cc_sieve(target_cc)
    primes_list, bad_flags = _flatten_badflags(mdllist)

    k_start = lo ÷ wheel
    k_end = (hi - 1) ÷ wheel
    total_cycles = k_end - k_start + 1
    verbose && println("  cycles=$(total_cycles) threads=$(Threads.nthreads())")

    # 並列篩: k 範囲をスレッド分割
    nt = Threads.nthreads()
    chunk = max(1, total_cycles ÷ nt)
    ranges = NTuple{2, Int64}[]
    for t in 1:nt
        ks = k_start + (t - 1) * chunk
        ke = (t == nt) ? k_end : min(ks + chunk - 1, k_end)
        ks > ke && break
        push!(ranges, (ks, ke))
    end

    all_candidates = Int[]
    alock = ReentrantLock()
    done = Threads.Atomic{Int}(0)
    n_ranges = length(ranges)

    Threads.@threads for ri in 1:n_ranges
        ks, ke = ranges[ri]
        cand = _sieve_worker64(ks, ke, wheel, wheel_n, lo, hi,
                                primes_list, bad_flags)
        if !isempty(cand)
            lock(alock) do; append!(all_candidates, cand); end
        end
        if verbose
            nd = Threads.atomic_add!(done, 1) + 1
            if nd % max(1, total_cycles ÷ 20) == 0 || nd == n_ranges
                elapsed = round(time() - t_start, digits=1)
                pct = round(100 * nd / n_ranges, digits=1)
                @info "  sieve: $(pct)% | candidates=$(length(all_candidates)) | $(elapsed)s"
            end
        end
    end

    verbose && @info "  sieve done: $(length(all_candidates)) candidates in $(round(time() - t_start, digits=2))s"

    # MR フィルター
    results = _filter_cc64(all_candidates, target_cc)

    t_end = time()
    verbose && println("=== Done: $(length(results)) CC$(target_cc) in $(round(t_end - t_start, digits=2))s ===")
    if verbose
        for r in results
            println("CC$(target_cc): $r")
        end
    end
    return results
end

# ============================================================
# 本体探索 (Int128)
# ============================================================

function search_cc_cpu(lo::Int128, hi::Int128, target_cc::Int; verbose::Bool=true)
    verbose && println("=== CC$(target_cc) CPU search (Int128) [$(lo), $(hi)] ===")
    t_start = time()

    wheel, wheel_n, mdllist = build_cc_sieve(target_cc)
    primes_list, bad_flags = _flatten_badflags(mdllist)

    w128 = Int128(wheel)
    k_start = lo ÷ w128
    k_end = (hi - 1) ÷ w128
    total_cycles = k_end - k_start + 1
    verbose && println("  cycles=$(total_cycles) threads=$(Threads.nthreads())")

    nt = Threads.nthreads()
    chunk = max(1, total_cycles ÷ nt)
    ranges = NTuple{2, Int128}[]
    for t in 1:nt
        ks = k_start + (t - 1) * chunk
        ke = (t == nt) ? k_end : min(ks + chunk - 1, k_end)
        ks > ke && break
        push!(ranges, (ks, ke))
    end

    all_candidates = Int128[]
    alock = ReentrantLock()
    done = Threads.Atomic{Int}(0)
    n_ranges = length(ranges)

    Threads.@threads for ri in 1:n_ranges
        ks, ke = ranges[ri]
        cand = _sieve_worker128(ks, ke, wheel, wheel_n, lo, hi,
                                 primes_list, bad_flags)
        if !isempty(cand)
            lock(alock) do; append!(all_candidates, cand); end
        end
        if verbose
            nd = Threads.atomic_add!(done, 1) + 1
            if nd % max(1, n_ranges ÷ 10) == 0 || nd == n_ranges
                elapsed = round(time() - t_start, digits=1)
                pct = round(100 * nd / n_ranges, digits=1)
                @info "  sieve: $(pct)% | candidates=$(length(all_candidates)) | $(elapsed)s"
            end
        end
    end

    verbose && @info "  sieve done: $(length(all_candidates)) candidates in $(round(time() - t_start, digits=2))s"

    results = _filter_cc128(all_candidates, target_cc)

    t_end = time()
    verbose && println("=== Done: $(length(results)) CC$(target_cc) in $(round(t_end - t_start, digits=2))s ===")
    if verbose
        for r in results
            println("CC$(target_cc): $r")
        end
    end
    return results
end

# ============================================================
# 本体探索 (Int64) — 第6章 ビットベクトルOR法
# ============================================================

function search_cc_cpu_bv(lo::Int64, hi::Int64, target_cc::Int; B::Int=64, verbose::Bool=true)
    verbose && println("=== CC$(target_cc) CPU-BV search [$(lo), $(hi)] (B=$B) ===")
    t_start = time()

    wheel, wheel_n, mdllist = build_cc_sieve(target_cc)
    primes_list, bad_flags = _flatten_badflags(mdllist)

    k_start = lo ÷ wheel
    k_end = (hi - 1) ÷ wheel
    total_cycles = k_end - k_start + 1
    verbose && println("  cycles=$(total_cycles) threads=$(Threads.nthreads()) wheel_n=$(length(wheel_n))")

    bd = _build_batch_data(wheel_n, B)
    verbose && println("  batches=$(bd.n_batches) per_batch=$(bd.n_per_batch)")

    nt = Threads.nthreads()
    chunk = max(1, total_cycles ÷ nt)
    ranges = NTuple{2, Int64}[]
    for t in 1:nt
        ks = k_start + (t - 1) * chunk
        ke = (t == nt) ? k_end : min(ks + chunk - 1, k_end)
        ks > ke && break
        push!(ranges, (ks, ke))
    end

    all_candidates = Int[]
    alock = ReentrantLock()
    done = Threads.Atomic{Int}(0)
    n_ranges = length(ranges)

    Threads.@threads for ri in 1:n_ranges
        ks, ke = ranges[ri]
        cand = _sieve_worker64_bv(ks, ke, wheel, bd, lo, hi,
                                   primes_list, bad_flags)
        if !isempty(cand)
            lock(alock) do; append!(all_candidates, cand); end
        end
        if verbose
            nd = Threads.atomic_add!(done, 1) + 1
            if nd % max(1, n_ranges ÷ 10) == 0 || nd == n_ranges
                elapsed = round(time() - t_start, digits=1)
                pct = round(100 * nd / n_ranges, digits=1)
                @info "  sieve-BV: $(pct)% | candidates=$(length(all_candidates)) | $(elapsed)s"
            end
        end
    end

    verbose && @info "  sieve done: $(length(all_candidates)) candidates in $(round(time() - t_start, digits=2))s"
    results = _filter_cc64(all_candidates, target_cc)

    t_end = time()
    verbose && println("=== Done: $(length(results)) CC$(target_cc) in $(round(t_end - t_start, digits=2))s ===")
    if verbose
        for r in results
            println("CC$(target_cc): $r")
        end
    end
    return results
end

# ============================================================
# 本体探索 (Int128) — 第6章 ビットベクトルOR法
# ============================================================

function search_cc_cpu_bv(lo::Int128, hi::Int128, target_cc::Int; B::Int=64, verbose::Bool=true)
    verbose && println("=== CC$(target_cc) CPU-BV search (Int128) [$(lo), $(hi)] (B=$B) ===")
    t_start = time()

    wheel, wheel_n, mdllist = build_cc_sieve(target_cc)
    primes_list, bad_flags = _flatten_badflags(mdllist)

    w128 = Int128(wheel)
    k_start = lo ÷ w128
    k_end = (hi - 1) ÷ w128
    total_cycles = k_end - k_start + 1
    verbose && println("  cycles=$(total_cycles) threads=$(Threads.nthreads()) wheel_n=$(length(wheel_n))")

    bd = _build_batch_data(wheel_n, B)
    verbose && println("  batches=$(bd.n_batches) per_batch=$(bd.n_per_batch)")

    nt = Threads.nthreads()
    chunk = max(1, total_cycles ÷ nt)
    ranges = NTuple{2, Int128}[]
    for t in 1:nt
        ks = k_start + (t - 1) * chunk
        ke = (t == nt) ? k_end : min(ks + chunk - 1, k_end)
        ks > ke && break
        push!(ranges, (ks, ke))
    end

    all_candidates = Int128[]
    alock = ReentrantLock()
    done = Threads.Atomic{Int}(0)
    n_ranges = length(ranges)

    Threads.@threads for ri in 1:n_ranges
        ks, ke = ranges[ri]
        cand = _sieve_worker128_bv(ks, ke, wheel, bd, lo, hi,
                                    primes_list, bad_flags)
        if !isempty(cand)
            lock(alock) do; append!(all_candidates, cand); end
        end
        if verbose
            nd = Threads.atomic_add!(done, 1) + 1
            if nd % max(1, n_ranges ÷ 10) == 0 || nd == n_ranges
                elapsed = round(time() - t_start, digits=1)
                pct = round(100 * nd / n_ranges, digits=1)
                @info "  sieve-BV: $(pct)% | candidates=$(length(all_candidates)) | $(elapsed)s"
            end
        end
    end

    verbose && @info "  sieve done: $(length(all_candidates)) candidates in $(round(time() - t_start, digits=2))s"
    results = _filter_cc128(all_candidates, target_cc)

    t_end = time()
    verbose && println("=== Done: $(length(results)) CC$(target_cc) in $(round(t_end - t_start, digits=2))s ===")
    if verbose
        for r in results
            println("CC$(target_cc): $r")
        end
    end
    return results
end

# ============================================================
# 本体探索 (Int64) — 高速 prime-outer 篩
# ============================================================

function search_cc_cpu_fast(lo::Int64, hi::Int64, target_cc::Int;
                             N::Int=30, verbose::Bool=true)
    verbose && println("=== CC$(target_cc) CPU-fast [$(lo), $(hi)] (N=$N) ===")
    t_start = time()

    wheel, wheel_n, mdllist = build_cc_sieve(target_cc)
    primes_list, bad_flags = _flatten_badflags(mdllist)

    fast = _build_fast_sieve(wheel_n, primes_list, N)
    verbose && println("  precomputed $(fast.n_pre) primes, remaining $(length(primes_list)-fast.n_pre)")

    k_start = lo ÷ wheel
    k_end = (hi - 1) ÷ wheel
    total_cycles = k_end - k_start + 1
    verbose && println("  cycles=$(total_cycles) threads=$(Threads.nthreads()) wheel_n=$(length(wheel_n))")

    nt = Threads.nthreads()
    chunk = max(1, total_cycles ÷ nt)
    ranges = NTuple{2, Int64}[]
    for t in 1:nt
        ks = k_start + (t - 1) * chunk
        ke = (t == nt) ? k_end : min(ks + chunk - 1, k_end)
        ks > ke && break
        push!(ranges, (ks, ke))
    end

    all_candidates = Int[]
    alock = ReentrantLock()
    done = Threads.Atomic{Int}(0)
    n_ranges = length(ranges)

    Threads.@threads for ri in 1:n_ranges
        ks, ke = ranges[ri]
        cand = _sieve_worker64_fast(ks, ke, wheel, wheel_n, lo, hi,
                                     primes_list, bad_flags, fast)
        if !isempty(cand)
            lock(alock) do; append!(all_candidates, cand); end
        end
        if verbose
            nd = Threads.atomic_add!(done, 1) + 1
            if nd % max(1, n_ranges ÷ 10) == 0 || nd == n_ranges
                elapsed = round(time() - t_start, digits=1)
                pct = round(100 * nd / n_ranges, digits=1)
                @info "  sieve-fast: $(pct)% | candidates=$(length(all_candidates)) | $(elapsed)s"
            end
        end
    end

    verbose && @info "  sieve done: $(length(all_candidates)) candidates in $(round(time() - t_start, digits=2))s"
    results = _filter_cc64(all_candidates, target_cc)

    t_end = time()
    verbose && println("=== Done: $(length(results)) CC$(target_cc) in $(round(t_end - t_start, digits=2))s ===")
    if verbose
        for r in results
            println("CC$(target_cc): $r")
        end
    end
    return results
end

# ============================================================
# 本体探索 (Int128) — 高速 prime-outer 篩
# ============================================================

function search_cc_cpu_fast(lo::Int128, hi::Int128, target_cc::Int;
                             N::Int=30, verbose::Bool=true)
    verbose && println("=== CC$(target_cc) CPU-fast (Int128) [$(lo), $(hi)] (N=$N) ===")
    t_start = time()

    wheel, wheel_n, mdllist = build_cc_sieve(target_cc)
    primes_list, bad_flags = _flatten_badflags(mdllist)

    fast = _build_fast_sieve(wheel_n, primes_list, N)
    verbose && println("  precomputed $(fast.n_pre) primes, remaining $(length(primes_list)-fast.n_pre)")

    w128 = Int128(wheel)
    k_start = lo ÷ w128
    k_end = (hi - 1) ÷ w128
    total_cycles = k_end - k_start + 1
    verbose && println("  cycles=$(total_cycles) threads=$(Threads.nthreads()) wheel_n=$(length(wheel_n))")

    nt = Threads.nthreads()
    chunk = max(1, total_cycles ÷ nt)
    ranges = NTuple{2, Int128}[]
    for t in 1:nt
        ks = k_start + (t - 1) * chunk
        ke = (t == nt) ? k_end : min(ks + chunk - 1, k_end)
        ks > ke && break
        push!(ranges, (ks, ke))
    end

    all_candidates = Int128[]
    alock = ReentrantLock()
    done = Threads.Atomic{Int}(0)
    n_ranges = length(ranges)

    Threads.@threads for ri in 1:n_ranges
        ks, ke = ranges[ri]
        cand = _sieve_worker128_fast(ks, ke, wheel, wheel_n, lo, hi,
                                      primes_list, bad_flags, fast)
        if !isempty(cand)
            lock(alock) do; append!(all_candidates, cand); end
        end
        if verbose
            nd = Threads.atomic_add!(done, 1) + 1
            if nd % max(1, n_ranges ÷ 10) == 0 || nd == n_ranges
                elapsed = round(time() - t_start, digits=1)
                pct = round(100 * nd / n_ranges, digits=1)
                @info "  sieve-fast: $(pct)% | candidates=$(length(all_candidates)) | $(elapsed)s"
            end
        end
    end

    verbose && @info "  sieve done: $(length(all_candidates)) candidates in $(round(time() - t_start, digits=2))s"
    results = _filter_cc128(all_candidates, target_cc)

    t_end = time()
    verbose && println("=== Done: $(length(results)) CC$(target_cc) in $(round(t_end - t_start, digits=2))s ===")
    if verbose
        for r in results
            println("CC$(target_cc): $r")
        end
    end
    return results
end

# ============================================================
# 本体探索 (Int64) — バッチ処理版
# ============================================================

function search_cc_cpu_batch(lo::Int64, hi::Int64, target_cc::Int;
                              N_pre::Int=10, B::Int=2048, verbose::Bool=true)
    verbose && println("=== CC$(target_cc) CPU-batch [$(lo), $(hi)] (N=$N_pre, B=$B) ===")
    t_start = time()

    wheel, wheel_n, mdllist = build_cc_sieve(target_cc)
    primes_list, bad_flags = _flatten_badflags(mdllist)

    k_start = lo ÷ wheel
    k_end = (hi - 1) ÷ wheel
    total_cycles = k_end - k_start + 1
    verbose && println("  cycles=$(total_cycles) threads=$(Threads.nthreads()) wheel_n=$(length(wheel_n))")

    nt = Threads.nthreads()
    chunk = max(1, total_cycles ÷ nt)
    ranges = NTuple{2, Int64}[]
    for t in 1:nt
        ks = k_start + (t - 1) * chunk
        ke = (t == nt) ? k_end : min(ks + chunk - 1, k_end)
        ks > ke && break
        push!(ranges, (ks, ke))
    end

    N = min(N_pre, length(primes_list))
    mods = N > 0 ? _build_mods_matrix(wheel_n, primes_list, N) : Matrix{UInt16}(undef, 0, 0)

    all_candidates = Int[]
    alock = ReentrantLock()
    done = Threads.Atomic{Int}(0)
    n_ranges = length(ranges)

    Threads.@threads for ri in 1:n_ranges
        ks, ke = ranges[ri]
        cand = _sieve_worker64_batch(ks, ke, wheel, wheel_n, lo, hi,
                                     primes_list, bad_flags;
                                     N_pre=N_pre, B=B, mods=mods)
        if !isempty(cand)
            lock(alock) do; append!(all_candidates, cand); end
        end
        if verbose
            nd = Threads.atomic_add!(done, 1) + 1
            if nd % max(1, n_ranges ÷ 10) == 0 || nd == n_ranges
                elapsed = round(time() - t_start, digits=1)
                pct = round(100 * nd / n_ranges, digits=1)
                @info "  sieve-batch: $(pct)% | candidates=$(length(all_candidates)) | $(elapsed)s"
            end
        end
    end

    verbose && @info "  sieve done: $(length(all_candidates)) candidates in $(round(time() - t_start, digits=2))s"
    results = _filter_cc64(all_candidates, target_cc)

    t_end = time()
    verbose && println("=== Done: $(length(results)) CC$(target_cc) in $(round(t_end - t_start, digits=2))s ===")
    if verbose
        for r in results
            println("CC$(target_cc): $r")
        end
    end
    return results
end

# ============================================================
# 本体探索 (Int128) — バッチ処理版
# ============================================================

function search_cc_cpu_batch(lo::Int128, hi::Int128, target_cc::Int;
                              N_pre::Int=10, B::Int=2048, verbose::Bool=true)
    verbose && println("=== CC$(target_cc) CPU-batch (Int128) [$(lo), $(hi)] (N=$N_pre, B=$B) ===")
    t_start = time()

    wheel, wheel_n, mdllist = build_cc_sieve(target_cc)
    primes_list, bad_flags = _flatten_badflags(mdllist)

    w128 = Int128(wheel)
    k_start = lo ÷ w128
    k_end = (hi - 1) ÷ w128
    total_cycles = k_end - k_start + 1
    verbose && println("  cycles=$(total_cycles) threads=$(Threads.nthreads()) wheel_n=$(length(wheel_n))")

    nt = Threads.nthreads()
    chunk = max(1, total_cycles ÷ nt)
    ranges = NTuple{2, Int128}[]
    for t in 1:nt
        ks = k_start + (t - 1) * chunk
        ke = (t == nt) ? k_end : min(ks + chunk - 1, k_end)
        ks > ke && break
        push!(ranges, (ks, ke))
    end

    N = min(N_pre, length(primes_list))
    mods = N > 0 ? _build_mods_matrix(wheel_n, primes_list, N) : Matrix{UInt16}(undef, 0, 0)

    all_candidates = Int128[]
    alock = ReentrantLock()
    done = Threads.Atomic{Int}(0)
    n_ranges = length(ranges)

    Threads.@threads for ri in 1:n_ranges
        ks, ke = ranges[ri]
        cand = _sieve_worker128_batch(ks, ke, wheel, wheel_n, lo, hi,
                                      primes_list, bad_flags;
                                      N_pre=N_pre, B=B, mods=mods)
        if !isempty(cand)
            lock(alock) do; append!(all_candidates, cand); end
        end
        if verbose
            nd = Threads.atomic_add!(done, 1) + 1
            if nd % max(1, n_ranges ÷ 10) == 0 || nd == n_ranges
                elapsed = round(time() - t_start, digits=1)
                pct = round(100 * nd / n_ranges, digits=1)
                @info "  sieve-batch: $(pct)% | candidates=$(length(all_candidates)) | $(elapsed)s"
            end
        end
    end

    verbose && @info "  sieve done: $(length(all_candidates)) candidates in $(round(time() - t_start, digits=2))s"
    results = _filter_cc128(all_candidates, target_cc)

    t_end = time()
    verbose && println("=== Done: $(length(results)) CC$(target_cc) in $(round(t_end - t_start, digits=2))s ===")
    if verbose
        for r in results
            println("CC$(target_cc): $r")
        end
    end
    return results
end

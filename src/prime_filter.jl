using CUDA

function rem_gpu(a::Int64, b::Int64)::Int64
    a - (a ÷ b) * b
end

function mulmod(a::Int64, b::Int64, m::Int64)::Int64
    r = UInt64(0)
    au = UInt64(rem_gpu(a, m))
    bu = UInt64(rem_gpu(b, m))
    mu = UInt64(m)
    while bu > 0
        if bu & 1 == 1
            r += au
            if r >= mu
                r -= mu
            end
        end
        au += au
        if au >= mu
            au -= mu
        end
        bu >>= 1
    end
    return Int64(r)
end

function powermod_gpu(base::Int64, exp::Int64, mod::Int64)::Int64
    r = Int64(1)
    base = rem_gpu(base, mod)
    e = exp
    while e > 0
        if e & 1 == 1
            r = mulmod(r, base, mod)
        end
        base = mulmod(base, base, mod)
        e >>= 1
    end
    return r
end

function is_prime_mr(n::Int64)::Bool
    n <= 1 && return false
    n <= 3 && return true
    rem_gpu(n, 2) == 0 && return false

    d, s = n - 1, 0
    while rem_gpu(d, 2) == 0
        d ÷= 2; s += 1
    end

    # bases: [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
    a = 2
    if rem_gpu(a, n) != 0
        x = powermod_gpu(a, d, n)
        if x != 1 && x != n - 1
            composite = true
            for _ in 1:(s - 1)
                x = mulmod(x, x, n)
                if x == n - 1
                    composite = false
                    break
                end
            end
            composite && return false
        end
    end

    a = 325
    if rem_gpu(a, n) != 0
        x = powermod_gpu(a, d, n)
        if x != 1 && x != n - 1
            composite = true
            for _ in 1:(s - 1)
                x = mulmod(x, x, n)
                if x == n - 1
                    composite = false
                    break
                end
            end
            composite && return false
        end
    end

    a = 9375
    if rem_gpu(a, n) != 0
        x = powermod_gpu(a, d, n)
        if x != 1 && x != n - 1
            composite = true
            for _ in 1:(s - 1)
                x = mulmod(x, x, n)
                if x == n - 1
                    composite = false
                    break
                end
            end
            composite && return false
        end
    end

    a = 28178
    if rem_gpu(a, n) != 0
        x = powermod_gpu(a, d, n)
        if x != 1 && x != n - 1
            composite = true
            for _ in 1:(s - 1)
                x = mulmod(x, x, n)
                if x == n - 1
                    composite = false
                    break
                end
            end
            composite && return false
        end
    end

    a = 450775
    if rem_gpu(a, n) != 0
        x = powermod_gpu(a, d, n)
        if x != 1 && x != n - 1
            composite = true
            for _ in 1:(s - 1)
                x = mulmod(x, x, n)
                if x == n - 1
                    composite = false
                    break
                end
            end
            composite && return false
        end
    end

    a = 9780504
    if rem_gpu(a, n) != 0
        x = powermod_gpu(a, d, n)
        if x != 1 && x != n - 1
            composite = true
            for _ in 1:(s - 1)
                x = mulmod(x, x, n)
                if x == n - 1
                    composite = false
                    break
                end
            end
            composite && return false
        end
    end

    a = 1795265022
    if rem_gpu(a, n) != 0
        x = powermod_gpu(a, d, n)
        if x != 1 && x != n - 1
            composite = true
            for _ in 1:(s - 1)
                x = mulmod(x, x, n)
                if x == n - 1
                    composite = false
                    break
                end
            end
            composite && return false
        end
    end

    return true
end

function filter_primes(arr::Vector{Int})::Vector{Int}
    d_arr = CuArray(arr)
    d_mask = map(is_prime_mr, d_arr)
    return arr[Array(d_mask)]
end

# ============================================================
# 128-bit ソフトウェアエミュレーション (for CUDA GPU)
# 128-bit 値 = (lo::UInt64, hi::UInt64) のタプルで表現
# ============================================================

# 128-bit 加算: a + b (128-bit 結果 + 桁上がりフラグ)
@inline function add128(lo1::UInt64, hi1::UInt64, lo2::UInt64, hi2::UInt64)::Tuple{UInt64, UInt64, Bool}
    lo = lo1 + lo2
    carry1 = (lo < lo1) ? UInt64(1) : UInt64(0)
    hi = hi1 + hi2 + carry1
    overflow = (hi1 > typemax(UInt64) - hi2 - carry1)
    return (lo, hi, overflow)
end

# 結果が 2^128 未満と分かっている場合の簡易加算
@inline function add128_safe(lo1::UInt64, hi1::UInt64, lo2::UInt64, hi2::UInt64)::Tuple{UInt64, UInt64}
    lo = lo1 + lo2
    carry1 = (lo < lo1) ? UInt64(1) : UInt64(0)
    hi = hi1 + hi2 + carry1
    return (lo, hi)
end

# 128-bit 減算: a - b (a >= b を前提)
@inline function sub128(lo1::UInt64, hi1::UInt64, lo2::UInt64, hi2::UInt64)::Tuple{UInt64, UInt64}
    lo = lo1 - lo2
    borrow = (lo > lo1) ? UInt64(1) : UInt64(0)
    hi = hi1 - hi2 - borrow
    return (lo, hi)
end

# 128-bit 比較: a >= b ?
@inline function ge128(lo1::UInt64, hi1::UInt64, lo2::UInt64, hi2::UInt64)::Bool
    return hi1 > hi2 || (hi1 == hi2 && lo1 >= lo2)
end

# 128-bit がゼロか
@inline function is_zero128(lo::UInt64, hi::UInt64)::Bool
    return lo == 0 && hi == 0
end

# ------------------------------------------------------------
# 128-bit modular 加算: (r + a) mod m
# r, a < m < 2^128
# ------------------------------------------------------------
@inline function mod_add128(r_lo::UInt64, r_hi::UInt64,
                            a_lo::UInt64, a_hi::UInt64,
                            m_lo::UInt64, m_hi::UInt64)::Tuple{UInt64, UInt64}
    r_lo, r_hi, overflow = add128(r_lo, r_hi, a_lo, a_hi)
    if overflow
        # r + a >= 2^128: wrapped = r + a - 2^128
        # (r + a) mod m = (wrapped + 2^128) mod m
        # m > 2^127 (otherwise no overflow for a,m < 2^128), so 2^128 mod m = 2^128 - m
        adj_lo, adj_hi = sub128(UInt64(0), UInt64(0), m_lo, m_hi)  # = 2^128 - m
        r_lo, r_hi = add128_safe(r_lo, r_hi, adj_lo, adj_hi)
        # r + a - m < m (since r + a < 2m)
    else
        if ge128(r_lo, r_hi, m_lo, m_hi)
            r_lo, r_hi = sub128(r_lo, r_hi, m_lo, m_hi)
        end
        if ge128(r_lo, r_hi, m_lo, m_hi)
            r_lo, r_hi = sub128(r_lo, r_hi, m_lo, m_hi)
        end
    end
    return (r_lo, r_hi)
end

# ------------------------------------------------------------
# 128-bit modular 2倍: (2*a) mod m (a < m < 2^128)
# ------------------------------------------------------------
@inline function mod_double128(a_lo::UInt64, a_hi::UInt64,
                               m_lo::UInt64, m_hi::UInt64)::Tuple{UInt64, UInt64}
    a_lo, a_hi, overflow = add128(a_lo, a_hi, a_lo, a_hi)
    if overflow
        adj_lo, adj_hi = sub128(UInt64(0), UInt64(0), m_lo, m_hi)
        a_lo, a_hi = add128_safe(a_lo, a_hi, adj_lo, adj_hi)
    else
        if ge128(a_lo, a_hi, m_lo, m_hi)
            a_lo, a_hi = sub128(a_lo, a_hi, m_lo, m_hi)
        end
        if ge128(a_lo, a_hi, m_lo, m_hi)
            a_lo, a_hi = sub128(a_lo, a_hi, m_lo, m_hi)
        end
    end
    return (a_lo, a_hi)
end

# ------------------------------------------------------------
# 128-bit ロシア農民法 mulmod: a * b mod m
# ------------------------------------------------------------
@inline function mulmod128(a_lo::UInt64, a_hi::UInt64,
                           b_lo::UInt64, b_hi::UInt64,
                           m_lo::UInt64, m_hi::UInt64)::Tuple{UInt64, UInt64}
    r_lo, r_hi = UInt64(0), UInt64(0)
    # a を m 未満に (高々2回の減算で十分)
    while ge128(a_lo, a_hi, m_lo, m_hi)
        a_lo, a_hi = sub128(a_lo, a_hi, m_lo, m_hi)
    end
    while ge128(b_lo, b_hi, m_lo, m_hi)
        b_lo, b_hi = sub128(b_lo, b_hi, m_lo, m_hi)
    end
    while !is_zero128(b_lo, b_hi)
        if (b_lo & 1) == 1
            r_lo, r_hi = mod_add128(r_lo, r_hi, a_lo, a_hi, m_lo, m_hi)
        end
        a_lo, a_hi = mod_double128(a_lo, a_hi, m_lo, m_hi)
        # b >>= 1
        b_lo = (b_lo >> 1) | ((b_hi & UInt64(1)) << 63)
        b_hi >>= 1
    end
    return (r_lo, r_hi)
end

# ------------------------------------------------------------
# 128-bit powermod: base^exp mod mod
# ------------------------------------------------------------
@inline function powermod128(base_lo::UInt64, base_hi::UInt64,
                             exp_lo::UInt64, exp_hi::UInt64,
                             mod_lo::UInt64, mod_hi::UInt64)::Tuple{UInt64, UInt64}
    r_lo, r_hi = UInt64(1), UInt64(0)
    # base を mod 未満に
    while ge128(base_lo, base_hi, mod_lo, mod_hi)
        base_lo, base_hi = sub128(base_lo, base_hi, mod_lo, mod_hi)
    end
    e_lo, e_hi = exp_lo, exp_hi
    while !is_zero128(e_lo, e_hi)
        if (e_lo & 1) == 1
            r_lo, r_hi = mulmod128(r_lo, r_hi, base_lo, base_hi, mod_lo, mod_hi)
        end
        base_lo, base_hi = mulmod128(base_lo, base_hi, base_lo, base_hi, mod_lo, mod_hi)
        # e >>= 1
        e_lo = (e_lo >> 1) | ((e_hi & UInt64(1)) << 63)
        e_hi >>= 1
    end
    return (r_lo, r_hi)
end

# ------------------------------------------------------------
# 128-bit Miller-Rabin: n < 2^128 の素数判定
# 7基底 [2,325,9375,28178,450775,9780504,1795265022]
# ------------------------------------------------------------
@inline function is_prime_mr128(n_lo::UInt64, n_hi::UInt64)::Bool
    # n <= 1
    is_zero128(n_lo, n_hi) && return false
    n_lo == 1 && n_hi == 0 && return false
    # n <= 3
    n_lo == 2 && n_hi == 0 && return true
    n_lo == 3 && n_hi == 0 && return true
    # n even
    (n_lo & 1) == 0 && return false

    # d = n - 1, s = 0
    d_lo, d_hi = sub128(n_lo, n_hi, UInt64(1), UInt64(0))
    s = UInt64(0)
    while (d_lo & 1) == 0
        d_lo = (d_lo >> 1) | ((d_hi & UInt64(1)) << 63)
        d_hi >>= 1
        s += 1
    end

    # 7 基底をループ展開 (GC 回避のためタプル不使用)
    # base = 2
    if n_lo != 2 || n_hi != 0
        x_lo, x_hi = powermod128(UInt64(2), UInt64(0), d_lo, d_hi, n_lo, n_hi)
        if !(x_lo == 1 && x_hi == 0) && !(x_lo == n_lo - 1 && x_hi == n_hi - (n_lo == 0 ? 1 : 0))
            comp = true
            for _ in 1:(s - 1)
                x_lo, x_hi = mulmod128(x_lo, x_hi, x_lo, x_hi, n_lo, n_hi)
                if x_lo == n_lo - 1 && x_hi == n_hi - (n_lo == 0 ? UInt64(1) : UInt64(0))
                    comp = false
                    break
                end
            end
            comp && return false
        end
    end

    # base = 325
    if !(n_lo == 325 && n_hi == 0)
        b_lo = UInt64(325)
        b_hi = UInt64(0)
        while ge128(b_lo, b_hi, n_lo, n_hi)
            b_lo, b_hi = sub128(b_lo, b_hi, n_lo, n_hi)
        end
        if !is_zero128(b_lo, b_hi)
            x_lo, x_hi = powermod128(b_lo, b_hi, d_lo, d_hi, n_lo, n_hi)
            if !(x_lo == 1 && x_hi == 0) && !(x_lo == n_lo - 1 && x_hi == n_hi - (n_lo == 0 ? 1 : 0))
                comp = true
                for _ in 1:(s - 1)
                    x_lo, x_hi = mulmod128(x_lo, x_hi, x_lo, x_hi, n_lo, n_hi)
                    if x_lo == n_lo - 1 && x_hi == n_hi - (n_lo == 0 ? UInt64(1) : UInt64(0))
                        comp = false
                        break
                    end
                end
                comp && return false
            end
        end
    end

    # base = 9375
    if !(n_lo == 9375 && n_hi == 0)
        b_lo = UInt64(9375)
        b_hi = UInt64(0)
        while ge128(b_lo, b_hi, n_lo, n_hi)
            b_lo, b_hi = sub128(b_lo, b_hi, n_lo, n_hi)
        end
        if !is_zero128(b_lo, b_hi)
            x_lo, x_hi = powermod128(b_lo, b_hi, d_lo, d_hi, n_lo, n_hi)
            if !(x_lo == 1 && x_hi == 0) && !(x_lo == n_lo - 1 && x_hi == n_hi - (n_lo == 0 ? 1 : 0))
                comp = true
                for _ in 1:(s - 1)
                    x_lo, x_hi = mulmod128(x_lo, x_hi, x_lo, x_hi, n_lo, n_hi)
                    if x_lo == n_lo - 1 && x_hi == n_hi - (n_lo == 0 ? UInt64(1) : UInt64(0))
                        comp = false
                        break
                    end
                end
                comp && return false
            end
        end
    end

    # base = 28178
    if !(n_lo == 28178 && n_hi == 0)
        b_lo = UInt64(28178)
        b_hi = UInt64(0)
        while ge128(b_lo, b_hi, n_lo, n_hi)
            b_lo, b_hi = sub128(b_lo, b_hi, n_lo, n_hi)
        end
        if !is_zero128(b_lo, b_hi)
            x_lo, x_hi = powermod128(b_lo, b_hi, d_lo, d_hi, n_lo, n_hi)
            if !(x_lo == 1 && x_hi == 0) && !(x_lo == n_lo - 1 && x_hi == n_hi - (n_lo == 0 ? 1 : 0))
                comp = true
                for _ in 1:(s - 1)
                    x_lo, x_hi = mulmod128(x_lo, x_hi, x_lo, x_hi, n_lo, n_hi)
                    if x_lo == n_lo - 1 && x_hi == n_hi - (n_lo == 0 ? UInt64(1) : UInt64(0))
                        comp = false
                        break
                    end
                end
                comp && return false
            end
        end
    end

    # base = 450775
    if !(n_lo == 450775 && n_hi == 0)
        b_lo = UInt64(450775)
        b_hi = UInt64(0)
        while ge128(b_lo, b_hi, n_lo, n_hi)
            b_lo, b_hi = sub128(b_lo, b_hi, n_lo, n_hi)
        end
        if !is_zero128(b_lo, b_hi)
            x_lo, x_hi = powermod128(b_lo, b_hi, d_lo, d_hi, n_lo, n_hi)
            if !(x_lo == 1 && x_hi == 0) && !(x_lo == n_lo - 1 && x_hi == n_hi - (n_lo == 0 ? 1 : 0))
                comp = true
                for _ in 1:(s - 1)
                    x_lo, x_hi = mulmod128(x_lo, x_hi, x_lo, x_hi, n_lo, n_hi)
                    if x_lo == n_lo - 1 && x_hi == n_hi - (n_lo == 0 ? UInt64(1) : UInt64(0))
                        comp = false
                        break
                    end
                end
                comp && return false
            end
        end
    end

    # base = 9780504
    if !(n_lo == 9780504 && n_hi == 0)
        b_lo = UInt64(9780504)
        b_hi = UInt64(0)
        while ge128(b_lo, b_hi, n_lo, n_hi)
            b_lo, b_hi = sub128(b_lo, b_hi, n_lo, n_hi)
        end
        if !is_zero128(b_lo, b_hi)
            x_lo, x_hi = powermod128(b_lo, b_hi, d_lo, d_hi, n_lo, n_hi)
            if !(x_lo == 1 && x_hi == 0) && !(x_lo == n_lo - 1 && x_hi == n_hi - (n_lo == 0 ? 1 : 0))
                comp = true
                for _ in 1:(s - 1)
                    x_lo, x_hi = mulmod128(x_lo, x_hi, x_lo, x_hi, n_lo, n_hi)
                    if x_lo == n_lo - 1 && x_hi == n_hi - (n_lo == 0 ? UInt64(1) : UInt64(0))
                        comp = false
                        break
                    end
                end
                comp && return false
            end
        end
    end

    # base = 1795265022
    if !(n_lo == 1795265022 && n_hi == 0)
        b_lo = UInt64(1795265022)
        b_hi = UInt64(0)
        while ge128(b_lo, b_hi, n_lo, n_hi)
            b_lo, b_hi = sub128(b_lo, b_hi, n_lo, n_hi)
        end
        if !is_zero128(b_lo, b_hi)
            x_lo, x_hi = powermod128(b_lo, b_hi, d_lo, d_hi, n_lo, n_hi)
            if !(x_lo == 1 && x_hi == 0) && !(x_lo == n_lo - 1 && x_hi == n_hi - (n_lo == 0 ? 1 : 0))
                comp = true
                for _ in 1:(s - 1)
                    x_lo, x_hi = mulmod128(x_lo, x_hi, x_lo, x_hi, n_lo, n_hi)
                    if x_lo == n_lo - 1 && x_hi == n_hi - (n_lo == 0 ? UInt64(1) : UInt64(0))
                        comp = false
                        break
                    end
                end
                comp && return false
            end
        end
    end

    return true
end

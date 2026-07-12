# algorithms/A8_bpsw.jl
# 自前の BPSW（強 MR 基底2 ＋ 強 Lucas / Selfridge）。BigInt 対応。
# Primes.isprime と一致することを自己検証する。

using Primes: isprime

# Jacobi 記号 (a/n)、n は正の奇数。Primes.jl に依存しないよう自前実装。
function jacobi_symbol(a::BigInt, n::BigInt)
    a = mod(a, n)
    result = 1
    while a != 0
        while iseven(a)
            a >>= 1
            if mod(n, 8) in (3, 5); result = -result; end
        end
        a, n = n, a
        if mod(a, 4) == 3 && mod(n, 4) == 3; result = -result; end
        a = mod(a, n)
    end
    return n == 1 ? result : 0
end

# 強 Miller-Rabin（基底 a）: n が強擬素数なら true
function strong_mr(n::BigInt, a::BigInt)
    n <= 1 && return false
    n == 2 && return true
    iseven(n) && return false
    d, s = n - 1, 0
    while iseven(d); d >>= 1; s += 1; end
    x = powermod(a, d, n)
    x == 1 && return true
    x == n - 1 && return true
    for _ in 1:(s-1)
        x = mod(x*x, n)
        x == n - 1 && return true
    end
    return false
end

# Lucas 倍加: (U,V,Q) -> (U2,V2,Q2)
function lucas_double(U, V, Q, D, n)
    U2 = mod(U * V, n)
    V2 = mod(V*V - 2*Q, n)
    Q2 = mod(Q*Q, n)
    return U2, V2, Q2
end

# Lucas 加算: (Un,Vn,Qn) + (Um,Vm,Qm) -> (U(n+m), V(n+m), Q(n+m))
function lucas_add(Un, Vn, Qn, Um, Vm, Qm, D, n)
    inv2 = (n + 1) ÷ 2                       # 2 の逆元 mod n（n 奇数）
    U = mod(mod(Un*Vm + Um*Vn, n) * inv2, n)
    V = mod(mod(Vn*Vm + Un*Um*D, n) * inv2, n)
    Q = mod(Qn * Qm, n)
    return U, V, Q
end

# 強 Lucas テスト（Selfridge の D 選択）。n が強 Lucas 擬素数なら true。
function strong_lucas(n::BigInt)
    n <= 1 && return false
    n == 2 && return true
    iseven(n) && return false
    Dval = BigInt(5); sign = 1; D = BigInt(5)
    while true
        Dcur = sign > 0 ? BigInt(Dval) : -BigInt(Dval)
        j = jacobi_symbol(Dcur, n)
        if j == 0
            # Dcur と n が非互素。n が素数なら最初に hit するのは Dcur=±n。
            # 巨大 n では j==0 は起きない。小さい n 用の措置。
            return mod(Dcur, n) == 0
        end
        j == -1 && (D = Dcur; break)
        Dval += 2; sign = -sign
        Dval > 100000 && error("D not found")
    end
    P = BigInt(1)
    Q = mod((BigInt(1) - D) * invmod(BigInt(4), n), n)
    delta = n + 1
    d, s = delta, 0
    while iseven(d); d >>= 1; s += 1; end
    U, V, Qc = BigInt(1), P, Q
    bits = reverse(digits(d, base=2))
    for b in bits[2:end]
        U, V, Qc = lucas_double(U, V, Qc, D, n)
        if b == 1
            U, V, Qc = lucas_add(U, V, Qc, BigInt(1), P, Q, D, n)
        end
    end
    U == 0 && return true
    V == 0 && return true          # r = 0 の場合（V_d ≡ 0）
    for _ in 1:(s-1)
        U, V, Qc = lucas_double(U, V, Qc, D, n)
        V == 0 && return true
    end
    return false
end

# BPSW: 強 MR(2) かつ 強 Lucas なら素数と判定
function is_prime_bpsw(n::BigInt)
    n < 2 && return false
    n == 2 && return true
    iseven(n) && return false
    strong_mr(n, BigInt(2)) || return false
    strong_lucas(n) || return false
    return true
end

# 自己検証: Primes.isprime と一致（小範囲全数 ＋ 巨大乱数標本 ＋ 平方数合成）
function selftest_bpsw(; samples=2000)
    for n in BigInt(3):BigInt(3000)
        iseven(n) && continue
        if is_prime_bpsw(n) != isprime(n); return false, n; end
    end
    for _ in 1:samples
        n = BigInt(rand(1:1000)) * BigInt(10)^20 + BigInt(rand(1:1000))
        if is_prime_bpsw(n) != isprime(n); return false, n; end
        n2 = n * n
        if is_prime_bpsw(n2); return false, n2; end
    end
    return true, nothing
end

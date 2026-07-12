# algorithms/A13_montgomery.jl
# 多倍長 Montgomery 乗算の参照実装。mod(a*b, N) と一致することを検証。

# Montgomery パラメタ。N は奇数。
function mont_setup(N::BigInt)
    s = ndigits(N, base=2)               # N のビット長
    R = BigInt(1) << s                  # R = 2^s > N
    Rinv = invmod(R, N)
    Ninv = mod(-invmod(N, R), R)        # -N^{-1} mod R
    return (R=R, Rinv=Rinv, Ninv=Ninv)
end

# REDC: T < N*R を仮定 → [0, 2N) の値を返す
function redc(T::BigInt, N::BigInt, R::BigInt, Ninv::BigInt)
    m = mod(mod(T, R) * Ninv, R)
    t = (T + m * N) ÷ R
    t >= N && (t -= N)
    return t
end

# a*b mod N（Montgomery 經由）
function mulmod_mont(a::BigInt, b::BigInt, N::BigInt)
    R = BigInt(1) << ndigits(N, base=2)
    Rinv = invmod(R, N); Ninv = mod(-invmod(N, R), R)
    aM = mod(a * R, N)                   # Mont(a)
    bM = mod(b * R, N)                   # Mont(b)
    T = aM * bM
    zM = redc(T, N, R, Ninv)             # Mont(a*b) = a*b*R mod N
    return mod(zM * Rinv, N)             # Mont^{-1}
end

# 自己検証: 通常の mod(a*b, N) と一致
function selftest_montgomery(; trials=5000)
    for _ in 1:trials
        N = BigInt(rand(3:1000)) * BigInt(10)^30 + BigInt(rand(1:1000))
        iseven(N) && continue
        a = BigInt(rand(1:10)^20 + rand(1:1000))
        b = BigInt(rand(1:10)^20 + rand(1:1000))
        if mulmod_mont(a, b, N) != mod(a * b, N)
            return false, (a, b, N)
        end
    end
    return true, nothing
end

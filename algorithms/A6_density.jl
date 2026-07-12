# algorithms/A6_density.jl
# H-W 重み W_k と期待個数の計算。「総なめが不可能なこと」の定量化。
# W_k = ∏_p (1-1/p)^{-k} (1 - ν(p)/p)、ν(p)=BadSet_n の大きさ（p=2 では 1）。

using Printf
using Primes: primes

include("A2_bad_residues.jl")

# 特異級数 W_k を P までの素数で計算（大きい p で収束するので有限精度で出る）。
function singular_series(k::Int; P::Int=1_000_000)
    plist = primes(P)
    W = 1.0
    for p in plist
        if p == 2
            nu = 1                      # n ≡ 0 のみ
        else
            nu = length(bad_residues_n(p, k))
        end
        W *= (1.0 - 1.0/p)^(-k) * (1.0 - nu/p)
    end
    return W
end

# [a,b] 内の長さ k の CC 先頭の期待個数。
# 鎖要素 f_i(n) ≈ 2^i n は幾何的に増大するので、各要素の素数密度は 1/(log t + i·log2)。
# よって密度 = W_k / ∏_{i=0}^{k-1}(log t + i·log2)。素朴な 1/(log t)^k は小 t で発散し
# （prob>1 となり）k と共に過大評価・非単調になるため、この幾何対数版を用いる。
# u = log t で ∫ e^u / ∏_i(u + i·log2) du を台形則（対数刻み）で積分。
const _LN2 = log(2.0)

function expected_count(k::Int, a::Number, b::Number; P::Int=1_000_000, N::Int=20_000)
    W = singular_series(k; P=P)
    la, lb = log(Float64(a)), log(Float64(b))
    du = (lb - la) / N
    function g(u)
        d = 1.0
        for i in 0:(k-1)
            d *= (u + i * _LN2)
        end
        return exp(u) / d
    end
    s = 0.5 * (g(la) + g(lb))
    for i in 1:(N-1)
        s += g(la + i * du)
    end
    return W * s * du
end

# 自己検証: k=2 の W_2 が Sophie Germain 定数 ≈ 1.32032363 に一致
function selftest_density(; P=200_000)
    W2 = singular_series(2; P=P)
    target = 1.32032363
    return abs(W2 - target) < 1e-3, (W2, target)
end

# verify.jl から呼び出す密度レポート
function report_density(; P=200_000)
    println("k | W_k")
    println("--|--------")
    for k in 1:16
        @printf("%2d | %.6f\n", k, singular_series(k; P=P))
    end
    println()
    for (k, a, b) in [(10, 10, big(10)^15), (12, 10, big(10)^18), (15, 10, big(10)^20)]
        @printf("E(CC%d in [10, %.3g]) ≈ %.3e\n", k, Float64(b), expected_count(k, a, b; P=P))
    end
end

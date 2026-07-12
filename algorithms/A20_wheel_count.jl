# algorithms/A20_wheel_count.jl
# wheel 生存残基数を「解析的」に数える: count = ∏_q (q − ν(q))。
# A4 の DFS で列挙した個数と一致するはず（検証）。A6 の W_k の各素因子と対応。

using Primes: primes
include("A2_bad_residues.jl")
include("A4_wheel_crt.jl")

# 各 wheel 素数 q について生存残基数 = q − |BadSet_q^(m)|。
# 全素数で掛け合わせたものが全体の生存残基数（CRT の積行性より）。
function wheel_residue_count(k::Int, wheel_primes::Vector{Int})
    c = 1
    for q in wheel_primes
        nu = length(bad_residues_m(q, k))
        c *= (q - nu)
    end
    return c
end

# 自己検証: A4 の列挙数と一致
function selftest_wheel_count(; k=6, w=6)
    wheel = wheel_primes_for(w)
    analytic = wheel_residue_count(k, wheel)
    enumerated = length(wheel_residues(k, wheel))
    return analytic == enumerated, (analytic, enumerated)
end

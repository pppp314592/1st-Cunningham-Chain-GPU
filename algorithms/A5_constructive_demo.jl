# algorithms/A5_constructive_demo.jl
# A5 プリモリアル・バンド探索の実践デモ。
# 較正済み A6 で「E ≳ 1 となる x 帯」を選び、既知の CC15/CC16 最小頭を
# 総当たりではなく「wheel 生存残基 r だけ」から構成的に再発見する。
# （m = n+1 形式。候補は m = r + t·M の等差数列のみ。Int64 上限を超えても BigInt で直接狙える）

using Primes, Printf, Base.Threads
include("A4_wheel_crt.jl")
include("A2_bad_residues.jl")

function constructive_search(k::Int; m_target::BigInt, H::Int, w::Int, P::Int)
    wheel = wheel_primes_for(w)
    M = prod(BigInt, wheel)
    residues = BigInt[BigInt(r) for r in wheel_residues(k, wheel)]

    extra = primes(P)
    filter!(p -> p > last(wheel), extra)
    extrabad = Dict{Int,Vector{Int}}()
    for p in extra
        seen = Set{Int}()
        for i in 0:(k-1)
            push!(seen, invmod(powermod(2, i, p), p))   # m 形式の悪残基
        end
        extrabad[p] = collect(seen)
    end

    m0 = m_target - BigInt(H)
    m1 = m_target + BigInt(H)
    results = BigInt[]
    lockres = SpinLock()

    @threads for r in residues
        m = r + cld(m0 - r, M) * M          # 最初の m ≥ m0
        localbuf = BigInt[]
        while m <= m1
            ok = true
            for p in extra
                if mod(m, p) in extrabad[p]
                    ok = false; break
                end
            end
            if ok
                good = true
                for i in 0:(k-1)
                    if !isprime(m * BigInt(2)^i - 1); good = false; break; end
                end
                good && push!(localbuf, m - 1)
            end
            m += M
        end
        if !isempty(localbuf)
            lock(lockres) do; append!(results, localbuf); end
        end
    end
    return sort!(results)
end

# ---- 較正済み A6 で選んだバンド（E ≳ 1 の x 帯は 1e20 級） ----
println("=== CC15 構成的再発見 (最小頭 90616211958465842219, 20桁) ===")
t_cc15 = BigInt(90616211958465842219) + 1
@time r15 = constructive_search(15; m_target=t_cc15, H=10_000_000_000, w=8, P=200_000)
println("  検出頭数: ", length(r15))
for n in r15; println("   head = ", n, "  digits=", ndigits(n)); end

println()
println("=== CC16 構成的再発見 (最小頭 810433818265726529159, 21桁・Int64超) ===")
t_cc16 = BigInt(810433818265726529159) + 1
@time r16 = constructive_search(16; m_target=t_cc16, H=500_000_000_000, w=9, P=300_000)
println("  検出頭数: ", length(r16))
for n in r16; println("   head = ", n, "  digits=", ndigits(n)); end

include("../algorithms/A8_bpsw.jl")

function chain_len(n::BigInt)
    m = n + 1
    i = 0
    while is_prime_bpsw(m * (BigInt(2)^i) - 1)
        i += 1
    end
    return i
end

known = [
    ("Wiki len11", parse(BigInt, "554688278429")),
    ("Wiki len12", parse(BigInt, "4090932431513069")),
    ("Wiki len13", parse(BigInt, "95405042230542329")),
    ("Wiki len14", parse(BigInt, "666141958065774597791")),
]
for (lbl, n) in known
    L = chain_len(n)
    over = n > BigInt(typemax(Int64))
    println("$lbl  n=$n  digits=$(ndigits(n))  chain_len=$L  over_Int64=$over")
end

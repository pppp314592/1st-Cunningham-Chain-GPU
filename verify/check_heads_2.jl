include("../algorithms/A8_bpsw.jl")

function chain_len_2(n::BigInt)
    m = n - 1
    i = 0
    while is_prime_bpsw(m * BigInt(2)^i + 1)
        i += 1
    end
    return i
end

# 第二種の既知値（必要に応じて追記）
known = [
    ("CC1_2 head=3",   parse(BigInt, "3")),
    ("CC2_2 head=2",   parse(BigInt, "2")),
]
for (lbl, n) in known
    L = chain_len_2(n)
    over = n > BigInt(typemax(Int64))
    println("$lbl  n=$n  digits=$(ndigits(n))  chain_len=$L  over_Int64=$over")
end

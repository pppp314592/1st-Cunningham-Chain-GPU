# algorithms/A16_known_chains.jl
# 既知の（小さい）第一種カニンガム鎖を独立な経路（直接 MR）で検証し、
# A3/A5 のテストオラクルとする。ハードコード表は直接チェックするので、
# 篩ロジックと無関係に「本当に鎖か」を確かめられる。

using Primes: isprime

# (先頭 p, 主張される長さ k) の表。いずれも下で再検証する。
const KNOWN_CHAINS = [
    (2, 5),    # 2,5,11,23,47 (95 合成)
    (3, 2),    # 3,7 (15 合成)
    (5, 4),    # 5,11,23,47 (95 合成)
    (11, 3),   # 11,23,47 (95 合成)
    (29, 2),   # 29,59 (119=7*17 合成)
    (41, 3),   # 41,83,167 (335 合成)
    (89, 6),   # A3 セルフテストで確認済
    (63419, 6),# A5 セルフテストで確認済
    (127139, 6),
]

# OEIS A005602「最小の（完全な）第一種カニンガム鎖を始める素数」の最小頭。
# 2026-07-12、A8(BPSW)の chain_length で独立検証済（GPUセッションの全走査/128bit
# カーネル検出とも一致）。注: GPUセッション check_heads.jl の定数はラベルが1つ
# ズレており、21桁の 666141958065774597791 は無効（chain_len=0）。正しい最小CC15
# 頭は 90616211958465842219（20桁・Int64超）。
const MINIMAL_HEADS = [
    (665043081119,          11),  # Löh 1989
    (554688278429,          12),  # Löh 1989
    (4090932431513069,      13),  # Brennen 1998
    (95405042230542329,     14),  # Jobling 1999
    (90616211958465842219,  15),  # Sorenson & Webster 2017
    (810433818265726529159, 16),  # Carmody & Jobling 2003 (Int64超)
]

# p が長さ k の第一種鎖を始めるか（直接 MR）。最大長も返す。
function chain_length(p::Integer)
    m = BigInt(p) + 1
    k = 0
    while isprime(m * BigInt(2)^k - 1)
        k += 1
    end
    return k
end

# 自己検証: 表の各 (p,k) について、chain_length(p) == k であること
function selftest_known_chains()
    for (p, k) in KNOWN_CHAINS
        if chain_length(p) != k
            return false, (p, k, chain_length(p))
        end
    end
    return true, nothing
end

# 自己検証: OEIS 最小頭（MINIMAL_HEADS）が主張通りの長さを持つこと
function selftest_minimal_heads()
    for (p, k) in MINIMAL_HEADS
        if chain_length(p) != k
            return false, (p, k, chain_length(p))
        end
    end
    return true, nothing
end

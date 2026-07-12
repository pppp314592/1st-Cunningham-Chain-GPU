include("algorithms/A2_bad_residues.jl")
using Primes: primes
k = 16
nu(q) = length(bad_residues_m(q, k))   # 生存 = q - nu(q)
# 現状 wheel (k>=14)
cur = [3,5,7,11,13,17,19,23,37,41,43]   # 2は別扱い
allsmall = filter(p->p>2, primes(200))
# 貪欲: density削減/対数(wheel成長) が大きい順に足す指標も出す
function analyze(wset)
    wheelodd = prod(BigInt.(wset))
    wheel = 2*wheelodd
    R = prod(BigInt(q)-nu(q) for q in wset)  # 奇素数部の生存数(2は係数1)
    dens = prod((BigInt(q)-nu(q))//q for q in wset)  # 奇部密度
    return wheel, R, Float64(dens)
end
w0,R0,d0 = analyze(cur)
println("=== 現状 wheel (奇素数 $(length(cur))個) ===")
println("wheel=$(2*prod(BigInt.(cur)))  R(奇)=$R0  density=$(d0)")
println()
println("=== 素数を1個ずつ追加した効果 (k=16) ===")
println("add | ord2 | nu | 生存/q | 新density比 | 新R | 新wheel(<2^51=$(2^51)?)")
for q in [29,31,47,53,59,61,67,71,73,79,83]
    ws = sort(vcat(cur,[q]))
    w,R,d = analyze(ws)
    ord = let o=1,x=2%q; while x!=1; x=(2x)%q; o+=1; end; o end
    rel = d/d0
    fits = w < (BigInt(2)^51)
    println("$q | $ord | $(nu(q)) | $(q-nu(q))/$q | ×$(round(rel,digits=3)) | $R | wheel=$(w) fits2^51=$fits")
end

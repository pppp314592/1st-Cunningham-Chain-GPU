include("algorithms/A6_density.jl")
using Printf
act = Dict(12=>105, 13=>8, 14=>1, 15=>0, 16=>0)
for k in 12:16
    e = expected_count(k, 10, BigInt(10)^17)
    @printf("E(CC%d, 1e17) = %10.4f   実測=%d\n", k, e, act[k])
end

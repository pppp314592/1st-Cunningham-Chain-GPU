include("prime_filter.jl")
using Primes

println("=== 10^7 〜 10^8 ===")
arr = collect(10^7:10^8)
gpu_cnt = length(filter_primes(arr))
cpu_cnt = length(primes(10^7, 10^8))
println("GPU: $gpu_cnt")
println("CPU: $cpu_cnt")
println("一致: $(gpu_cnt == cpu_cnt)")

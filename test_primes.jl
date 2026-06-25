using Primes
include("prime_filter.jl")

println("=== 小型テスト ===")
arr = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
gpu_result = filter_primes(arr)
cpu_result = [n for n in arr if isprime(n)]
println("GPU:  $gpu_result")
println("CPU:  $cpu_result")
println("一致: $(gpu_result == cpu_result)")

println("\n=== 中規模テスト (10^4〜10^4+1000) ===")
arr2 = collect(10_000:11_000)
gpu_result2 = filter_primes(arr2)
cpu_result2 = [n for n in arr2 if isprime(n)]
println("GPU 件数: $(length(gpu_result2))")
println("CPU 件数: $(length(cpu_result2))")
println("一致: $(gpu_result2 == cpu_result2)")

println("\n=== 大規模テスト (10^8〜10^8+10000) ===")
arr3 = collect(100_000_000:100_010_000)
gpu_result3 = filter_primes(arr3)
cpu_result3 = primes(100_000_000, 100_010_000)
println("GPU 件数: $(length(gpu_result3))")
println("CPU 件数: $(length(cpu_result3))")
println("一致: $(gpu_result3 == cpu_result3)")

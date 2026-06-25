include("cc_gpu.jl")

lo = 3300000000000000000
hi = lo + 10000000000000000

wheel, _, _, wheel_n_d, primes_d, bad_data_d, offs_d, np = _get_cached_sieve(15)
start_k = lo ÷ wheel
end_k = (hi - 1) ÷ wheel
ncycles = end_k - start_k + 1

# 初回（コンパイル込み）
k = start_k
t1 = @elapsed c1 = _gpu_sieve_cycle64(k, wheel, lo, hi, wheel_n_d, primes_d, bad_data_d, offs_d, np)
println("初回（コンパイル込み）: $(length(c1)) candidates, $(round(t1, digits=3))s")

# 2回目以降（コンパイルなし）
k = start_k + 1
t2 = @elapsed c2 = _gpu_sieve_cycle64(k, wheel, lo, hi, wheel_n_d, primes_d, bad_data_d, offs_d, np)
println("2回目: $(length(c2)) candidates, $(round(t2, digits=3))s")

k = start_k + 2
t3 = @elapsed c3 = _gpu_sieve_cycle64(k, wheel, lo, hi, wheel_n_d, primes_d, bad_data_d, offs_d, np)
println("3回目: $(length(c3)) candidates, $(round(t3, digits=3))s")

println()
println("推定: 1サイクル平均 $(round((t2+t3)/2 * 1000, digits=1))ms × $ncycles サイクル = $(round((t2+t3)/2 * ncycles, digits=1))s/バッチ")

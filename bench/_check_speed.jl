include("../src/cc_gpu.jl")

lo = 3300000000000000000
hi = lo + 10000000000000000

wheel, _, _, wheel_n_d, primes_d, bad_data_d, offs_d, np = _get_cached_sieve(15)
start_k = lo ﾃｷ wheel
end_k = (hi - 1) ﾃｷ wheel
ncycles = end_k - start_k + 1

# 蛻晏屓・医さ繝ｳ繝代う繝ｫ霎ｼ縺ｿ・・k = start_k
t1 = @elapsed c1 = _gpu_sieve_cycle64(k, wheel, lo, hi, wheel_n_d, primes_d, bad_data_d, offs_d, np)
println("蛻晏屓・医さ繝ｳ繝代う繝ｫ霎ｼ縺ｿ・・ $(length(c1)) candidates, $(round(t1, digits=3))s")

# 2蝗樒岼莉･髯搾ｼ医さ繝ｳ繝代う繝ｫ縺ｪ縺暦ｼ・k = start_k + 1
t2 = @elapsed c2 = _gpu_sieve_cycle64(k, wheel, lo, hi, wheel_n_d, primes_d, bad_data_d, offs_d, np)
println("2蝗樒岼: $(length(c2)) candidates, $(round(t2, digits=3))s")

k = start_k + 2
t3 = @elapsed c3 = _gpu_sieve_cycle64(k, wheel, lo, hi, wheel_n_d, primes_d, bad_data_d, offs_d, np)
println("3蝗樒岼: $(length(c3)) candidates, $(round(t3, digits=3))s")

println()
println("謗ｨ螳・ 1繧ｵ繧､繧ｯ繝ｫ蟷ｳ蝮・$(round((t2+t3)/2 * 1000, digits=1))ms ﾃ・$ncycles 繧ｵ繧､繧ｯ繝ｫ = $(round((t2+t3)/2 * ncycles, digits=1))s/繝舌ャ繝・)

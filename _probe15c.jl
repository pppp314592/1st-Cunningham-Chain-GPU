include("cc_gpu.jl")

# 1回目 (コンパイル込み)
t1 = @elapsed res1 = search_cc_gpu(10^18, 10^18 + 10^16, 15, verbose=false)
println("1回目: $(length(res1)) 件, $(round(t1, digits=1))s")

# 2回目 (キャッシュ済み)
t2 = @elapsed res2 = search_cc_gpu(10^18 + 10^16, 10^18 + 2*10^16, 15, verbose=false)
println("2回目: $(length(res2)) 件, $(round(t2, digits=1))s")

# 3回目
t3 = @elapsed res3 = search_cc_gpu(10^18 + 2*10^16, 10^18 + 3*10^16, 15, verbose=false)
println("3回目: $(length(res3)) 件, $(round(t3, digits=1))s")

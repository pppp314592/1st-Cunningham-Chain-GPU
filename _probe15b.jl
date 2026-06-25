include("cc_gpu.jl")

# CC15 1バッチ (10^18〜10^18+10^16) の時間計測
t = @elapsed res = search_cc_gpu(10^18, 10^18 + 10^16, 15, verbose=false)
println("CC15 [10^18, 10^18+10^16): $(length(res)) 件, $(round(t, digits=1))s")

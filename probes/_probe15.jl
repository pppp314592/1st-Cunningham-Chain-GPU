include("../src/cc_gpu.jl")
t = @elapsed res = search_cc_gpu(10^18, 10^18 + 10^16, 15, verbose=false)
println("結果: $(length(res)) 件, $(round(t, digits=1))s")

include("../src/cc_gpu.jl")
println("cc_gpu.jl loaded OK")
@time res = search_cc_gpu(1, 100000, 5, verbose=false)
println("CC5 [1,100000): $(length(res))件, 結果: $res")

include("src/cc_gpu.jl"); include("src/cc_gpu_wheel.jl")
cases = [(8, Int64(10)^9, Int64(10)^9+2*10^8),
         (13, Int64(4_090_932_431_000_000), Int64(4_090_932_432_000_000))]
allok=true
for (k,lo,hi) in cases
  r64 = sort(Int128.(search_cc_gpu_wheel(lo,hi,k;verbose=false)))
  r128 = sort(search_cc_gpu_wheel128(Int128(lo),Int128(hi),k;verbose=false))
  m = r64==r128; global allok &= m
  println("CC$k: 64bit=$(length(r64)) 128bit=$(length(r128)) match=$m")
  m || println("  64=$r64\n 128=$r128")
end
println(allok ? "64bit vs 128bit ALL MATCH" : "不一致")

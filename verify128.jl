include("src/cc_gpu.jl"); include("src/cc_gpu_wheel.jl")
target = parse(Int128,"90616211958465842219")
lo = target - Int128(5)*10^14
hi = target + Int128(5)*10^14
println("band=[$lo, $hi]  target=$target (>Int64=$(target>typemax(Int64)))")
r = search_cc_gpu_wheel128(lo, hi, 15; verbose=true)
println("found=$r")
println(target in r ? "✔ SUCCESS: CC15 を再発見 (Int128カーネル)" : "�’EFAIL: 見つからず")

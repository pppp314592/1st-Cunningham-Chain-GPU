include("src/cc_gpu.jl"); include("src/cc_gpu_wheel.jl")
target = parse(Int128,"810433818265726529159")   # OEIS A005602 最小CC16
lo = target - Int128(5)*10^14
hi = target + Int128(5)*10^14
println("CC16 target=$target (>Int64=$(target>typemax(Int64)))  band=±5e14")
r = search_cc_gpu_wheel128(lo, hi, 16; verbose=true)
println("found=$r")
println(target in r ? "✔ SUCCESS: 最小CC16 (8.1e20) を検出 — GPU融合カーネルがCC16域に到達" : "�’EFAIL")

include("src/cc_cpu.jl")
include("src/cc_gpu_wheel.jl")
using CUDA

# stream 拡張wheel(31,47) で既知 CC16 を実機再発見 (1 cyc = 全生存残基を走査)
head = Int128(810433818265726529159)   # 既知最小 CC16 (8.1e20)
k = 16
lo = head - Int128(10)^15
hi = head + Int128(10)^15
println("rediscover CC16 head=$head  band=±1e15")
t0 = time()
r = search_cc_gpu_wheel_stream128(lo, hi, k; progress=true)
dt = time() - t0
println("found=$r")
println(head in r ? "PASS: CC16 再発見 in $(round(dt,digits=1))s" : "FAIL")

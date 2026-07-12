include("src/cc_cpu.jl")
include("src/cc_gpu_wheel.jl")

println("threads=$(Threads.nthreads())")

# warmup
search_cc_cpu(Int64(10)^9, Int64(10)^9+10^6, 10; verbose=false)
search_cc_gpu_wheel(Int64(10)^9, Int64(10)^9+10^6, 10; verbose=false)

function bench(lo, hi, k)
    span = hi-lo
    t0=time(); r_cpu = search_cc_cpu(Int64(lo),Int64(hi),k; verbose=false); tc=time()-t0
    t0=time(); r_gpu = search_cc_gpu_wheel(lo,hi,k; verbose=false); tg=time()-t0
    m = sort(Int64.(r_cpu))==sort(r_gpu)
    println("CC$k span=$span cpu=$(length(r_cpu))/$( round(tc,digits=2))s  gpu=$(length(r_gpu))/$(round(tg,digits=2))s  match=$m  speedup=$(round(tc/max(tg,1e-9),digits=2))x")
end

# 大きめレンジで実測 (CPUが数十秒になる規模)
bench(Int64(10)^12, Int64(10)^12 + Int64(5)*10^11, 10)
bench(Int64(10)^14, Int64(10)^14 + Int64(2)*10^12, 12)

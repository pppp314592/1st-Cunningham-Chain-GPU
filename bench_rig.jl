include("src/cc_cpu.jl")
include("src/cc_gpu_wheel.jl")

function cpu_scan(wheel, wheel_n, primes_list, bad_flags, lo::Int64, hi::Int64, k::Int)
    w = Int64(wheel)
    k_start = lo ÷ w; k_end = (hi-1) ÷ w
    nt = Threads.nthreads()
    total = k_end-k_start+1
    chunk = max(1, total ÷ nt)
    ranges = NTuple{2,Int64}[]
    for t in 1:nt
        ks = k_start+(t-1)*chunk
        ke = (t==nt) ? k_end : min(ks+chunk-1,k_end)
        ks>ke && break
        push!(ranges,(ks,ke))
    end
    allc = Int[]
    alock = ReentrantLock()
    Threads.@threads for ri in 1:length(ranges)
        ks,ke = ranges[ri]
        c = _sieve_worker64(ks,ke,w,wheel_n,lo,hi,primes_list,bad_flags)
        isempty(c) || lock(alock) do; append!(allc,c); end
    end
    return sort(Int64.(_filter_cc64(allc,k)))
end

function run(k, lo::Int64, hi::Int64)
    wheel,wheel_n,mdllist = build_cc_sieve(k)
    primes_list,bad_flags = _flatten_badflags(mdllist)
    st = gpu_wheel_setup(k; verbose=false)
    cpu_scan(wheel,wheel_n,primes_list,bad_flags,lo,lo+Int64(wheel)*2,k)  # warm
    gpu_wheel_scan!(st, lo, lo+Int64(wheel)*2)                           # warm
    tc = @elapsed cpu_scan(wheel,wheel_n,primes_list,bad_flags,lo,hi,k)
    tc = @elapsed r_cpu = cpu_scan(wheel,wheel_n,primes_list,bad_flags,lo,hi,k)
    tg = @elapsed gpu_wheel_scan!(st, lo, hi)
    tg = @elapsed r_gpu = gpu_wheel_scan!(st, lo, hi)
    cyc = (hi-1)÷Int64(wheel) - lo÷Int64(wheel) + 1
    println("CC$k span=$(hi-lo) cycles=$cyc")
    println("  CPU $(length(r_cpu)) $(round(tc,digits=3))s   GPU $(length(r_gpu)) $(round(tg,digits=3))s   match=$(r_cpu==r_gpu)  speedup=$(round(tc/tg,digits=2))x")
end

println("threads=$(Threads.nthreads())")
run(12, Int64(10)^16, Int64(10)^16 + Int64(680)*10^12)   # ~2000 cycles
run(13, Int64(10)^16, Int64(10)^16 + Int64(680)*10^12)
run(10, Int64(10)^15, Int64(10)^15 + Int64(5)*10^11)     # ~2240 cycles (small wheel)

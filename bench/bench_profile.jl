include("src/cc_cpu.jl")

function profile_scan(lo::Int128, hi::Int128, k::Int)
    wheel, wheel_n, mdllist = build_cc_sieve(k)
    primes_list, bad_flags = _flatten_badflags(mdllist)
    w128 = Int128(wheel)
    k_start = lo ÷ w128
    k_end = (hi - 1) ÷ w128
    nt = Threads.nthreads()
    total = k_end - k_start + 1
    chunk = max(1, total ÷ nt)
    ranges = NTuple{2,Int128}[]
    for t in 1:nt
        ks = k_start + (t-1)*chunk
        ke = (t==nt) ? k_end : min(ks+chunk-1, k_end)
        ks > ke && break
        push!(ranges, (ks,ke))
    end
    # sieve phase
    all_c = Int128[]
    alock = ReentrantLock()
    t0 = time()
    Threads.@threads for ri in 1:length(ranges)
        ks,ke = ranges[ri]
        c = _sieve_worker128(ks,ke,wheel,wheel_n,lo,hi,primes_list,bad_flags)
        isempty(c) || lock(alock) do; append!(all_c,c); end
    end
    t_sieve = time()-t0
    nsurv = length(all_c)
    # filter phase
    t0 = time()
    res = _filter_cc128(all_c, k)
    t_filter = time()-t0
    span = hi - lo
    println("CC$k span=$span  survivors=$nsurv  found=$(length(res))")
    println("   sieve=$(round(t_sieve,digits=3))s  filter=$(round(t_filter,digits=3))s  total=$(round(t_sieve+t_filter,digits=3))s")
    println("   survivors/sieve-sec=$(round(nsurv/max(t_sieve,1e-9),digits=0))  scan-rate=$(round(Float64(span)/max(t_sieve+t_filter,1e-9),digits=0))/s")
    return res
end

println("threads=$(Threads.nthreads())")
profile_scan(Int128(10)^9, Int128(10)^9 + Int128(10)^9, 10)   # warmup+CC10
println("---")
profile_scan(Int128(10)^12, Int128(10)^12 + Int128(2)*10^9, 10)
println("---")
profile_scan(Int128(10)^12, Int128(10)^12 + Int128(2)*10^9, 12)

include("src/cc_cpu.jl")        # build_cc_sieve
include("src/cc_gpu_wheel.jl")

const LO = Int64(10)
const HI = Int64(10)^17   # 100000000000000000

expected = Dict(12=>105, 13=>8, 14=>1, 15=>0)

open("logs/newgpu_scan.log", "w") do io
    println(io, "=== 新GPU (search_cc_gpu_wheel) 全走査 [$LO, $HI] ===")
    println(io, "threads=$(Threads.nthreads())")
    flush(io)
    for k in [12, 13, 14, 15, 16]
        println("### CC$k scan start")
        t0 = time()
        st = gpu_wheel_setup(k; verbose=true)
        tsetup = time() - t0
        t1 = time()
        res = gpu_wheel_scan!(st, LO, HI; progress=true)
        tscan = time() - t1
        exp = get(expected, k, -1)
        okstr = exp >= 0 ? (length(res)==exp ? "OK(=$exp)" : "MISMATCH(exp=$exp)") : "NEW"
        line = "CC$k: found=$(length(res))  setup=$(round(tsetup,digits=1))s scan=$(round(tscan,digits=1))s  [$okstr]"
        println(line)
        println(io, line)
        for r in res
            println(io, "  CC$k: $r")
        end
        flush(io)
        # free device residue array before next k
        st = nothing
        GC.gc(); CUDA.reclaim()
    end
    println(io, "=== DONE ===")
end
println("=== ALL DONE, see logs/newgpu_scan.log ===")

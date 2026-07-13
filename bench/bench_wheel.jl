include("src/cc_cpu.jl")
include("src/cc_gpu_wheel.jl")

println("threads=$(Threads.nthreads())")

# --- warmup / correctness on a small range with known CC10 ---
# 一致検証: CPU リファレンス vs GPU-wheel
function check(lo, hi, k)
    r_cpu = sort(Int64.(search_cc_cpu(Int64(lo), Int64(hi), k; verbose=false)))
    r_gpu = sort(search_cc_gpu_wheel(lo, hi, k; verbose=false))
    ok = r_cpu == r_gpu
    println("CC$k [$lo,$hi]: cpu=$(length(r_cpu)) gpu=$(length(r_gpu)) match=$ok")
    if !ok
        println("  cpu=$r_cpu")
        println("  gpu=$r_gpu")
    end
    return ok
end

# 既知 CC10 開始値: 33send... use known small CC: 1418575498567 is CC10? use range scanning
# 小範囲で存在確認しつつ一致だけ検証 (warmup)
check(Int64(10)^9, Int64(10)^9 + 5*10^7, 6)
check(Int64(10)^9, Int64(10)^9 + 5*10^7, 8)
check(Int64(10)^11, Int64(10)^11 + 2*10^8, 10)

# --- speed benchmark CC10 ---
for (lo,hi) in [(Int64(10)^12, Int64(10)^12 + Int64(10)^10)]
    span = hi-lo
    t0=time(); r_cpu = search_cc_cpu(Int64(lo),Int64(hi),10; verbose=false); tc=time()-t0
    t0=time(); r_gpu = search_cc_gpu_wheel(lo,hi,10; verbose=false); tg=time()-t0
    println("\nCC10 span=$span")
    println("  CPU  found=$(length(r_cpu)) time=$(round(tc,digits=3))s")
    println("  GPU  found=$(length(r_gpu)) time=$(round(tg,digits=3))s")
    println("  match=$(sort(Int64.(r_cpu))==sort(r_gpu))  speedup=$(round(tc/max(tg,1e-9),digits=2))x")
end

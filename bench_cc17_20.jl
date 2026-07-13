# CC17..CC20 の同一帯域でのスキャン所要時間測定（各 2 チャンク）。
# CC16 が ~30s/chunk なので、鎖が長くなると密度低下で若干速くなるはず。
include("src/cc_cpu.jl")
include("src/cc_gpu_wheel.jl")

const LO = Int128(10)^19
const CHUNK = Int128(10)^17
const NCHUNK = 2

open("logs/bench_cc17_20.result.txt", "w") do out
    println(out, "CC17..CC20 scan time, 2 chunks each, band [1e19, 1e19+2e17), method=ext")
    for k in 17:20
        println("==== CC$k ====")
        st = gpu_wheel_setup(k; wl = _cc_wl_ext(k), verbose=true)
        scanfn(lo, hi) = gpu_wheel_scan128!(st, lo, hi; progress=false)
        ts = Float64[]
        cursor = LO
        for c in 1:NCHUNK
            chi = cursor + CHUNK
            t0 = time()
            scanfn(cursor, chi)
            push!(ts, time() - t0)
            cursor = chi
        end
        avg = sum(ts) / length(ts)
        line = "CC$k: chunks=$NCHUNK times=$(round.(ts,digits=2))s avg=$(round(avg,digits=2))s/chunk"
        println(line)
        println(out, line)
        CUDA.unsafe_free!(st.d_out)
    end
end

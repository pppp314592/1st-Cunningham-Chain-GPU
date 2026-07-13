# 3-way comparison benchmark (current definitions):
#   base   = _cc_wl(16)         = [2,3,5,7,11,13,17,19,23,37,41,43]
#   ext    = _cc_wl_ext(16)     = [2,3,5,7,11,13,17,19,23,29,37,41,43]
#   stream = _cc_wl_stream(16)  = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]
# Run each for DURATION seconds over the same band; compare measured throughput (range width / sec).
include("src/cc_cpu.jl")
include("src/cc_gpu_wheel.jl")

const OUT = "logs/bench_cmp3.result.txt"
open(OUT, "w") do f; println(f, "bench start $(Libc.strftime(time()))"); end
logmsg(msg) = (open(OUT, "a") do f; println(f, msg); end; println(msg);)

const K = 16
const LO = Int128(10)^19
const SPAN = Int128(11) * 10^17
const CHUNK = Int128(10)^17
const DURATION = 300.0

function run_one(label, scanfn, wl_desc)
    logmsg("==================== $label ====================")
    logmsg("  wheel: $wl_desc")
    t0 = time()
    cursor = LO
    nchunk = 0
    while cursor < LO + SPAN
        chi = min(cursor + CHUNK, LO + SPAN)
        scanfn(cursor, chi)
        nchunk += 1
        if time() - t0 >= DURATION
            break
        end
        cursor = chi
    end
    el = time() - t0
    covered = cursor - LO
    thr = Float64(covered) / el
    logmsg("  chunks=$nchunk  elapsed=$(round(el,digits=1))s  covered=$(covered)  throughput=$(round(thr/1e15,digits=3))e15 range/s")
    return thr
end

st_base = gpu_wheel_setup(K; wl = _cc_wl(K), verbose=true)
scan_base(lo, hi) = gpu_wheel_scan128!(st_base, lo, hi; progress=false)
thr_base = run_one("BASE (_cc_wl)", scan_base, "2,3,5,7,11,13,17,19,23,37,41,43")

st_ext = gpu_wheel_setup(K; wl = _cc_wl_ext(K), verbose=true)
scan_ext(lo, hi) = gpu_wheel_scan128!(st_ext, lo, hi; progress=false)
thr_ext = run_one("EXT (_cc_wl_ext)", scan_ext, "2,3,5,7,11,13,17,19,23,29,37,41,43")

st_s = gpu_wheel_stream_setup(K; verbose=true)
scan_s(lo, hi) = gpu_wheel_scan_stream128!(st_s, lo, hi; progress=false)
thr_s = run_one("STREAM (_cc_wl_stream)", scan_s, "2,3,5,7,11,13,17,19,23,29,31,37,41,43,47")

logmsg("==================== RESULT (range/s, larger=faster) ====================")
logmsg("  BASE   (_cc_wl)        : $(round(thr_base/1e15,digits=3))e15")
logmsg("  EXT    (_cc_wl_ext)    : $(round(thr_ext/1e15,digits=3))e15  ($(round(thr_ext/thr_base,digits=3))x of BASE)")
logmsg("  STREAM (_cc_wl_stream) : $(round(thr_s/1e15,digits=3))e15  ($(round(thr_s/thr_base,digits=3))x of BASE)")

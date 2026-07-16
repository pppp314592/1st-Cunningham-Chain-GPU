# ============================================================
# 本番用 第二種カニンガム鎖 実走査スクリプト
# ------------------------------------------------------------
# 第一種版 scan.jl の第二種対応。鎖は p, 2p-1, 4p-3, … (進行: x -> 2x-1)。
# 使い方も第一種と同じ:
#   julia -g0 -t 12 scan2.jl <k> <start> <width> [chunk] [ext|stream] [--fresh]
#
# 例 (CC16 を第二種で [8.1e20, 8.2e20) 走査):
#   julia -g0 -t 12 scan2.jl 16 810000000000000000000 10000000000000000000
#
# 出力:
#   logs/scan2_k<k>.found.log … 検証済み発見鎖 (追記)
#   logs/scan2_k<k>.ckpt       … 再開用チェックポイント
#   logs/scan2_k<k>.run.log    … 実行ログ (追記)
# ============================================================
include("src/cc_cpu.jl")        # build_cc_sieve_2, is_prime_mr_cpu, is_prime_mr128_cpu
include("src/cc_gpu_wheel.jl")  # gpu_wheel_setup_2 / scan128_2 / stream_2

# ---- CPU 独立検証: n, 2n-1, 4n-3, … が何項まで素数か (第二種)。実鎖長を返す ----
# k 項をクリアしたらさらに延伸して実鎖長を測る（k=K の篩で CC(K+1) 等も検出するため、上限 k+EXTEND まで）。
const _CC2_VERIFY_EXTEND = 8
function verify_cc_2(n::Int128, k::Int)::Int
    n <= 0 && return 0
    x_lo = UInt64(n & 0xFFFFFFFFFFFFFFFF)
    x_hi = UInt64((n >> 64) & 0xFFFFFFFFFFFFFFFF)
    len = 0
    @inbounds for _ in 1:(k + _CC2_VERIFY_EXTEND)
        ok = if x_hi == 0 && x_lo <= 0x7FFFFFFFFFFFFFFF
            is_prime_mr_cpu(Int64(x_lo))
        else
            x_hi > 0x7FFFFFFFFFFFFFFF ? false : is_prime_mr128_cpu(x_lo, x_hi)
        end
        ok || break
        len += 1
        carry = (x_lo >> 63) & UInt64(1)
        x_lo = x_lo << 1
        x_hi = (x_hi << 1) | carry
        if x_lo == 0
            x_lo = 0xFFFFFFFFFFFFFFFF
            x_hi -= UInt64(1)
        else
            x_lo -= UInt64(1)
        end
    end
    return len
end

# ---- 引数パース ----
if length(ARGS) < 3
    println("使い方: julia -g0 -t <N> scan2.jl <k> <start> <width> [chunk] [ext|stream] [--fresh]")
    exit(1)
end
fresh  = "--fresh" in ARGS
pos    = filter(a -> !startswith(a, "--"), ARGS)
k      = parse(Int, pos[1])
start  = parse(Int128, pos[2])
width  = parse(Int128, pos[3])
rend   = start + width
method = length(pos) >= 5 ? lowercase(pos[5]) : "ext"
@assert method in ("ext","stream") "method は ext か stream"

# ---- GPU 状態セットアップ (1回のみ) ----
println("=== CC$k (2nd kind) 実走査 [$start, $rend)  method=$method ===")
if method == "stream"
    st = gpu_wheel_stream_setup_2(k; verbose=true)
    scanfn = (lo, hi) -> gpu_wheel_scan_stream128_2!(st, lo, hi; progress=true)
else
    st = gpu_wheel_setup_2(k; wl=_cc_wl_ext(k), verbose=true)
    scanfn = (lo, hi) -> gpu_wheel_scan128_2!(st, lo, hi; progress=true)
end
wheel = Int128(st.wheel)
println("wheel=$(st.wheel) R=$(st.R) primes=$(st.nprimes)")

# チャンク幅: 未指定 or 0以下 なら ~50 分割 (最低でも wheel 幅)
chunk = (length(pos) >= 4 && parse(Int128, pos[4]) > 0) ? parse(Int128, pos[4]) : max(wheel, width ÷ 50)

# ---- チェックポイント (再開) ----
ckpt = "logs/scan2_k$(k).ckpt"
cursor = start
if !fresh && isfile(ckpt)
    lines = readlines(ckpt)
    if length(lines) >= 3
        c_start = parse(Int128, lines[1]); c_width = parse(Int128, lines[2]); c_cur = parse(Int128, lines[3])
        if c_start == start && c_width == width && c_cur >= rend
            println(">> この範囲は既に完了済み (checkpoint cursor=$c_cur ≥ 終端)。再走査するには --fresh を付けてください。")
            method == "ext" && CUDA.unsafe_free!(st.d_out)
            exit(0)
        elseif c_start == start && c_width == width && c_cur > start
            cursor = c_cur
            println(">> チェックポイントから再開: cursor=$cursor ($(round(100*Float64((cursor-start))/Float64(width),digits=1))% 済)")
        end
    end
end

runlog = "logs/scan2_k$(k).run.log"
foundlog = "logs/scan2_k$(k).found.log"
log(msg) = (open(runlog,"a") do io; println(io, "$(Libc.strftime(time()))  $msg"); end; println(msg))

log("START k=$k range=[$start,$rend) chunk=$chunk method=$method cursor=$cursor")

# ---- 走査ループ ----
t0 = time(); nfound = 0; nchunk = 0
while cursor < rend
    global cursor, nfound, nchunk
    chi = min(cursor + chunk, rend)
    nchunk += 1
    cres = scanfn(cursor, chi)
    # 発見鎖を CPU で独立検証（実鎖長を判定）してから記録
    for x in cres
        len = verify_cc_2(x, k)
        if len >= k
            nfound += 1
            open(foundlog, "a") do io
                println(io, "$x  # CC$len (2nd) verified $(Libc.strftime(time())) chunk=[$cursor,$chi)")
            end
            log("★FOUND & VERIFIED  CC$len (2nd) head = $x")
        elseif len > 0
            open(foundlog, "a") do io
                println(io, "$x  # CC$len (2nd, <k=$k) verified $(Libc.strftime(time())) chunk=[$cursor,$chi)")
            end
            log("◇SUBCHAIN  CC$len (2nd) head = $x  (k=$k 篩ヒットだが実鎖長 $len)")
        else
            log("⚠GPUヒットが検証不合格 (要調査): $x  chunk=[$cursor,$chi)")
        end
    end
    # チェックポイント更新
    open(ckpt, "w") do io
        println(io, start); println(io, width); println(io, chi)
    end
    frac = Float64(chi - start) / Float64(width)
    el = time() - t0
    eta = frac > 0 ? el/frac*(1-frac) : 0.0
    log("chunk$nchunk [$cursor,$chi) 完了 | $(round(100*frac,digits=2))% | 経過$(round(el/3600,digits=2))h | ETA$(round(eta/3600,digits=2))h | 累計発見=$nfound")
    cursor = chi
end

log("DONE k=$k found=$nfound in $(round((time()-t0)/3600,digits=2))h")
println("=== 完了: $nfound 本 (検証済み)。詳細 $foundlog ===")
method == "ext" && CUDA.unsafe_free!(st.d_out)

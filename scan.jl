# ============================================================
# 本番用 第一種カニンガム鎖 実走査スクリプト
# ------------------------------------------------------------
# 特徴:
#   - 範囲をチャンク分割し、チャンクごとにチェックポイントを保存 → 中断/再開可能
#   - 発見した鎖先頭は CPU (is_cc128) で独立検証してからログに追記保存
#   - ext(materialize拡張wheel) / stream(巨大wheel) を選択可能
#   - 進捗 (%/経過/ETA/発見数) を随時表示
#
# 使い方:
#   julia -g0 -t 12 scan.jl <k> <start> <width> [chunk] [ext|stream] [--fresh]
#
# 例 (CC16 を [8.1e20, 8.2e20) 走査, 既定チャンク):
#   julia -g0 -t 12 scan.jl 16 810000000000000000000 10000000000000000000
#
# 例 (巨大範囲を stream 方式で, 明示チャンク 2e18):
#   julia -g0 -t 12 scan.jl 16 810000000000000000000 100000000000000000000 2000000000000000000 stream
#
# 出力:
#   logs/scan_k<k>.found.log … 検証済み発見鎖 (追記)
#   logs/scan_k<k>.ckpt       … 再開用チェックポイント (start,width,cursor)
#   logs/scan_k<k>.run.log    … 実行ログ (追記)
# ============================================================
include("src/cc_cpu.jl")        # build_cc_sieve, is_prime_mr_cpu, is_prime_mr128_cpu
include("src/cc_gpu_wheel.jl")  # gpu_wheel_setup / scan128 / stream

# ---- CPU 独立検証: n, 2n+1, 4n+3, … の k 項が全て素数か (is_cc128 相当) ----
function verify_cc(n::Int128, k::Int)::Bool
    n <= 0 && return false
    x_lo = UInt64(n & 0xFFFFFFFFFFFFFFFF)
    x_hi = UInt64((n >> 64) & 0xFFFFFFFFFFFFFFFF)
    @inbounds for _ in 1:k
        if x_hi == 0 && x_lo <= 0x7FFFFFFFFFFFFFFF
            is_prime_mr_cpu(Int64(x_lo)) || return false
        else
            x_hi > 0x7FFFFFFFFFFFFFFF && return false
            is_prime_mr128_cpu(x_lo, x_hi) || return false
        end
        carry = (x_lo > 0x7FFFFFFFFFFFFFFF) ? UInt64(1) : UInt64(0)
        x_lo = (x_lo << 1) | UInt64(1)
        x_hi = (x_hi << 1) | carry
    end
    return true
end

# ---- 引数パース ----
if length(ARGS) < 3
    println("使い方: julia -g0 -t <N> scan.jl <k> <start> <width> [chunk] [ext|stream] [--fresh]")
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
println("=== CC$k 実走査 [$start, $rend)  method=$method ===")
if method == "stream"
    st = gpu_wheel_stream_setup(k; verbose=true)
    scanfn = (lo, hi) -> gpu_wheel_scan_stream128!(st, lo, hi; progress=true)
else
    st = gpu_wheel_setup(k; wl=_cc_wl_ext(k), verbose=true)
    scanfn = (lo, hi) -> gpu_wheel_scan128!(st, lo, hi; progress=true)
end
wheel = Int128(st.wheel)
println("wheel=$(st.wheel) R=$(st.R) primes=$(st.nprimes)")

# チャンク幅: 未指定 or 0以下 なら ~50 分割 (最低でも wheel 幅)
chunk = (length(pos) >= 4 && parse(Int128, pos[4]) > 0) ? parse(Int128, pos[4]) : max(wheel, width ÷ 50)

# ---- チェックポイント (再開) ----
ckpt = "logs/scan_k$(k).ckpt"
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

runlog = "logs/scan_k$(k).run.log"
foundlog = "logs/scan_k$(k).found.log"
log(msg) = (open(runlog,"a") do io; println(io, "$(Libc.strftime(time()))  $msg"); end; println(msg))

log("START k=$k range=[$start,$rend) chunk=$chunk method=$method cursor=$cursor")

# ---- 走査ループ ----
t0 = time(); nfound = 0; nchunk = 0
while cursor < rend
    global cursor, nfound, nchunk
    chi = min(cursor + chunk, rend)
    nchunk += 1
    cres = scanfn(cursor, chi)
    # 発見鎖を CPU で独立検証してから記録
    for x in cres
        if verify_cc(x, k)
            nfound += 1
            open(foundlog, "a") do io
                println(io, "$x  # CC$k verified $(Libc.strftime(time())) chunk=[$cursor,$chi)")
            end
            log("★FOUND & VERIFIED  CC$k head = $x")
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

include("cc_gpu.jl")
using Printf

function run_search()
    TARGET_CC = 15
    STEP = 10^16
    HI_MAX = 9_220_000_000_000_000_000

    println("=" ^ 70)
    println("CC$(TARGET_CC) 探索開始")
    println("基底範囲: 10^18 〜 Int64上限")
    println("1バッチ = $(STEP÷10^16) × 10^16")
    println("=" ^ 70)

    t_start = time()
    found = Int[]
    batch = 0
    r = 10^18

    while r < HI_MAX
        batch += 1
        r_end = min(r + STEP, HI_MAX)

        t_batch = @elapsed begin
            res = search_cc_gpu(r, r_end, TARGET_CC, verbose=false)
        end
        append!(found, res)

        elapsed = round(time() - t_start, digits=0)
        cumul = length(found)
        @printf "[%4d] %20d〜%20d | %6.1fs | found: %2d | total: %2d | elapsed: %ds\n" batch r r_end t_batch length(res) cumul elapsed

        if !isempty(res)
            println(" *** CC$(TARGET_CC) FOUND: $res ***")
        end

        r = r_end
        flush(stdout)
    end

    t_total = round(time() - t_start, digits=0)
    println("\n" ^ 2)
    println("=" ^ 70)
    println("探索終了")
    println("総バッチ数: $batch")
    println("総時間: $(t_total)s ($(round(t_total/60, digits=1))分)")
    println("発見 CC$(TARGET_CC): $(length(found)) 件")
    for f in found
        println("  CC$(TARGET_CC): $f")
    end
    println("=" ^ 70)
end

run_search()

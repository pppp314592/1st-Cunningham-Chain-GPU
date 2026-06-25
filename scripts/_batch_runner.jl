include("../src/cc_gpu.jl")
using Printf
using Dates

TARGET_CC = 15
STEP = 10^16
HI_MAX = 9_220_000_000_000_000_000
LOG_FILE = "logs/cc15_progress.log"
RESULT_FILE = "logs/cc15_results.txt"

function log_msg(msg)
    ts = now()
    open(LOG_FILE, "a") do f
        println(f, "[$ts] $msg")
    end
    println(msg)
    flush(stdout)
end

function main()
    open(LOG_FILE, "a") do f
        println(f, "[$(now())] --- resume ---")
        println(f, "[$(now())] CC15 search resumed")
        println(f, "[$(now())] range: 10^18 - $(HI_MAX)")
        println(f, "[$(now())] batch size: $STEP")
        println(f, "[$(now())] threads: $(Threads.nthreads())")
    end

    t_start = time()
    found = Int[]
    batch = 401
    r = 5_010_000_000_000_000_000
    total_batches = (HI_MAX - 10^18) ÷ STEP

    while r < HI_MAX
        batch += 1
        r_end = min(r + STEP, HI_MAX)

        t_batch = @elapsed begin
            res = search_cc_gpu(r, r_end, TARGET_CC, verbose=false)
        end
        append!(found, res)

        if !isempty(res)
            open(RESULT_FILE, "a") do f
                for x in res
                    println(f, x)
                end
            end
        end

        elapsed = round(time() - t_start, digits=0)
        cumul = length(found)
        pct = round(100 * (r - 10^18) / (HI_MAX - 10^18), digits=1)
        eta_s = elapsed / max(1, batch) * (total_batches - batch)
        eta_min = round(Int, eta_s / 60)

        log_msg(
            "[$(batch)/$(total_batches)] $(r)-$(r_end) | $(round(t_batch, digits=1))s | " *
            "found=$(length(res)) total=$cumul | " *
            "elapsed=$(elapsed)s ETA=$(eta_min)min $(pct)%"
        )

        if !isempty(res)
            log_msg(" *** CC$(TARGET_CC) FOUND: $res ***")
        end

        r = r_end
        flush(stdout)
    end

    t_total = round(time() - t_start, digits=0)
    log_msg("="^60)
    log_msg("DONE! total=$(t_total)s ($(round(t_total/60, digits=1))min)")
    log_msg("CC$(TARGET_CC) found: $(length(found))")
    for f in found
        log_msg("  CC$(TARGET_CC): $f")
    end
    log_msg("="^60)
end

main()

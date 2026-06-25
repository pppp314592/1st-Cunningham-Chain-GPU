include("../src/cc_gpu.jl")
using Printf
using Dates

RESULTS = "logs/cc12plus_results.txt"
LOG = "logs/cc12plus_progress.log"

function log_msg(msg)
    ts = now()
    open(LOG, "a") do f
        println(f, "[$ts] $msg")
    end
    println(msg)
    flush(stdout)
end

function scan(target_cc, lo, hi)
    log_msg("=== CC$(target_cc) scan [$lo, $hi] ===")
    t = @elapsed res = search_cc_gpu(lo, hi, target_cc, verbose=false)
    log_msg("  found $(length(res)) CC$(target_cc) in $(round(t, digits=1))s")
    for r in res
        log_msg("  CC$(target_cc): $r")
    end
    return res
end

function main()
    open(LOG, "w") do f
        println(f, "[$(now())] CC12+ scan started")
        println(f, "[$(now())] range: 10 - 10^17")
    end
    open(RESULTS, "w") do f
        println(f, "# CC12+ results (10 to 10^17)")
    end

    all = Dict{Int, Vector{Int}}()
    for cc in [15, 14, 13, 12]
        all[cc] = scan(cc, 10, 10^17)
    end

    open(RESULTS, "a") do f
        for cc in [12, 13, 14, 15]
            println(f, "\n=== CC$(cc): $(length(all[cc])) found ===")
            for r in sort(all[cc])
                println(f, "CC$(cc): $r")
            end
        end
    end

    log_msg("=" ^ 60)
    log_msg("DONE!")
    for cc in [12, 13, 14, 15]
        log_msg("  CC$(cc): $(length(all[cc]))")
    end
    log_msg("=" ^ 60)
end

main()

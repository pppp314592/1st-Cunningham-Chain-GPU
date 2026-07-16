using Dates

function main()
    logf = "logs/scan_k16.run.log"
    lines = readlines(logf)
    re_line = r"^(\d{4}/\d{2}/\d{2}) (\d{2}:\d{2}:\d{2})\s+chunk(\d+)\s+\[(\d+),(\d+)\)\s+完了"
    Elem = Tuple{DateTime,Int,Int128,Int128}
    events = Elem[]
    for l in lines
        m = match(re_line, l)
        if m !== nothing
            ts = DateTime("$(m[1]) $(m[2])", "yyyy/mm/dd HH:MM:SS")
            push!(events, (ts, parse(Int, m[3]), parse(Int128, m[4]), parse(Int128, m[5])))
        end
    end
    @assert !isempty(events) "no chunk events found"

    sessions = Vector{Elem}[]
    cur = Elem[]
    prev = events[1]
    for e in events
        if e[2] < prev[2] || e[1] < prev[1]
            push!(sessions, cur)
            cur = Elem[]
        end
        push!(cur, e)
        prev = e
    end
    push!(sessions, cur)

    println("resume sessions: $(length(sessions))")
    total_scan_sec = 0.0
    for (i, s) in enumerate(sessions)
        if length(s) >= 2
            gap = 0.0
            for j in 2:length(s)
                gap += Dates.value(s[j][1] - s[j-1][1]) / 1000.0
            end
            total_scan_sec += gap
            firstc = s[1][2]; lastc = s[end][2]
            rng = s[end][4] - s[1][3]
            println("  session $i: chunks $(firstc)..$(lastc)  scan=$(round(gap/3600,digits=2))h  covered=$(rng)")
        else
            println("  session $i: only 1 chunk ($(s[1][2])), skipped")
        end
    end

    println()
    println("TOTAL pure scan time (chunk-interval gaps): $(round(total_scan_sec/3600,digits=2)) h = $(round(total_scan_sec/86400,digits=2)) days")
    last = events[end]
    println("last reached cursor: $(last[4])  (chunk $(last[2]))")
    println("total chunks logged: $(length(events))")
end

main()

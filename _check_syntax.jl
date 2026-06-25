try
    include("prime_filter.jl")
    include("cc_gpu.jl")
    println("SYNTAX OK")
catch e
    println("ERROR: ", e)
    for s in stacktrace(catch_backtrace())
        println(s)
    end
end

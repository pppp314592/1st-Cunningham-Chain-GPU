using CUDA
using Primes: primes

include("prime_filter.jl")

# GPU kernel: カニンガム鎖カウント
function cc_count_kernel(n::Int64, target_cc::Int)::Int32
    x_lo = UInt64(n)
    x_hi = UInt64(0)
    for i in 1:target_cc
        if x_hi == 0 && x_lo <= 0x7FFFFFFFFFFFFFFF
            if !is_prime_mr(Int64(x_lo))
                return Int32(i - 1)
            end
        else
            if !is_prime_mr128(x_lo, x_hi)
                return Int32(i - 1)
            end
        end
        i == target_cc && return Int32(target_cc)
        x_hi > 0x7FFFFFFFFFFFFFFF && return Int32(i)
        carry = (x_lo > 0x7FFFFFFFFFFFFFFF) ? UInt64(1) : UInt64(0)
        x_lo = (x_lo << 1) | UInt64(1)
        x_hi = (x_hi << 1) | carry
    end
    return Int32(target_cc)
end

# バッチ GPU フィルター
function filter_cc_gpu(candidates::Vector{Int}, target_cc::Int)::Vector{Int}
    isempty(candidates) && return Int[]
    d_cand = CuArray(candidates)
    d_counts = map(n -> cc_count_kernel(n, target_cc), d_cand)
    counts = Array(d_counts)
    results = Int[]
    for (i, cnt) in enumerate(counts)
        if cnt == target_cc
            push!(results, candidates[i])
        end
    end
    return sort(results)
end

# ------------------------------------------------------------
# Notebook 互換の篩構築
# ------------------------------------------------------------
function build_cc_sieve(target_cc::Int)
    wl = if target_cc ≤ 9
        [2,3,5,7,11,13,17]
    elseif target_cc ≤ 11
        [2,3,5,7,11,13,17,19,23]
    elseif target_cc ≤ 13
        [2,3,5,7,11,13,17,19,23,37,41]
    else
        [2,3,5,7,11,13,17,19,23,37,41,43]
    end
    wheel = prod(wl)

    function cc_len_mod(x::Int, p::Int, target_cc::Int)::Int
        n = 0
        y = x
        while n < target_cc && y % p != 0
            y = (2 * y + 1) % p
            n += 1
        end
        return n
    end

    modsieve_wl = [(p, Set(i for i in 0:p-1 if cc_len_mod(i, p, target_cc) < target_cc)) for p in wl]

    wheel_n = Int[1]
    pprod = 1
    for (p, badset) in modsieve_wl
        tmp = copy(wheel_n)
        wheel_n = Int[]
        for i in 0:p-1
            append!(wheel_n, filter(x -> x % p ∉ badset, tmp .+ i * pprod))
        end
        pprod *= p
    end
    eff = wheel ÷ length(wheel_n)
    println("CC$(target_cc) sieve: wheel=$(wheel) residues=$(length(wheel_n)) efficiency=$(eff)")

    max_prime = if target_cc ≤ 12
        1000
    elseif target_cc ≤ 13
        3000
    else
        5000
    end
    extra_primes = filter(p -> !(p in wl) && p ≤ max_prime, primes(max_prime))
    mdllist = [(p, Set(i for i in 0:p-1 if cc_len_mod(i, p, target_cc) < target_cc)) for p in extra_primes]

    return wheel, wheel_n, mdllist
end

# ------------------------------------------------------------
# 探索本体 (逐次 GPU フィルター)
# ------------------------------------------------------------
function search_cc_gpu(lo::Int, hi::Int, target_cc::Int;
                       verbose::Bool=true)
    verbose && println("=== CC$(target_cc) search [$(lo), $(hi)] ===")
    t_start = time()

    wheel, wheel_n, mdllist = build_cc_sieve(target_cc)

    start_k = lo ÷ wheel
    end_k = (hi - 1) ÷ wheel
    total_cycles = end_k - start_k + 1
    cycle_count = 0
    global_results = Int[]

    batch = Int[]
    for k in start_k:end_k
        A = copy(wheel_n)
        base = k * wheel
        for (p, badset) in mdllist
            filter!(x -> begin
                n = x + base
                n <= p || n % p ∉ badset
            end, A)
        end
        for r in A
            n = r + base
            (lo < n < hi) || continue
            push!(batch, n)
        end
        cycle_count += 1
        if verbose && cycle_count % max(1, total_cycles ÷ 20) == 0
            elapsed = round(time() - t_start, digits=1)
            pct = round(100 * cycle_count / total_cycles, digits=1)
            @info "  progress: $(pct)% ($(cycle_count)/$(total_cycles)) | batch: $(length(batch)) | elapsed: $(elapsed)s"
        end
        # 100K ごとに GPU へ
        if length(batch) >= 100_000
            verbose && println("  -> GPU batch ($(length(batch)) candidates)")
            append!(global_results, filter_cc_gpu(batch, target_cc))
            batch = Int[]
        end
    end
    # 残り
    if !isempty(batch)
        verbose && println("  -> GPU final batch ($(length(batch)) candidates)")
        append!(global_results, filter_cc_gpu(batch, target_cc))
    end

    t_end = time()
    sort!(global_results)
    verbose && println("=== Done: $(length(global_results)) CC$(target_cc) found in $(round(t_end - t_start, digits=2))s ===")
    for r in global_results
        println("CC$(target_cc): $(r)")
    end
    return global_results
end

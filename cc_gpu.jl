using CUDA
using Primes: primes

include("prime_filter.jl")

# ============================================================
# Sieve cache (global, build once per target_cc)
# ============================================================
const _SIEVE_CACHE = Dict{Int, Any}()

function _get_cached_sieve(target_cc::Int)
    haskey(_SIEVE_CACHE, target_cc) && return _SIEVE_CACHE[target_cc]

    wheel, wheel_n, mdllist = build_cc_sieve(target_cc)

    # mdllist を GPU 用フラット配列に変換 (p は Int32 で十分)
    num = length(mdllist)
    primes_arr = Int32[p for (p, _) in mdllist]
    bad_data = UInt8[]
    prime_offsets = Int[]
    for (p, badset) in mdllist
        push!(prime_offsets, length(bad_data))
        for i in 0:p-1
            push!(bad_data, i in badset ? UInt8(1) : UInt8(0))
        end
    end

    wheel_n_d  = CuArray(wheel_n)
    primes_d   = CuArray(primes_arr)
    bad_data_d = CuArray(bad_data)
    offs_d     = CuArray(prime_offsets)

    entry = (wheel, wheel_n, mdllist, wheel_n_d, primes_d, bad_data_d, offs_d, num)
    _SIEVE_CACHE[target_cc] = entry
    return entry
end

# ============================================================
# GPU sieve kernel (Int64 版)
# ============================================================
function _sieve_kernel64(wheel_n, base, lo, hi,
                         primes_arr, bad_data, prime_offsets, num_primes,
                         output, out_count)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx > length(wheel_n) && return

    n = UInt64(wheel_n[idx]) + UInt64(base)
    (UInt64(lo) < n < UInt64(hi)) || return

    n_i = Int(n)
    for pi in 1:num_primes
        p = primes_arr[pi]
        n_i <= p && continue  # n=p のとき p で割り切れても n は素数なので除外しない
        r = n % UInt64(p)
        off = prime_offsets[pi]
        if bad_data[off + Int(r) + 1] != 0
            return nothing
        end
    end

    pos = CUDA.atomic_add!(pointer(out_count, 1), Int32(1))
    output[pos + 1] = n_i
    return nothing
end

# ============================================================
# GPU sieve kernel (Int128 版) — バッチ処理: 全サイクルを一括
#   各スレッドが1つの wheel_n 要素を担当し、全サイクルをループ
#   base は前サイクルからの increment で計算 (128-bit 乗算不要)
# ============================================================
function _sieve_kernel128_batch(wheel_n,
                                base_lo, base_hi, wheel,
                                ncycles, lo_lo, lo_hi, hi_lo, hi_hi,
                                primes_arr, bad_data, prime_offsets, num_primes,
                                output_lo, output_hi, out_count, max_out)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx > length(wheel_n) && return

    r = UInt64(wheel_n[idx])
    b_lo = base_lo
    b_hi = base_hi
    w = UInt64(wheel)

    for cyc in 1:ncycles
        # n = r + b (128-bit)
        n_lo = r + b_lo
        n_hi = b_hi
        if n_lo < r
            n_hi += UInt64(1)
        end

        # bounds: lo < n < hi
        if !(n_hi < lo_hi || (n_hi == lo_hi && n_lo <= lo_lo) ||
             n_hi > hi_hi || (n_hi == hi_hi && n_lo >= hi_lo))

            ok = true
            for pi in 1:num_primes
                p = primes_arr[pi]
                rn = rem128by32(n_lo, n_hi, p)
                if bad_data[prime_offsets[pi] + rn + 1] != 0
                    ok = false
                    break
                end
            end
            if ok
                pos = CUDA.atomic_add!(pointer(out_count, 1), Int32(1))
                if pos < max_out
                    output_lo[pos + 1] = n_lo
                    output_hi[pos + 1] = n_hi
                end
            end
        end

        # b += wheel (128-bit increment)
        b_lo += w
        if b_lo < w
            b_hi += UInt64(1)
        end
    end
    return nothing
end

# 128-bit % 32-bit (GPU デバイス関数)
function rem128by32(lo::UInt64, hi::UInt64, p::Int32)::Int32
    hi_r = hi % UInt64(p)
    pow2_32 = UInt64(1) << 32
    pow2_64_mod_p = (pow2_32 % UInt64(p)) * (pow2_32 % UInt64(p)) % UInt64(p)
    r = (hi_r * pow2_64_mod_p + lo % UInt64(p)) % UInt64(p)
    return Int32(r)
end

# ============================================================
# 1 cycle を GPU で篩う (Int64)
# ============================================================
function _gpu_sieve_cycle64(k, wheel, lo, hi,
                            wheel_n_d, primes_d, bad_data_d, offs_d, num_primes)
    base = k * wheel
    n = length(wheel_n_d)
    out_count = CuArray{Int32}([0])
    output_d = CuArray{Int}(undef, n)

    threads = 256
    blocks = cld(n, threads)
    @cuda blocks=blocks threads=threads _sieve_kernel64(
        wheel_n_d, base, lo, hi, primes_d, bad_data_d, offs_d, num_primes,
        output_d, out_count)

    cnt = Array(out_count)[1]
    cnt == 0 && return Int[]
    return Array(output_d[1:cnt])
end

# ============================================================
# GPU kernel: カニンガム鎖カウント
# ============================================================
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

# ============================================================
# GPU フィルター (Int64)
# ============================================================
function filter_cc_gpu(candidates::Vector{Int}, target_cc::Int)::Vector{Int}
    isempty(candidates) && return Int[]
    d_cand = CuArray(candidates)
    d_counts = map(n -> cc_count_kernel(n, target_cc), d_cand)
    counts = Array(d_counts)
    results = Int[]
    for (i, cnt) in enumerate(counts)
        cnt == target_cc && push!(results, candidates[i])
    end
    return sort(results)
end

# ============================================================
# GPU フィルター (Int128)
# ============================================================
function filter_cc_gpu128(candidates::Vector{Int128}, target_cc::Int)::Vector{Int128}
    isempty(candidates) && return Int128[]
    # Int128 → (lo, hi) のペア配列に分解
    los = UInt64[c & 0xFFFFFFFFFFFFFFFF for c in candidates]
    his = UInt64[(c >> 64) & 0xFFFFFFFFFFFFFFFF for c in candidates]
    d_lo = CuArray(los)
    d_hi = CuArray(his)

    n = length(candidates)
    d_counts = CuArray{Int32}(undef, n)
    threads = 256
    blocks = cld(n, threads)
    @cuda blocks=blocks threads=threads _mr128_kernel(d_lo, d_hi, target_cc, d_counts, n)

    counts = Array(d_counts)
    results = Int128[]
    for (i, cnt) in enumerate(counts)
        cnt == target_cc && push!(results, candidates[i])
    end
    return sort(results)
end

function _mr128_kernel(los, his, target_cc, counts, n)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx > n && return nothing
    counts[idx] = _cc_count128(los[idx], his[idx], target_cc)
    return nothing
end

function _cc_count128(x_lo::UInt64, x_hi::UInt64, target_cc::Int)::Int32
    for i in 1:target_cc
        if x_hi > 0x7FFFFFFFFFFFFFFF
            return Int32(i - 1)
        end
        if !is_prime_mr128(x_lo, x_hi)
            return Int32(i - 1)
        end
        i == target_cc && return Int32(target_cc)
        carry = (x_lo > 0x7FFFFFFFFFFFFFFF) ? UInt64(1) : UInt64(0)
        x_lo = (x_lo << 1) | UInt64(1)
        x_hi = (x_hi << 1) | carry
    end
    return Int32(target_cc)
end

# ============================================================
# Notebook 互換の篩構築 (変更なし)
# ============================================================
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

# ============================================================
# 探索本体 (Int64) — GPU 篩版
# ============================================================
function search_cc_gpu(lo::Int, hi::Int, target_cc::Int; verbose::Bool=true)
    verbose && println("=== CC$(target_cc) search [$(lo), $(hi)] ===")
    t_start = time()

    wheel, _, _, wheel_n_d, primes_d, bad_data_d, offs_d, np = _get_cached_sieve(target_cc)

    start_k = lo ÷ wheel
    end_k = (hi - 1) ÷ wheel
    total_cycles = end_k - start_k + 1

    all_candidates = Int[]
    all_lock = Threads.ReentrantLock()
    done = Threads.Atomic{Int}(0)

    Threads.@threads for k in start_k:end_k
        cand = _gpu_sieve_cycle64(k, wheel, lo, hi,
                                  wheel_n_d, primes_d, bad_data_d, offs_d, np)
        if !isempty(cand)
            Threads.lock(all_lock) do
                append!(all_candidates, cand)
            end
        end
        if verbose
            n_done = Threads.atomic_add!(done, 1) + 1
            if n_done % max(1, total_cycles ÷ 10) == 0 || n_done == total_cycles
                elapsed = round(time() - t_start, digits=1)
                pct = round(100 * n_done / total_cycles, digits=1)
                @info "  progress: $(pct)% ($(n_done)/$(total_cycles)) | candidates: $(length(all_candidates)) | elapsed: $(elapsed)s"
            end
        end
    end

    res = filter_cc_gpu(all_candidates, target_cc)
    t_end = time()
    verbose && println("=== Done: $(length(res)) CC$(target_cc) in $(round(t_end - t_start, digits=2))s ===")
    for r in res
        println("CC$(target_cc): $(r)")
    end
    return res
end

# ============================================================
# 探索本体 (Int128) — GPU 篩版
# ============================================================
function search_cc_gpu128(lo::Int128, hi::Int128, target_cc::Int; verbose::Bool=true)
    verbose && println("=== CC$(target_cc) search (Int128) [$(lo), $(hi)] ===")
    t_start = time()

    wheel, _, _, wheel_n_d, primes_d, bad_data_d, offs_d, np = _get_cached_sieve(target_cc)

    lo_lo = UInt64(lo & 0xFFFFFFFFFFFFFFFF)
    lo_hi = UInt64((lo >> 64) & 0xFFFFFFFFFFFFFFFF)
    hi_lo = UInt64(hi & 0xFFFFFFFFFFFFFFFF)
    hi_hi = UInt64((hi >> 64) & 0xFFFFFFFFFFFFFFFF)

    k_lo = lo ÷ wheel
    k_hi = (hi - 1) ÷ wheel
    total_cycles = k_hi - k_lo + 1

    CYCLES_PER_KERNEL = 200
    n = length(wheel_n_d)
    est_per_cycle = max(1, n ÷ 500_000)

    all_candidates_lo = UInt64[]
    all_candidates_hi = UInt64[]

    k_start = k_lo
    batch_idx = 0
    while k_start <= k_hi
        batch_idx += 1
        k_end = min(k_start + CYCLES_PER_KERNEL - 1, k_hi)
        ncycles = k_end - k_start + 1

        init_base = Int128(k_start) * Int128(wheel)
        base_lo = UInt64(init_base & 0xFFFFFFFFFFFFFFFF)
        base_hi = UInt64(init_base >> 64)

        max_out = Int32(min(500_000_000, est_per_cycle * ncycles * 4))
        out_count = CuArray{Int32}([0])
        out_lo = CuArray{UInt64}(undef, max_out)
        out_hi = CuArray{UInt64}(undef, max_out)

        threads = 256
        blocks = cld(n, threads)
        @cuda blocks=blocks threads=threads _sieve_kernel128_batch(
            wheel_n_d, base_lo, base_hi, UInt64(wheel), ncycles,
            lo_lo, lo_hi, hi_lo, hi_hi,
            primes_d, bad_data_d, offs_d, np,
            out_lo, out_hi, out_count, max_out)

        cnt = Array(out_count)[1]
        if cnt > 0
            append!(all_candidates_lo, Array(out_lo[1:cnt]))
            append!(all_candidates_hi, Array(out_hi[1:cnt]))
        end

        if verbose
            pct = round(100 * (k_end - k_lo + 1) / total_cycles, digits=1)
            @info "  batch $(batch_idx): $(cnt) candidates (total $(length(all_candidates_lo)), $(pct)%)"
        end

        k_start = k_end + 1
    end

    verbose && @info "  sieve done: $(length(all_candidates_lo)) candidates in $(round(time() - t_start, digits=2))s"

    isempty(all_candidates_lo) && return Int128[]

    d_lo = CuArray(all_candidates_lo)
    d_hi = CuArray(all_candidates_hi)
    nc = length(all_candidates_lo)
    d_counts = CuArray{Int32}(undef, nc)
    threads2 = 256
    blocks2 = cld(nc, threads2)
    @cuda blocks=blocks2 threads=threads2 _mr128_kernel(d_lo, d_hi, target_cc, d_counts, nc)

    counts = Array(d_counts)
    results = Int128[]
    for i in 1:nc
        if counts[i] == target_cc
            push!(results, Int128(all_candidates_hi[i]) << 64 | Int128(all_candidates_lo[i]))
        end
    end

    t_end = time()
    verbose && println("=== Done: $(length(results)) CC$(target_cc) in $(round(t_end - t_start, digits=2))s ===")
    for r in results
        println("CC$(target_cc): $r")
    end
    return results
end

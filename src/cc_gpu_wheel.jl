# cc_gpu_wheel.jl — 融合GPUホイール篩
# 1スレッド=1ホイール候補。追加素数篩 + 鎖MR判定をカーネル内で完結し、
# 確定した第一種カニンガム鎖の先頭値だけを書き戻す。
#
# 使い方:
#   include("src/cc_cpu.jl")        # build_cc_sieve を使う
#   include("src/cc_gpu_wheel.jl")
#   search_cc_gpu_wheel(lo, hi, k)
#
# 起動は必ず  julia -g0 -t N  (非ASCIIパス対策)

include("prime_filter.jl")   # デバイス側 MR: is_prime_mr, is_prime_mr128
using CUDA

# ------------------------------------------------------------
# 融合カーネル
#   g in 1..(ncyc*R): cyc = k_base + (g-1)÷R,  ridx = (g-1)%R + 1
#   n = cyc*wheel + wheel_n[ridx]   (n < ~9.2e18 なので Int64 で保持)
# ------------------------------------------------------------
function _cc_wheel_kernel!(out::CuDeviceVector{Int64},
                           counter::CuDeviceVector{Int32}, cap::Int32,
                           d_wheel_n::CuDeviceVector{Int64}, R::Int64,
                           wheel::Int64, k_base::Int64, ncyc::Int64,
                           lo::Int64, hi::Int64,
                           d_primes::CuDeviceVector{Int32},
                           d_wheel_mod::CuDeviceVector{Int32},
                           d_pow20::CuDeviceVector{Int32},
                           d_mu::CuDeviceVector{UInt32},
                           d_bad_off::CuDeviceVector{Int32},
                           d_bad::CuDeviceVector{UInt8}, nprimes::Int32,
                           kk::Int32)
    g = (blockIdx().x - Int64(1)) * blockDim().x + threadIdx().x
    total = ncyc * R
    if g <= total
        cyc = k_base + (g - Int64(1)) ÷ R
        ridx = (g - Int64(1)) % R + Int64(1)
        @inbounds r = d_wheel_n[ridx]
        n = cyc * wheel + r
        if n > lo && n < hi
            # --- 追加素数篩 (32bit分解 + Barrett還元で除算レス) ---
            # r < wheel < 2^51 なので r = r_hi*2^20 + r_lo に分解
            r_hi = UInt32((r >> 20) & 0x7FFFFFFF)
            r_lo = UInt32(r & 0x00000000000FFFFF)
            cyc32 = UInt32(cyc & 0x7FFFFFFF)
            ok = true
            @inbounds for pi in 1:nprimes
                p = UInt32(d_primes[pi])
                # p が n 以上なら篩わない (p が鎖要素自身の場合の誤除去を防ぐ; CPU版と一致)
                # 素数は昇順なので以降も全て n 以上 → break
                Int64(p) >= n && break
                mu = d_mu[pi]
                rhp = bmod(r_hi, p, mu)
                rp = bmod(rhp * UInt32(d_pow20[pi]) + r_lo, p, mu)
                cp = bmod(cyc32, p, mu)
                np = bmod(cp * UInt32(d_wheel_mod[pi]) + rp, p, mu)
                if d_bad[d_bad_off[pi] + Int64(np) + Int64(1)] != 0x00
                    ok = false
                    break
                end
            end
            if ok
                # --- 鎖 MR 判定 ---
                x_lo = reinterpret(UInt64, n)
                x_hi = UInt64(0)
                good = true
                @inbounds for _ in 1:kk
                    isp = if x_hi == UInt64(0) && x_lo <= 0x7FFFFFFFFFFFFFFF
                        is_prime_mr(Int64(x_lo))
                    else
                        (x_hi > 0x7FFFFFFFFFFFFFFF) ? false : is_prime_mr128(x_lo, x_hi)
                    end
                    if !isp
                        good = false
                        break
                    end
                    carry = (x_lo > 0x7FFFFFFFFFFFFFFF) ? UInt64(1) : UInt64(0)
                    x_lo = (x_lo << 1) | UInt64(1)
                    x_hi = (x_hi << 1) | carry
                end
                if good
                    idx = CUDA.atomic_add!(pointer(counter, 1), Int32(1)) + Int32(1)
                    if idx <= cap
                        @inbounds out[idx] = n
                    end
                end
            end
        end
    end
    return nothing
end

# ------------------------------------------------------------
# Int128 頭 対応カーネル (巨大数域: 頭 n > 9.2e18, 例:10^19〜10^22)
#   篩は32bit分解のまま (cyc = n÷wheel < 2^31 をホスト側でassert)。
#   128bit演算は n生成・範囲比較・鎖MR のみ (スレッド当たり僅少)。
# ------------------------------------------------------------
function _cc_wheel_kernel128!(out::CuDeviceVector{Int128},
                           counter::CuDeviceVector{Int32}, cap::Int32,
                           d_wheel_n::CuDeviceVector{Int64}, R::Int64,
                           wheel::Int64, k_base::Int64, ncyc::Int64,
                           lo::Int128, hi::Int128,
                           d_primes::CuDeviceVector{Int32},
                           d_wheel_mod::CuDeviceVector{Int32},
                           d_pow20::CuDeviceVector{Int32},
                           d_mu::CuDeviceVector{UInt32},
                           d_bad_off::CuDeviceVector{Int32},
                           d_bad::CuDeviceVector{UInt8}, nprimes::Int32,
                           kk::Int32)
    g = (blockIdx().x - Int64(1)) * blockDim().x + threadIdx().x
    total = ncyc * R
    if g <= total
        cyc = k_base + (g - Int64(1)) ÷ R
        ridx = (g - Int64(1)) % R + Int64(1)
        @inbounds r = d_wheel_n[ridx]
        n = Int128(cyc) * Int128(wheel) + Int128(r)
        if n > lo && n < hi
            r_hi = UInt32((r >> 20) & 0x7FFFFFFF)
            r_lo = UInt32(r & 0x00000000000FFFFF)
            cyc32 = UInt32(cyc & 0x7FFFFFFF)
            ok = true
            @inbounds for pi in 1:nprimes
                p = UInt32(d_primes[pi]); mu = d_mu[pi]
                rhp = bmod(r_hi, p, mu)
                rp = bmod(rhp * UInt32(d_pow20[pi]) + r_lo, p, mu)
                cp = bmod(cyc32, p, mu)
                np = bmod(cp * UInt32(d_wheel_mod[pi]) + rp, p, mu)
                if d_bad[d_bad_off[pi] + Int64(np) + Int64(1)] != 0x00
                    ok = false
                    break
                end
            end
            if ok
                x_lo = UInt64(n & Int128(0xFFFFFFFFFFFFFFFF))
                x_hi = UInt64((n >> 64) & Int128(0xFFFFFFFFFFFFFFFF))
                good = true
                @inbounds for _ in 1:kk
                    isp = if x_hi == UInt64(0) && x_lo <= 0x7FFFFFFFFFFFFFFF
                        is_prime_mr(Int64(x_lo))
                    else
                        (x_hi > 0x7FFFFFFFFFFFFFFF) ? false : is_prime_mr128(x_lo, x_hi)
                    end
                    if !isp
                        good = false
                        break
                    end
                    carry = (x_lo > 0x7FFFFFFFFFFFFFFF) ? UInt64(1) : UInt64(0)
                    x_lo = (x_lo << 1) | UInt64(1)
                    x_hi = (x_hi << 1) | carry
                end
                if good
                    idx = CUDA.atomic_add!(pointer(counter, 1), Int32(1)) + Int32(1)
                    if idx <= cap
                        @inbounds out[idx] = n
                    end
                end
            end
        end
    end
    return nothing
end

# ============================================================
# hot/cold パス分割カーネル (篩専用 + MR専用)
# ------------------------------------------------------------
# 融合カーネルは MR(128bit)コードが全スレッドのレジスタを占有し(88reg)、篩の占有率を
# 下げていた。篩専用カーネルは 24reg で占有率↑ → 篩(98.4%)が ~1.4倍高速。
# 生存候補(~1.4e-5%)を d_surv に蓄積し、稀に(末尾+一定タイルごと)MRカーネルでドレインする。
# タイル毎にMRを挟むと非同期パイプラインにバブルが出て利得が消えるため、蓄積が要。
# ============================================================

# 篩専用 (Int128頭). 生存 n を d_surv に atomic 追記 (MRを含まない → 低レジスタ)。
function _cc_sieve_kernel128!(d_surv::CuDeviceVector{Int128},
                           scnt::CuDeviceVector{Int32}, scap::Int32,
                           d_wheel_n::CuDeviceVector{Int64}, R::Int64,
                           wheel::Int64, k_base::Int64, ncyc::Int64,
                           lo::Int128, hi::Int128,
                           d_primes::CuDeviceVector{Int32},
                           d_wheel_mod::CuDeviceVector{Int32},
                           d_pow20::CuDeviceVector{Int32},
                           d_mu::CuDeviceVector{UInt32},
                           d_bad_off::CuDeviceVector{Int32},
                           d_bad::CuDeviceVector{UInt8}, nprimes::Int32)
    g = (blockIdx().x - Int64(1)) * blockDim().x + threadIdx().x
    total = ncyc * R
    if g <= total
        cyc = k_base + (g - Int64(1)) ÷ R
        ridx = (g - Int64(1)) % R + Int64(1)
        @inbounds r = d_wheel_n[ridx]
        n = Int128(cyc) * Int128(wheel) + Int128(r)
        if n > lo && n < hi
            r_hi = UInt32((r >> 20) & 0x7FFFFFFF)
            r_lo = UInt32(r & 0x00000000000FFFFF)
            cyc32 = UInt32(cyc & 0x7FFFFFFF)
            ok = true
            @inbounds for pi in 1:nprimes
                p = UInt32(d_primes[pi]); mu = d_mu[pi]
                rhp = bmod(r_hi, p, mu)
                rp = bmod(rhp * UInt32(d_pow20[pi]) + r_lo, p, mu)
                cp = bmod(cyc32, p, mu)
                np = bmod(cp * UInt32(d_wheel_mod[pi]) + rp, p, mu)
                if d_bad[d_bad_off[pi] + Int64(np) + Int64(1)] != 0x00
                    ok = false
                    break
                end
            end
            if ok
                idx = CUDA.atomic_add!(pointer(scnt, 1), Int32(1)) + Int32(1)
                idx <= scap && (@inbounds d_surv[idx] = n)
            end
        end
    end
    return nothing
end

# MR専用 (Int128). 生存候補 d_surv[1..scnt] のみ鎖MRし、確定鎖頭を out に書く。
# nsurv はデバイスの scnt から読む → ホスト同期不要 (非同期ドレイン可)。
function _cc_mr_kernel128!(d_surv::CuDeviceVector{Int128},
                           scnt::CuDeviceVector{Int32},
                           out::CuDeviceVector{Int128},
                           counter::CuDeviceVector{Int32}, cap::Int32, kk::Int32)
    i = (blockIdx().x - Int64(1)) * blockDim().x + threadIdx().x
    @inbounds nsurv = scnt[1]
    if i <= nsurv
        @inbounds n = d_surv[i]
        x_lo = UInt64(n & Int128(0xFFFFFFFFFFFFFFFF))
        x_hi = UInt64((n >> 64) & Int128(0xFFFFFFFFFFFFFFFF))
        good = true
        @inbounds for _ in 1:kk
            isp = if x_hi == UInt64(0) && x_lo <= 0x7FFFFFFFFFFFFFFF
                is_prime_mr(Int64(x_lo))
            else
                (x_hi > 0x7FFFFFFFFFFFFFFF) ? false : is_prime_mr128(x_lo, x_hi)
            end
            if !isp
                good = false
                break
            end
            carry = (x_lo > 0x7FFFFFFFFFFFFFFF) ? UInt64(1) : UInt64(0)
            x_lo = (x_lo << 1) | UInt64(1)
            x_hi = (x_hi << 1) | carry
        end
        if good
            idx = CUDA.atomic_add!(pointer(counter, 1), Int32(1)) + Int32(1)
            if idx <= cap
                @inbounds out[idx] = n
            end
        end
    end
    return nothing
end

# wl (ホイールに使う素数) — build_cc_sieve と一致させる
function _cc_wl(k::Int)
    k ≤ 9  ? [2,3,5,7,11,13,17] :
    k ≤ 11 ? [2,3,5,7,11,13,17,19,23] :
    k ≤ 13 ? [2,3,5,7,11,13,17,19,23,37,41] :
             [2,3,5,7,11,13,17,19,23,37,41,43]
end

# 拡張 wl: 29 を追加 (密度 ×0.45, ~1.9倍高速)。wheel=4.2e14<2^51 で 32bit篩維持。
# R=256M (2GB GPU常駐) のため巨大数域の長時間スキャン向け。
function _cc_wl_ext(k::Int)
    k ≤ 13 ? _cc_wl(k) : [2,3,5,7,11,13,17,19,23,29,37,41,43]
end

# 任意 wl から wheel 生存残基を生成 (build_cc_sieve の CRT 展開を汎用化)
function build_wheel_custom(k::Int, wl::Vector{Int})
    cc_len_mod(x, p) = (n = 0; y = x; while n < k && y % p != 0; y = (2y + 1) % p; n += 1; end; n)
    wheel_n = Int[1]; pprod = 1
    for p in wl
        bad = Set(i for i in 0:p-1 if cc_len_mod(i, p) < k)
        tmp = wheel_n; wheel_n = Int[]
        sizehint!(wheel_n, length(tmp) * (p - length(bad)))
        for i in 0:p-1
            @inbounds for x in tmp
                v = x + i * pprod
                (v % p) in bad || push!(wheel_n, v)
            end
        end
        pprod *= p
    end
    return prod(wl), wheel_n
end

# 追加素数の bad 残差リストを任意の max_prime で構築 (GPU用, wl を除く)
# p < 46340 (=√2^31) 未満に制限: カーネルの 32bit 演算が安全
function gpu_build_extra(k::Int, max_prime::Int; wl::Vector{Int} = _cc_wl(k))
    @assert max_prime < 46340 "max_prime は 46340 未満に (Int32 32bit演算の制約)"
    function cc_len_mod(x::Int, p::Int)::Int
        n = 0; y = x
        while n < k && y % p != 0
            y = (2y + 1) % p; n += 1
        end
        return n
    end
    extra = filter(p -> !(p in wl) && p ≤ max_prime, primes(max_prime))
    return [(p, Set(i for i in 0:p-1 if cc_len_mod(i, p) < k)) for p in extra]
end

# ホスト側: 追加素数の bad テーブルを平坦化
function _flatten_bad_gpu(mdllist)
    primes32 = Int32[Int32(p) for (p, _) in mdllist]
    offsets = Int32[]
    bad = UInt8[]
    off = 0
    for (p, s) in mdllist
        push!(offsets, Int32(off))
        for rr in 0:p-1
            push!(bad, (rr in s) ? 0x01 : 0x00)
        end
        off += p
    end
    return primes32, offsets, bad
end

# GPU 側の常駐状態 (ホイール残差・素数 bad テーブル・出力バッファをデバイスに常駐)
struct GpuWheelState
    k::Int
    wheel::Int64
    R::Int64
    d_wheel_n::CuArray{Int64,1}
    d_primes::CuArray{Int32,1}
    d_wheel_mod::CuArray{Int32,1}
    d_pow20::CuArray{Int32,1}
    d_mu::CuArray{UInt32,1}
    d_bad_off::CuArray{Int32,1}
    d_bad::CuArray{UInt8,1}
    nprimes::Int32
    d_out::CuArray{Int64,1}
    d_cnt::CuArray{Int32,1}
    cap::Int
end

# 各素数の Barrett 定数 mu = floor(2^32 / p)
_barrett_mu(primes32::Vector{Int32}) = UInt32[UInt32(fld(UInt64(2)^32, UInt64(p))) for p in primes32]

# セットアップ: ホイール構築 + デバイス転送 (一度だけ実行、以降のスキャンで再利用)
function gpu_wheel_setup(k::Int; cap::Int = 1 << 16, max_prime::Int = 0,
                         wl::Union{Nothing,Vector{Int}} = nothing, verbose::Bool = true)
    if wl === nothing
        wheel, wheel_n, mdllist = build_cc_sieve(k)
    else
        wheel, wheel_n = build_wheel_custom(k, wl)   # 拡張 wl (例: 29追加)
        mdllist = Tuple{Int,Set{Int}}[]
    end
    if max_prime > 0 || wl !== nothing
        mp = max_prime > 0 ? max_prime : _default_max_prime(k)
        wlset = wl === nothing ? _cc_wl(k) : wl
        mdllist = gpu_build_extra(k, mp; wl = wlset)   # 篩を強化してMR呼び出しを削減
    end
    R = Int64(length(wheel_n))
    primes32, offsets, badflat = _flatten_bad_gpu(mdllist)
    wheel_mod = Int32[Int32(mod(Int64(wheel), Int64(p))) for p in primes32]
    pow20 = Int32[Int32(mod(Int64(1) << 20, Int64(p))) for p in primes32]
    mu = _barrett_mu(primes32)
    @assert wheel < (Int64(1) << 51) "wheel exceeds 2^51; 32bit分解の前提が崩れる (CC$k)"
    verbose && println("  setup CC$k: R=$R primes=$(length(primes32)) wheel=$wheel")
    return GpuWheelState(k, Int64(wheel), R,
        CuArray(Int64.(wheel_n)), CuArray(primes32),
        CuArray(wheel_mod), CuArray(pow20), CuArray(mu), CuArray(offsets),
        CuArray(badflat), Int32(length(primes32)),
        CUDA.zeros(Int64, cap), CUDA.zeros(Int32, 1), cap)
end

# 純スキャン (セットアップ済み state を使用、ビルド/転送コストを含まない)
function gpu_wheel_scan!(st::GpuWheelState, lo::Integer, hi::Integer;
                         work_tile::Int = 1 << 25, progress::Bool = false,
                         threads::Int = 64)
    lo64 = Int64(lo); hi64 = Int64(hi)
    w64 = st.wheel
    k_start = lo64 ÷ w64
    k_end   = (hi64 - 1) ÷ w64
    total_cyc = k_end - k_start + 1
    results = Int64[]
    cyc_per_launch = max(1, work_tile ÷ Int(st.R))
    fill!(st.d_cnt, Int32(0))
    t0 = time()
    last_report = 0.0
    cyc = k_start
    while cyc <= k_end
        ncyc = min(cyc_per_launch, k_end - cyc + 1)
        work = ncyc * st.R
        blocks = cld(work, threads)
        @cuda threads=threads blocks=blocks _cc_wheel_kernel!(
            st.d_out, st.d_cnt, Int32(st.cap), st.d_wheel_n, st.R, w64, cyc, ncyc,
            lo64, hi64, st.d_primes, st.d_wheel_mod, st.d_pow20, st.d_mu,
            st.d_bad_off, st.d_bad, st.nprimes, Int32(st.k))
        cyc += ncyc
        if progress
            frac = (cyc - k_start) / total_cyc
            now = time() - t0
            if now - last_report > 15.0
                CUDA.synchronize()
                eta = frac > 0 ? now/frac*(1-frac) : 0.0
                @info "  CC$(st.k) scan $(round(100*frac,digits=1))% | $(round(now,digits=0))s | ETA $(round(eta,digits=0))s | found=$(Array(st.d_cnt)[1])"
                last_report = now
            end
        end
    end
    CUDA.synchronize()
    cnt = Array(st.d_cnt)[1]
    if cnt > 0
        got = min(Int(cnt), st.cap)
        append!(results, Array(view(st.d_out, 1:got)))
    end
    sort!(results)
    return results
end

# k ごとの推奨 max_prime (篩とMRのバランス。密なCCほど篩を強化)
_default_max_prime(k::Int) = k ≤ 13 ? 20000 : 30000

# 全部入りの便利関数 (セットアップ + スキャン)
function search_cc_gpu_wheel(lo::Integer, hi::Integer, k::Int;
                             work_tile::Int = 1 << 25,
                             cap::Int = 1 << 16,
                             max_prime::Int = -1,
                             progress::Bool = false,
                             verbose::Bool = true)
    mp = max_prime < 0 ? _default_max_prime(k) : max_prime
    t0 = time()
    st = gpu_wheel_setup(k; cap=cap, max_prime=mp, verbose=verbose)
    results = gpu_wheel_scan!(st, lo, hi; work_tile=work_tile, progress=progress)
    verbose && println("=== GPU-wheel CC$k: $(length(results)) in $(round(time()-t0,digits=3))s ===")
    return results
end

# ------------------------------------------------------------
# Int128 頭 スキャン (巨大数域). setup済み state を使用。
# ------------------------------------------------------------
function gpu_wheel_scan128!(st::GpuWheelState, lo::Int128, hi::Int128;
                            work_tile::Int = 1 << 25, cap::Int = 1 << 12,
                            progress::Bool = false, threads::Int = 64,
                            mr_threads::Int = 64, scap::Int = 1 << 20,
                            drain_every::Int = 256)
    w = Int128(st.wheel)
    k_start = Int64(lo ÷ w)
    k_end   = Int64((hi - Int128(1)) ÷ w)
    @assert k_end < (Int64(1) << 31) "cyc≥2^31: 範囲が大きすぎ/wheelが小さすぎ (32bit篩の前提が崩れる)"
    d_out = CUDA.zeros(Int128, cap)
    d_cnt = CUDA.zeros(Int32, 1)
    d_surv = CUDA.zeros(Int128, scap)          # 篩生存候補バッファ (hot/cold 分割)
    d_scnt = CUDA.zeros(Int32, 1)
    mr_blocks = cld(scap, mr_threads)
    cyc_per_launch = max(1, work_tile ÷ Int(st.R))
    total_cyc = k_end - k_start + 1
    t0 = time(); last_report = 0.0
    cyc = k_start; tile = 0
    while cyc <= k_end
        ncyc = min(cyc_per_launch, k_end - cyc + 1)
        work = ncyc * st.R
        blocks = cld(work, threads)
        @cuda threads=threads blocks=blocks _cc_sieve_kernel128!(
            d_surv, d_scnt, Int32(scap), st.d_wheel_n, st.R, st.wheel, cyc, ncyc,
            lo, hi, st.d_primes, st.d_wheel_mod, st.d_pow20, st.d_mu,
            st.d_bad_off, st.d_bad, st.nprimes)
        cyc += ncyc; tile += 1
        if tile % drain_every == 0                # 稀にドレイン (蓄積で非同期維持)
            @cuda threads=mr_threads blocks=mr_blocks _cc_mr_kernel128!(
                d_surv, d_scnt, d_out, d_cnt, Int32(cap), Int32(st.k))
            CUDA.fill!(d_scnt, Int32(0))
        end
        if progress
            now = time() - t0
            if now - last_report > 15.0
                CUDA.synchronize()
                @assert Array(d_scnt)[1] <= scap "survivor overflow: drain_every↓ か scap↑"
                frac = (cyc - k_start) / total_cyc
                eta = frac > 0 ? now/frac*(1-frac) : 0.0
                @info "  CC$(st.k)/128 $(round(100*frac,digits=1))% | $(round(now,digits=0))s | ETA $(round(eta,digits=0))s | found=$(Array(d_cnt)[1])"
                last_report = now
            end
        end
    end
    @cuda threads=mr_threads blocks=mr_blocks _cc_mr_kernel128!(   # 末尾ドレイン
        d_surv, d_scnt, d_out, d_cnt, Int32(cap), Int32(st.k))
    CUDA.synchronize()
    cnt = Array(d_cnt)[1]
    results = Int128[]
    if cnt > 0
        got = min(Int(cnt), cap)
        append!(results, Array(view(d_out, 1:got)))
    end
    CUDA.unsafe_free!(d_out); CUDA.unsafe_free!(d_cnt)
    CUDA.unsafe_free!(d_surv); CUDA.unsafe_free!(d_scnt)
    sort!(results)
    return results
end

# 全部入り (Int128 頭). 巨大数域 CC 探索用。
function search_cc_gpu_wheel128(lo::Integer, hi::Integer, k::Int;
                                max_prime::Int = -1, progress::Bool = false,
                                verbose::Bool = true)
    mp = max_prime < 0 ? _default_max_prime(k) : max_prime
    t0 = time()
    st = gpu_wheel_setup(k; max_prime=mp, verbose=verbose)
    r = gpu_wheel_scan128!(st, Int128(lo), Int128(hi); progress=progress)
    verbose && println("=== GPU-wheel128 CC$k: $(length(r)) in $(round(time()-t0,digits=3))s ===")
    CUDA.unsafe_free!(st.d_out)
    return r
end

# ============================================================
# A4 ストリーミング CRT ホイール (GPU オドメータ)
# ------------------------------------------------------------
# 従来カーネルは wheel 生存残基 R 個を d_wheel_n に materialize する
# (拡張wheelで R=256M ≈ 2GB が上限)。これが wheel 素数追加の壁だった。
#
# 本方式は各スレッドが自分の生存残基を「その場で」合成する:
#   1. 作業番号 g → cyc と 生存インデックス sidx に分解
#   2. sidx を mixed-radix (基数 = 各素数の生存残基数) で桁分解
#   3. 各桁が指す生存残基の CRT 寄与を加算 → 残基 r (mod wheel)
# メモリは O(Σ生存残基数) = 数百要素のみ。R の上限が消え、
# wheel 素数 (31,47…) を追加して密度を下げられる (= MR呼び出し削減)。
#
# 大きい wheel (>2^51) に対応するため篩を 3-limb 32bit 分解に拡張:
#   r = r2·2^40 + r1·2^20 + r0  (各 limb < 2^20), wheel < 2^60 まで可。
# ============================================================

# 生存残基の CRT 寄与 (contrib = a·crtcoef mod wheel) を平坦化して返す。
# materialize しないので R が巨大でもメモリを消費しない。
function build_wheel_streaming(k::Int, wl::Vector{Int})
    function cc_len_mod(x::Int, p::Int)::Int
        n = 0; y = x
        while n < k && y % p != 0
            y = (2y + 1) % p; n += 1
        end
        return n
    end
    wheel128 = prod(Int128, wl)
    @assert wheel128 < (Int128(1) << 60) "wheel exceeds 2^60 (3-limb 篩の前提が崩れる)"
    goods = [[i for i in 0:p-1 if cc_len_mod(i, p) >= k] for p in wl]
    bases = Int64[length(g) for g in goods]
    R = prod(bases)                          # 生存残基総数 (materialize しない)
    radixprod = Int64[]; acc = Int64(1)      # radixprod[j] = ∏_{l<j} bases[l]
    for b in bases
        push!(radixprod, acc); acc *= b
    end
    contrib_flat = Int64[]; contrib_off = Int32[]; off = 0
    for (j, p) in enumerate(wl)
        M = wheel128 ÷ Int128(p)             # wheel / q_j
        y = Int128(invmod(Int64(M % Int128(p)), p))
        coef = mod(M * y, wheel128)          # ≡1 (mod q_j), ≡0 (mod 他)
        push!(contrib_off, Int32(off))
        for a in goods[j]
            push!(contrib_flat, Int64(mod(Int128(a) * coef, wheel128)))
        end
        off += length(goods[j])
    end
    return Int64(wheel128), contrib_flat, contrib_off, bases, radixprod, R
end

# ストリーミング用 GPU 常駐状態 (CRT寄与テーブル + 3-limb篩テーブル)
struct GpuWheelStreamState
    k::Int
    wheel::Int64
    R::Int64
    w::Int32
    d_contrib::CuArray{Int64,1}
    d_contrib_off::CuArray{Int32,1}
    d_bases::CuArray{Int64,1}
    d_radixprod::CuArray{Int64,1}
    d_primes::CuArray{Int32,1}
    d_wheel_mod::CuArray{Int32,1}
    d_pow20::CuArray{Int32,1}
    d_pow40::CuArray{Int32,1}
    d_mu::CuArray{UInt32,1}
    d_bad_off::CuArray{Int32,1}
    d_bad::CuArray{UInt8,1}
    nprimes::Int32
end

# 巨大数域向け拡張 wl: ストリーミングで materialize 不要になったため 31,47 を追加可能。
# wheel = 4.22e14 ×31×47 ≈ 6.15e17 < 2^60 (3-limb篩で扱える上限内)。
function _cc_wl_stream(k::Int)
    k ≤ 13 ? _cc_wl(k) : [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]
end

function gpu_wheel_stream_setup(k::Int; wl::Vector{Int} = _cc_wl_stream(k),
                                max_prime::Int = -1, verbose::Bool = true)
    wheel, contrib, contrib_off, bases, radixprod, R = build_wheel_streaming(k, wl)
    mp = max_prime < 0 ? _default_max_prime(k) : max_prime
    mdllist = gpu_build_extra(k, mp; wl = wl)
    primes32, offsets, badflat = _flatten_bad_gpu(mdllist)
    wheel_mod = Int32[Int32(mod(wheel, Int64(p))) for p in primes32]
    pow20 = Int32[Int32(mod(Int64(1) << 20, Int64(p))) for p in primes32]
    pow40 = Int32[Int32(mod(Int64(1) << 40, Int64(p))) for p in primes32]
    mu = _barrett_mu(primes32)
    if verbose
        dens = R / wheel
        println("  stream setup CC$k: R=$R density=$(round(dens, sigdigits=3)) primes=$(length(primes32)) wheel=$wheel")
        println("    wl=$wl")
    end
    return GpuWheelStreamState(k, wheel, R, Int32(length(wl)),
        CuArray(contrib), CuArray(contrib_off), CuArray(bases), CuArray(radixprod),
        CuArray(primes32), CuArray(wheel_mod), CuArray(pow20), CuArray(pow40),
        CuArray(mu), CuArray(offsets), CuArray(badflat), Int32(length(primes32)))
end

function _cc_wheel_kernel_stream128!(out::CuDeviceVector{Int128},
        counter::CuDeviceVector{Int32}, cap::Int32,
        d_contrib::CuDeviceVector{Int64}, d_contrib_off::CuDeviceVector{Int32},
        d_bases::CuDeviceVector{Int64}, d_radixprod::CuDeviceVector{Int64},
        w::Int32, wheel::Int64, k_start::Int64, R::Int64,
        work_base::Int64, total_work::Int64,
        lo::Int128, hi::Int128,
        d_primes::CuDeviceVector{Int32}, d_wheel_mod::CuDeviceVector{Int32},
        d_pow20::CuDeviceVector{Int32}, d_pow40::CuDeviceVector{Int32},
        d_mu::CuDeviceVector{UInt32},
        d_bad_off::CuDeviceVector{Int32}, d_bad::CuDeviceVector{UInt8},
        nprimes::Int32, kk::Int32)
    t = (blockIdx().x - Int64(1)) * blockDim().x + threadIdx().x
    g = work_base + t - Int64(1)                 # 0-based グローバル作業番号
    if g < total_work
        cyc = k_start + g ÷ R
        sidx = g % R
        # --- オドメータ (mixed-radix) + CRT で生存残基 r を合成 ---
        r = Int64(0)
        @inbounds for j in 1:w
            ij = (sidx ÷ d_radixprod[j]) % d_bases[j]
            r = (r + d_contrib[d_contrib_off[j] + ij + Int64(1)]) % wheel
        end
        n = Int128(cyc) * Int128(wheel) + Int128(r)
        if n > lo && n < hi
            # --- 3-limb 32bit 篩 (wheel < 2^60) + Barrett還元で除算レス ---
            r2 = UInt32((r >> 40) & 0x00000000000FFFFF)
            r1 = UInt32((r >> 20) & 0x00000000000FFFFF)
            r0 = UInt32(r & 0x00000000000FFFFF)
            cyc32 = UInt32(cyc & 0x7FFFFFFF)
            ok = true
            @inbounds for pi in 1:nprimes
                p = UInt32(d_primes[pi]); mu = d_mu[pi]
                t2 = bmod(bmod(r2, p, mu) * UInt32(d_pow40[pi]), p, mu)
                t1 = bmod(bmod(r1, p, mu) * UInt32(d_pow20[pi]) + r0, p, mu)
                rp = bmod(t2 + t1, p, mu)
                cp = bmod(cyc32, p, mu)
                np = bmod(cp * UInt32(d_wheel_mod[pi]) + rp, p, mu)
                if d_bad[d_bad_off[pi] + Int64(np) + Int64(1)] != 0x00
                    ok = false
                    break
                end
            end
            if ok
                x_lo = UInt64(n & Int128(0xFFFFFFFFFFFFFFFF))
                x_hi = UInt64((n >> 64) & Int128(0xFFFFFFFFFFFFFFFF))
                good = true
                @inbounds for _ in 1:kk
                    isp = if x_hi == UInt64(0) && x_lo <= 0x7FFFFFFFFFFFFFFF
                        is_prime_mr(Int64(x_lo))
                    else
                        (x_hi > 0x7FFFFFFFFFFFFFFF) ? false : is_prime_mr128(x_lo, x_hi)
                    end
                    if !isp
                        good = false
                        break
                    end
                    carry = (x_lo > 0x7FFFFFFFFFFFFFFF) ? UInt64(1) : UInt64(0)
                    x_lo = (x_lo << 1) | UInt64(1)
                    x_hi = (x_hi << 1) | carry
                end
                if good
                    idx = CUDA.atomic_add!(pointer(counter, 1), Int32(1)) + Int32(1)
                    if idx <= cap
                        @inbounds out[idx] = n
                    end
                end
            end
        end
    end
    return nothing
end

# 篩専用 (ストリーミング CRT オドメータ). 生存 n を d_surv に追記 (MR含まず → 低レジスタ)。
function _cc_sieve_kernel_stream128!(d_surv::CuDeviceVector{Int128},
        scnt::CuDeviceVector{Int32}, scap::Int32,
        d_contrib::CuDeviceVector{Int64}, d_contrib_off::CuDeviceVector{Int32},
        d_bases::CuDeviceVector{Int64}, d_radixprod::CuDeviceVector{Int64},
        w::Int32, wheel::Int64, k_start::Int64, R::Int64,
        work_base::Int64, total_work::Int64,
        lo::Int128, hi::Int128,
        d_primes::CuDeviceVector{Int32}, d_wheel_mod::CuDeviceVector{Int32},
        d_pow20::CuDeviceVector{Int32}, d_pow40::CuDeviceVector{Int32},
        d_mu::CuDeviceVector{UInt32},
        d_bad_off::CuDeviceVector{Int32}, d_bad::CuDeviceVector{UInt8},
        nprimes::Int32)
    t = (blockIdx().x - Int64(1)) * blockDim().x + threadIdx().x
    g = work_base + t - Int64(1)
    if g < total_work
        cyc = k_start + g ÷ R
        sidx = g % R
        r = Int64(0)
        @inbounds for j in 1:w
            ij = (sidx ÷ d_radixprod[j]) % d_bases[j]
            r = (r + d_contrib[d_contrib_off[j] + ij + Int64(1)]) % wheel
        end
        n = Int128(cyc) * Int128(wheel) + Int128(r)
        if n > lo && n < hi
            r2 = UInt32((r >> 40) & 0x00000000000FFFFF)
            r1 = UInt32((r >> 20) & 0x00000000000FFFFF)
            r0 = UInt32(r & 0x00000000000FFFFF)
            cyc32 = UInt32(cyc & 0x7FFFFFFF)
            ok = true
            @inbounds for pi in 1:nprimes
                p = UInt32(d_primes[pi]); mu = d_mu[pi]
                t2 = bmod(bmod(r2, p, mu) * UInt32(d_pow40[pi]), p, mu)
                t1 = bmod(bmod(r1, p, mu) * UInt32(d_pow20[pi]) + r0, p, mu)
                rp = bmod(t2 + t1, p, mu)
                cp = bmod(cyc32, p, mu)
                np = bmod(cp * UInt32(d_wheel_mod[pi]) + rp, p, mu)
                if d_bad[d_bad_off[pi] + Int64(np) + Int64(1)] != 0x00
                    ok = false
                    break
                end
            end
            if ok
                idx = CUDA.atomic_add!(pointer(scnt, 1), Int32(1)) + Int32(1)
                idx <= scap && (@inbounds d_surv[idx] = n)
            end
        end
    end
    return nothing
end

function gpu_wheel_scan_stream128!(st::GpuWheelStreamState, lo::Int128, hi::Int128;
        work_tile::Int = 1 << 25, cap::Int = 1 << 12, progress::Bool = false,
        threads::Int = 64, mr_threads::Int = 64, scap::Int = 1 << 20,
        drain_every::Int = 256)
    w = Int128(st.wheel)
    k_start = Int64(lo ÷ w)
    k_end   = Int64((hi - Int128(1)) ÷ w)
    @assert k_end < (Int64(1) << 31) "cyc≥2^31: 範囲が大きすぎ/wheelが小さすぎ (32bit篩の前提が崩れる)"
    total_cyc = k_end - k_start + 1
    total_work = total_cyc * st.R
    d_out = CUDA.zeros(Int128, cap)
    d_cnt = CUDA.zeros(Int32, 1)
    d_surv = CUDA.zeros(Int128, scap)
    d_scnt = CUDA.zeros(Int32, 1)
    mr_blocks = cld(scap, mr_threads)
    t0 = time(); last_report = 0.0
    wb = Int64(0); tile = 0
    while wb < total_work
        nwork = min(Int64(work_tile), total_work - wb)
        blocks = cld(nwork, threads)
        @cuda threads=threads blocks=blocks _cc_sieve_kernel_stream128!(
            d_surv, d_scnt, Int32(scap),
            st.d_contrib, st.d_contrib_off, st.d_bases, st.d_radixprod,
            st.w, st.wheel, k_start, st.R, wb, total_work,
            lo, hi, st.d_primes, st.d_wheel_mod, st.d_pow20, st.d_pow40, st.d_mu,
            st.d_bad_off, st.d_bad, st.nprimes)
        wb += nwork; tile += 1
        if tile % drain_every == 0
            @cuda threads=mr_threads blocks=mr_blocks _cc_mr_kernel128!(
                d_surv, d_scnt, d_out, d_cnt, Int32(cap), Int32(st.k))
            CUDA.fill!(d_scnt, Int32(0))
        end
        if progress
            now = time() - t0
            if now - last_report > 15.0
                CUDA.synchronize()
                @assert Array(d_scnt)[1] <= scap "survivor overflow: drain_every↓ か scap↑"
                frac = wb / total_work
                eta = frac > 0 ? now/frac*(1-frac) : 0.0
                @info "  CC$(st.k)/stream $(round(100*frac,digits=1))% | $(round(now,digits=0))s | ETA $(round(eta,digits=0))s | found=$(Array(d_cnt)[1])"
                last_report = now
            end
        end
    end
    @cuda threads=mr_threads blocks=mr_blocks _cc_mr_kernel128!(
        d_surv, d_scnt, d_out, d_cnt, Int32(cap), Int32(st.k))
    CUDA.synchronize()
    cnt = Array(d_cnt)[1]
    results = Int128[]
    if cnt > 0
        got = min(Int(cnt), cap)
        append!(results, Array(view(d_out, 1:got)))
    end
    CUDA.unsafe_free!(d_out); CUDA.unsafe_free!(d_cnt)
    CUDA.unsafe_free!(d_surv); CUDA.unsafe_free!(d_scnt)
    sort!(results)
    return results
end

# 全部入り (ストリーミング CRT, Int128 頭)。巨大数域 CC 探索の推奨経路。
function search_cc_gpu_wheel_stream128(lo::Integer, hi::Integer, k::Int;
        wl::Vector{Int} = _cc_wl_stream(k), max_prime::Int = -1,
        work_tile::Int = 1 << 25, progress::Bool = false, verbose::Bool = true)
    t0 = time()
    st = gpu_wheel_stream_setup(k; wl=wl, max_prime=max_prime, verbose=verbose)
    r = gpu_wheel_scan_stream128!(st, Int128(lo), Int128(hi);
                                  work_tile=work_tile, progress=progress)
    verbose && println("=== GPU-stream128 CC$k: $(length(r)) in $(round(time()-t0,digits=3))s ===")
    return r
end

# ============================================================
# 第二種カニンガム鎖 (p, 2p-1, 4p-3, ...) の GPU ホイール篩
#   第一種との唯一の違いは鎖進行が x -> 2x - 1 (下位ビットが 0 になる) であること。
#   篩 bad 残基も y -> 2y - 1 で計算。カーネル/setup/scan は全て *_2 別名で複製。
# ============================================================

# 任意 wl から wheel 生存残基を生成 (第二種: 2y-1)
function build_wheel_custom_2(k::Int, wl::Vector{Int})
    cc_len_mod(x, p) = (n = 0; y = x; while n < k && y % p != 0; y = (2y - 1) % p; n += 1; end; n)
    wheel_n = Int[1]; pprod = 1
    for p in wl
        bad = Set(i for i in 0:p-1 if cc_len_mod(i, p) < k)
        tmp = wheel_n; wheel_n = Int[]
        sizehint!(wheel_n, length(tmp) * (p - length(bad)))
        for i in 0:p-1
            @inbounds for x in tmp
                v = x + i * pprod
                (v % p) in bad || push!(wheel_n, v)
            end
        end
        pprod *= p
    end
    return prod(wl), wheel_n
end

# 追加素数の bad 残差リスト (第二種: 2y-1)
function gpu_build_extra_2(k::Int, max_prime::Int; wl::Vector{Int} = _cc_wl(k))
    @assert max_prime < 46340 "max_prime は 46340 未満に (Int32 32bit演算の制約)"
    function cc_len_mod(x::Int, p::Int)::Int
        n = 0; y = x
        while n < k && y % p != 0
            y = (2y - 1) % p; n += 1
        end
        return n
    end
    extra = filter(p -> !(p in wl) && p <= max_prime, primes(max_prime))
    return [(p, Set(i for i in 0:p-1 if cc_len_mod(i, p) < k)) for p in extra]
end

# ストリーミング用 CRT 寄与 (第二種: 2y-1)
function build_wheel_streaming_2(k::Int, wl::Vector{Int})
    function cc_len_mod(x::Int, p::Int)::Int
        n = 0; y = x
        while n < k && y % p != 0
            y = (2y - 1) % p; n += 1
        end
        return n
    end
    wheel128 = prod(Int128, wl)
    @assert wheel128 < (Int128(1) << 60) "wheel exceeds 2^60 (3-limb 篩の前提が崩れる)"
    goods = [[i for i in 0:p-1 if cc_len_mod(i, p) >= k] for p in wl]
    bases = Int64[length(g) for g in goods]
    R = prod(bases)
    radixprod = Int64[]; acc = Int64(1)
    for b in bases
        push!(radixprod, acc); acc *= b
    end
    contrib_flat = Int64[]; contrib_off = Int32[]; off = 0
    for (j, p) in enumerate(wl)
        M = wheel128 ÷ Int128(p)
        y = Int128(invmod(Int64(M % Int128(p)), p))
        coef = mod(M * y, wheel128)
        push!(contrib_off, Int32(off))
        for a in goods[j]
            push!(contrib_flat, Int64(mod(Int128(a) * coef, wheel128)))
        end
        off += length(goods[j])
    end
    return Int64(wheel128), contrib_flat, contrib_off, bases, radixprod, R
end
# 融合カーネル (第二種) — 64bit 頭 : 鎖進行 x = 2x - 1
function _cc_wheel_kernel_2!(out::CuDeviceVector{Int64},
                            counter::CuDeviceVector{Int32}, cap::Int32,
                            d_wheel_n::CuDeviceVector{Int64}, R::Int64,
                            wheel::Int64, k_base::Int64, ncyc::Int64,
                            lo::Int64, hi::Int64,
                            d_primes::CuDeviceVector{Int32},
                            d_wheel_mod::CuDeviceVector{Int32},
                            d_pow20::CuDeviceVector{Int32},
                            d_mu::CuDeviceVector{UInt32},
                            d_bad_off::CuDeviceVector{Int32},
                            d_bad::CuDeviceVector{UInt8}, nprimes::Int32,
                            kk::Int32)
    g = (blockIdx().x - Int64(1)) * blockDim().x + threadIdx().x
    total = ncyc * R
    if g <= total
        cyc = k_base + (g - Int64(1)) ÷ R
        ridx = (g - Int64(1)) % R + Int64(1)
        @inbounds r = d_wheel_n[ridx]
        n = cyc * wheel + r
        if n > lo && n < hi
            r_hi = UInt32((r >> 20) & 0x7FFFFFFF)
            r_lo = UInt32(r & 0x00000000000FFFFF)
            cyc32 = UInt32(cyc & 0x7FFFFFFF)
            ok = true
            @inbounds for pi in 1:nprimes
                p = UInt32(d_primes[pi])
                Int64(p) >= n && break
                mu = d_mu[pi]
                rhp = bmod(r_hi, p, mu)
                rp = bmod(rhp * UInt32(d_pow20[pi]) + r_lo, p, mu)
                cp = bmod(cyc32, p, mu)
                np = bmod(cp * UInt32(d_wheel_mod[pi]) + rp, p, mu)
                if d_bad[d_bad_off[pi] + Int64(np) + Int64(1)] != 0x00
                    ok = false; break
                end
            end
            if ok
                x_lo = reinterpret(UInt64, n)
                x_hi = UInt64(0)
                good = true
                @inbounds for _ in 1:kk
                    isp = if x_hi == UInt64(0) && x_lo <= 0x7FFFFFFFFFFFFFFF
                        is_prime_mr(Int64(x_lo))
                    else
                        (x_hi > 0x7FFFFFFFFFFFFFFF) ? false : is_prime_mr128(x_lo, x_hi)
                    end
                    if !isp
                        good = false; break
                    end
                    carry = (x_lo > 0x7FFFFFFFFFFFFFFF) ? UInt64(1) : UInt64(0)
                    x_lo = x_lo << 1
                    x_hi = (x_hi << 1) | carry
                    if x_lo == UInt64(0)
                        x_lo = 0xFFFFFFFFFFFFFFFF
                        x_hi -= UInt64(1)
                    else
                        x_lo -= UInt64(1)
                    end
                end
                if good
                    idx = CUDA.atomic_add!(pointer(counter, 1), Int32(1)) + Int32(1)
                    if idx <= cap
                        @inbounds out[idx] = n
                    end
                end
            end
        end
    end
    return nothing
end

# 融合カーネル (第二種, Int128 頭)
function _cc_wheel_kernel128_2!(out::CuDeviceVector{Int128},
                            counter::CuDeviceVector{Int32}, cap::Int32,
                            d_wheel_n::CuDeviceVector{Int64}, R::Int64,
                            wheel::Int64, k_base::Int64, ncyc::Int64,
                            lo::Int128, hi::Int128,
                            d_primes::CuDeviceVector{Int32},
                            d_wheel_mod::CuDeviceVector{Int32},
                            d_pow20::CuDeviceVector{Int32},
                            d_mu::CuDeviceVector{UInt32},
                            d_bad_off::CuDeviceVector{Int32},
                            d_bad::CuDeviceVector{UInt8}, nprimes::Int32,
                            kk::Int32)
    g = (blockIdx().x - Int64(1)) * blockDim().x + threadIdx().x
    total = ncyc * R
    if g <= total
        cyc = k_base + (g - Int64(1)) ÷ R
        ridx = (g - Int64(1)) % R + Int64(1)
        @inbounds r = d_wheel_n[ridx]
        n = Int128(cyc) * Int128(wheel) + Int128(r)
        if n > lo && n < hi
            r_hi = UInt32((r >> 20) & 0x7FFFFFFF)
            r_lo = UInt32(r & 0x00000000000FFFFF)
            cyc32 = UInt32(cyc & 0x7FFFFFFF)
            ok = true
            @inbounds for pi in 1:nprimes
                p = UInt32(d_primes[pi]); mu = d_mu[pi]
                rhp = bmod(r_hi, p, mu)
                rp = bmod(rhp * UInt32(d_pow20[pi]) + r_lo, p, mu)
                cp = bmod(cyc32, p, mu)
                np = bmod(cp * UInt32(d_wheel_mod[pi]) + rp, p, mu)
                if d_bad[d_bad_off[pi] + Int64(np) + Int64(1)] != 0x00
                    ok = false; break
                end
            end
            if ok
                x_lo = UInt64(n & Int128(0xFFFFFFFFFFFFFFFF))
                x_hi = UInt64((n >> 64) & Int128(0xFFFFFFFFFFFFFFFF))
                good = true
                @inbounds for _ in 1:kk
                    isp = if x_hi == UInt64(0) && x_lo <= 0x7FFFFFFFFFFFFFFF
                        is_prime_mr(Int64(x_lo))
                    else
                        (x_hi > 0x7FFFFFFFFFFFFFFF) ? false : is_prime_mr128(x_lo, x_hi)
                    end
                    if !isp
                        good = false; break
                    end
                    carry = (x_lo > 0x7FFFFFFFFFFFFFFF) ? UInt64(1) : UInt64(0)
                    x_lo = x_lo << 1
                    x_hi = (x_hi << 1) | carry
                    if x_lo == UInt64(0)
                        x_lo = 0xFFFFFFFFFFFFFFFF
                        x_hi -= UInt64(1)
                    else
                        x_lo -= UInt64(1)
                    end
                end
                if good
                    idx = CUDA.atomic_add!(pointer(counter, 1), Int32(1)) + Int32(1)
                    if idx <= cap
                        @inbounds out[idx] = n
                    end
                end
            end
        end
    end
    return nothing
end

# 篩専用 (第二種, Int128). 生存 n を d_surv に atomic 追記。
function _cc_sieve_kernel128_2!(d_surv::CuDeviceVector{Int128},
                            scnt::CuDeviceVector{Int32}, scap::Int32,
                            d_wheel_n::CuDeviceVector{Int64}, R::Int64,
                            wheel::Int64, k_base::Int64, ncyc::Int64,
                            lo::Int128, hi::Int128,
                            d_primes::CuDeviceVector{Int32},
                            d_wheel_mod::CuDeviceVector{Int32},
                            d_pow20::CuDeviceVector{Int32},
                            d_mu::CuDeviceVector{UInt32},
                            d_bad_off::CuDeviceVector{Int32},
                            d_bad::CuDeviceVector{UInt8}, nprimes::Int32)
    g = (blockIdx().x - Int64(1)) * blockDim().x + threadIdx().x
    total = ncyc * R
    if g <= total
        cyc = k_base + (g - Int64(1)) ÷ R
        ridx = (g - Int64(1)) % R + Int64(1)
        @inbounds r = d_wheel_n[ridx]
        n = Int128(cyc) * Int128(wheel) + Int128(r)
        if n > lo && n < hi
            r_hi = UInt32((r >> 20) & 0x7FFFFFFF)
            r_lo = UInt32(r & 0x00000000000FFFFF)
            cyc32 = UInt32(cyc & 0x7FFFFFFF)
            ok = true
            @inbounds for pi in 1:nprimes
                p = UInt32(d_primes[pi]); mu = d_mu[pi]
                rhp = bmod(r_hi, p, mu)
                rp = bmod(rhp * UInt32(d_pow20[pi]) + r_lo, p, mu)
                cp = bmod(cyc32, p, mu)
                np = bmod(cp * UInt32(d_wheel_mod[pi]) + rp, p, mu)
                if d_bad[d_bad_off[pi] + Int64(np) + Int64(1)] != 0x00
                    ok = false; break
                end
            end
            if ok
                idx = CUDA.atomic_add!(pointer(scnt, 1), Int32(1)) + Int32(1)
                idx <= scap && (@inbounds d_surv[idx] = n)
            end
        end
    end
    return nothing
end

# MR専用 (第二種, Int128). 生存候補 d_surv[1..scnt] のみ鎖MR (x=2x-1)。
function _cc_mr_kernel128_2!(d_surv::CuDeviceVector{Int128},
                            scnt::CuDeviceVector{Int32},
                            out::CuDeviceVector{Int128},
                            counter::CuDeviceVector{Int32}, cap::Int32, kk::Int32)
    i = (blockIdx().x - Int64(1)) * blockDim().x + threadIdx().x
    @inbounds nsurv = scnt[1]
    if i <= nsurv
        @inbounds n = d_surv[i]
        x_lo = UInt64(n & Int128(0xFFFFFFFFFFFFFFFF))
        x_hi = UInt64((n >> 64) & Int128(0xFFFFFFFFFFFFFFFF))
        good = true
        @inbounds for _ in 1:kk
            isp = if x_hi == UInt64(0) && x_lo <= 0x7FFFFFFFFFFFFFFF
                is_prime_mr(Int64(x_lo))
            else
                (x_hi > 0x7FFFFFFFFFFFFFFF) ? false : is_prime_mr128(x_lo, x_hi)
            end
            if !isp
                good = false; break
            end
            carry = (x_lo > 0x7FFFFFFFFFFFFFFF) ? UInt64(1) : UInt64(0)
            x_lo = x_lo << 1
            x_hi = (x_hi << 1) | carry
            if x_lo == UInt64(0)
                x_lo = 0xFFFFFFFFFFFFFFFF
                x_hi -= UInt64(1)
            else
                x_lo -= UInt64(1)
            end
        end
        if good
            idx = CUDA.atomic_add!(pointer(counter, 1), Int32(1)) + Int32(1)
            if idx <= cap
                @inbounds out[idx] = n
            end
        end
    end
    return nothing
end

# ストリーミング融合カーネル (第二種, Int128 頭)
function _cc_wheel_kernel_stream128_2!(out::CuDeviceVector{Int128},
        counter::CuDeviceVector{Int32}, cap::Int32,
        d_contrib::CuDeviceVector{Int64}, d_contrib_off::CuDeviceVector{Int32},
        d_bases::CuDeviceVector{Int64}, d_radixprod::CuDeviceVector{Int64},
        w::Int32, wheel::Int64, k_start::Int64, R::Int64,
        work_base::Int64, total_work::Int64,
        lo::Int128, hi::Int128,
        d_primes::CuDeviceVector{Int32}, d_wheel_mod::CuDeviceVector{Int32},
        d_pow20::CuDeviceVector{Int32}, d_pow40::CuDeviceVector{Int32},
        d_mu::CuDeviceVector{UInt32},
        d_bad_off::CuDeviceVector{Int32}, d_bad::CuDeviceVector{UInt8},
        nprimes::Int32, kk::Int32)
    t = (blockIdx().x - Int64(1)) * blockDim().x + threadIdx().x
    g = work_base + t - Int64(1)
    if g < total_work
        cyc = k_start + g ÷ R
        sidx = g % R
        r = Int64(0)
        @inbounds for j in 1:w
            ij = (sidx ÷ d_radixprod[j]) % d_bases[j]
            r = (r + d_contrib[d_contrib_off[j] + ij + Int64(1)]) % wheel
        end
        n = Int128(cyc) * Int128(wheel) + Int128(r)
        if n > lo && n < hi
            r2 = UInt32((r >> 40) & 0x00000000000FFFFF)
            r1 = UInt32((r >> 20) & 0x00000000000FFFFF)
            r0 = UInt32(r & 0x00000000000FFFFF)
            cyc32 = UInt32(cyc & 0x7FFFFFFF)
            ok = true
            @inbounds for pi in 1:nprimes
                p = UInt32(d_primes[pi]); mu = d_mu[pi]
                t2 = bmod(bmod(r2, p, mu) * UInt32(d_pow40[pi]), p, mu)
                t1 = bmod(bmod(r1, p, mu) * UInt32(d_pow20[pi]) + r0, p, mu)
                rp = bmod(t2 + t1, p, mu)
                cp = bmod(cyc32, p, mu)
                np = bmod(cp * UInt32(d_wheel_mod[pi]) + rp, p, mu)
                if d_bad[d_bad_off[pi] + Int64(np) + Int64(1)] != 0x00
                    ok = false; break
                end
            end
            if ok
                x_lo = UInt64(n & Int128(0xFFFFFFFFFFFFFFFF))
                x_hi = UInt64((n >> 64) & Int128(0xFFFFFFFFFFFFFFFF))
                good = true
                @inbounds for _ in 1:kk
                    isp = if x_hi == UInt64(0) && x_lo <= 0x7FFFFFFFFFFFFFFF
                        is_prime_mr(Int64(x_lo))
                    else
                        (x_hi > 0x7FFFFFFFFFFFFFFF) ? false : is_prime_mr128(x_lo, x_hi)
                    end
                    if !isp
                        good = false; break
                    end
                    carry = (x_lo > 0x7FFFFFFFFFFFFFFF) ? UInt64(1) : UInt64(0)
                    x_lo = x_lo << 1
                    x_hi = (x_hi << 1) | carry
                    if x_lo == UInt64(0)
                        x_lo = 0xFFFFFFFFFFFFFFFF
                        x_hi -= UInt64(1)
                    else
                        x_lo -= UInt64(1)
                    end
                end
                if good
                    idx = CUDA.atomic_add!(pointer(counter, 1), Int32(1)) + Int32(1)
                    if idx <= cap
                        @inbounds out[idx] = n
                    end
                end
            end
        end
    end
    return nothing
end

# ストリーミング篩専用 (第二種, Int128)
function _cc_sieve_kernel_stream128_2!(d_surv::CuDeviceVector{Int128},
        scnt::CuDeviceVector{Int32}, scap::Int32,
        d_contrib::CuDeviceVector{Int64}, d_contrib_off::CuDeviceVector{Int32},
        d_bases::CuDeviceVector{Int64}, d_radixprod::CuDeviceVector{Int64},
        w::Int32, wheel::Int64, k_start::Int64, R::Int64,
        work_base::Int64, total_work::Int64,
        lo::Int128, hi::Int128,
        d_primes::CuDeviceVector{Int32}, d_wheel_mod::CuDeviceVector{Int32},
        d_pow20::CuDeviceVector{Int32}, d_pow40::CuDeviceVector{Int32},
        d_mu::CuDeviceVector{UInt32},
        d_bad_off::CuDeviceVector{Int32}, d_bad::CuDeviceVector{UInt8},
        nprimes::Int32)
    t = (blockIdx().x - Int64(1)) * blockDim().x + threadIdx().x
    g = work_base + t - Int64(1)
    if g < total_work
        cyc = k_start + g ÷ R
        sidx = g % R
        r = Int64(0)
        @inbounds for j in 1:w
            ij = (sidx ÷ d_radixprod[j]) % d_bases[j]
            r = (r + d_contrib[d_contrib_off[j] + ij + Int64(1)]) % wheel
        end
        n = Int128(cyc) * Int128(wheel) + Int128(r)
        if n > lo && n < hi
            r2 = UInt32((r >> 40) & 0x00000000000FFFFF)
            r1 = UInt32((r >> 20) & 0x00000000000FFFFF)
            r0 = UInt32(r & 0x00000000000FFFFF)
            cyc32 = UInt32(cyc & 0x7FFFFFFF)
            ok = true
            @inbounds for pi in 1:nprimes
                p = UInt32(d_primes[pi]); mu = d_mu[pi]
                t2 = bmod(bmod(r2, p, mu) * UInt32(d_pow40[pi]), p, mu)
                t1 = bmod(bmod(r1, p, mu) * UInt32(d_pow20[pi]) + r0, p, mu)
                rp = bmod(t2 + t1, p, mu)
                cp = bmod(cyc32, p, mu)
                np = bmod(cp * UInt32(d_wheel_mod[pi]) + rp, p, mu)
                if d_bad[d_bad_off[pi] + Int64(np) + Int64(1)] != 0x00
                    ok = false; break
                end
            end
            if ok
                idx = CUDA.atomic_add!(pointer(scnt, 1), Int32(1)) + Int32(1)
                idx <= scap && (@inbounds d_surv[idx] = n)
            end
        end
    end
    return nothing
end
# ============================================================
# 第二種用 setup / scan / search
#   GpuWheelState / GpuWheelStreamState は第一種と同一構造を再利用。
#   篩 bad 残基のみ build_cc_sieve_2 / gpu_build_extra_2 / build_wheel_streaming_2 で構築。
# ============================================================

# セットアップ (第二種)
function gpu_wheel_setup_2(k::Int; cap::Int = 1 << 16, max_prime::Int = 0,
                          wl::Union{Nothing,Vector{Int}} = nothing, verbose::Bool = true)
    if wl === nothing
        wheel, wheel_n, mdllist = build_cc_sieve_2(k)
    else
        wheel, wheel_n = build_wheel_custom_2(k, wl)
        mdllist = Tuple{Int,Set{Int}}[]
    end
    if max_prime > 0 || wl !== nothing
        mp = max_prime > 0 ? max_prime : _default_max_prime(k)
        wlset = wl === nothing ? _cc_wl(k) : wl
        mdllist = gpu_build_extra_2(k, mp; wl = wlset)
    end
    R = Int64(length(wheel_n))
    primes32, offsets, badflat = _flatten_bad_gpu(mdllist)
    wheel_mod = Int32[Int32(mod(Int64(wheel), Int64(p))) for p in primes32]
    pow20 = Int32[Int32(mod(Int64(1) << 20, Int64(p))) for p in primes32]
    mu = _barrett_mu(primes32)
    @assert wheel < (Int64(1) << 51) "wheel exceeds 2^51; 32bit分解の前提が崩れる (CC$k)"
    verbose && println("  setup CC$(k)-2nd: R=$R primes=$(length(primes32)) wheel=$wheel")
    return GpuWheelState(k, Int64(wheel), R,
        CuArray(Int64.(wheel_n)), CuArray(primes32),
        CuArray(wheel_mod), CuArray(pow20), CuArray(mu), CuArray(offsets),
        CuArray(badflat), Int32(length(primes32)),
        CUDA.zeros(Int64, cap), CUDA.zeros(Int32, 1), cap)
end

# 純スキャン (第二種, セットアップ済み state を使用)
function gpu_wheel_scan_2!(st::GpuWheelState, lo::Integer, hi::Integer;
                          work_tile::Int = 1 << 25, progress::Bool = false,
                          threads::Int = 64)
    lo64 = Int64(lo); hi64 = Int64(hi)
    w64 = st.wheel
    k_start = lo64 ÷ w64
    k_end   = (hi64 - 1) ÷ w64
    total_cyc = k_end - k_start + 1
    results = Int64[]
    cyc_per_launch = max(1, work_tile ÷ Int(st.R))
    fill!(st.d_cnt, Int32(0))
    t0 = time()
    last_report = 0.0
    cyc = k_start
    while cyc <= k_end
        ncyc = min(cyc_per_launch, k_end - cyc + 1)
        work = ncyc * st.R
        blocks = cld(work, threads)
        @cuda threads=threads blocks=blocks _cc_wheel_kernel_2!(
            st.d_out, st.d_cnt, Int32(st.cap), st.d_wheel_n, st.R, w64, cyc, ncyc,
            lo64, hi64, st.d_primes, st.d_wheel_mod, st.d_pow20, st.d_mu,
            st.d_bad_off, st.d_bad, st.nprimes, Int32(st.k))
        cyc += ncyc
        if progress
            frac = (cyc - k_start) / total_cyc
            now = time() - t0
            if now - last_report > 15.0
                CUDA.synchronize()
                eta = frac > 0 ? now/frac*(1-frac) : 0.0
                @info "  CC$(st.k)-2nd scan $(round(100*frac,digits=1))% | $(round(now,digits=0))s | ETA $(round(eta,digits=0))s | found=$(Array(st.d_cnt)[1])"
                last_report = now
            end
        end
    end
    CUDA.synchronize()
    cnt = Array(st.d_cnt)[1]
    if cnt > 0
        got = min(Int(cnt), st.cap)
        append!(results, Array(view(st.d_out, 1:got)))
    end
    sort!(results)
    return results
end

# 全部入りの便利関数 (第二種)
function search_cc_gpu_wheel_2(lo::Integer, hi::Integer, k::Int;
                              work_tile::Int = 1 << 25,
                              cap::Int = 1 << 16,
                              max_prime::Int = -1,
                              progress::Bool = false,
                              verbose::Bool = true)
    mp = max_prime < 0 ? _default_max_prime(k) : max_prime
    t0 = time()
    st = gpu_wheel_setup_2(k; cap=cap, max_prime=mp, verbose=verbose)
    results = gpu_wheel_scan_2!(st, lo, hi; work_tile=work_tile, progress=progress)
    verbose && println("=== GPU-wheel CC$(k)-2nd: $(length(results)) in $(round(time()-t0,digits=3))s ===")
    return results
end

# Int128 頭 スキャン (第二種, 巨大数域). setup済み state を使用。
function gpu_wheel_scan128_2!(st::GpuWheelState, lo::Int128, hi::Int128;
                             work_tile::Int = 1 << 25, cap::Int = 1 << 12,
                             progress::Bool = false, threads::Int = 64,
                             mr_threads::Int = 64, scap::Int = 1 << 20,
                             drain_every::Int = 256)
    w = Int128(st.wheel)
    k_start = Int64(lo ÷ w)
    k_end   = Int64((hi - Int128(1)) ÷ w)
    @assert k_end < (Int64(1) << 31) "cyc>=2^31: 範囲が大きすぎ/wheelが小さすぎ (32bit篩の前提が崩れる)"
    d_out = CUDA.zeros(Int128, cap)
    d_cnt = CUDA.zeros(Int32, 1)
    d_surv = CUDA.zeros(Int128, scap)
    d_scnt = CUDA.zeros(Int32, 1)
    mr_blocks = cld(scap, mr_threads)
    cyc_per_launch = max(1, work_tile ÷ Int(st.R))
    total_cyc = k_end - k_start + 1
    t0 = time(); last_report = 0.0
    cyc = k_start; tile = 0
    while cyc <= k_end
        ncyc = min(cyc_per_launch, k_end - cyc + 1)
        work = ncyc * st.R
        blocks = cld(work, threads)
        @cuda threads=threads blocks=blocks _cc_sieve_kernel128_2!(
            d_surv, d_scnt, Int32(scap), st.d_wheel_n, st.R, st.wheel, cyc, ncyc,
            lo, hi, st.d_primes, st.d_wheel_mod, st.d_pow20, st.d_mu,
            st.d_bad_off, st.d_bad, st.nprimes)
        cyc += ncyc; tile += 1
        if tile % drain_every == 0
            @cuda threads=mr_threads blocks=mr_blocks _cc_mr_kernel128_2!(
                d_surv, d_scnt, d_out, d_cnt, Int32(cap), Int32(st.k))
            CUDA.fill!(d_scnt, Int32(0))
        end
        if progress
            now = time() - t0
            if now - last_report > 15.0
                CUDA.synchronize()
                @assert Array(d_scnt)[1] <= scap "survivor overflow: drain_every down か scap up"
                frac = (cyc - k_start) / total_cyc
                eta = frac > 0 ? now/frac*(1-frac) : 0.0
                @info "  CC$(st.k)-2nd/128 $(round(100*frac,digits=1))% | $(round(now,digits=0))s | ETA $(round(eta,digits=0))s | found=$(Array(d_cnt)[1])"
                last_report = now
            end
        end
    end
    @cuda threads=mr_threads blocks=mr_blocks _cc_mr_kernel128_2!(
        d_surv, d_scnt, d_out, d_cnt, Int32(cap), Int32(st.k))
    CUDA.synchronize()
    cnt = Array(d_cnt)[1]
    results = Int128[]
    if cnt > 0
        got = min(Int(cnt), cap)
        append!(results, Array(view(d_out, 1:got)))
    end
    CUDA.unsafe_free!(d_out); CUDA.unsafe_free!(d_cnt)
    CUDA.unsafe_free!(d_surv); CUDA.unsafe_free!(d_scnt)
    sort!(results)
    return results
end

# 全部入り (第二種, Int128 頭)
function search_cc_gpu_wheel128_2(lo::Integer, hi::Integer, k::Int;
                                 max_prime::Int = -1, progress::Bool = false,
                                 verbose::Bool = true)
    mp = max_prime < 0 ? _default_max_prime(k) : max_prime
    t0 = time()
    st = gpu_wheel_setup_2(k; max_prime=mp, verbose=verbose)
    r = gpu_wheel_scan128_2!(st, Int128(lo), Int128(hi); progress=progress)
    verbose && println("=== GPU-wheel128 CC$(k)-2nd: $(length(r)) in $(round(time()-t0,digits=3))s ===")
    CUDA.unsafe_free!(st.d_out)
    return r
end

# ストリーミング setup (第二種)
function gpu_wheel_stream_setup_2(k::Int; wl::Vector{Int} = _cc_wl_stream(k),
                                 max_prime::Int = -1, verbose::Bool = true)
    wheel, contrib, contrib_off, bases, radixprod, R = build_wheel_streaming_2(k, wl)
    mp = max_prime < 0 ? _default_max_prime(k) : max_prime
    mdllist = gpu_build_extra_2(k, mp; wl = wl)
    primes32, offsets, badflat = _flatten_bad_gpu(mdllist)
    wheel_mod = Int32[Int32(mod(wheel, Int64(p))) for p in primes32]
    pow20 = Int32[Int32(mod(Int64(1) << 20, Int64(p))) for p in primes32]
    pow40 = Int32[Int32(mod(Int64(1) << 40, Int64(p))) for p in primes32]
    mu = _barrett_mu(primes32)
    if verbose
        dens = R / wheel
        println("  stream setup CC$(k)-2nd: R=$R density=$(round(dens, sigdigits=3)) primes=$(length(primes32)) wheel=$wheel")
        println("    wl=$wl")
    end
    return GpuWheelStreamState(k, wheel, R, Int32(length(wl)),
        CuArray(contrib), CuArray(contrib_off), CuArray(bases), CuArray(radixprod),
        CuArray(primes32), CuArray(wheel_mod), CuArray(pow20), CuArray(pow40),
        CuArray(mu), CuArray(offsets), CuArray(badflat), Int32(length(primes32)))
end

# ストリーミングスキャン (第二種, Int128 頭)
function gpu_wheel_scan_stream128_2!(st::GpuWheelStreamState, lo::Int128, hi::Int128;
        work_tile::Int = 1 << 25, cap::Int = 1 << 12, progress::Bool = false,
        threads::Int = 64, mr_threads::Int = 64, scap::Int = 1 << 20,
        drain_every::Int = 256)
    w = Int128(st.wheel)
    k_start = Int64(lo ÷ w)
    k_end   = Int64((hi - Int128(1)) ÷ w)
    @assert k_end < (Int64(1) << 31) "cyc>=2^31: 範囲が大きすぎ/wheelが小さすぎ (32bit篩の前提が崩れる)"
    total_cyc = k_end - k_start + 1
    total_work = total_cyc * st.R
    d_out = CUDA.zeros(Int128, cap)
    d_cnt = CUDA.zeros(Int32, 1)
    d_surv = CUDA.zeros(Int128, scap)
    d_scnt = CUDA.zeros(Int32, 1)
    mr_blocks = cld(scap, mr_threads)
    t0 = time(); last_report = 0.0
    wb = Int64(0); tile = 0
    while wb < total_work
        nwork = min(Int64(work_tile), total_work - wb)
        blocks = cld(nwork, threads)
        @cuda threads=threads blocks=blocks _cc_sieve_kernel_stream128_2!(
            d_surv, d_scnt, Int32(scap),
            st.d_contrib, st.d_contrib_off, st.d_bases, st.d_radixprod,
            st.w, st.wheel, k_start, st.R, wb, total_work,
            lo, hi, st.d_primes, st.d_wheel_mod, st.d_pow20, st.d_pow40, st.d_mu,
            st.d_bad_off, st.d_bad, st.nprimes)
        wb += nwork; tile += 1
        if tile % drain_every == 0
            @cuda threads=mr_threads blocks=mr_blocks _cc_mr_kernel128_2!(
                d_surv, d_scnt, d_out, d_cnt, Int32(cap), Int32(st.k))
            CUDA.fill!(d_scnt, Int32(0))
        end
        if progress
            now = time() - t0
            if now - last_report > 15.0
                CUDA.synchronize()
                @assert Array(d_scnt)[1] <= scap "survivor overflow: drain_every down か scap up"
                frac = wb / total_work
                eta = frac > 0 ? now/frac*(1-frac) : 0.0
                @info "  CC$(st.k)-2nd/stream $(round(100*frac,digits=1))% | $(round(now,digits=0))s | ETA $(round(eta,digits=0))s | found=$(Array(d_cnt)[1])"
                last_report = now
            end
        end
    end
    @cuda threads=mr_threads blocks=mr_blocks _cc_mr_kernel128_2!(
        d_surv, d_scnt, d_out, d_cnt, Int32(cap), Int32(st.k))
    CUDA.synchronize()
    cnt = Array(d_cnt)[1]
    results = Int128[]
    if cnt > 0
        got = min(Int(cnt), cap)
        append!(results, Array(view(d_out, 1:got)))
    end
    CUDA.unsafe_free!(d_out); CUDA.unsafe_free!(d_cnt)
    CUDA.unsafe_free!(d_surv); CUDA.unsafe_free!(d_scnt)
    sort!(results)
    return results
end

# 全部入り (第二種, ストリーミング CRT, Int128 頭)
function search_cc_gpu_wheel_stream128_2(lo::Integer, hi::Integer, k::Int;
        wl::Vector{Int} = _cc_wl_stream(k), max_prime::Int = -1,
        work_tile::Int = 1 << 25, progress::Bool = false, verbose::Bool = true)
    t0 = time()
    st = gpu_wheel_stream_setup_2(k; wl=wl, max_prime=max_prime, verbose=verbose)
    r = gpu_wheel_scan_stream128_2!(st, Int128(lo), Int128(hi);
                                   work_tile=work_tile, progress=progress)
    verbose && println("=== GPU-stream128 CC$(k)-2nd: $(length(r)) in $(round(time()-t0,digits=3))s ===")
    return r
end

# cc_gpu_block.jl — ブロック篩の生存者に対する GPU 上の鎖 MR テスト
# 使い方:
#   include("src/cc_cpu.jl")          # または search_cc が使える状態
#   include("src/cc_block.jl")
#   include("src/cc_gpu_block.jl")    # これで cc_chain_test_gpu が定義される
#   search_cc(lo, hi, k; gpu=true)
#
# 注意: CUDA.jl がシステムの CUDA ドライバと異なるバージョンで事前コンパイル
#       されている場合は、一度 `using CUDA` して再コンパイルするか、
#       ]build CUDA を実行すること。

include("prime_filter.jl")   # GPU 用 128bit MR: is_prime_mr, is_prime_mr128
using CUDA

# GPU カーネル: 候補 (lo,hi) が長さ k の第一カニンガム鎖開始値か判定
function _cc_gpu_kernel!(res::CuDeviceVector{Int32},
                         lo::CuDeviceVector{UInt64},
                         hi::CuDeviceVector{UInt64},
                         kk::Int32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(lo)
        x_lo = lo[i]; x_hi = hi[i]
        ok = true
        @inbounds for j in 1:kk
            if x_hi == 0x0000000000000000 && x_lo <= 0x7FFFFFFFFFFFFFFF
                is_prime_mr(Int64(x_lo)) || (ok = false; break)
            else
                (x_hi > 0x7FFFFFFFFFFFFFFF) && (ok = false; break)
                is_prime_mr128(x_lo, x_hi) || (ok = false; break)
            end
            carry = (x_lo > 0x7FFFFFFFFFFFFFFF) ? UInt64(1) : UInt64(0)
            x_lo = (x_lo << 1) | UInt64(1)
            x_hi = (x_hi << 1) | carry
        end
        res[i] = ok ? Int32(1) : Int32(0)
    end
    return nothing
end

function cc_chain_test_gpu(cands::Vector{Int128}, k::Int)::Vector{Int128}
    isempty(cands) && return Int128[]
    d_lo = CuArray(UInt64.(cands .& 0xFFFFFFFFFFFFFFFF))
    d_hi = CuArray(UInt64.((cands .>> 64) .& 0xFFFFFFFFFFFFFFFF))
    d_res = CUDA.fill(Int32(0), length(cands))

    threads = 256
    blocks = cld(length(cands), threads)
    @cuda threads=threads blocks=blocks _cc_gpu_kernel!(d_res, d_lo, d_hi, Int32(k))
    res = Array(d_res)
    return cands[res .== 1]
end

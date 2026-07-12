include("src/cc_gpu.jl"); include("src/cc_gpu_wheel.jl")
using Primes: primes

# 任意 wl で wheel 残基を生成 (build_cc_sieveのCRT展開を汎用化)
function build_wheel_custom(k::Int, wl::Vector{Int})
    cc_len_mod(x,p) = (n=0;y=x; while n<k && y%p!=0; y=(2y+1)%p; n+=1; end; n)
    badof(p) = Set(i for i in 0:p-1 if cc_len_mod(i,p)<k)
    wheel_n = Int[1]; pprod = 1
    for p in wl
        bad = badof(p); tmp = wheel_n; wheel_n = Int[]
        sizehint!(wheel_n, length(tmp)*(p-length(bad)))
        for i in 0:p-1
            @inbounds for x in tmp
                v = x + i*pprod
                (v % p) in bad || push!(wheel_n, v)
            end
        end
        pprod *= p
    end
    return prod(wl), wheel_n
end

function setup_custom(k, wl; max_prime)
    wheel, wheel_n = build_wheel_custom(k, wl)
    @assert wheel < (Int64(1)<<51) "wheel>2^51"
    extra = filter(p-> !(p in wl) && p<=max_prime, primes(max_prime))
    mdl = [(p, Set(i for i in 0:p-1 if (let n=0,y=i; while n<k && y%p!=0; y=(2y+1)%p; n+=1; end; n end)<k)) for p in extra]
    primes32, offsets, badflat = _flatten_bad_gpu(mdl)
    wheel_mod = Int32[Int32(mod(Int64(wheel),Int64(p))) for p in primes32]
    pow20 = Int32[Int32(mod(Int64(1)<<20,Int64(p))) for p in primes32]
    R = Int64(length(wheel_n))
    println("  custom CC$k wl=$wl -> wheel=$wheel R=$R (mem=$(round(R*8/1e9,digits=2))GB) primes=$(length(primes32))")
    return GpuWheelState(k, Int64(wheel), R, CuArray(Int64.(wheel_n)), CuArray(primes32),
        CuArray(wheel_mod), CuArray(pow20), CuArray(offsets), CuArray(badflat),
        Int32(length(primes32)), CUDA.zeros(Int64,1<<12), CUDA.zeros(Int32,1), 1<<12)
end

k=16; base=Int128(10)^18
# 基準(現状 wheel)
st0 = gpu_wheel_setup(k; verbose=false)
gpu_wheel_scan128!(st0, base, base+Int128(st0.wheel)*2)  # warmup
nc=100
t0 = @elapsed gpu_wheel_scan128!(st0, base, base+Int128(st0.wheel)*nc)
r0 = t0/nc/st0.wheel*1e15  # ms相当... 正規化: s per (unit range 1e15)
println("BASE wheel=$(st0.wheel) R=$(st0.R): $(round(t0/nc*1000,digits=2)) ms/cycle, $(round(t0/nc/st0.wheel*1e18,digits=4)) s/1e18range")
CUDA.unsafe_free!(st0.d_out)

# 29追加
wl2 = sort([2,3,5,7,11,13,17,19,23,37,41,43,29])
st1 = setup_custom(k, wl2; max_prime=30000)
gpu_wheel_scan128!(st1, base, base+Int128(st1.wheel))  # warmup
ncy = 8
t1 = @elapsed gpu_wheel_scan128!(st1, base, base+Int128(st1.wheel)*ncy)
println("29追加 wheel=$(st1.wheel) R=$(st1.R): $(round(t1/ncy*1000,digits=2)) ms/cycle, $(round(t1/ncy/st1.wheel*1e18,digits=4)) s/1e18range")
println("単位レンジ速度比 = ×$(round((t1/ncy/st1.wheel)/(t0/nc/st0.wheel),digits=3)) (予測0.448)")

include("src/cc_cpu.jl")
include("src/cc_gpu_wheel.jl")
using CUDA

for k in (13, 15, 16)
    wl = _cc_wl_stream(k)
    wheel, contrib, coff, bases, rp, R = build_wheel_streaming(k, wl)
    dens = R / wheel
    wl0 = _cc_wl(k)
    w0, _, _, _, _, R0 = build_wheel_streaming(k, wl0)
    dens0 = R0 / w0
    println("CC$k stream wl=$wl")
    println("  wheel=$wheel  R=$R  density=$dens  contriblen=$(length(contrib))")
    println("  base   wheel=$w0  R=$R0  density=$dens0")
    println("  density ratio stream/base = $(dens/dens0)  (=> unit-range work factor)")
end

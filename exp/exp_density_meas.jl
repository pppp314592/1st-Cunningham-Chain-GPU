include("src/cc_gpu.jl"); include("src/cc_gpu_wheel.jl")
base = Int128(10)^20
runs = [(12, Int128(2)*10^15), (13, Int128(8)*10^16)]
dens = Dict{Int,Float64}()
for (k,w) in runs
    st = gpu_wheel_setup(k; verbose=false)
    t = @elapsed r = gpu_wheel_scan128!(st, base, base+w)
    d = length(r)/Float64(w)*1e18
    dens[k]=d
    println("CC$k: width=$(Float64(w)) hits=$(length(r)) density=$(round(d,digits=3))/1e18 ($(round(t,digits=1))s)")
    CUDA.unsafe_free!(st.d_out)
end
rr = dens[13]/dens[12]
println("\n実測減衰比 r = d13/d12 = $(round(rr,digits=4))  (理論 ~1/ln(2^13·1e20)≈$(round(1/(log(1e20)+13*log(2)),digits=4)))")
d14=dens[13]*rr; d15=d14*rr; d16=d15*rr
println("外挿 density/1e18 @1e20: CC14≈$(round(d14,digits=4)) CC15≈$(round(d15,digits=5)) CC16≈$(round(d16,digits=6))")
for (k,spr,d) in [(15,1985.0,d15),(16,1107.0,d16)]
    hrs1 = (1/d)*spr/3600
    in1h = d*(3600/spr); in2h = d*(7200/spr)
    println("CC$k: 期待1本≈$(round(hrs1,digits=1))時間 | 1h期待$(round(in1h,digits=3))本 | 2h期待$(round(in2h,digits=3))本")
end

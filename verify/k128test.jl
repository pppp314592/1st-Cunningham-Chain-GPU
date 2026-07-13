using CUDA
function ktest!(out, a::Int128, b::Int128)
  i=threadIdx().x
  n = a*Int128(i) + b
  x_lo = UInt64(n & Int128(0xFFFFFFFFFFFFFFFF))
  x_hi = UInt64((n>>64) & Int128(0xFFFFFFFFFFFFFFFF))
  @inbounds out[i] = (n > b) ? Int64(x_lo % 1000000) + Int64(x_hi)*1000000 : Int64(-1)
  return
end
out=CUDA.zeros(Int64,8)
@cuda threads=8 ktest!(out, Int128(14552571002970), Int128(90616211958465842219))
CUDA.synchronize()
println("Int128 kernel OK: ", Array(out))

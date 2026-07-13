using CUDA
println("CUDA.functional() = ", CUDA.functional())
try
    println("device = ", CUDA.name(CUDA.device()))
catch e
    println("no device: ", e)
end
a = CUDA.rand(Float32, 1_000_000)
CUDA.@sync a .+= 1f0
println("array type = ", typeof(a))
println("kernel round-trip OK, sum=", sum(Array(a[1:5])))

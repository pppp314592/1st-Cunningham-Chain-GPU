include("../src/cc_gpu.jl")
using Primes

n = Int128(90616211958465842219)
println("Testing CC15 candidate: $n")

x_lo = UInt64(n & 0xFFFFFFFFFFFFFFFF)
x_hi = UInt64((n >> 64) & 0xFFFFFFFFFFFFFFFF)

println("\nChain terms (Miller-Rabin):")
for i in 1:15
    global x_lo, x_hi
    is_prime = is_prime_mr128(x_lo, x_hi)
    println("  term $i: $(Int128(x_hi) << 64 | Int128(x_lo)) prime=$is_prime")
    i == 15 && break
    carry = (x_lo > 0x7FFFFFFFFFFFFFFF) ? UInt64(1) : UInt64(0)
    x_lo = (x_lo << 1) | UInt64(1)
    x_hi = (x_hi << 1) | carry
end

x_lo = UInt64(n & 0xFFFFFFFFFFFFFFFF)
x_hi = UInt64((n >> 64) & 0xFFFFFFFFFFFFFFFF)
cnt = _cc_count128(x_lo, x_hi, 15)
println("\n_cc_count128 result: chain length = $cnt")
println("  => $(cnt >= 15 ? "CC15 CONFIRMED ✁E : "NOT CC15 ✁E)")

println("\nVerification with Primes.jl:")
x = n
for i in 1:15
    global x
    is_p = isprime(x)
    println("  term $i: $x prime=$is_p")
    i == 15 && break
    x = 2x + 1
end

include("cc_gpu.jl")
using Printf

println("=" ^ 70)
println("バッチパフォーマンス測定")
println("=" ^ 70)

# Int64 版 (従来): 1 × 10^16 の範囲
lo64 = 3_300_000_000_000_000_000
hi64 = lo64 + 10^16
println("\n--- Int64 版: 1バッチ 10^16 [$lo64, $hi64) ---")
t = @elapsed res = search_cc_gpu(lo64, hi64, 15, verbose=false)
@printf "  時間: %.1fs, 発見: %d件\n" t length(res)

# Int128 版: 同規模 (10^16 相当のサイクル数)
wheel, _, _, _, _, _, _, _ = _get_cached_sieve(15)
cyc_per_batch = 10^16 ÷ wheel  # 約 68
step128 = Int128(wheel) * cyc_per_batch * 2  # 2倍のサイクル数で
lo128 = Int128(10)^19
hi128 = lo128 + step128
ncycles128 = (hi128 - 1) ÷ wheel - lo128 ÷ wheel + 1
println("\n--- Int128 版: $(ncycles128)サイクル (約 $(step128÷10^16)×10^16相当) [$lo128, $hi128) ---")
t128 = @elapsed res128 = search_cc_gpu128(lo128, hi128, 15, verbose=false)
@printf "  時間: %.1fs, 発見: %d件\n" t128 length(res128)

println("\n" ^ 2)
println("=" ^ 70)
println("比較")
println("=" ^ 70)
@printf "  Int64 版:  %.1fs  (10^16, 約%dサイクル)\n" t cyc_per_batch
@printf "  Int128 版: %.1fs  (%dサイクル)\n" t128 ncycles128
@printf "  1サイクルあたり: Int64=%.3fs, Int128=%.3fs\n" (t/cyc_per_batch) (t128/ncycles128)

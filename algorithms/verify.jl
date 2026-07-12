# algorithms/verify.jl
# 全モジュールの自己検証を実行し、密度レポートを表示する。
# 起動:  julia -g0 algorithms/verify.jl

println("=== A2 悪残基 等価性 ===")
include("A2_bad_residues.jl")
ok2, d2 = selftest_bad_residues()
println("  A2 self-test: ", ok2 ? "PASS" : "FAIL $d2")

println("=== A3 segmented sieve ===")
include("A3_segmented_sieve.jl")
ok3, d3 = selftest_segmented()
println("  A3 self-test (CC6 in [2,5000]): ", ok3 ? "PASS ($(length(d3[1])) chains)" : "FAIL $d3")
println("    found: ", d3[1])

println("=== A4 wheel CRT walk ===")
include("A4_wheel_crt.jl")
ok4, d4 = selftest_wheel_crt()
println("  A4 self-test: ", ok4 ? "PASS ($(d4[1]) residues)" : "FAIL $d4")

println("=== A5 primorial band ===")
include("A5_primorial_band.jl")
ok5, d5 = selftest_band()
println("  A5 self-test (CC6 band [10,200000]): ", ok5 ? "PASS ($(length(d5[1])) chains)" : "FAIL $d5")
println("    found: ", d5[1])

println("=== A6 density / singular series ===")
include("A6_density.jl")
ok6, d6 = selftest_density()
println("  A6 self-test (W2 ≈ SG const): ", ok6 ? "PASS (W2=$(round(d6[1],digits=6)))" : "FAIL $d6")

println("=== A8 BPSW primality ===")
include("A8_bpsw.jl")
ok8, d8 = selftest_bpsw()
println("  A8 self-test (vs Primes.isprime): ", ok8 ? "PASS" : "FAIL $d8")

println("=== A13 Montgomery mulmod ===")
include("A13_montgomery.jl")
ok13, d13 = selftest_montgomery()
println("  A13 self-test (vs mod(a*b,N)): ", ok13 ? "PASS" : "FAIL $d13")

println("=== A15 two-stage wheel ===")
include("A15_two_stage_wheel.jl")
ok15, d15 = selftest_two_stage()
println("  A15 self-test (== A4 DFS): ", ok15 ? "PASS ($(d15[1]) residues)" : "FAIL $d15")

println("=== A16 known chains oracle ===")
include("A16_known_chains.jl")
ok16, d16 = selftest_known_chains()
println("  A16 self-test (hardcoded chains): ", ok16 ? "PASS" : "FAIL $d16")
ok16b, d16b = selftest_minimal_heads()
println("  A16 self-test (OEIS minimal heads len11..16): ", ok16b ? "PASS" : "FAIL $d16b")

println("=== A17 second-kind chains ===")
include("A17_second_kind.jl")
ok17, d17 = selftest_band2()
println("  A17 self-test (2nd-kind band): ", ok17 ? "PASS ($(length(d17[1])) chains)" : "FAIL $d17")

println("=== A20 wheel residue count ===")
include("A20_wheel_count.jl")
ok20, d20 = selftest_wheel_count()
println("  A20 self-test (== A4 count): ", ok20 ? "PASS ($(d20[1]) residues)" : "FAIL $d20")

println()
println("=== 密度レポート ===")
report_density()

allok = ok2 && ok3 && ok4 && ok5 && ok6 && ok8 && ok13 && ok15 && ok16 && ok16b && ok17 && ok20
println()
println(allok ? "ALL SELF-TESTS PASS" : "SOME SELF-TESTS FAILED")

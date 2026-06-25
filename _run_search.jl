include("cc_gpu.jl")

TARGET_CC = 15

# CC15 の篩情報
wheel, wheel_n, mdllist = build_cc_sieve(TARGET_CC)
println("wheel = $wheel")
println("residues = $(length(wheel_n))")
println("efficiency = $(wheel ÷ length(wheel_n))")

# 1バッチのテスト (10^16 〜 10^16 + 10^15 で簡易確認)
lo = 10^18
hi = lo + 10^15
println("\n1バッチテスト: [$lo, $hi)")
t = @elapsed begin
    res = search_cc_gpu(lo, hi, TARGET_CC, verbose=true)
end
println("結果: $(length(res)) 件, 所要時間: $(round(t, digits=1))s")
for r in res
    println("  CC$TARGET_CC: $r")
end

# 推定
STEP = 10^16
HI_MAX = 9_220_000_000_000_000_000
n_batches = (HI_MAX - 10^18) ÷ STEP
println("\n推定: 1バッチ約 $(round(t, digits=1))s × $(n_batches)バッチ = $(round(t * n_batches / 3600, digits=1)) 時間")

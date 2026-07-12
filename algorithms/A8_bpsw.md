# A8 — BPSW 決定的素数判定（巨大数の本判定に必須）

## なぜ必要か

A3/A5 の最終判定に `Primes.isprime`（BigInt）を使ったが、生産コードや GPU 実装
では自前の判定が要る。単なる Miller–Rabin は確率的（偽陽性あり）だが、
**BPSW テスト**（強 MR + 強 Lucas、Selfridge の D 選択）は、64bit 以下で
決定的であり、それ以上でも**既知の反例が存在しない**（実用上決定的）。

巨大数 CC 探索では鎖の各項を確実に素数と判定したいので、BPSW が標準。

## 構成

1. **強 Miller–Rabin（基底 2）**: `n-1 = d·2^s` として `2^d` からの二乗列。
2. **強 Lucas（Selfridge 法）**:
   * `Jacobi(D, n) = -1` となる最初の `D ∈ {5, -7, 9, -11, 13, …}` を選ぶ。
   * `P = 1, Q = (1 - D)/4`。`δ = n + 1 = d'·2^{s'}`（d' 奇数）。
   *  Lucas 列 `U_m, V_m`（パラメタ P, Q）を倍加・加算で計算し、
     `U_{d'} ≡ 0 (mod n)` またはある `V_{d'·2^r} ≡ 0 (mod n)` なら素数。

## 巨大数での使い方

* 候補 `m·2^i − 1` が 64bit 以下なら `src/cc_cpu.jl` の 7 基底 MR で決定的。
* それ以上は BPSW。GPU 化するなら Montgomery 乗算で `mulmod/powermod` を
  置換し、Lucas の倍加・加算も limb 演算で実装する（A7 参照）。

## 検証

`selftest_bpsw()` で `Primes.isprime` と一致することを確認（小範囲全数 ＋
巨大乱数標本 ＋ 平方数合成）。`algorithms/verify.jl` に組み込み可。

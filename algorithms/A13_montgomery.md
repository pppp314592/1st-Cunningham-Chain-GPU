# A13 — 多倍長 Montgomery 乗算（GPU/多倍長 MR の基礎）

## なぜ必要か

A8 の BPSW は `mod`/`powermod` を BigInt 剰余でやっている。巨大数ではこれが
遅い。Montgomery 乗算は「剰余を掛け算の中に吸収」し、多倍長でも定数倍速い。
GPU 上の多倍長 MR（A7）はこれを limb 演算で実装する。ここでは BigInt 版を
参照実装し、通常の `mod(a*b, N)` と一致することを検証する。

## アルゴリズム

奇数 N に対し `R = 2^s > N`（s = N のビット長）。
* `Mont(a) = a·R mod N`（Montgomery 形式）。
* `REDC(T)`: `m = (T mod R)·N⁻¹ mod R`, `t = (T + m·N)/R`; `t ≥ N` なら `−N`。
  入力が `T < N·R` なら `t ∈ [0, 2N)` になる。
* `MontMul(x,y) = REDC(x·y)` は `x·y·R⁻¹ mod N`（x,y が Mont 形式）。
* `a·b mod N = Mont⁻¹(MontMul(Mont(a), Mont(b)))`。

## 巨大数での使い方

* 候補 `m·2^i − 1` が 64bit 以下なら `src/cc_cpu.jl` の 7 基底 MR で決定的。
* それ以上は BPSW。GPU 化するなら Montgomery 乗算で `mulmod/powermod` を
  置換し、Lucas の倍加・加算も limb 演算で実装する（A7 参照）。

## 検証

`selftest_montgomery()` で `mod(a*b, N)` と一致することを確認。
`algorithms/verify.jl` に組み込み可。

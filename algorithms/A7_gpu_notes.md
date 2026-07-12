# A7 — GPU / 多倍長化の指針（巨大数向け）

`src/cc_gpu_wheel.jl` は 64bit ホイール篩＋鎖 MR を GPU で融合しているが、
128bit まで・wheel 列挙型という限界がある。巨大数領域へ持ち上げる方針。

## 1. 篩の多倍長化（host 側で準備、device で消費）

* n の base が BigInt でも、A3 のインクリメンタル更新により各ブロック内は
  `Int64` 演算で済む。この `bmod` 配列と悪残基テーブルをデバイスに転送し、
  カーネル内で `np = (cp*wheelmod + rp) % p` を計算する（`cc_gpu_wheel.jl`
  の 32bit 分解と同じ考え方）。p は篩上限 P までなので `Int32` に収まる。
* 巨大数では wheel 自体を A4 の DFS で生成し、ホストで `d_wheel_n` 相当を
  ブロックごとに供給する。1 ブロック = 1 つの wheel 残基 × 複数 cyc。

## 2. 鎖 MR の多倍長化（device 側）

* 鎖の項 `m·2^i − 1` は i が大きいと 128bit を超える。GPU 上で
  多倍長 Montgomery 乗算（例: 256/512bit を 32bit limb で）を実装する。
* 基底は BPSW（強 MR 基底 2 ＋ 強 Lucas）が実用上決定的。device 側に
  Lucas テストを入れるか、MR を 3 〜 7 基底（巨大数でも基底ごとに決定的な
  上限付き）で代用する。
* 消費電力・時間のトレードオフ：高 CC では「篩が支配的、MR はほぼ発生しない」
  （STATUS の分析）なので、篩カーネルのスループット最大化が最優先。
  MR は滅多に発火しないため、CPU に回してもボトルネックにならない場合が多い。

## 3. 64bit `n % p` が遅い問題への対処

GTX 1660 Ti 等のコンシューマ GPU は 64bit 整数除算が遅い（STATUS 確認済）。
* 篩では 32bit 分解（`r = r_hi·2^20 + r_lo`, Barrett 還元）を維持。
* `p` が大きく 32bit に収まらない P（> 2^31）を使う場合は、64bit 除算を
  避けるため p を 32bit に収まる上限に据え置き、追加篩は host/wheel 側で
  吸収する設計が現実的。

## 4. 推奨構成（巨大数・記録狙い）

```
host:  A4 DFS で wheel 残基生成 → バンド [mlo,mhi] を決定
       （A6 で E ≳ 1 になる x, w を選ぶ）
GPU :  各ブロック base（Int64/BigInt から mod p は host で事前計算）
       → 篩カーネルで候補絞り → 生存者を多倍長 BPSW で本判定
```

## 5. 現状コードからの差分（TODO）

* `cc_gpu_wheel.jl` の `_cc_wheel_kernel!` を多倍長 MR カーネルに差し替え。
* `gpu_wheel_setup` の wheel 構築を A4 DFS に変更（materialize 廃止）。
* 篩上限 P を GPU 向けに 32bit 内（≤ 2^31）に固定し、それ以上は host 篩。
* `is_prime_mr128` の上位を拡張する limb 演算 MR/BPSW を `prime_filter.jl` に追加。

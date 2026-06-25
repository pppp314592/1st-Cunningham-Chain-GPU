# 1st Cunningham Chain GPU

GPU (CUDA.jl) を用いて第1種カニンガム鎖 (Cunningham Chain) を高速探索する Julia プログラム。

## 概要

カニンガム鎖とは `n, 2n+1, 4n+3, 8n+7, ...` と全て素数が連続する数列。このプログラムは CPU 篩で候補を約 1/5000 に絞り込み、GPU 上で Miller-Rabin 素数判定を並列実行して高速に探索する。

### アーキテクチャ

1. **Phase 1 (CPU):** Wheel 篩 [2,3,5,7,11,13] + modulo 篩で候補削減（マルチスレッド）
2. **Phase 2 (GPU):** 全候補のチェインを `CuArray.map` で並列判定
3. **128-bit 対応:** `UInt64` 2個のソフトエミュレーションで `2^128` を超える範囲の素数判定に対応

## ファイル構成

```
src/
  prime_filter.jl   — GPU Miller-Rabin 素数判定 (Int64 / 128-bit)
  cc_gpu.jl         — カニンガム鎖探索エンジン（篩 + GPU 判定）
scripts/
  search_cc15.jl           — CC15 探索 (Int64 範囲)
  scan_cc12plus.jl         — CC12+ 一斉スキャン (10〜10^17)
  cc15_128.jl              — CC15 探索 (Int128 範囲)
  cc15_128_10-10^20.jl     — CC15 探索 (10〜10^20, 引数で開始位置指定)
  _batch_runner.jl         — 中断再開対応バッチ実行
tests/
  test_primes.jl           — 素数判定テスト（小規模）
  test_large.jl            — 素数判定テスト（大規模 10^7〜10^8）
  test_cc_gpu.jl           — GPU 探索テスト (CC5/12/13/14)
  _test_128.jl             — Int128 版テスト
bench/
  bench_cc13.jl            — GPU vs CPU ベンチマーク
notebook/
  Cunningham Chain records -revenge- ver2.ipynb — CPU 版リファレンス実装
```

## 使い方

```julia
# GPU 素数判定
include("src/prime_filter.jl")
filter_primes(collect(10^7:10^8)) |> length

# カニンガム鎖探索（CC5, 1〜10^6）
include("src/cc_gpu.jl")
search_cc_gpu(1, 10^6, 5)
```

## 実績

| 鎖長 | 値 | 状態 |
|------|-----|------|
| CC12 | 554688278429 | 確認済み |
| CC13 | 4090932431513069 | 確認済み |
| CC14 | 95405042230542329 | 確認済み |
| CC15 | 90616211958465842219 (候補) | 全15項素数確認済み |

## システム要件

- Julia 1.9+
- CUDA.jl（NVIDIA GPU）
- Primes.jl（検証用）

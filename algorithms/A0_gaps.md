# A0 — 既存 src との対応と優先 TODO

このフォルダのアルゴリズムを、別セッションの `src/*.jl` にどう組み込むか。

## 現状コードの位置づけ（src/）

| ファイル | やっていること | このフォルダの対応 | 限界 |
|----------|----------------|--------------------|------|
| `cc_block.jl` | ブロック・ビット篩＋128bit MR | A3 の Int128 版 | Int128 まで |
| `cc_cpu.jl` | マルチスレッド wheel 篩＋MR | A2/A3 + A8(64/128bit) | wheel 列挙型 |
| `cc_gpu.jl` | GPU 逐次 MR（篩は弱い） | A3 の篩を GPU 化すべき | 篩がボトルネック |
| `cc_gpu_wheel.jl` | 融合 GPU ホイール篩＋鎖 MR | A3+A8 の GPU 版 | 64bit・wheel 列挙 |
| `prime_filter.jl` | デバイス MR(64/128bit) | A8 の多倍長化 | 多倍長未対応 |

## 優先 TODO（影響大 → 小）

1. **多倍長 MR（BPSW）の実装** — `A8_bpsw.jl` を `prime_filter.jl` / `cc_cpu.jl`
   に取り込む。これがないと 128bit 超の CC が証明できない（最重要）。
2. **インクリメンタル segmented sieve の共通化** — `A3_segmented_sieve.jl` の
   `build_badtable`+ブロック走査を `cc_block.jl` の `block_sieve!` と統合。
   BigInt base ＋ インクリメンタル `bmod` で任意範囲に対応。
3. **wheel の DFS 化** — `cc_gpu.jl` の `build_cc_sieve`（列挙型）を
   `A4_wheel_crt.jl` の DFS に置換。CC16+ で >GB になる wheel を解消。
4. **プリモリアル・バンド探索のドライバ** — `A5_primorial_band.jl` を本番用に
   （チェックポイント A9、分散 A9、確率的早落ち A11-5 を加味）。
5. **GPU 多倍長化** — A7 の指針に従い、`_cc_wheel_kernel!` を多倍長 MR に。
   `n % p` は 32bit 分解を維持し、P ≤ 2^31 に固定。
6. **密度に基づく範囲選択** — `A6_density.jl` で `E ≳ 1` になる x, w を選び、
   探索対象バンドを決定（「何も見つからない」を避ける）。

## 検証フロー（推奨）

```
julia -g0 algorithms/verify.jl        # 全自己検証 + 密度レポート
# 個別: include("algorithms/A3_segmented_sieve.jl"); segmented_sieve_cc(2,5000,6)
```

既存 `src` のテスト（`tests/`）ともクロスチェックし、cc_cpu / cc_gpu の
結果と `A3/A5` の結果が一致することを確認すること。

# 第一種カニンガム鎖 — 巨大数領域向けアルゴリズム法

このフォルダは、別セッションで書かれている Julia 探索コード（`src/cc_*.jl`）
に対する**アルゴリズム面の補完**である。目標は「巨大数領域（10^20 以上、
128bit を超える桁数）で、第一種カニンガム鎖（CC）をどう探すか」の方法論を
整理し、検証可能な参照実装（`.jl`）を置くこと。

> 第一種カニンガム鎖とは、p, 2p+1, 4p+3, 8p+7, … と続く素数の列。
> 長さ k の鎖の先頭 p とは、k 個の数
> `f_i(n) = 2^i n + (2^i - 1)  (i = 0..k-1, n = p)` が全て素数なこと。

## 既存コードとの関係と限界（`src/` の現状）

| 実装 | 手法 | 限界 |
|------|------|------|
| `cc_block.jl` | ブロック・ビット篩 + 128bit MR | Int128 まで（CC20 超は不可） |
| `cc_cpu.jl` | マルチスレッド wheel 篩 | wheel 列挙型なので CC16+ で >GB |
| `cc_gpu.jl` / `cc_gpu_wheel.jl` | GPU 融合ホイール篩 | 64bit `n%p` が GTX1660Ti で遅い |
| `prime_filter.jl` | デバイス MR（64/128bit） | 多倍長未対応 |

`logs/STATUS.md` の結論どおり、**高 CC では篩が支配的**で、かつ
**CC15 を 10^20 まで走らせても 0 件**（密度的に見つかる見込みなし）という
現実がある。したがって巨大数領域では「幅広い区間を総なめ」は不可能であり、
以下のアルゴリズム的転換が必要となる。

## このフォルダの構成

| ファイル | 内容 | 巨大数で効く理由 |
|----------|------|------------------|
| `A0_gaps.md` | 既存 `src/` との対応 ＋ 優先 TODO | どこから手を付けるか |
| `A1_reformulation.md` | **m = n+1 変形**：鎖 = `m·2^i − 1` (i=0..k-1) が全て素数 | 悪残基が「2 の逆べき」のみになり構造が透明化 |
| `A2_bad_residues.jl` | 悪残基の 2 通りの計算（n 形式 / m 形式）＋ 自己検証 | 篩の正しさの基盤 |
| `A3_segmented_sieve.jl` | **多倍長・インクリメンタル segmented sieve** ＋ 自己検証 | 任意の start にスケール；ブロック間で `base mod p` を更新し BigInt% を 1 回だけに |
| `A4_wheel_crt.jl` | **再帰 CRT ウォーク**（wheel を列挙せずに生存残基を生成）＋ 自己検証 | wheel の >GB 爆発を回避（O(素数数) メモリで列挙） |
| `A5_primorial_band.jl` | **プリモリアル・バンド探索**（記録狙いの実用ドライバ）＋ 自己検証 | 巨大数で唯一現実的な手法 |
| `A5_constructive_demo.jl` | A5 の実践: 既知 CC15/CC16 最小頭を構成的に再発見（Int64超も BigInt で直接） | `julia -g0 -t 12 algorithms/A5_constructive_demo.jl` |
| `A6_density.jl` | **H-W 重み（特異級数）と期待個数**の計算 ＋ 検証 | 「どこを・どれだけ探せば見つかるか」を定量化 |
| `A7_gpu_notes.md` | GPU / 多倍長化の指針 | `cc_gpu_wheel.jl` の拡張案 |
| `A8_bpsw.jl` + `A8_bpsw.md` | **BPSW 決定的素数判定**（BigInt）＋ 自己検証 | 巨大項を確定的に証明 |
| `A9_distributed.md` | 分散処理・チェックポイント戦略 | 候補の r/t ブロック分割で並列 |
| `A10_covering.md` | ord_p(2) による被覆と篩優先順位 | 小位数素数ほど篩が強い |
| `A11_search_angles.md` | 別角度（末尾項・SG 種・早落ち） | 実装により MR コスト削減 |
| `A12_records.md` | 記録・文脈（PrimeGrid/NewPGen）｜ 標準手法との対応 |
| `A13_montgomery.jl` + `.md` | **多倍長 Montgomery 乗算**（BigInt）＋ 自己検証 | GPU/多倍長 MR の基礎 |
| `A14_bucket_sieve.md` | バケツ篩・ワード圧縮でキャッシュを効かせる | P を大きくしても篩が速い |
| `A15_two_stage_wheel.jl` | **2 段 wheel**（小列挙＋大 DFS）＋ 自己検証 | CC16+ で DFS 分岐爆発を抑制 |
| `A16_known_chains.jl` | 既知鎖のテストオラクル（直接 MR）＋ 自己検証 | A3/A5 の独立検証用 |
| `A17_second_kind.jl` + `.md` | **第二種鎖**へ悪残基符号反転で転用 ＋ 自己検証 | 第一種と同じ枠組みで両対応 |
| `A18_cache_blocking.md` | キャッシュブロッキング・B 選択・並列割当 | 篩スループット最大化 |
| `A19_pipeline.md` | 篩と MR を 2 パス分離（頑健パイプライン） | クラッシュ復旧・実装差し替え |

## 巨大数で効く核心の 3 つ

1. **m = n+1 変形**（A1）で悪残基を `{2^{-i} mod q}` とし、各素数 q が
   `ord_q(2)` 周期で高々 1 個の i しか殺せないことを利用する。
2. **インクリメンタル segmented sieve**（A3）で、ブロック base が BigInt でも
   1 ブロック内の篩は全て Int64 演算。ブロック間の `base mod p` を足し算で更新し
   BigInt 剰余を素数ごとに 1 回だけにする。
3. **再帰 CRT ウォーク**（A4）で wheel 生存残基をストリーミング生成し、
   巨大な wheel（`prod(素数)` が 2^51 超）でも O(w) メモリで全候補を尽くす。
   これを A5 のバンド探索に組み込むのが、記録級 CC を探す標準的手段。

## 検証ステータス（verify.jl の実行結果）

`verify.jl` を `julia -g0 algorithms/verify.jl` で実行すると全モジュールの
自己検証が走る。本リポジトリ作成時点の結果：

```
=== A2 悪残基 等価性 ===  PASS
=== A3 segmented sieve === PASS (CC6 in [2,5000] -> 1 chain: [89])
=== A4 wheel CRT walk === PASS (140 residues)
=== A5 primorial band === PASS (CC6 band [10,200000] -> 3 chains: [89,63419,127139])
=== A6 density === PASS (W2=1.320324 ≈ Sophie Germain 定数)
=== A8 BPSW === PASS (vs Primes.isprime, 小範囲全数 + 巨大乱数 + 平方数)
=== A13 Montgomery === PASS (vs mod(a*b,N))
=== A15 two-stage wheel === PASS (1540 residues, == A4 DFS)
=== A16 known chains === PASS (hardcoded chains verified)
=== A17 second-kind === PASS (2nd-kind band matches brute force)
ALL SELF-TESTS PASS

W_k テーブル: W_2=1.32, W_6=71.96, W_10=3393, W_12=2.82e4, W_15=3.15e6, W_16=1.38e7
E(CC10 in [10,1e15])   ≈ 8.3e+02
E(CC12 in [10,1e18])   ≈ 5.4e+02
E(CC15 in [10,1e20])   ≈ 1.2e+01   (修正後・単調)
```

注: 旧 `logs/cc15_*` の「0 件」は走査の途中経過だが、下記A6修正後の見込み
（CC15 @1e17 ≈ 0.11 個）とも整合する（1e17 では元々ほぼ期待できない）。

ユーザーが別セッションで何かを検証した場合は、この下へ結果を追記してよい。

> **[GPUセッションより] 検証結果は `algorithms/VERIFY_FROM_GPU_SESSION.md` に記録済み。**
> 要点: (1) verify.jl 全PASS を再現、(2) A3↔GPU 相互検証で CC6/CC8/CC12 が値まで一致、
> (3) A6 の `expected_count` が k≥14 で過大・非単調（要修正）。
>
> **[アルゴリズムセッション → 修正済 2026-07-12]** A6 の期待個数を修正。真因は
> 特異級数の裾打ち切り（∑_{p>1e6}1/p² は無視可能）ではなく、**積分の被積分関数
> `1/(log t)^k` が小 t で発散**していたこと（t=10,k=15 で W/(log t)^k≈11.6>1 の
> 非物理な「確率」）。鎖要素 `f_i(n)≈2^i·n` の対数は `log t + i·log2` と増大するので、
> 密度を `W_k / ∏_{i=0}^{k-1}(log t + i·log2)` に修正。結果、**単調かつ GPU実測と一致**:
> CC12=102.5(実105), CC13=11.1(実8), CC14=1.14(実1), CC15=0.11(実0), CC16=0.010(実0)。
> 1e20 での見込み: CC15≈11.7, CC16≈0.93, CC17≈0.06（旧「CC15≈77」は撤回）。
>
> **[構成的探索デモ 2026-07-12]** `A5_constructive_demo.jl` で A5 の実効性を実証。
> 較正済み A6 で選んだ 1e20 級バンド上で、**総当たりではなく wheel 生存残基 r のみ**
> （m = r + t·M・BigInt）から既知最小頭を再発見:
> - CC15 最小頭 `90616211958465842219`（20桁）→ 1.35s で検出
> - CC16 最小頭 `810433818265726529159`（21桁・Int64超）→ 13.1s で検出
> GPUセッション §5「CC15+ は多倍長頭必須（A3/A5 路線）」の独立裏付けにもなった。


## 使い方

```julia
include("algorithms/verify.jl")          # 全自己検証を実行
include("algorithms/A3_segmented_sieve.jl")
include("algorithms/A5_primorial_band.jl")
r = segmented_sieve_cc(2, 5000, 6)       # CC6 の先頭 p を [2,5000] から
r2 = primorial_band_search(6; w=6, P=1000, mlo=10, mhi=30030*5)
```

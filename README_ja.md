# buqi-minimax-h3-multigpu

[English](README.md) | [中文](README_zh.md) | **日本語**

ComfyUI 用 **MiniMax-H3** マルチ GPU(シーケンス並列)推論ノード — `UNETLoader` の
差し替えだけで使え、**近似も精度低下も一切ありません**。

MiniMax-H3 はパック トークン DiT で**動画と音声を同時生成**します。本ノードは
[DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509) 方式でそのシーケンスを
2/4/7/8 GPU に分割します。各 GPU は自分が受け持ったヘッドについて**完全で厳密な
アテンション**を計算するため、数学的に等価です。

### 精度について

各 GPU は完全なシーケンス上で厳密なアテンションを計算し、近似・追加量子化・マスク・
キャッシュは一切行いません。したがって結果はシングル GPU と**数学的に等価**ですが、
**ビット単位で同一ではありません**。各ランクが扱うヘッドは 56/`world_size` 個であり、
アテンション カーネルはヘッド数に応じて縮約順序を選ぶため、浮動小数点の丸めが変わり
ます。DiT の出力で実測した偏差はシングル GPU に対し**相対 4.2e-6 以下**(音声は 0)で、
bfloat16 の分解能(7.8e-3)より 3 桁小さく、系統的な偏りもありません。

`tests/latent_parity.py` でご自身で検証できます。エンコード後の動画のハッシュではなく、
DiT が実際に出力する速度場テンソルを比較します(H.264 は不可逆なので、この規模の差は
隠すことも誇張することもあります)。

コミュニティで最も多い構成である **2 GPU** を最優先に設計(1〜8 GPU 対応)。

## 動作要件

- Linux(Windows は WSL2 必須 — ネイティブ Windows に NCCL はありません)
- ComfyUI **>= 0.30.0**(`comfy.ldm.minimax.model` を含むバージョン)。古いビルドでは
  読み込み時に明確なエラーを出します
- MiniMax-H3 のモデルファイル(DiT、Qwen3-VL テキストエンコーダ、動画/音声 VAE)を
  通常どおり `diffusion_models/`、`text_encoders/`、`vae/` に配置してください。
  モデルは MiniMax-H3 公式から入手してください(本リポジトリはコードのみ)
- NCCL 入り PyTorch(ComfyUI 公式の wheels に含まれます)
- `world_size` は 56 のアテンションヘッドを割り切れる必要があります:**1, 2, 4, 7, 8**

## インストール

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/buqi-code/buqi-minimax-h3-multigpu.git
```

追加の Python 依存はありません。

## クイックスタート(2 GPU)

1. 両方の GPU を可視にして ComfyUI を起動:

   ```bash
   python main.py --cuda-device 0,1 --highvram
   ```

2. ワークフローの `UNETLoader` を **MiniMax H3 Multi-GPU Loader (Ulysses SP)** に
   置き換えます。`world_size=2` がデフォルト、`devices=auto` のままで OK。

3. キューに投入。初回は追加 GPU ごとにワーカープロセスを起動し、それぞれに DiT を
   読み込みます(fp8 で約 21 GB/枚)。2回目以降は再利用されます。

API 形式のサンプルは [`examples/workflow_api_2gpu.json`](examples/workflow_api_2gpu.json)。
UI にドラッグ&ドロップでそのまま読み込めるワークフローは
[`examples/workflow_ui_2gpu.json`](examples/workflow_ui_2gpu.json)。

## ノード入力

| 入力 | デフォルト | 説明 |
|---|---|---|
| `unet_name` | — | H3 DiT のチェックポイント。`UNETLoader` と同じ |
| `weight_dtype` | `default` | `UNETLoader` と同じ。fp8 推奨 |
| `world_size` | `2` | 分割する GPU 数(1/2/4/7/8)。`1` は通常のシングル GPU 読み込み |
| `devices` | `auto` | 物理 CUDA ID(例: `"0,1"`)。最初の ID は ComfyUI 自体が使う GPU である必要があります。`auto` は環境変数 `MINIMAX_SP_DEVICES` があればそれを、なければ先頭 `world_size` 枚を使用 |

`world_size=1` の場合はそのままパススルーするので、シングル GPU 環境でもこのノードを
含むワークフローがそのまま動きます。

## VRAM と解像度の目安

シーケンス並列は各 GPU に DiT 全体を複製します(重みは分割しない)ので、
1枚あたりの VRAM はシングル GPU と同じです。メリットは速度です。

| 重み | 1枚あたり DiT | 実用的な GPU |
|---|---|---|
| fp8(推奨) | 約 21 GB | 24 GB カードで 720p まで、480p は余裕 |
| bf16(bf16 チェックポイント + `default`) | 約 40 GB | 48 GB 以上 |

| 区間 | 1 GPU | SP2 | 高速化 |
|---|---|---|---|
| デノイズ ループ | 78.3s | 41.0s | **1.91×** |
| VAE デコード | 5.9s | 3.5s | **1.69×** |
| エンドツーエンド | 84.9s | 46.2s | **1.84×** |

測定環境:2× RTX PRO 5000 Blackwell(72GB、PCIe 5.0、**NVLink なし**)、fp8、
832×480、20 ステップ。

オペレータ単位の内訳(`MINIMAX_SP_PROFILE_OPS=1`、1 ステップ / 1 ランク):
アテンション+出力交換 856ms(重なるため合算)、MLP 657ms、gather+qkv_proj 325ms、
out_proj 83ms、変調+norm 76ms、QK-norm+RoPE 27ms。本体の GEMM はすでに 270〜370 TFLOPS(fp8)、
アテンションは約 176 TFLOPS で動いており、**演算そのものに伸ばす余地はありません**。
GPU 数で割れない唯一のコストは GPU 間通信で、当初はステップの約 21%(481ms)でしたが、
現在はほぼ演算の裏に隠れています。

VAE の行は `MiniMaxH3SPVAEDecode`(後述)が必要で、上限は時間チャンクの分割で決まり
ます。この例は 7 チャンクなので、2 GPU の 4/3 分割では 1.75× が上限です。セッション
最初のデコードでは VAE 重みの受け渡しに約 2.3s が別途かかります。

1 GPU のまま残るのはテキスト エンコーダ(短いプロンプトで約 1s)と動画の多重化です。
長尺・高解像度ほどよくスケールします。デノイズが伸びる一方で後段は一定で、VAE も
分配できるチャンクが増えるためです。

より大きな解像度(1080p 以上)では、アクティベーションが分割される効果も加わります。
1 GPU ではオフロードが始まるのに SP2 なら常駐できる、という状況です。こうした条件で
超線形に見えるのはそのためで、シーケンス並列が自らの上限を超えたわけではありません。

## 解像度の制約

MiniMax-H3 標準のルールに従います。キャンバスは 32px 単位に丸められ、
latent(px/16)はパッチ(2)で割り切れる必要があるため、標準グリッドを使って
ください — 832×480、1280×736、1920×1088 など。`height=720` は**不可**
(latent 高 45 が奇数)。736 を使ってください。

## マルチ GPU VAE デコード(オプション)

`VAEDecode` を **MiniMax H3 Multi-GPU VAE Decode**(`MiniMaxH3SPVAEDecode`)に
差し替えると、入出力はそのままで、動画 VAE の時間チャンクが SP グループの GPU に
分散されます。

`decode_temporal` は時間チャンクを 1 つずつデコードし、各チャンクは潜在表現の一部
しか読まないため互いに独立です。分散するのはその末端の計算だけで、空間タイルの
ブレンド、時間ブレンド、キャンバス書き込みはすべて rank 0 の公式コード経路に残り、
完成したチャンクを受け取ります。結果はビット単位で同一です(検証は下記)。

worker の VAE 重みは初回使用時に rank 0 から NCCL で受け渡されます。ファイルを自分で
探さないので、実際に読み込んだチェックポイント(独自パスを含む)が必ず使われます。
転送量は約 4.85 GiB、セッションごとに一度、約 2.3s です。

この機能はオプトインです。無料ではなく、各 worker が DiT に加えて動画 VAE(約 5GB)
を保持するため fp8 で 1 枚あたり約 26GB になります。**24GB カードでは標準の
`VAEDecode` を使ってください。** SP グループが無い、`world_size` が 1、H3 動画 VAE
ではない、チャンク数が足りない場合は、ログを出して標準デコードにフォールバックします。

## 正しさの検証

2 つのテストはいずれも**テンソル**を比較します。これが唯一意味のある水準です。DiT:

```bash
cd custom_nodes/buqi-minimax-h3-multigpu
torchrun --nproc_per_node=2 tests/latent_parity.py \
    --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors
```

公式のシングル GPU `_forward` を基準に、同一入力で 2 種類の通信方式のシーケンス並列
フォワードを実行し、最大絶対偏差と相対偏差を表示します。`ag vs a2a … exact=True`
(2 方式が完全一致)と、`vs 1gpu` の相対偏差 4e-6 程度が期待値です。

VAE:

```bash
torchrun --nproc_per_node=2 tests/vae_parity.py --handover
```

各 worker が自分のチャンクをデコードし、rank 0 も同じ範囲を自分でデコードして比較
します。`--handover` は worker の VAE を rank 0 のブロードキャスト重みから再構築し、
本番と同じ経路を通します。すべてのチャンクが `exact=True` になるはずです。

### エンコード済み動画で比較してはいけません

`tests/selftest.py` は両方の経路を通しで実行しますが、**スモーク テストにすぎません**。
出力動画のハッシュは何も証明しません。エンコード済みコンテナは**バイト単位で再現
されない**からです。同じシード・同じ入力・同じ標準経路で 3 回実行して、3 つの異なる
ハッシュが出ました(`529324c6…` 736142 バイト、`c1c1c01c…` と `1bd76f6c…` はどちらも
736204 バイト)。ここでハッシュを比較すると、存在しない失敗を報告し、本当の失敗を
隠します。

## プロファイリング

```bash
MINIMAX_SP_PROFILE_OPS=1 MINIMAX_SP_PROFILE=1 python main.py --cuda-device 0,1 --highvram
```

領域別(アテンション、MLP、gather、通信…)のステップ内訳と、ステップごとのディスパッチ
コストを記録します。上記のチューニングはこのデータに基づいています。
`tests/profile_phases.py` は websocket API 経由でノード粒度の同じ計測を行います。

## 環境変数

| 変数 | 説明 |
|---|---|
| `MINIMAX_SP_DEVICES` | `devices=auto` 時のフォールバックデバイスリスト(例: `0,1`) |
| `MINIMAX_SP_LOGDIR` | ワーカーログのディレクトリ(既定: システム一時ディレクトリ) |
| `MINIMAX_SP_AG_CHUNKS` | 隠れ状態の gather を分割してプロジェクションと重ねる数(既定 4、1 で無効) |
| `MINIMAX_SP_ATTN_CHUNKS` | アテンションをヘッド分割して出力交換と重ねる数(既定 4、1 で無効) |
| `MINIMAX_SP_PROFILE` | ステップごとのディスパッチとフォワード時間を記録 |
| `MINIMAX_SP_PROFILE_OPS` | ステップごとの領域別内訳を記録 |

## トラブルシューティング

- **"world_size N must divide the 56 attention heads"** — 1/2/4/7/8 を使用してください。
- **worker died, see .../minimax_sp_worker1.log** — 多くは VRAM 不足。解像度を下げるか
  fp8 チェックポイントを使用してください。実際のトレースバックはログにあります。
- **"process group up, waiting for workers to load weights" で止まる** — 各 GPU に
  約 21 GB の読み込みが発生し、初回は数分かかります。ワーカーログを確認してください。
- **`devices` の不一致** — `devices` の最初の ID は ComfyUI が動作している GPU
  (`--cuda-device`)でなければなりません。
- ComfyUI プロセスにつき SP グループは1つです。チェックポイント/dtype/world_size を
  変更するにはサーバーの再起動が必要です。

## 仕組み

H3 DiT は `[text|cond|audio|video]` のパックされた1本のシーケンスを 56 個の
アテンションヘッドで処理します。Ulysses SP ではシーケンスを行方向に分割して各ランクに
配置。トークンごとの演算(patch proj、変調、RoPE、MLP)は行内で完結し通信不要。
跨ぐのはアテンションだけで、各ランクは 56/`world_size` 個のヘッドについて
**全シーケンス**の完全かつ厳密なアテンションを計算します。

ヘッドとシーケンスを揃える方法は 2 つあり、どちらが安いかは GPU 数で決まります:

- **all-to-all**(従来の Ulysses):ローカル行で全 56 ヘッドの Q/K/V を計算し、
  ヘッド次元とシーケンス次元を交換する。1 行あたり `3 × inner / world` バイト。
- **all_gather**(`world_size ≤ 4` で使用):代わりに変調後の隠れ状態をブロードキャスト
  し、自ランク分の `qkv_proj` の行だけで射影する。1 行あたり `hidden` バイトで、
  Q/K/V は最初から全シーケンスにまたがるため、**3 回の transpose+コピーが消滅**します。

H3(`hidden` 5376、`inner` 7168)では `world_size == 4` で両者が同じバイト数になり、
それ未満では all_gather が有利です。480p / PCIe の実測ではブロックあたり
6.04ms → 2.34ms、達成帯域は 24.9 → 32.2 GB/s(経路に permute が無くなるため)。
4 を超える場合は all-to-all を使い続けます。

`qkv_proj` の行分割は近似ではなく厳密です。fp8 の重みは per-tensor のスケールを 1 つ
だけ持つため、行を選んでも各出力要素は元と同じ内積のままです。代償として、その重みの
`1/world` 分を各カードに余分に保持します(fp8・2 GPU で約 2.9GB)。

さらに 2 つの転送はいずれも周辺の演算と重ねられます。gather は複数の非同期サブ転送に
分割され(`MINIMAX_SP_AG_CHUNKS`、既定 4)、各チャンクの射影が次のチャンクの転送と
並行します。アテンションはヘッド分割で計算され(`MINIMAX_SP_ATTN_CHUNKS`、既定 4)、
各チャンクの出力交換が次のチャンクのアテンションと並行します。合わせて 480p で
約 150ms/ステップの短縮です。どちらも演算を変えません(アテンションのヘッドは独立、
射影の行は独立)。`tests/latent_parity.py` は 1/2/4/8 分割のいずれでもビット単位で
同一と報告します。

## ライセンス

MIT([LICENSE](LICENSE) 参照)。利用・改変・商用利用すべて自由です。

---

キーワード:MiniMax H3、MiniMax-H3、ComfyUI、ComfyUI カスタムノード、マルチ GPU、
並列推論、シーケンス並列、Ulysses、動画生成、ビデオ生成、音声同時生成、
video generation、multi-GPU。

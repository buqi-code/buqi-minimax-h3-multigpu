# buqi-minimax-h3-multigpu

[English](README.md) | [中文](README_zh.md) | **日本語**

ComfyUI 用 **MiniMax-H3** の本当のマルチ GPU 推論 — サンプリングだけでなく
**VAE デコードも並列化**します。差し替え可能な 2 ノード、**近似も精度低下も一切
ありません**。

MiniMax-H3 はパック トークン DiT で**動画と音声を同時生成**し、動画 VAE が latent
をピクセルに戻します。本リポジトリはこの両方を 2/4/7/8 GPU に分散します:

- **`MiniMaxH3SPUNETLoader`** — `UNETLoader` の差し替え。
  [DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509) で DiT のパック シーケンス
  を各ランクに分割:各 GPU は自分が担当するヘッドについて**完全で厳密な
  アテンション**を計算し、数学的に等価です。2 GPU のデノイズで **1.94×**、8 GPU で **6.65×**(下記)。
- **`MiniMaxH3SPVAEDecode`** — 任意、`VAEDecode` の差し替え。動画 VAE の時間チャンク
  を同じランクに分散。2 GPU の VAE デコードで **1.70×**、8 GPU で **5.99×**。
- 480p の 2 GPU でエンドツーエンド **1.95×**、1080p の 8 GPU で **6.28×**。
  プロファイル駆動、NVLink なしで測定。

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

測定環境:最大 8× RTX PRO 5000 Blackwell(72GB、PCIe 5.0、**NVLink なし**)、fp8、
124 フレーム、20 ステップ。**すべて定常状態の数値**です。セッション最初の分散デコード
は一度だけ重みの受け渡しを払うため、各構成は 2 回目の実行で測定しています:

| 形状 | GPU 数 | 1 GPU | マルチ GPU | エンドツーエンド | デノイズ | VAE | 1 秒あたりのコスト |
|---|---|---|---|---|---|---|---|
| 832×480 | 2 | 89.7s | 46.7s | **1.95×** | 1.94× | 1.70× | **1.03×** |
| 832×480 | 4 | 91.2s | 27.0s | **3.37×** | 3.41× | 3.27× | **1.19×** |
| 832×480 | 8 | 91.1s | 19.2s | **4.74×** | 4.86× | 5.99× | 1.69× |
| 1280×736 | 8 | 301.6s | 57.8s | **5.22×** | 5.46× | 5.11× | 1.53× |
| 1920×1088 | 8 | 1131.8s | 180.2s | **6.28×** | 6.65× | 5.09× | **1.27×** |

最後の列は `GPU 数 / 高速化` — 動画 1 秒を作るのに 1 枚構成より何倍の GPU 時間が
かかるかです。2 GPU はほぼ無償(+3%)、4 GPU は安価(+19%)。8 GPU が報われるのは
大きなキャンバスのときだけで、そこではデノイズが伸びる一方、固定の後段(テキスト
エンコーダ、多重化)は一定で、通信量は計算量に比例します。

VAE の列は `MiniMaxH3SPVAEDecode`(後述)が必要です。上限は時間チャンクの分割で決まり、
これらのクリップは 7 チャンクなので 2 GPU では 4/3 分割で 1.75× が上限、8 GPU では
1 枚 1 チャンクになります。

480p・2 GPU でのデノイズ 1 ステップの内訳(`MINIMAX_SP_PROFILE_OPS=1`、1 ランク):
アテンション+出力交換 856ms(重なるため合算)、MLP 657ms、gather+qkv_proj 325ms、
out_proj 83ms、変調+norm 76ms、QK-norm+RoPE 27ms。本体の GEMM はすでに 270〜370
TFLOPS(fp8)、アテンションは約 176 TFLOPS で、**演算に伸ばす余地はありません**。
GPU 数で割れない唯一のコストは GPU 間通信で、当初はステップの約 21%(481ms)でしたが、
今はほぼ演算の裏に隠れています。

1920×1088 の行が最もよくスケールする理由の一部は、その形状では 1 GPU が
メモリ圧迫のコストを払い始める一方、分散実行では避けられることです。

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

いずれのノードも**ステージ全体**を分散します。カーネルだけの並列化ではなく、
近似も追加量子化もありません。

**DiT(デノイズ ループ)** — H3 DiT は `[text|cond|audio|video]` の 1 本のパック
シーケンスを 56 ヘッドで処理します。Ulysses SP がシーケンスを行方向に分割して常駐、
トークンごとの演算(patch proj、変調、RoPE、MLP)は行内で完結し通信不要。跨ぐのは
アテンションだけで、各ランクは 56/`world_size` ヘッドについて**全シーケンス**の
厳密なアテンションを計算します。GPU 数に応じて 2 種類の交換方式を自動選択します:

- `world_size ≤ 4`:変調後の隠れ状態をブロードキャストし、自ランク分の `qkv_proj`
  の行だけで射影。1 行 `hidden` バイトで、従来方式の 3 回の transpose+コピーが消滅。
  fp8 重みの行分割は per-tensor スケールなので厳密です。
- `world_size ≥ 7`:従来の all-to-all(1 行 `3 × inner / world` バイト)。

H3(`hidden` 5376、`inner` 7168)では world 4 で両者が同じバイト数、それ未満で
all_gather 有利。480p / PCIe 実測ではブロックあたり 6.04ms → 2.34ms、達成帯域は
24.9 → 32.2 GB/s(経路に permute が無いため)。両方の転送は周辺演算と重ねられます:
gather は非同期サブ転送に分割(`MINIMAX_SP_AG_CHUNKS`、既定 4)、各チャンクの射影が
次のチャンクの転送と並行。アテンションもヘッド分割(`MINIMAX_SP_ATTN_CHUNKS`、既定
4)、各チャンクの出力交換が次のチャンクのアテンションと並行。合わせて 480p で
約 150ms/ステップの短縮。演算は変えず、`tests/latent_parity.py` が 1/2/4/8 分割で
ビット単位で同一と報告します。

**VAE デコード** — `decode_temporal` は時間チャンクを 1 つずつデコードし、各チャンク
は latent の一部しか読まないため互いに独立です。この末端の計算だけをランクにラウンド
ロビン分散し、空間タイル ブレンド・時間ブレンド・キャンバス書き込みはすべて rank 0
の公式コード経路に残します。worker の VAE 重みは初回使用時に rank 0 から NCCL で
受け渡し(約 4.85 GiB、セッションごとに一度、約 2.3s)、実際にロードしたチェックポ
イント(独自パスを含む)が必ず使われます。

**試したが採用しなかった項目(将来の担当者が同じ道を歩まないよう記録)。** プロファ
イリングで却下したもの:AdaLN 事前計算(上流の pruned 重みが `t_dim` を 8 次元の
曲線基底に因数分解済み、分岐全体で 44M パラメータ・1 ステップ 0.06ms)、テキスト
エンコーダの取り込み(短プロンプトで合計 1.0s)、gloo 制御チャネルを常駐 NCCL に
置き換える(ステップごとの meta ブロードキャストは 0.3ms)、変調+norm を Triton
カーネルに融合(変調+norm 領域はステップの 3.4%)。詳細はコミット履歴。

## ライセンス

MIT([LICENSE](LICENSE) 参照)。利用・改変・商用利用すべて自由です。

---

キーワード:MiniMax H3、MiniMax-H3、ComfyUI、ComfyUI カスタムノード、マルチ GPU、
並列推論、シーケンス並列、Ulysses、動画生成、ビデオ生成、音声同時生成、
video generation、multi-GPU。

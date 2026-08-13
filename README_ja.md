# buqi-minimax-h3-multigpu

[English](README.md) | [中文](README_zh.md) | **日本語**

ComfyUI 用 **MiniMax-H3** マルチ GPU(シーケンス並列)推論ノード — `UNETLoader` の
差し替えだけで使え、**出力はビット単位で同一**、品質劣化なし。

MiniMax-H3 はパック トークン DiT で**動画と音声を同時生成**します。本ノードは
[DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509) の all-to-all 方式でその
シーケンスを 2/4/7/8 GPU に分割します。各 GPU は自分が受け持ったヘッドについて
**完全で厳密なアテンション**を計算するため数学的に等価であり、マルチ GPU の結果は
シングル GPU とビット単位で一致します(検証済み。`tests/selftest.py` で
お手元の環境でも確認できます)。

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

実測(RTX PRO 5000 48 GB、fp8、20 ステップ、エンドツーエンド秒):

| 形状 | 1 GPU | SP2 | 高速化 |
|---|---|---|---|
| 480p × 5s | 89.3 | 62.3 | 1.43× |
| 720p × 5s | 300.8 | 223.5 | 1.35× |
| 720p × 10s | 867.6 | 403.7 | 2.15× |
| 1080p × 5s | 1127.4 | 375.6 | 3.00× |

SP2 のスケーリングが線形に満たないのは想定内です。2 ランクではステップごとの
all-to-all オーバーヘッドの割合が大きく、短尺動画では VAE デコードに対して
サンプリングの比重が小さいためです。長尺・高解像度ほどよくスケールします
(8 GPU では最大 6.9× を実測)。

PCIe / NVLink なしのマシンでも動作します(RTX 5090 で確認済み)。上記の数値より
スケーリングは若干低くなります。

## 解像度の制約

MiniMax-H3 標準のルールに従います。キャンバスは 32px 単位に丸められ、
latent(px/16)はパッチ(2)で割り切れる必要があるため、標準グリッドを使って
ください — 832×480、1280×736、1920×1088 など。`height=720` は**不可**
(latent 高 45 が奇数)。736 を使ってください。

## 正しさの検証

```bash
# サーバー起動中に:
python custom_nodes/buqi-minimax-h3-multigpu/tests/selftest.py \
    --server http://127.0.0.1:18188 --sp 2 \
    --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors \
    --image your_first_frame.png
```

同一入力でシングル GPU ジョブと 2 GPU ジョブを各1回実行し、出力動画の SHA-256 を
比較します。`PASS: sp2 output is bit-identical` が表示されれば成功です。

## 環境変数

| 変数 | 説明 |
|---|---|
| `MINIMAX_SP_DEVICES` | `devices=auto` 時のフォールバックデバイスリスト(例: `0,1`) |
| `MINIMAX_SP_LOGDIR` | ワーカーログのディレクトリ(既定: システム一時ディレクトリ) |

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
配置。トークンごとの演算(patch proj、AdaLN、RoPE、MLP)は行内で完結し通信不要。
跨ぐのはアテンションだけで、2回の all-to-all でヘッド次元とシーケンス次元を交換し、
各ランクが 56/P 個のヘッドについて**全シーケンス**の完全かつ厳密なアテンションを
計算します。近似なし、マスクなし、劣化なし。

## ライセンス

MIT([LICENSE](LICENSE) 参照)。利用・改変・商用利用すべて自由です。

---

キーワード:MiniMax H3、MiniMax-H3、ComfyUI、ComfyUI カスタムノード、マルチ GPU、
並列推論、シーケンス並列、Ulysses、動画生成、ビデオ生成、音声同時生成、
video generation、multi-GPU。

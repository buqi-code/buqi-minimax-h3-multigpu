# buqi-minimax-h3-multigpu

[English](README.md) | [中文](README_zh.md) | **日本語**

ComfyUI MiniMax H3 用の Ulysses シーケンス並列ローダーです。

## 互換性

- ComfyUI commit **`8d534945`** 以降で導入された MiniMax H3 API が必要です。release tag ではなく起動時の capability check で判定します。
- 現在の ComfyUI 0.37 development commit **`7fbcfa8be9a8f47cf905ec47978b5bd754959ea7`** で検証済みです。
- ComfyUI 0.30 は旧 release の対象で、main branch の対象外です。
- Linux または WSL2 + NCCL。Windows native の multi-GPU は非対応です。
- `world_size`: 1 / 2 / 4 / 7 / 8

CPU テストは package discovery、現在の H3 API、fail-fast、patch 同期、2-rank Gloo parity を確認します。実 GPU の可否・性能は対象マシンで GPU acceptance を実行して確認してください。

## インストールと起動

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/buqi-code/buqi-minimax-h3-multigpu.git
cd /path/to/ComfyUI
MINIMAX_SP_DEVICES=0,1 python main.py --cuda-device 0,1 --highvram
```

ワークフローでは `UNETLoader` だけを `MiniMaxH3SPUNETLoader` に置き換えます。SP は 1 prompt の遅延を下げる方式で、各 GPU に完全な DiT weight を保持します。別の ComfyUI MultiGPU wrapper と重ねないでください。DP は GPU ごとに別 prompt を処理する throughput 向け方式です。

`devices="auto"` は `MINIMAX_SP_DEVICES`、次に `CUDA_VISIBLE_DEVICES` の物理 ID（`--cuda-device 2,3` を含む）を使います。論理番号 `0,1` へ置き換えません。明示指定は重複不可で `world_size` と同数、すべて可視リスト内、先頭は ComfyUI の primary GPU と一致する必要があります。

## INT8 と VAE

事前量子化済み INT8/ConvRot checkpoint は **`weight_dtype=default`** を使用します。例では次を使用します。

- `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- `minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- `minimax_h3_video_vae_int8_convrot.safetensors`
- `minimax_h3_audio_vae_fp32.safetensors`
- `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`

公式 INT8/FP16 video VAE はどちらも標準 **`VAEDecode`** を使い、ComfyUI の quantized metadata、streaming、memory policy を保持します。旧 `MiniMaxH3SPVAEDecode` node ID は workflow 互換性のためだけに残り、warning を一度出して公式 `vae.decode` を呼ぶ deprecated alias です。VAE の並列実行は行いません。Audio は標準 `VAEDecodeAudio` を使います。

## サポート範囲

対応: T2V、I2V first、I2V first+last、R2V、`MiniMaxH3AddGuide`、mask、PDD/static options、実行前に適用する static Turbo LoRA。

非対応（distributed 実行前に停止）: Fun ControlNet、Sparse Attention、dynamic hooks/patches、ComfyUI MultiGPU/threaded MultiGPU との併用。

## サンプル

- [`examples/workflow_ui_2gpu.json`](examples/workflow_ui_2gpu.json): 固定 commit の公式 MiniMax H3 UI blueprint をコピーし、`UNETLoader` のみ SP loader に変更。T2V / first / first+last / Turbo を設定できます。
- API: [`T2V`](examples/workflow_api_t2v_2gpu.json), [`I2V first`](examples/workflow_api_2gpu.json), [`I2V first+last`](examples/workflow_api_i2v_first_last_2gpu.json), [`R2V`](examples/workflow_api_r2v_2gpu.json), [`AddGuide`](examples/workflow_api_addguide_2gpu.json), [`Turbo LoRA`](examples/workflow_api_turbo_lora_2gpu.json)。

API の `REPLACE_WITH_...` は `ComfyUI/input` 内の自分の画像名へ変更してください。API variant は現在の node schema に合わせた例で、upstream 公式 export を装ってはいません。実環境では UI workflow を確認後、Developer Mode の **Save (API Format)** で再 export してください。

## 検証とトラブルシューティング

```bash
COMFYUI_ROOT=/path/to/ComfyUI PYTHONPATH=/path/to/ComfyUI \
  python -m unittest discover -s tests -p 'test_*.py' -v
COMFYUI_ROOT=/path/to/ComfyUI torchrun --standalone --nproc-per-node=2 tests/test_current_api.py
python -m compileall -q __init__.py minimax_sp tests
```

GPU tensor parity と実際の loader/SPGroup lifecycle command は [English README](README.md#verification) を参照してください。失敗は nonzero で終了します。

主な環境変数:

- `MINIMAX_SP_DEVICES=0,1`: device mapping。先頭は ComfyUI の main GPU。
- `MINIMAX_SP_STARTUP_TIMEOUT=300`: startup timeout（秒）。
- `MINIMAX_SP_COLLECTIVE_TIMEOUT=120`: collective timeout（秒）。
- `MINIMAX_SP_LOGDIR=/path`: `minimax_sp_worker<N>.log` の保存先。

ノードが出ない場合は clone 先と root `__init__.py`、ComfyUI 起動ログを確認してください。P2P/startup/collective failure は `nvidia-smi`、device mapping、NCCL と各 worker log を確認します。旧 VAE node の deprecated warning が出たら標準 `VAEDecode` に置き換えてください。checkpoint/dtype/world size を変える場合は ComfyUI を再起動してください。

MIT License。詳細は [LICENSE](LICENSE) を参照してください。

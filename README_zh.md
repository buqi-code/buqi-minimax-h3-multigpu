# buqi-minimax-h3-multigpu

[English](README.md) | **中文** | [日本語](README_ja.md)

用于 ComfyUI MiniMax H3 的 Ulysses 序列并行多 NVIDIA GPU 加载器。

## 兼容性与验证范围

- 要求 ComfyUI commit **`8d534945`** 或之后引入的 MiniMax H3 API；启动时由能力检查判断，不声明精确 release tag 范围。
- 已测试当前 ComfyUI 0.37 开发提交：**`7fbcfa8be9a8f47cf905ec47978b5bd754959ea7`**。
- ComfyUI 0.30 接口只属于旧版本；main 不再声明兼容。
- 支持 Linux 或 WSL2，并要求 PyTorch/NCCL；不支持原生 Windows 多卡。
- `world_size` 必须整除 56 个注意力头：**1、2、4、7、8**。

CPU 测试覆盖安装发现、当前 H3 API 契约、fail-fast、patch 同步和双进程 Gloo parity。当前提交的真实 GPU 能力仍需在目标机器运行手动 GPU 验收；CPU 通过不等于 GPU 性能已验证。

## 安装

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/buqi-code/buqi-minimax-h3-multigpu.git
```

重启 ComfyUI。仓库根目录的 `__init__.py` 会直接导出扩展入口，不需要改 `sys.path`。本节点没有额外 pip 依赖；`torch` 和 ComfyUI 不作为 pip 依赖重复声明。

## 当前模型名与 INT8

- DiT：`minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- R2V DiT：`minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- 文本编码器：`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- 视频 VAE：`minimax_h3_video_vae_int8_convrot.safetensors`
- 音频 VAE：`minimax_h3_audio_vae_fp32.safetensors`
- Turbo LoRA：`minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`

预量化 INT8/ConvRot checkpoint 必须使用 **`weight_dtype=default`**，由 ComfyUI 读取并保留量化 metadata，不要再次强制转换 dtype。

官方 INT8 和 FP16 视频 VAE 都使用标准 **`VAEDecode`**，以保留 ComfyUI 的量化 metadata、streaming 和显存策略。旧 `MiniMaxH3SPVAEDecode` node ID 只为兼容旧工作流保留：它仅警告一次并调用官方 `vae.decode`，不再执行并行 VAE decode。

| VAE 路径 | 实际支持 |
|---|---|
| 标准 `VAEDecode` + 官方 INT8 或 FP16 视频 VAE | 推荐；走官方 decode/streaming 路径 |
| 旧 `MiniMaxH3SPVAEDecode` 节点 | 标准 `vae.decode` 的 deprecated 兼容别名，不并行 |
| 音频 VAE | 使用标准 `VAEDecodeAudio`，本项目不分片 |

## 功能矩阵

| 功能 | 状态 |
|---|---|
| T2V | 支持当前 H3 conditioning 路径 |
| I2V first | 支持 |
| I2V first + last | 支持 |
| R2V / 链式 `MiniMaxH3AddGuide` | 支持 |
| denoise mask | 支持 |
| PDD / 静态 model options | 支持 |
| 运行前加载的静态 Turbo LoRA | 支持，去噪前同步 patch |
| Fun ControlNet | 不支持，进入分布式执行前失败 |
| Sparse Attention | 不支持，提前失败 |
| 动态 hooks/patches | 不支持，提前失败 |
| 与 ComfyUI MultiGPU/threaded MultiGPU 叠加 | 不支持，提前失败 |

这里的“支持”指通过运行时能力检查的 H3 接口；部署前仍应对你的硬件、模型文件和工作流运行 GPU 验收。

## SP 与 DP

- **SP（本项目）**：同一个 prompt 的序列/注意力计算分到多张卡，目标是降低单任务延迟。每张卡仍保存完整 DiT 权重，不会按卡数降低权重显存。
- **DP**：每张卡独立跑不同 prompt，提升吞吐量，不降低单个任务延迟，需要外部队列或调度器。

不要把本 SP loader 再套到其他 MultiGPU wrapper 上。

## 启动与设备映射

```bash
cd /path/to/ComfyUI
MINIMAX_SP_DEVICES=0,1 python main.py --cuda-device 0,1 --highvram
```

工作流中只把 `UNETLoader` 换成 `MiniMaxH3SPUNETLoader`；尤其使用 INT8 视频 VAE 时，继续使用标准 `VAEDecode`。

`devices="auto"` 优先读取 `MINIMAX_SP_DEVICES`，否则读取 `CUDA_VISIBLE_DEVICES` 中的物理 ID（包括 `--cuda-device 2,3` 写入的值），不会误用逻辑重编号后的 `0,1`。显式映射必须唯一、数量恰好等于 `world_size`、全部来自当前可见物理列表，且第一项与 ComfyUI 主卡一致。`world_size=1` 为普通单卡加载。

## 示例

| 文件 | 格式/用途 |
|---|---|
| [`examples/workflow_ui_2gpu.json`](examples/workflow_ui_2gpu.json) | 直接复制固定 ComfyUI 提交中的官方 “Image to Video (MiniMax H3)” UI 模板，只替换 `UNETLoader`；可通过可选 first/last 输入和静态 Turbo 开关覆盖 T2V、I2V first、first+last、Turbo |
| [`examples/workflow_api_t2v_2gpu.json`](examples/workflow_api_t2v_2gpu.json) | API：T2V |
| [`examples/workflow_api_2gpu.json`](examples/workflow_api_2gpu.json) | API：I2V first |
| [`examples/workflow_api_i2v_first_last_2gpu.json`](examples/workflow_api_i2v_first_last_2gpu.json) | API：I2V first+last |
| [`examples/workflow_api_r2v_2gpu.json`](examples/workflow_api_r2v_2gpu.json) | API：R2V reference image |
| [`examples/workflow_api_addguide_2gpu.json`](examples/workflow_api_addguide_2gpu.json) | API：任意帧 AddGuide |
| [`examples/workflow_api_turbo_lora_2gpu.json`](examples/workflow_api_turbo_lora_2gpu.json) | API：静态 8-step Turbo LoRA |

API 变体按已测试开发提交的节点 schema 编写，但不冒充上游已导出的官方 API 模板。生产使用时，请先导入 UI 模板、按本机模型配置并实际验证，再打开 ComfyUI Developer Mode，使用 **Save (API Format)** 导出。

API 中 `REPLACE_WITH_` 开头的图片名是必须由用户替换的占位值：把自己的文件放进 `ComfyUI/input` 后修改 JSON。UI 模板不依赖仓库内图片。

## 验证命令

CPU：

```bash
cd /path/to/ComfyUI/custom_nodes/buqi-minimax-h3-multigpu
COMFYUI_ROOT=/path/to/ComfyUI PYTHONPATH=/path/to/ComfyUI \
  python -m unittest discover -s tests -p 'test_*.py' -v
COMFYUI_ROOT=/path/to/ComfyUI torchrun --standalone --nproc-per-node=2 tests/test_current_api.py
python -m compileall -q __init__.py minimax_sp tests
python -c 'import json,pathlib; [json.loads(p.read_text()) for p in pathlib.Path("examples").glob("*.json")]'
```

双卡验收：

```bash
COMFYUI_ROOT=/path/to/ComfyUI torchrun --standalone --nproc-per-node=2 \
  tests/latent_parity.py --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors
COMFYUI_ROOT=/path/to/ComfyUI python tests/runtime_recovery.py \
  --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors --devices auto
```

张量 parity 和真实 loader/SPGroup 生命周期脚本失败都会返回非零。`tests/selftest.py` 是连接已运行 ComfyUI 服务的端到端冒烟测试，也会传递提交/执行错误；编码视频 hash 只作信息展示，不代表数值一致性。

## 超时、worker 日志与排障

| 环境变量 | 默认 | 作用 |
|---|---:|---|
| `MINIMAX_SP_DEVICES` | `CUDA_VISIBLE_DEVICES` | `devices=auto` 的可选物理设备映射 |
| `MINIMAX_SP_STARTUP_TIMEOUT` | 300 秒 | worker/进程组启动超时 |
| `MINIMAX_SP_COLLECTIVE_TIMEOUT` | 120 秒 | NCCL collective 超时 |
| `MINIMAX_SP_LOGDIR` | 系统临时目录 | `minimax_sp_worker<N>.log` 所在目录 |
| `MINIMAX_SP_VERIFY=1` | 关闭 | 首次 SP 输出与原生 H3 对比后才接受 |
| `MINIMAX_SP_PROFILE=1` | 关闭 | forward 耗时 |
| `MINIMAX_SP_PROFILE_OPS=1` | 关闭 | 分区耗时 |

- **clone 后没有节点**：确认仓库目录本身位于 `custom_nodes`，根目录有 `__init__.py`，查看 ComfyUI 启动 traceback。
- **原生 Windows 报错**：改用 WSL2 或 Linux，SP 需要 NCCL。
- **GPU visibility/P2P/startup probe 失败**：检查 `--cuda-device`、`MINIMAX_SP_DEVICES`、`nvidia-smi` 与 NCCL/P2P。只有确实加载较慢时才提高 startup timeout。
- **collective timeout/worker 退出**：查看 `MINIMAX_SP_LOGDIR` 中每个 rank 的日志；先修复 OOM、设备映射或不支持 patch，不要只反复加大 timeout。
- **VAE 节点 deprecated warning**：把 `MiniMaxH3SPVAEDecode` 替换为标准 `VAEDecode`；两者现在都走官方解码。
- **更换 checkpoint/dtype/world size**：重启 ComfyUI 后再建新 SP 组。

## 许可证

MIT，见 [LICENSE](LICENSE)。

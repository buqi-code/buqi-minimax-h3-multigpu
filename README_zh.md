# buqi-minimax-h3-multigpu

[English](README.md) | **中文** | [日本語](README_ja.md)

ComfyUI 上 **MiniMax-H3** 的多卡(序列并行)推理节点 —— 替换 `UNETLoader` 即可使用,
**输出逐比特一致**,零质量损失。

MiniMax-H3 用一个打包 token 的 DiT 同时生成**视频+音频**。本节点按
[DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509) 的 all-to-all 方案把该序列
切分到 2/4/7/8 张 GPU:每张卡对自己分到的一批注意力头做**完整精确的注意力**,
数学上完全等价 —— 多卡结果与单卡逐比特一致(已验证,可用
`tests/selftest.py` 在你自己的机器上复验)。

优先适配社区最常见配置:**双卡**(1~8 卡均可)。

## 环境要求

- Linux(Windows 请用 WSL2 —— 原生 Windows 没有 NCCL)
- ComfyUI **>= 0.30.0**(自带 `comfy.ldm.minimax.model` 的版本);旧版本加载时会明确报错
- MiniMax-H3 模型文件(DiT、Qwen3-VL 文本编码器、视频/音频 VAE)照常放入
  `diffusion_models/`、`text_encoders/`、`vae/` —— 请从 MiniMax-H3 官方发布渠道获取,
  本仓库只含代码
- 带 NCCL 的 PyTorch(ComfyUI 官方依赖自带)
- `world_size` 必须整除 56 个注意力头:**1、2、4、7、8**

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/buqi-code/buqi-minimax-h3-multigpu.git
```

无额外 Python 依赖。

## 快速上手(双卡)

1. 启动 ComfyUI 时让两张卡都可见:

   ```bash
   python main.py --cuda-device 0,1 --highvram
   ```

2. 把工作流里的 `UNETLoader` 换成 **MiniMax H3 Multi-GPU Loader (Ulysses SP)**。
   `world_size` 默认 2,`devices` 保持 `auto` 即可。

3. 提交任务。首次运行会为每张额外的卡拉起一个 worker 进程并各加载一份 DiT
   (fp8 约 21GB/卡),之后的任务复用这些 worker。

API 格式示例见 [`examples/workflow_api_2gpu.json`](examples/workflow_api_2gpu.json);
可直接拖入 ComfyUI 画布导入的图形工作流见
[`examples/workflow_ui_2gpu.json`](examples/workflow_ui_2gpu.json)。

## 节点输入

| 输入 | 默认 | 说明 |
|---|---|---|
| `unet_name` | — | H3 DiT 权重,与 `UNETLoader` 相同 |
| `weight_dtype` | `default` | 与 `UNETLoader` 相同;推荐 fp8 |
| `world_size` | `2` | 并行卡数(1/2/4/7/8),填 1 即普通单卡加载 |
| `devices` | `auto` | 物理 CUDA 卡号,如 `"0,1"`;第一个必须是 ComfyUI 本身所用的卡。`auto` = 优先读环境变量 `MINIMAX_SP_DEVICES`,否则取前 `world_size` 张卡 |

`world_size=1` 时节点直通不加任何改动,单卡机器也能直接用本节点的工作流。

## 显存与分辨率建议

序列并行是**每卡复制完整 DiT**(不切权重),单卡显存与单卡推理相同 ——
收益在速度。

| 权重 | 每卡 DiT 占用 | 适用显卡 |
|---|---|---|
| fp8(推荐) | ~21GB | 24GB 卡可跑至 720p,480p 宽裕 |
| bf16(bf16 权重 + `default`) | ~40GB | 48GB 及以上 |

实测(RTX PRO 5000 48GB,fp8,20 步,端到端秒):

| 档位 | 单卡 | SP2 | 加速 |
|---|---|---|---|
| 480p × 5s | 89.3 | 62.3 | 1.43× |
| 720p × 5s | 300.8 | 223.5 | 1.35× |
| 720p × 10s | 867.6 | 403.7 | 2.15× |
| 1080p × 5s | 1127.4 | 375.6 | 3.00× |

双卡加速比偏低属预期:仅 2 路时每步 all-to-all 开销摊得少,且短视频里采样
占比小(VAE 解码串行)。视频更长、分辨率更高时扩展性更好(8 卡实测最高 6.9×)。

PCIe / 无 NVLink 的机器可用(已在 RTX 5090 验证),扩展性略低于上表。

## 分辨率约束

遵循 MiniMax-H3 官方规则:画布按 32px 取整,latent(像素/16)须被 patch(2)
整除,请用标准档位 —— 832×480、1280×736、1920×1088 等。
`height=720` **不合法**(latent 高 45 为奇数),请用 736。

## 在你机器上验证正确性

```bash
# 服务运行时:
python custom_nodes/buqi-minimax-h3-multigpu/tests/selftest.py \
    --server http://127.0.0.1:18188 --sp 2 \
    --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors \
    --image your_first_frame.png
```

用完全相同的输入分别跑一次单卡、一次双卡,比对输出视频的 SHA-256。
看到 `PASS: sp2 output is bit-identical` 即通过。

## 环境变量

| 变量 | 说明 |
|---|---|
| `MINIMAX_SP_DEVICES` | `devices=auto` 时的备选卡表,如 `0,1` |
| `MINIMAX_SP_LOGDIR` | worker 日志目录(默认系统临时目录) |

## 常见问题

- **"world_size N must divide the 56 attention heads"** —— 只用 1/2/4/7/8。
- **worker died, see .../minimax_sp_worker1.log** —— 多为显存不足;降分辨率或换
  fp8 权重,真实报错在日志里。
- **卡在 "process group up, waiting for workers to load weights"** —— 每卡要加载
  约 21GB 权重,首跑需几分钟,看 worker 日志确认进度。
- **`devices` 不匹配** —— `devices` 第一个卡号必须是 ComfyUI 启动用的卡
  (`--cuda-device`)。
- 每个 ComfyUI 进程只有一个 SP 组:更换权重/dtype/world_size 需重启服务。

## 原理

H3 DiT 处理一条打包序列 `[text|cond|audio|video]`,共 56 个注意力头。Ulysses SP
让序列按行切分常驻各卡:所有逐 token 运算(patch proj、AdaLN、RoPE、MLP)
都是行内本地计算,无需通信;只有注意力跨卡,通过两次 all-to-all 把"头维"换成
"序列维" —— 每卡随后对自己负责的 56/P 个头在**完整序列**上做精确注意力。
无近似、无掩码、无损失。

## 许可证

MIT(见 [LICENSE](LICENSE)),可自由使用、修改、商用。

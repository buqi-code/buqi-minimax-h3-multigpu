# buqi-minimax-h3-multigpu

[English](README.md) | **中文** | [日本語](README_ja.md)

ComfyUI 上 **MiniMax-H3** 的真·多卡推理 —— 不只并行采样,**VAE 解码也并行**。
两个即插即用节点,**无任何近似、无精度降低**。

MiniMax-H3 用一个打包 token 的 DiT 同时生成**视频+音频**,再由视频 VAE 把 latent
解成像素。本仓库把这两段都分散到 2/4/7/8 张 GPU:

- **`MiniMaxH3SPUNETLoader`** —— 替换 `UNETLoader`。按
  [DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509) 把 DiT 的打包序列切分到
  各 rank:每张卡对自己那批注意力头做**完整精确的注意力**,数学等价。
  双卡去噪 **1.94×**、8 卡 **6.65×**(数据见下)。
- **`MiniMaxH3SPVAEDecode`** —— 可选,替换 `VAEDecode`。把视频 VAE 的时间块
  分散到同一批 GPU。双卡 VAE **1.70×**、8 卡 **5.99×**。
- 480p 双卡端到端 **1.95×**,1080p 八卡 **6.28×**;剖析驱动,无 NVLink 也适用。

### 关于精度

每张卡仍然在完整序列上做精确注意力,没有任何近似、额外量化、掩码或缓存。因此结果与
单卡**数学等价**,但**并非逐比特相同**:每个 rank 只处理 56/`world_size` 个注意力头,
而注意力 kernel 会根据头数选择归约顺序,浮点舍入落点因此不同。在 DiT 输出上实测,与
单卡的偏差为**相对 ≤ 4.2e-6**(音频为 0),比 bf16 的机器精度(7.8e-3)小三个数量级,
且无系统性偏向。

可用 `tests/latent_parity.py` 自行复验 —— 它比较 DiT 实际输出的速度场张量,而不是编码
后视频的哈希(H.264 有损,这个量级的差异它既可能掩盖也可能放大)。

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



实测环境:最多 8× RTX PRO 5000 Blackwell(72GB,PCIe 5.0,**无 NVLink**),fp8,
124 帧,20 步。**所有数字都是稳态** —— 每个会话首次分片解码要付一次性的权重交接,
因此每种配置都取第二次运行:

| 档位 | 卡数 | 单卡 | 多卡 | 端到端 | 去噪 | VAE | 每秒视频成本 |
|---|---|---|---|---|---|---|---|
| 832×480 | 2 | 89.7s | 46.7s | **1.95×** | 1.94× | 1.70× | **1.03×** |
| 832×480 | 4 | 91.2s | 27.0s | **3.37×** | 3.41× | 3.27× | **1.19×** |
| 832×480 | 8 | 91.1s | 19.2s | **4.74×** | 4.86× | 5.99× | 1.69× |
| 1280×736 | 8 | 301.6s | 57.8s | **5.22×** | 5.46× | 5.11× | 1.53× |
| 1920×1088 | 8 | 1131.8s | 180.2s | **6.28×** | 6.65× | 5.09× | **1.27×** |

最后一列是 `卡数 / 加速比` —— 即产出一秒视频所耗的 GPU 时间比单卡多多少。双卡几乎
免费(+3%),四卡很便宜(+19%);八卡只在大画幅上划算,因为那时去噪时间增长,而固定
尾部(文本编码、封装)不变,通信量则与计算量同比例增长。

VAE 那几列需要启用 `MiniMaxH3SPVAEDecode`(见下)。其上限由时间块划分决定 —— 这些
片段有 7 块,双卡 4:3 分配所以不可能超过 1.75×,八卡则每卡一块。

480p 双卡下去噪每步的构成(`MINIMAX_SP_PROFILE_OPS=1`,每步每卡):attention+输出交换
856ms(两者重叠故合并计时)、MLP 657ms、gather+qkv_proj 325ms、out_proj 83ms、
调制+norm 76ms、QK-norm+RoPE 27ms。主干 GEMM 已跑到 270–370 TFLOPS(fp8),attention
约 176 TFLOPS,**算子本身已无空间**;唯一不随卡数摊薄的就是卡间通信 —— 起初占每步
约 21%(481ms),现已基本被计算掩盖。

1080p 那一行扩展性最好,还有一个原因:单卡在该画幅已开始承受显存压力,而分片运行不会。

## 分辨率约束

遵循 MiniMax-H3 官方规则:画布按 32px 取整,latent(像素/16)须被 patch(2)
整除,请用标准档位 —— 832×480、1280×736、1920×1088 等。
`height=720` **不合法**(latent 高 45 为奇数),请用 736。

## 多卡 VAE 解码(可选)

把 `VAEDecode` 换成 **MiniMax H3 Multi-GPU VAE Decode**(`MiniMaxH3SPVAEDecode`),
输入输出完全一致,视频 VAE 的时间块会分散到 SP 组已持有的那些卡上。

`decode_temporal` 逐个时间块解码,每块只读 latent 的一个切片,因此块间独立。只有
这层叶子计算被分发:**空间 tile 的 blend、时间 blend、canvas 写入全部留在 rank0 的
官方代码路径里**,由它接收算好的块。结果逐比特相同(验证见下)。

worker 的 VAE 权重是首次使用时由 rank0 通过 NCCL 交接的,而不是自己去找文件,所以
分片解码用的一定就是你加载的那份权重(含自定义路径)。该传输约 4.85 GiB,每会话
一次约 2.3s。

这个特性是 opt-in 的,因为它不免费:每个 worker 要在 DiT 之外再放一份视频 VAE
(约 5GB),fp8 下每卡约 26GB。**24GB 卡请继续用官方 `VAEDecode`。** 以下情况节点会
自动回退到单卡解码(只打日志):没有运行中的 SP 组、`world_size` 为 1、传入的不是
H3 video VAE、或 latent 的块数不足以切分。

## 在你机器上验证正确性

两个测试都在**张量层面**比对,这是唯一有意义的层面。DiT:

```bash
cd custom_nodes/buqi-minimax-h3-multigpu
torchrun --nproc_per_node=2 tests/latent_parity.py \
    --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors
```

它以官方单卡 `_forward` 为基准,对同一输入跑两种通信策略的并行前向,输出最大绝对
与相对偏差。预期看到 `ag vs a2a … exact=True`(两种策略完全一致),以及 `vs 1gpu`
的相对偏差约 4e-6。

VAE:

```bash
torchrun --nproc_per_node=2 tests/vae_parity.py --handover
```

让各 worker 解自己的块、rank0 自己也解同样的切片,然后逐元素比对;`--handover` 表示
worker 的 VAE 从 rank0 广播的权重重建,与生产路径完全一致。预期每块 `exact=True`。

### 不要拿编码后的视频做比对

`tests/selftest.py` 会端到端各跑一遍,但它**只是冒烟测试**。对输出视频求哈希证明不了
任何事:编码后的容器**不是逐字节可复现的**。同样的种子、同样的输入、同样的官方代码
路径跑三次,得到三个不同哈希(`529324c6…` 736142 字节、`c1c1c01c…` 与 `1bd76f6c…`
均为 736204 字节)。在那里做哈希比对会报出不存在的失败,也会掩盖真实的失败。

## 性能剖析

```bash
MINIMAX_SP_PROFILE_OPS=1 MINIMAX_SP_PROFILE=1 python main.py --cuda-device 0,1 --highvram
```

会按区域(注意力、MLP、gather、通信……)打印每步耗时,以及每步的调度开销 ——
上面的调优就是基于这些数据做的。`tests/profile_phases.py` 通过 websocket API
在节点粒度上做同样的事。

## 环境变量

| 变量 | 说明 |
|---|---|
| `MINIMAX_SP_DEVICES` | `devices=auto` 时的备选卡表,如 `0,1` |
| `MINIMAX_SP_LOGDIR` | worker 日志目录(默认系统临时目录) |
| `MINIMAX_SP_AG_CHUNKS` | 隐状态 gather 拆成几个子块以与投影重叠(默认 4;设 1 关闭重叠) |
| `MINIMAX_SP_ATTN_CHUNKS` | attention 按 head 拆成几块以与输出交换重叠(默认 4;设 1 关闭重叠) |
| `MINIMAX_SP_PROFILE` | 打印每步调度与前向耗时 |
| `MINIMAX_SP_PROFILE_OPS` | 打印每步按区域的耗时拆分 |

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

两个节点都是**整段并行**,不只是 kernel 层;没有任何近似或额外量化。

**DiT(去噪循环)** —— H3 DiT 处理一条打包序列 `[text|cond|audio|video]`,共 56 个
注意力头。Ulysses SP 让序列按行切分常驻各卡:所有逐 token 运算(patch proj、调制、
RoPE、MLP)都是行内本地计算,无需通信;只有注意力跨卡 —— 每卡对自己负责的
56/`world_size` 个头在**完整序列**上做精确注意力。支持两种通信方案,按卡数自动选择:

- `world_size ≤ 4`:广播调制后的隐状态,只用本卡那部分 `qkv_proj` 权重行做投影,
  每行搬运 `hidden` 字节,**经典方案里三次 transpose+拷贝彻底消失**。
  fp8 权重按行切分是精确的,因为 scale 是 per-tensor。
- `world_size ≥ 7`:走经典 all-to-all(每行 `3 × inner / world` 字节)。

对 H3(`hidden` 5376、`inner` 7168),两者在 world 4 时字节相等,小于 4 时
all_gather 更优 —— 480p PCIe 实测:每 block 通信 6.04ms → 2.34ms,达成带宽从
24.9 升到 32.2 GB/s(路径上不再有 permute)。两处传输都与周边计算重叠:gather 拆成
异步子传输(`MINIMAX_SP_AG_CHUNKS`,默认 4),每块的投影与下一块的传输并行;
attention 按 head 分块(`MINIMAX_SP_ATTN_CHUNKS`,默认 4),每块的输出交换与下一块的
attention 并行。两者合计在 480p 上约省 150ms/步。都不改变任何算术,
`tests/latent_parity.py` 在 1/2/4/8 块下均报告逐比特相同。

**VAE 解码** —— `decode_temporal` 逐时间块解码,每块只读 latent 的一个切片,块间
独立。只把这层叶子计算按 rank 轮询分发,**空间 tile 的 blend、时间 blend、canvas
写入全部留在 rank0 的官方代码路径里**。worker 的 VAE 权重是首次使用时由 rank0
通过 NCCL 交接的(约 4.85 GiB,每会话一次约 2.3s),所以用的一定是你实际加载的
那份权重(含自定义路径)。

**试过但被否决的方案,免得再走一遍。** 剖析显示以下都不值得做:AdaLN 预计算
(上游 pruned 权重已把 `t_dim` 因子化到 8 维曲线基,整条分支 44M 参数、每步 0.06ms)、
文本编码器折叠(短提示实测总共只有 1.0s)、把 gloo 控制通道换成常驻 NCCL(每步 meta
广播 0.3ms)、以及把调制+norm 融合为 Triton kernel(整块调制+norm 占每步 3.4%)。
具体见提交历史。

## 许可证

MIT(见 [LICENSE](LICENSE)),可自由使用、修改、商用。

---

关键词:MiniMax H3、MiniMax-H3、ComfyUI、ComfyUI 自定义节点、多卡、多卡并行、
并行推理、序列并行、Ulysses、视频生成、音画同步、video generation、
multi-GPU、H3 加速。

# buqi-minimax-h3-multigpu

[English](README.md) | **中文** | [日本語](README_ja.md)

ComfyUI 上 **MiniMax-H3** 的多卡(序列并行)推理节点 —— 替换 `UNETLoader` 即可使用,
**无任何近似、无精度降低**。

MiniMax-H3 用一个打包 token 的 DiT 同时生成**视频+音频**。本节点按
[DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509) 方案把该序列切分到 2/4/7/8
张 GPU:每张卡对自己分到的一批注意力头做**完整精确的注意力**,数学上完全等价。

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



| 阶段 | 单卡 | SP2 | 加速 |
|---|---|---|---|
| 去噪循环 | 78.3s | 41.0s | **1.91×** |
| VAE 解码 | 5.9s | 3.5s | **1.69×** |
| 端到端 | 84.9s | 46.2s | **1.84×** |

测试环境:2× RTX PRO 5000 Blackwell(72GB,PCIe 5.0,**无 NVLink**),fp8,
832×480,20 步。

算子级剖析(`MINIMAX_SP_PROFILE_OPS=1`,每步每卡):attention+输出交换 856ms
(两者重叠故合并计时)、MLP 657ms、gather+qkv_proj 325ms、out_proj 83ms、
调制+norm 76ms、QK-norm+RoPE 27ms。
主干 GEMM 已跑到 270–370 TFLOPS(fp8),attention 约 176 TFLOPS,**算子本身已无空间**;
唯一不随卡数摊薄的就是卡间通信 —— 起初占每步约 21%(481ms),现已基本被计算掩盖。

VAE 那一行需要启用 `MiniMaxH3SPVAEDecode`(见下),其上限由时间块的划分决定:本例
共 7 块,双卡 4:3 分配,理论上限就是 1.75×。每个会话首次解码另需约 2.3s 把 VAE
权重交给 worker。

仍在单卡上跑的部分:文本编码(短提示约 1s)与视频封装。视频更长、分辨率更高时
扩展性更好 —— 去噪时间增长而这段尾巴基本不变,且 VAE 能分到更多块。

更大画幅(1080p 及以上)还会因为激活被切分而受益:单卡可能已经在做显存换页,
而 SP2 仍完全驻留。这类场景看起来"超线性"是这个原因,不是序列并行突破了自身上限。

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

H3 DiT 处理一条打包序列 `[text|cond|audio|video]`,共 56 个注意力头。Ulysses SP
让序列按行切分常驻各卡:所有逐 token 运算(patch proj、调制、RoPE、MLP)都是行内
本地计算,无需通信;只有注意力跨卡 —— 每卡最终对自己负责的 56/`world_size` 个头在
**完整序列**上做精确注意力。

让"头"和"序列"对齐有两种做法,哪种更省取决于卡数:

- **all-to-all**(经典 Ulysses):在本地行上算出全部 56 头的 Q/K/V,再把"头维"换成
  "序列维"。每行搬运 `3 × inner / world` 字节。
- **all_gather**(`world_size ≤ 4` 时启用):改为广播调制后的隐状态,然后只用本卡
  那部分 `qkv_proj` 权重行去投影。每行搬运 `hidden` 字节,且 Q/K/V 出来就已经横跨
  完整序列,**三次 transpose+拷贝彻底消失**。

对 H3(`hidden` 5376、`inner` 7168),两者在 `world_size == 4` 时字节相等,小于 4 时
all_gather 更优 —— 480p PCIe 实测:每 block 通信 6.04ms → 2.34ms,达成带宽从
24.9 升到 32.2 GB/s(因为路径上不再有 permute)。大于 4 时仍走 all-to-all。

按行切分 `qkv_proj` 是精确的而非近似:fp8 权重只带一个 per-tensor scale,所以选取
若干行之后,每个输出元素仍是原来那个点积。代价是每卡多存该权重的 `1/world`
(fp8、双卡时约 2.9GB)。

最后,两处传输都与周边计算重叠:gather 拆成若干异步子传输
(`MINIMAX_SP_AG_CHUNKS`,默认 4),使每块的投影与下一块的传输并行;attention 按
head 分块计算(`MINIMAX_SP_ATTN_CHUNKS`,默认 4),使每块的输出交换与下一块的
attention 并行。两者合计在 480p 上约省 150ms/步。都不改变任何算术 —— attention 的
head 彼此独立,投影的行彼此独立 —— `tests/latent_parity.py` 在 1/2/4/8 块下均报告
逐比特相同。

## 许可证

MIT(见 [LICENSE](LICENSE)),可自由使用、修改、商用。

---

关键词:MiniMax H3、MiniMax-H3、ComfyUI、ComfyUI 自定义节点、多卡、多卡并行、
并行推理、序列并行、Ulysses、视频生成、音画同步、video generation、
multi-GPU、H3 加速。

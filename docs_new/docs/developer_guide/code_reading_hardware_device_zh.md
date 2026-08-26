---
title: "第三方硬件接入精读（DeepSeek-V4 / MoE+MLA）"
description: "实现级中文精读：如何把新设备接入 SGLang 以跑 DeepSeek-V4 与 MoE+MLA——hardware_backend、attention 注册、量化/MoE、kernels、communicator、platform plugins，以及 mydevice 最小清单。"
keywords:
  - sglang
  - hardware backend
  - DeepSeek-V4
  - MLA
  - DSA
  - MoE
  - NPU
  - XPU
  - platform plugin
  - code reading
---

本篇是精读系列第 **6** 篇：对着源码说明「新硬件 / 第三方设备」如何挂进 SGLang，并能跑 **DeepSeek-V4**（及一般 **MoE + MLA / DSA** 模型）。路径相对仓库根。

系列导航：[总目录](./code_reading_notes_zh.md) · [第 1 篇](./code_reading_srt_core_zh.md) · [第 2 篇](./code_reading_deep_dive_zh.md) · [第 3 篇](./code_reading_notes_advanced_zh.md) · [第 4 篇](./code_reading_serving_extensions_zh.md) · [第 5 篇](./code_reading_ecosystem_zh.md) · [第 6 篇 硬件接入](./code_reading_hardware_device_zh.md)

官方操作手册（非精读）：[Plugin System](/docs/hardware-platforms/plugin)、[Ascend NPUs](/docs/hardware-platforms/ascend-npus/getting-started/installation)、[XPU](/docs/hardware-platforms/xpu)。模型语义与 V4 forward 细节另见专题：[DeepSeek-V4 实现精读](./code_reading_deepseek_v4_zh.md)。

> **心智模型：** 核心 serving 默认按 CUDA 写；设备特化用 **三层厚度** 接入——(1) `is_*()` / `current_platform` 条件分支；(2) `@register_attention_backend` / MoE runner / MultiPlatformOp；(3) OOT `setuptools` entry_points（`sglang.srt.platforms` + `sglang.srt.plugins`）。NPU 是「厚 in-tree 第三方」参考；XPU 是「中等厚度 + 部分 OOT 工厂」参考；全新 `mydevice` 应优先走 **OOT Platform Plugin**，再按 DeepSeek-V4 清单补算子。

---

## 0. 总图：新设备要插进哪些缝

```text
CLI / Engine
  load_plugins()                          # srt/plugins/__init__.py
  current_platform lazy resolve           # srt/platforms/__init__.py
       │
       ▼
ModelRunner
  device = get_device()                   # srt/utils/common.py
  use_mla_backend = (AttentionArch.MLA)
  init_memory_pool → KV pool / allocator  # mem_cache/kv_cache_configurator.py
  init_attention_backends                 # model_runner_components/attention_backend_setup.py
       │  ATTENTION_BACKENDS[name](runner)
       ▼
layers/* (MultiPlatformOp.dispatch_forward)
  forward_cuda | forward_npu | forward_xpu | forward_<oot_key>
  MoE: moe_runner/* + token_dispatcher/* + a2a
  Quant: QuantizationConfig.get_quant_method → hardware_backend/*/quantization
       │
       ▼
kernels/ (BaseFusedOp / get_kernel)       # AOT vs JIT vs torch_npu …
distributed/ GroupCoordinator             # device_communicators/*
cuda_graph_setup → GraphRunner            # NPUGraphRunner / XPUGraphRunner / OOT cls
```

DeepSeek-V4 额外路径（相对「普通 MLA」）：

```text
models/deepseek_v4.py
  → attention_backend="dsv4"
  → DeepseekV4AttnBackend (+ Compressor + C4Indexer)
  → DeepSeekV4TokenToKVPool（全量 / SWA / c4 / c128 + indexer）
  → MoE gemm（常 FP8 / MXFP8）+ EP all-to-all
```

---

## 1. `hardware_backend/` 结构与设备探测

### 1.1 目录一览

路径：`python/sglang/srt/hardware_backend/`

| 子目录 | 角色 | 典型内容 |
|---|---|---|
| `gpu/quantization/` | CUDA 量化 kernel 封装 | `awq_kernels.py`、`gptq_kernels.py` |
| `cpu/quantization/` | CPU/AMX 量化 | 同上命名 |
| `npu/` | Ascend 厚插件（最完整第三方样板） | `attention/`、`moe/`、`quantization/`、`graph_runner/`、`memory_pool_npu.py`、`allocator_npu.py`、`dsv4/`、`modules/` |
| `xpu/` | Intel XPU | `graph_runner/`、`kernels/fla/`（线性注意力） |
| `musa/` | 摩尔线程 | `attention/`、`kernels/`、`utils/patch_torch.py` |
| `mlx/` | Apple MLX「整机替换」 | `model_runner.py`、`tp_worker.py`、`kv_cache/`、`scheduler_mixin.py` |

**没有**统一的 `hardware_backend/__init__.py` 自动发现：各子树由 `is_npu()` / `is_xpu()` / registry lazy import / OOT hook 拉进来。

NPU 子树（跑 DSV4 时几乎全用到）：

```text
hardware_backend/npu/
  attention/
    ascend_backend.py              # AscendAttnBackend（MHA/MLA/DSA 通用）
    ascend_dsv4_backend.py         # DeepseekV4AscendAttnBackend
    ascend_gdn_backend.py
    ascend_hybrid_linear_attn_backend.py
    mla_preprocess.py
  dsv4/
    dsv4_memory_pool.py            # NPUDeepSeekV4TokenToKVPool …
    dsv4_allocator.py
    dsv4_rope.py / dsv4_common_hooks.py / dsv4_req_to_token_pool.py
  moe/                             # init_routing, matmul, topk, fuseep, …
  quantization/                    # linear_method_npu, moe_methods, …
  graph_runner/                    # NPUGraphRunner 等
  memory_pool_npu.py               # NPUMHATokenToKVPool, NPUMLATokenToKVPool
  allocator_npu.py                 # NPUPagedTokenToKVPoolAllocator
```

### 1.2 设备如何被探测

主入口：`python/sglang/srt/utils/common.py`（硬件探测段注释写明：只放 detect / capability / device select）。

| 函数 | 判定逻辑（摘要） |
|---|---|
| `is_cuda()` | `torch.cuda.is_available() and torch.version.cuda is not None` |
| `is_hip()` | `torch.version.hip is not None` |
| `is_cuda_alike()` | CUDA 或 HIP |
| `is_xpu()` | `hasattr(torch, "xpu") and torch.xpu.is_available()` |
| `is_npu()` | 有 `torch.npu` 且 `is_available()`；**不可见则抛 RuntimeError** |
| `is_cpu()` | `SGLANG_USE_CPU_ENGINE=1` **且** 主机为 x86/arm64 |
| `is_musa()` | 能 `import torchada` 且 `torch.version.musa` |
| `is_mps()` | `torch.backends.mps.is_available()` |
| `get_device(device_id=None)` | 优先级：CPU 引擎 → cuda → xpu → npu → hpu → musa → mps → `current_platform.get_device` |

`get_device()` 签名：

```python
@lru_cache(maxsize=8)
def get_device(device_id: Optional[int] = None) -> str: ...
# 返回 "cuda" / "cuda:0" / "npu" / "xpu" / "cpu" / ...
```

平台单例：`from sglang.srt.platforms import current_platform`（见 §6）。`ModelRunner` 里：

```python
self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA
# 见 model_executor/model_runner.py
```

**给 `mydevice` 的含义：**

1. 短期：在 `common.py` 加 `is_mydevice()` + `get_device()` 分支（in-tree 路径，像 NPU）。
2. 推荐：实现 OOT `SRTPlatform`，让 `get_device()` 落到 `current_platform.get_device()`；探测放在 `activate()`。

---

## 2. Attention backend 注册与 MLA/DSA 接口

### 2.1 注册表

文件：`python/sglang/srt/layers/attention/attention_registry.py`

```python
ATTENTION_BACKENDS = {}

def register_attention_backend(name):
    def decorator(fn):
        ATTENTION_BACKENDS[name] = fn
        return fn
    return decorator
```

工厂签名约定：`create_*(runner: ModelRunner) -> AttentionBackend`。

装配：`model_runner_components/attention_backend_setup.py`

```python
def build_attention_backends(*, model_runner: ModelRunner) -> AttentionBackends: ...
# 最终：
return ATTENTION_BACKENDS[backend_str](model_runner)
# 再经 attn_backend_wrapper(runner, backend) 包 hybrid GDN/Mamba 等
```

CLI：`--attention-backend` / `--prefill-attention-backend` / `--decode-attention-backend`。

### 2.2 已注册名字（与硬件相关摘录）

| name | 工厂 | 设备备注 |
|---|---|---|
| `flashinfer` / `fa3` / `fa4` / `triton` / … | CUDA 主流 | MLA 时 flashinfer 切 `FlashInferMLAAttnBackend` |
| `ascend` | `AscendAttnBackend` | NPU |
| `dsv4` | NPU→`DeepseekV4AscendAttnBackend`；HIP→`DeepseekV4HipRadixBackend`；else→`DeepseekV4AttnBackend` | DeepSeek-V4 |
| `dsa`（`nsa` 弃用别名） | `DeepseekSparseAttnBackend` | DeepSeek V3.2 稀疏 |
| `intel_xpu` | `XPUAttentionBackend` | XPU |
| `intel_amx` | `IntelAMXAttnBackend` | CPU AMX |
| `aiter` / `wave` | ROCm 生态 | HIP |
| MLA 专用 | `trtllm_mla`、`flashmla`、`cutlass_mla`、`cutedsl_mla`、`tokenspeed_mla` | 要求 `runner.use_mla_backend` |

`dsv4` 工厂核心分支（同文件）：

```python
@register_attention_backend("dsv4")
def create_dsv4_backend(runner):
    if _is_npu:
        from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
            DeepseekV4AscendAttnBackend,
        )
        return DeepseekV4AscendAttnBackend(runner)
    elif _is_hip:
        ...
    else:
        from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
        return DeepseekV4AttnBackend(runner)
```

### 2.3 `AttentionBackend` 必须实现的接口

文件：`python/sglang/srt/layers/attention/base_attn_backend.py`

```python
class AttentionBackend(ABC):
    prefill_attention_backend_str: Optional[str] = None
    decode_attention_backend_str: Optional[str] = None
    supports_ragged_verify_graph: bool = False
    needs_cpu_seq_lens: bool = True
    use_captured_forward_metadata_for_breakable_cuda_graph: bool = False
    supports_full_cuda_graph_chunked_prefix: bool = False

    def init_forward_metadata(self, forward_batch: ForwardBatch): ...
    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False): ...
    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch): ...

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int): ...
    def get_cuda_graph_seq_len_fill_value(self): ...

    def forward(self, q, k, v, layer: RadixAttention, forward_batch, save_kv_cache=True, **kwargs): ...
    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs): ...
    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs): ...
    def forward_mixed(self, ...): ...   # NPU mixed batch 会走这里

    def get_indexer_metadata(self, layer_id: int, forward_batch) -> Optional[BaseIndexerMetadata]:
        """None = 不支持 DSA indexer。"""
```

**Graph 契约（重要）：**

- 旧 `init_forward_metadata_capture/replay_cuda_graph` 已从 ABC 删除；必须迁到 `init_forward_metadata_out_graph(fb, in_capture)` + `init_forward_metadata_in_graph(fb)`。
- `in_graph` 禁止 `.item()` / `.cpu()` / 动态 `torch.empty`（不可录进 device graph）。

### 2.4 MLA vs DSA vs DSV4 对 backend 的额外要求

| 架构 | 模型侧 | Backend 要点 |
|---|---|---|
| **MLA** | `AttentionArch.MLA`；`RadixAttention` 写 compressed KV | `set_mla_kv_buffer` / 单流 `kv_buffer`；decode/extend 走 MLA kernel |
| **DSA**（V3.2） | sparse top-k + indexer | `get_indexer_metadata`；池类型 `DSATokenToKVPool`；`--attention-backend dsa` |
| **DSV4** | `DeepseekV4AttnBackend` = `AttentionBackend` + `C4IndexerBackendMixin` + `CompressorBackendMixin` | 多池（full/SWA/c4/c128）、compressor、C4 indexer、稀疏 prefill/decode |

CUDA 参考实现：

- `layers/attention/deepseek_v4_backend.py` → `class DeepseekV4AttnBackend(...)`
- Mixins：`layers/attention/dsv4/compressor.py`、`indexer.py`、`metadata.py`
- Kernel：`kernels/ops/attention/dsv4/*`

NPU 参考：`hardware_backend/npu/attention/ascend_dsv4_backend.py` → `DeepseekV4AscendAttnBackend`，复用 Ascend sparse attn + NPU fused compressor。

**新设备最小 MLA backend：** 注册名字 → 实现 `forward_decode`/`forward_extend` + graph metadata + 对接 `MLATokenToKVPool`（或设备布局子类）。

**跑 DSV4：** 额外实现/移植 compressor + indexer + 多级 KV pool（见 §8）。

---

## 3. 量化 / MoE 设备钩子

### 3.1 量化：`get_quant_method` + hardware_backend

模式：`QuantizationConfig.get_quant_method(layer, prefix)` 内 `if is_npu(): lazy import hardware_backend.npu...`。

例：

- `layers/quantization/npu_mxfp4.py` / `npu_mxfp4_w4a4.py` → NPU linear method
- `layers/quantization/unquant.py` → NPU 时走 `hardware_backend.npu.quantization.moe_methods`
- GPU/CPU：`hardware_backend/{gpu,cpu}/quantization/{awq,gptq}_kernels.py`

NPU MoE 量化方法类（`hardware_backend/npu/quantization/moe_methods.py`）：

- `NPUW8A8Int8MoEMethod` / `NPUW4A8Int8MoEMethod` / `NPUMXFP8MoEMethod`
- 基类：`layers/quantization/base_config.FusedMoEMethodBase`
- 底层：`npu/moe/matmul.py`（`GroupedMatmul`、`GroupedMatmulSwigluQuant`）、`npu/moe/quant.py`（`HiddenStatesDynamicQuant`）

OOT 平台也可覆盖：

```python
# srt/platforms/interface.py
class SRTPlatform:
    supported_quantization: list[str] = []
    def get_quantization_config(self, quantization: str) -> Optional[Type[QuantizationConfig]]:
        return None  # None = 用默认；或返回设备专用 Config
    def supports_fp8(self) -> bool:
        return False
```

### 3.2 MoE runner / A2A

枚举：`layers/moe/utils.py`

```python
class MoeA2ABackend(Enum):
    NONE = "none"
    DEEPEP = "deepep"
    ...
    ASCEND_FUSEEP = "ascend_fuseep"
    ASCEND_TP = "ascend_tp"

class MoeRunnerBackend(Enum):
    TRITON = "triton"
    DEEP_GEMM = "deep_gemm"
    ASCEND = "ascend"
    ...
```

NPU runner：`layers/moe/moe_runner/ascend.py`

- `AscendRunnerCore` + `AscendRunnerInput` / `AscendRunnerOutput`
- `@register_pre_permute` / `@register_post_permute`（见 `moe_runner/base.py`）对接 DeepEP / AscendTP dispatcher

Token dispatcher：

- `layers/moe/token_dispatcher/deepep.py` — CUDA EP 主流
- `layers/moe/token_dispatcher/ascend_tp.py` — NPU TP 侧

`fused_moe_triton/layer.py` 里大量 `if _is_npu:` / `a2a_backend.is_none() and is_npu()` 分支；新设备通常需：

1. 新 `MoeRunnerBackend` 或复用 triton/deep_gemm（若能跑）
2. 新 A2A 或走 `torch.distributed` all-to-all
3. 在 `FusedMoE` / runner 选择处加 `is_mydevice()` 或 OOT hook

### 3.3 `MultiPlatformOp`（层内算子分发）

文件：`layers/utils/multi_platform.py`

```python
class MultiPlatformOp(nn.Module):
    _oot_forward_registry: ClassVar[dict[str, dict[type, Callable]]] = {}

    @classmethod
    def register_oot_forward(cls, op_cls: type, fn: Callable, platform_key: str): ...

    def forward_native / forward_cuda / forward_npu / forward_hip / forward_xpu / forward_musa / forward_cpu / forward_hpu

    def dispatch_forward(self):
        if current_platform.is_out_of_tree():
            key = current_platform.get_dispatch_key_name()
            # 1) OOT registry  2) forward_<key>  3) forward_native
        elif _is_cuda: return self.forward_cuda
        elif _is_hip:  return self.forward_hip
        ...
```

`SRTPlatform.get_dispatch_key_name()` 默认 `"native"`；OOT 应返回 `"mydevice"`，并实现 `forward_mydevice` 或 `register_oot_forward`。

---

## 4. Kernel 选择：`BaseFusedOp`、AOT vs JIT、强制 backend

### 4.1 两套机制

| 机制 | 路径 | 行为 |
|---|---|---|
| **统一 registry** | `kernels/selector.py` → `get_kernel(op, backend=None)` | 按平台硬过滤；多候选必须显式 `backend=` |
| **FusedOp 自动择优** | `kernels/fused_op.py` → `BaseFusedOp` | 按 `priority` + `CapabilityRequirement` 选第一个合格 |

### 4.2 `KernelBackend` / `DeviceType`

`kernels/spec.py`：

```python
class KernelBackend(str, Enum):
    TORCH = "torch"
    TORCH_COMPILE = "torch_compile"
    TRITON = "triton"
    JIT = "jit"          # sglang.kernels.jit（nvcc/hipcc）
    AOT = "aot"          # sgl_kernel wheel
    CUTE_DSL = "cute_dsl"
    FLASHINFER = "flashinfer"
    DEEPGEMM = "deepgemm"
    AITER = "aiter"      # HIP
    TORCH_NPU = "torch_npu"

class DeviceType(str, Enum):
    CUDA = "cuda"
    HIP = "hip"
    NPU = "npu"
    CPU = "cpu"
    # TODO: XPU / MUSA …
```

`PlatformInfo.detect()`：HIP → NPU → CUDA(+arch) → CPU。

### 4.3 `BaseFusedOp` 契约

```python
class BaseFusedOp(ABC):
    op: ClassVar[str]                           # e.g. "layernorm.rmsnorm"
    priority: ClassVar[Tuple[KernelBackend, ...]] = DEFAULT_PRIORITY
    capabilities: ClassVar[Mapping[KernelBackend, AbstractSet[CapabilityRequirement]]] = {}

    def forward_native(self, *args, **kwargs): ...          # 必须：正确性基准
    def forward_torch_compile / forward_triton / forward_jit / forward_aot / ...
    def forward_npu(self, *args, **kwargs): ...             # TORCH_NPU

    def forward(self, *args, backend: Optional[KernelBackend] = None, **kwargs): ...
```

默认优先级：`AOT → JIT → FLASHINFER → DEEPGEMM → CUTE_DSL → AITER → TORCH_NPU → TRITON → TORCH`。

强制全局后端：

```python
# environ: SGLANG_FORCE_FUSED_OP_BACKEND
get_fused_op_backend() -> Optional[KernelBackend]
set_fused_op_backend(backend: Optional[KernelBackend]) -> None
```

**新设备：**

1. 扩展 `DeviceType` + `PlatformInfo.detect` + `CapabilityRequirement`（若进统一 kernels）。
2. 或暂时只在 `hardware_backend/mydevice/` 调厂商库，不进 `BaseFusedOp`。
3. 调试：`SGLANG_FORCE_FUSED_OP_BACKEND=torch` 对齐数值。

---

## 5. Communicator / 分布式

### 5.1 目录

`python/sglang/srt/distributed/device_communicators/`

| 文件 | 类 | 用途 |
|---|---|---|
| `npu_communicator.py` | `NpuCommunicator` | HCCL all_reduce / all_gather；`quant_all_reduce`（int8 gather + 反量化 reduce） |
| `xpu_communicator.py` | `XpuCommunicator` | XCCL；`gather` 用 all_gather 规避 Ray 问题 |
| `hpu_communicator.py` | `HpuCommunicator` | Habana |
| `pynccl.py` / `custom_all_reduce*.py` / `quick_all_reduce.py` | CUDA/HIP 加速 AR | |
| `shm_broadcast.py` | CPU MQ | |

### 5.2 挂载点：`GroupCoordinator`

`distributed/parallel_state.py` 构造时：

```python
self.npu_communicator = NpuCommunicator(group=self.device_group)  # if use_npu_communicator
self.xpu_communicator = XpuCommunicator(group=self.device_group)  # if use_xpu_communicator
```

`all_reduce` 分派顺序（摘要）：CPU shm → HPU → XPU → **NPU** → pynccl/custom/mscclpp/…。

`NpuCommunicator` 接口：

```python
class NpuCommunicator:
    def __init__(self, group: ProcessGroup): ...
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...
    def quant_all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...
    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor: ...
```

### 5.3 分布式 backend 字符串

`platforms/device_mixin.py`：

```python
_DEVICE_TO_DISTRIBUTED_BACKEND = {
    "cuda": "nccl",
    "xpu": "xccl",
    "hpu": "hccl",
    "cpu": "gloo",
    "npu": "hccl",   # 或 zbal（env）
    "musa": "mccl",
}

def get_torch_distributed_backend_str(self) -> str:
    return _DEVICE_TO_DISTRIBUTED_BACKEND.get(self.device_type, "gloo")

def get_communicator_class(self) -> type | None:  # [Planned]
    return None
```

**EP all-to-all：** 不在 `device_communicators/`，而在 `layers/moe/token_dispatcher/*`（DeepEP / Ascend fuseep / flashinfer a2a）。新设备做 EP 时通常新增 dispatcher + 注册 fused pre/post permute。

---

## 6. Platform plugins / entry_points

### 6.1 两组 entry points

`srt/plugins/__init__.py`：

```python
PLATFORM_PLUGINS_GROUP = "sglang.srt.platforms"   # 硬件平台
GENERAL_PLUGINS_GROUP = "sglang.srt.plugins"      # HookRegistry 钩子
```

环境变量（`srt/environ.py`）：

- `SGLANG_PLATFORM` — 选定平台插件名
- `SGLANG_PLUGINS` — 通用插件白名单（逗号分隔）

`load_plugins()` 调用点：`cli/serve.py`、`launch_server.py`、`entrypoints/engine.py`、`managers/scheduler.py`（每个进程都要，idempotent）。

### 6.2 Platform 发现

`srt/platforms/__init__.py` → `_resolve_platform()`：

1. `SGLANG_PLATFORM` 已设：只 `ep.load()` 该插件的 `activate()`
2. 未设：激活全部；0 个则 fallback CPU/CUDA/ROCm/XPU/`SRTPlatform`；多个则报错要求设置 `SGLANG_PLATFORM`

`activate()` 返回 **全限定类名字符串** 或 `None`（硬件不可用）。

### 6.3 `SRTPlatform` 工厂（OOT 必看）

`srt/platforms/interface.py`：

```python
class SRTPlatform(DeviceMixin):
    supported_quantization: list[str] = []

    def apply_server_args_defaults(self, server_args) -> None: ...
    def get_default_attention_backend(self) -> str: ...
    def get_graph_runner_cls(self) -> type: ...
    def get_mha_kv_pool_cls(self) -> type: ...
    def get_mla_kv_pool_cls(self) -> type: ...
    def get_dsa_kv_pool_cls(self) -> type: ...
    def get_paged_allocator_cls(self) -> type: ...
    def get_compile_backend(self, mode: str | None = None) -> str: ...
    def get_piecewise_backend_cls(self) -> type: ...
    def get_quantization_config(self, quantization: str) -> Optional[Type[QuantizationConfig]]: ...
    def supports_fp8(self) -> bool: ...
    def support_cuda_graph(self) -> bool: ...
    def support_piecewise_cuda_graph(self) -> bool: ...
    def init_backend(self) -> None: ...
    def get_dispatch_key_name(self) -> str: ...  # MultiPlatformOp
```

In-tree 例：`platforms/xpu.py` → `XpuSRTPlatform`；`platforms/cuda.py` → `CudaSRTPlatform`。NPU **尚未**完全迁到 `platforms/npu.py`，仍散落在 `is_npu()` + `hardware_backend/npu/`。

### 6.4 General plugin hooks

`srt/plugins/hook_registry.py`：

```python
class HookType(Enum):
    BEFORE / AFTER / AROUND / REPLACE

class HookRegistry:
    @classmethod
    def register(cls, target: str, hook: Callable, hook_type: HookType = HookType.AFTER, *, source=None): ...
    # target 例: "sglang.srt.managers.scheduler.Scheduler.schedule"
```

可用于：替换 `ModelRunner` 某方法、注入 NPU 式 memory pool 选择、不改主仓 patch。

完整脚手架见官方 [Plugin System](/docs/hardware-platforms/plugin)。

---

## 7. 参考模式：NPU（厚）与 XPU（中）

### 7.1 NPU = in-tree 第三方样板

接入手法组合：

| 层 | NPU 做法 |
|---|---|
| 探测 | `is_npu()` |
| Attention | `@register_attention_backend("ascend"|"dsv4")` |
| KV | `NPUMHATokenToKVPool` / `NPUMLATokenToKVPool`；DSV4 → `NPUDeepSeekV4*`（`dsv4_memory_pool.py`） |
| Allocator | `NPUPagedTokenToKVPoolAllocator` |
| MoE | `MoeRunnerBackend.ASCEND` + `ascend_fuseep` / `ascend_tp` |
| Quant | `hardware_backend/npu/quantization/*` |
| Graph | `NPUGraphRunner`（`cuda_graph_setup.py` 按 `device=="npu"`） |
| Compile | `NPUPiecewiseBackend`（`compilation/backend.py`） |
| Dist | `NpuCommunicator` + HCCL |
| 包 | `pyproject_npu.toml`；依赖 `torch_npu` / `sgl_kernel_npu` |

DSV4 上 NPU 特化要点（`dsv4_memory_pool.py` 文档串）：

- CUDA 用 ring `CompressStatePool`；Atlas 要求 paged `cache_mode=1` → `NPUCompressStatePool`
- KV layout：`npu_sparse_attn_sharedkv` 要 PA_ND `(num_pages, kernel_page_size, 1, dim)` bf16
- 由 `ModelRunnerKVCacheMixin._init_pools` 在「DSV4 **且** NPU」时切换

### 7.2 XPU = 较薄平台

| 层 | XPU 做法 |
|---|---|
| Platform | in-tree `XpuSRTPlatform`（`platforms/xpu.py`） |
| Attention | `intel_xpu` → `layers/attention/xpu_backend.py`（多基于 FlashAttention 路径） |
| Graph | `XPUGraphRunner` / `xpu_full_graph_backend.py` |
| FLA kernels | `hardware_backend/xpu/kernels/fla/*` |
| Dist | `XpuCommunicator` + xccl |
| FP8 | `supports_fp8() -> False`（当前） |

### 7.3 MLX = 整机替换（极端）

`hardware_backend/mlx/{model_runner,tp_worker,scheduler_mixin,kv_cache}`：不复用 CUDA `ModelRunner` 热路径。新设备一般不必走到这层，除非 runtime 与 PyTorch 设备模型差异极大。

### 7.4 推荐对照学习顺序

1. 读官方 `docs/hardware-platforms/plugin.mdx` OOT 最小包  
2. 跟一次 `ascend` backend + `memory_pool_npu` 的条件导入链  
3. 跟 `dsv4` 在 CUDA vs NPU 的分流（registry + pool）  
4. 跟 `GroupCoordinator.all_reduce` 的 NPU/XPU 分支  
5. 跟 `cuda_graph_setup` / `make_backend` 的设备映射  

---

## 8. DeepSeek-V4 对新设备的具体需求

模型入口：`python/sglang/srt/models/deepseek_v4.py`（及 `deepseek_v4_nextn.py` / `deepseek_v4_dspark.py`）。

### 8.1 Attention：`dsv4` + MLA 稀疏压缩

| 能力 | CUDA 落点 | 新设备要交付 |
|---|---|---|
| 主 attn backend | `DeepseekV4AttnBackend` | 注册 `dsv4` 或设备名，实现 decode/extend + graph |
| C4 indexer | `C4IndexerBackendMixin` + `kernels/ops/attention/dsv4` | top-k / plan / index buffer 读写 |
| Compressor（c4/c128） | `CompressorBackendMixin` | fused 或 Python 压缩；NPU：`torch.ops.custom.compressor` |
| Sparse MLA kernel | FlashMLA / dsv4 sparse kernels | 厂商 sparse MLA 或等价实现 |
| RoPE + store | `fused_k_norm_rope_flashmla` / NPU `Dsv4NpuRoPE` | 与 KV layout 一致的写 cache |

### 8.2 DSA（若配置走稀疏索引路径）

V3.2 DSA 与 V4 indexer 共享思想：`DSATokenToKVPool` + `get_indexer_metadata`。

```python
class DSATokenToKVPool(MLATokenToKVPool):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8
    # index buffer: page 内 fp8 data + fp32 scale 打包
    # CUDA 要求 page_size == 64
```

新设备若只跑「稠密 MLA」可暂缓；跑官方 DSV4 权重则 **indexer + compressed KV 几乎不可省**。

### 8.3 MoE GEMM

- 路由：`topk` / hash_topk（`layers/moe/hash_topk.py`，NPU/HIP 有分支）
- Expert GEMM：DeepGemm / Triton / Ascend `GroupedMatmul*`
- 激活：SiLU+Mul；常与 FP8 后量化融合（`silu_and_mul_*_post_quant`，`kernels/ops/attention/dsv4`）

新设备最低：正确的 grouped GEMM + swiglu；有 EP 时再加 dispatch/combine。

### 8.4 FP8

贯穿：

- 权重 / 激活 FP8（`fp8_kernel`、`deep_gemm.fp8_einsum`、wo_a 路径）
- KV：DSA/DSV4 常 **nope FP8 + rope BF16** 打包（见 `NopeFp8RopeBf16Pack`、`DeepSeekV4SingleKVPool`）
- `SRTPlatform.supports_fp8()`；NPU MXFP8 需 `float8_e8m0fnu`（`moe_methods._require_e8m0_dtype`）

无硬件 FP8 时需：软件模拟（慢）或强制 BF16 权重路径（可能与官方 checkpoint 不兼容）。

### 8.5 通信：TP all-reduce + EP all-to-all

| 并行 | 需求 |
|---|---|
| TP | `GroupCoordinator.all_reduce` → 设备 communicator；graph capture 下需可录或走 custom op（参见 XPU inplace 路径） |
| EP | MoE A2A：DeepEP 或自研；注册 `register_pre_permute` / `register_post_permute` / fused func |
| DP-attn / CP | `layers/communicator*.py`、`layers/cp/*`；DeepSeek 系列有 DSA CP |

### 8.6 CUDA Graph 等价物

装配：`model_runner_components/cuda_graph_setup.py`

```python
if current_platform.is_out_of_tree():
    GraphRunnerCls = current_platform.get_graph_runner_cls()
else:
    graph_runners = { "npu": NPUGraphRunner, "xpu": XPUGraphRunner, ... }
```

Backend 必须实现 `init_cuda_graph_state`、`get_cuda_graph_seq_len_fill_value`、out/in graph metadata。无 graph 则 `support_cuda_graph() -> False` 并关 decode capture（吞吐会差）。

Piecewise：`compilation/backend.py::make_backend` → OOT `get_piecewise_backend_cls()`。

### 8.7 KV pool dtype / layout

| 池 | 文件 | Layout 要点 |
|---|---|---|
| MLA 通用 | `mem_cache/memory_pool.py` → `MLATokenToKVPool` | `[size+page, 1, kv_lora_rank+qk_rope]`；`set_mla_kv_buffer` |
| DSA | `DSATokenToKVPool` | 上 + `index_k_with_scale_buffer` uint8 打包；page=64 |
| DSV4 | `mem_cache/deepseek_v4_memory_pool.py` → `DeepSeekV4TokenToKVPool` | full / SWA / c4 / c128 + indexer + compress state |
| NPU MLA | `NPUMLATokenToKVPool` | Ascend 连续/分页布局，利于传输后端 |
| NPU DSV4 | `NPUDeepSeekV4SingleKVPool` 等 | PA_ND bf16；paged compress state |

选型中枢：`mem_cache/kv_cache_configurator.py`（NPU 分支显式 `NPUMHA*` / `NPUMLA*`）；OOT 应实现 `get_mla_kv_pool_cls` / `get_dsa_kv_pool_cls` 并保证 core 已调用这些工厂（部分仍 `[Planned]`，可能需 temporary hook）。

---

## 9. `mydevice` 最小实现清单

假设设备类型字符串 `"mydevice"`，包名 `my_platform_plugin`。

### 9.1 OOT 包骨架（推荐）

```text
my_platform_plugin/
  pyproject.toml
  my_platform_plugin/
    __init__.py          # activate()
    device.py            # MyDeviceMixin
    platform.py          # MySRTPlatform
    attention/
      mydevice_backend.py
      mydevice_dsv4_backend.py   # 若跑 V4
    memory/
      mla_pool.py
      dsv4_pool.py
      allocator.py
    moe/
      runner.py
      dispatcher.py
    quant/
      moe_methods.py
    graph/
      graph_runner.py
    distributed/
      communicator.py
    hooks.py             # 可选 general plugin
```

`pyproject.toml`：

```toml
[project.entry-points."sglang.srt.platforms"]
mydevice = "my_platform_plugin:activate"

[project.entry-points."sglang.srt.plugins"]
mydevice_hooks = "my_platform_plugin.hooks:register"
```

```python
# __init__.py
def activate():
    if not _mydevice_available():
        return None
    return "my_platform_plugin.platform.MySRTPlatform"
```

### 9.2 类与方法清单

| # | 类 / 符号 | 关键方法 / 签名 | 备注 |
|---|---|---|---|
| 1 | `MyDeviceMixin(DeviceMixin)` | `_enum=PlatformEnum.OOT`；`device_name/device_type="mydevice"`；`get_device_total_memory`；`get_torch_distributed_backend_str` | |
| 2 | `MySRTPlatform(SRTPlatform, MyDeviceMixin)` | §6.3 全部工厂 + `get_dispatch_key_name→"mydevice"`；`supports_fp8`；`support_cuda_graph` | |
| 3 | `@register_attention_backend("mydevice")` 或复用 `dsv4` 分支 | `create(runner)->AttentionBackend` | 可在 hooks 里 import 触发注册 |
| 4 | `MyDeviceAttnBackend(AttentionBackend)` | `forward_decode` / `forward_extend`；`init_cuda_graph_state`；`init_forward_metadata_*` | MLA 最低集 |
| 5 | `MyDeviceDsv4Backend` | + compressor/indexer mixins 或自研 | DeepSeek-V4 |
| 6 | `MyMLATokenToKVPool` / `MyDSATokenToKVPool` / `MyDeepSeekV4TokenToKVPool` | `_create_buffers`；`set_mla_kv_buffer`；indexer accessors | layout 对齐 kernel |
| 7 | `MyPagedAllocator` | 对齐 `TokenToKVPoolAllocator` / NPUPaged* | |
| 8 | `MyDeviceCommunicator` | `all_reduce`；`all_gather`；可选 quant AR | 挂到 GroupCoordinator（hook 或上游 PR） |
| 9 | MoE：`MyMoeRunnerCore` | `MoeRunnerBackend` 扩展或 hook；EP dispatcher | |
| 10 | Quant：`MyMoEMethod(FusedMoEMethodBase)` | `create_weights` / `apply` | FP8 必需则实现 |
| 11 | `MyGraphRunner` | 对标 `NPUGraphRunner` / `DecodeCudaGraphRunner` | `get_graph_runner_cls` |
| 12 | `MyPiecewiseBackend` | 对标 `NPUPiecewiseBackend` | 可选 |
| 13 | `MultiPlatformOp` | `forward_mydevice` 或 `register_oot_forward` | RMSNorm/RoPE/Linear 等 |
| 14 | （可选）`kernels` | `DeviceType.MYDEVICE` + `forward_*` / `CapabilityRequirement` | 长期 |

### 9.3 主仓可能仍需的最小补丁（若 OOT 接口未覆盖）

当前仍有硬编码 `is_npu()` / `device=="npu"` 之处；OOT 未完全替代前，常见补丁点：

1. `utils/common.py` — `get_device()` / `is_mydevice()`（若不用 platform fallback）
2. `distributed/parallel_state.py` — 构造 `mydevice_communicator` + `all_reduce` 分支
3. `device_mixin._DEVICE_TO_DISTRIBUTED_BACKEND["mydevice"] = "nccl_like"`
4. `attention_registry.create_dsv4_backend` — `elif is_mydevice(): ...`
5. `mem_cache/kv_cache_configurator.py` — 选 pool 类
6. `cuda_graph_setup.py` — 若未走 `is_out_of_tree()` 工厂

优先把逻辑放进 plugin + `HookRegistry.REPLACE`，减少主仓分叉；稳定后再上游化。

### 9.4 冒烟验收顺序

1. `SGLANG_PLATFORM=mydevice` → `current_platform` 打印正确  
2. 小 dense 模型 + `--attention-backend mydevice`（或 torch_native）正确性  
3. MLA 模型（DeepSeek-V2/V3 类）+ MLA pool + graph off  
4. MoE + TP all-reduce；再开 EP  
5. FP8 权重加载与一层 gemm 数值  
6. `--attention-backend dsv4` + DeepSeek-V4：compressor/indexer/KV 布局  
7. 打开 device graph / piecewise，对齐 greedy decode  

---

## 10. 关键文件索引（速查）

| 主题 | 路径 |
|---|---|
| 设备探测 | `python/sglang/srt/utils/common.py` |
| Platform | `python/sglang/srt/platforms/{__init__,interface,device_mixin,xpu,cuda}.py` |
| Plugins | `python/sglang/srt/plugins/{__init__,hook_registry}.py` |
| Attn 注册 | `python/sglang/srt/layers/attention/attention_registry.py` |
| Attn ABC | `python/sglang/srt/layers/attention/base_attn_backend.py` |
| DSV4 CUDA | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| DSV4 NPU | `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py` |
| DSA | `python/sglang/srt/layers/attention/dsa_backend.py` |
| KV MLA/DSA | `python/sglang/srt/mem_cache/memory_pool.py` |
| KV DSV4 | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| KV 选型 | `python/sglang/srt/mem_cache/kv_cache_configurator.py` |
| MultiPlatformOp | `python/sglang/srt/layers/utils/multi_platform.py` |
| MoE Ascend | `python/sglang/srt/layers/moe/moe_runner/ascend.py` |
| FusedOp | `python/sglang/kernels/fused_op.py` |
| Kernel spec | `python/sglang/kernels/spec.py` |
| Communicators | `python/sglang/srt/distributed/device_communicators/` |
| Graph 装配 | `python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py` |
| Compile | `python/sglang/srt/compilation/backend.py` |
| 模型 | `python/sglang/srt/models/deepseek_v4.py` |
| 官方插件文档 | `docs_new/docs/hardware-platforms/plugin.mdx` |

---

## 阅读顺序建议

1. 本文 §0 总图 + §6 Platform（建立 OOT 心智）  
2. §1–2 Attention 注册与 ABC（先能跑 MHA）  
3. §7 NPU 对照 + `ascend_backend.py` 开头  
4. §3 MoE/Quant + §5 Communicator  
5. §8 DeepSeek-V4 清单 → `deepseek_v4_backend.py` / `dsv4_memory_pool.py`  
6. §9 按表打勾实现 `mydevice`  

系列导航：[总目录](./code_reading_notes_zh.md) · [第 4 篇 Hardware 小节](./code_reading_serving_extensions_zh.md) · [第 5 篇](./code_reading_ecosystem_zh.md)

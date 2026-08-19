# 06 · Attention Backend 抽象与开发

> 本篇解析 `layers/attention/`:后端基类契约、注册与选择机制、元数据初始化三段式,
> 以及「为新硬件写一个 attention backend」的完整步骤。这是第三方设备适配**必做**的一件事。

---

## 1. 为什么 Attention 要抽象

Attention 占推理算力大头,且形态多样:

- 算法:MHA / MLA / SWA / 稀疏(DSA);
- 场景:prefill(变长、计算密集)vs decode(单 token、访存密集)vs 投机解码(树状 mask);
- 硬件:每个芯片有自己的高性能 kernel(flashinfer、FA3、AIter、Ascend FA、Triton…);
- 执行:eager vs CUDA Graph(要求静态形状、无 host 同步)。

SGLang 把这些差异收敛到一个接口 `AttentionBackend` 后面:模型层只见 `RadixAttention`,运行时按配置挑选后端实例。

---

## 2. 基类契约(`base_attn_backend.py`)

```19:40:python/sglang/srt/layers/attention/base_attn_backend.py
class AttentionBackend(ABC):
    """The base class of attention backends.

    Forward-data init contract (3 methods):

      - ``init_forward_metadata(fb)`` — eager entry point. Default is a wrapper
        that calls ``_out_graph(fb)`` then ``_in_graph(fb)``. Backends may
        override to keep an independent eager body.
      - ``init_forward_metadata_out_graph(fb, in_capture=False)`` — per-iter
        metadata prep, runs outside ``with graph.capture():``. Capture
        sites pass ``in_capture=True``; replay/eager use the default
        ``False``. Backends read ``in_capture`` only when capture / replay
        bodies diverge.
      - ``init_forward_metadata_in_graph(fb)`` — graph-recordable static-shape
        GPU op, runs inside ``with graph.capture():`` at capture time and
        is auto-replayed by ``graph.replay()``. Default is no-op.
    """
```

### 2.1 元数据初始化三段式(核心契约)

每个前向步开始前,runner 调后端的 init 方法准备元数据(cu_seqlens、page table、slot mapping 等):

| 方法 | 时机 | 放什么 |
|---|---|---|
| `init_forward_metadata_out_graph(fb, in_capture)` | capture 前 / replay 前 / eager 时 | host 操作、动态形状、不能入图的逻辑 |
| `init_forward_metadata_in_graph(fb)` | capture 时录制,replay 自动重放 | 静态形状的纯 GPU 操作 |
| `init_forward_metadata(fb)` | eager 默认 = 上面两个依次调用 | 也可整体重写 |

> 历史包袱提示:旧接口 `init_forward_metadata_capture_cuda_graph` / `..._replay_cuda_graph` 已废弃移除。阅读老代码或移植旧后端时注意迁移到三段式。

### 2.2 前向与图相关方法

```186:229:python/sglang/srt/layers/attention/base_attn_backend.py
    @debug_kernel_api
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run forward on an attention layer."""
        if forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(...)
        ...
        else:
            return self.forward_extend(...)
```

子类必须实现 `forward_extend` 与 `forward_decode`(NPU 混合模式另有 `forward_mixed`)。`forward` 基类实现负责按 `forward_mode` 分发。

支持 CUDA Graph 还需实现:

- `init_cuda_graph_state(max_bs, max_num_tokens)`:预分配图专用缓冲(workspace、元数据张量);
- `get_cuda_graph_seq_len_fill_value()`:padding 的 seq_len 填充值(0 或 1,取决于 kernel);
- (可选)BCG(breakable cuda graph)相关钩子、FullCG chunked-prefix 钩子——高级特性,初版可不实现。

---

## 3. 注册与选择

### 3.1 注册:`attention_registry.py`

```30:38:python/sglang/srt/layers/attention/attention_registry.py
ATTENTION_BACKENDS = {}


def register_attention_backend(name):
    def decorator(fn):
        ATTENTION_BACKENDS[name] = fn
        return fn

    return decorator
```

注册的是**工厂函数**(接收 `ModelRunner`,返回后端实例),而不是类本身——因为同一名字可能按模型类型返回不同变体(如 `flashinfer` 对 MHA/MLA 返回不同实现;`dsv4` 按设备分支)。

已注册的名字包括:`flashinfer`、`triton`、`torch_native`、`flex_attention`、`fa3`、`fa4`、`flashmla`、`cutlass_mla`、`trtllm_mla`、`trtllm_mha`、`aiter`(AMD)、`wave`(AMD)、`intel_amx`、`intel_xpu`、`ascend`(NPU)、`dsa`、`dsv4`、`hpc_ops`、`dual_chunk_flash_attn` 等。

### 3.2 选择链路

1. 用户 `--attention-backend xxx`(或分开的 `--prefill-attention-backend` / `--decode-attention-backend`);
2. 未指定时走 `ServerArgs._get_default_attn_backend()`:
   - OOT 平台 → `current_platform.get_default_attention_backend()`;
   - 否则按硬件:Hopper → `fa3`,SM100 → `trtllm_mha`/`fa4`,AMD → `aiter`,其它 → `flashinfer`/`triton`;MLA 模型另有分支;
3. 某些设备 handler 强制覆盖(如 NPU 强制 `ascend`、HPU 强制 `torch_native`);
4. 兼容层(`arg_groups/overrides.py`)做回退:无 AMX 的 CPU `intel_amx→torch_native`,无 XMX 的 XPU `intel_xpu→triton`。
5. `ModelRunner.init_attention_backend()` 调工厂建实例,并 stamp `prefill_attention_backend_str` / `decode_attention_backend_str`。

### 3.3 合法名单(与 OOT 免改主仓的注册法)

`server_args.py` 的 `ATTENTION_BACKEND_CHOICES` 是 CLI 的白名单,`--attention-backend` 会先校验名字。但**改主仓列表不是唯一途径**——`server_args.py` 提供了一组运行时扩展函数:

```363:364:python/sglang/srt/server_args.py
def add_attention_backend_choices(choices):
    ATTENTION_BACKEND_CHOICES.extend(choices)
```

同系列还有 `add_load_format_choices`、`add_quantization_method_choices`、`add_chunked_prefix_cache_attention_backend` 等。OOT 平台插件在自己的 `activate()` / hook 里调用 `add_attention_backend_choices(["mydevice"])` + `register_attention_backend("mydevice")(factory)`,即可**完全不改主仓**接入后端。

```175:203:python/sglang/srt/server_args.py
ATTENTION_BACKEND_CHOICES = [
    # Common
    "triton",
    "torch_native",
    "flex_attention",
    "dsa",
    ...
    # NVIDIA specific
    "cutlass_mla",
    "fa3",
    ...
    # AMD specific
    "aiter",
    "wave",
    # Other platforms
    "intel_amx",
    "ascend",
    "intel_xpu",
]
```

### 3.4 混合包装器

对混合模型(全注意力层 + 线性注意力/Mamba 层交替),`attn_backend_wrapper()` 会用 `HybridLinearAttnBackend` 等把两个后端包成一个(按层类型分发)。NPU 上 GDN/Mamba2 有专门变体。做适配时若目标模型是混合架构,需要关注这一层。

---

## 4. 一个后端内部长什么样(以通用结构讲)

以 `flashattention_backend.py`(FA3)为例,典型成员:

```python
class FlashAttentionBackend(AttentionBackend):
    def __init__(self, model_runner):
        self.page_size = model_runner.page_size
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        ...

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        # 组 cu_seqlens_q/k、max_seqlen、page_table(由 req_to_token 切片)
        ...

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
        # 调 varlen flash attention kernel
        ...

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        # 写 KV → paged decode kernel(page_table + cache_seqlens)
        ...

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        # 预分配 graph 专用元数据缓冲
        ...
```

**两条铁律**:

1. 写 KV 一律通过 `token_to_kv_pool.set_kv_buffer(layer, loc, k, v)`(量化池会在这里做量化转换);
2. 读 KV 一律 `get_key_buffer/get_value_buffer(layer_id)` + `req_to_token` 页表。

---

## 5. 实战:为新硬件写一个 Attention Backend

**阶段 0:先跑通 torch_native。** 任何新设备的第一步都是让 `--attention-backend torch_native` 正确出结果——它纯 PyTorch 实现,是你所有优化的数值基准。

**阶段 1:Triton 版(可选但推荐)。** 若目标支持 Triton,写一个 triton kernel 版后端,兼顾性能与可移植性。

**阶段 2:原生高性能版。** 步骤:

1. 在 `layers/attention/` 或 `hardware_backend/<device>/attention/` 新建 `my_backend.py`;
2. 实现 `AttentionBackend` 子类:
   - `__init__` 拿 `model_runner` 的池与配置;
   - `init_forward_metadata_out_graph`:组元数据(注意 in_capture 分支);
   - `forward_extend` / `forward_decode`(先支持 MHA,MLA 按需);
   - 支持 graph 时:实现 `init_cuda_graph_state` 等;不支持则平台 `support_cuda_graph()` 返回 False;
3. `register_attention_backend("my_backend")` 注册工厂;在 `attention_registry.py` 或自己的文件中 import 触发注册(OOT 插件可在 `activate()`/hook 中注册);
4. 名字加入 `ATTENTION_BACKEND_CHOICES`;
5. 平台类 `get_default_attention_backend()` 返回 `"my_backend"`;
6. 测试:
   - 与 `torch_native` 对数(logprob 级一致性);
   - `test/registered/attention/` 仿写测试;
   - prefill 长序列 + decode 大批量分别压测。

**常见坑**:

- page_size 对齐:kernel 不支持分页时要么 pad 页表,要么限制 `--page-size 1`;
- `out_cache_loc` 写 KV 要在 kernel 计算**之前或同帧**完成,否则 decode 读到旧值;
- CUDA Graph 下元数据张量**地址必须稳定**——在 `init_cuda_graph_state` 里预分配,replay 时原地更新;
- IDLE 模式(DP attention)下 batch 可能为空,`forward` 基类已短路返回空张量,自己的代码也要容忍 bs=0;
- MLA 的 `get_key_buffer` 返回的是 latent,别按 MHA 形状解释。

---

## 6. 本篇要点回顾

1. `AttentionBackend` 统一了 prefill/decode/投机/图执行的 attention 入口;模型只见 `RadixAttention`。
2. 元数据初始化三段式(out_graph / in_graph / 默认合成)是图兼容的关键契约。
3. 注册用 `register_attention_backend` 工厂 + `ATTENTION_BACKEND_CHOICES` 白名单;选择链路 = 用户指定 → 平台默认 → 兼容回退。
4. 写新后端:先 torch_native 对数,再上原生 kernel;KV 读写必须走池接口。
5. 适配新硬件时,attention backend 通常是性能收益最大、也最磨人的部分。

下一篇:[07 · 设备抽象层与第三方硬件适配](./07-设备抽象与第三方硬件适配.md)。

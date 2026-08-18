# 04 · 内存池、KV Cache 与 RadixCache

> 本篇解析 SGLang 最负盛名的部分:KV 内存如何管理、前缀如何用基数树共享、
> 以及「显存 → token 容量」的自动换算。这是性能与适配的核心战场。

---

## 1. 三个核心抽象

SGLang 的内存体系分三层,务必分清:

```
┌──────────────────────────────────────────────────────────────┐
│ RadixCache(逻辑层)                                          │
│   基数树,按 token 序列组织「哪些前缀的 KV 还在」              │
│   树节点存的不是 KV 数据,而是「槽位索引」                     │
└─────────────────────────────┬────────────────────────────────┘
                              │ 槽位索引(int)
┌─────────────────────────────▼────────────────────────────────┐
│ TokenToKVPoolAllocator(索引层)                                │
│   管理空闲槽位列表,alloc/free 返回的是「下标」                │
└─────────────────────────────┬────────────────────────────────┘
                              │ 下标 → 物理偏移
┌─────────────────────────────▼────────────────────────────────┐
│ KVCache(memory_pool,物理层)                                  │
│   真正的大块 GPU 张量,每层一对 k_buffer/v_buffer             │
│   按 [槽位, head, dim] 存放实际 K/V 数值                      │
└──────────────────────────────────────────────────────────────┘
```

另有一个正交的表:`ReqToTokenPool`,记录「请求 i 的第 j 个 token 用的是哪个槽位」。

---

## 2. 物理层:KVCache 与内存池

### 2.1 类层级(`mem_cache/memory_pool.py`)

```
KVCache (抽象基类)
├── MHATokenToKVPool        # 标准多头注意力:每层一对 k_buffer / v_buffer
│   └── 派生:FP4 / MXFP8 / PageMajor 等量化与布局变体
├── MLATokenToKVPool        # DeepSeek 系:每层单个融合 kv_buffer
│   └── DSATokenToKVPool    # DeepSeek V3.2 稀疏索引注意力
├── HybridLinearKVPool      # Mamba/线性注意力混合模型
└── …
ReqToTokenPool              # 请求 → token 槽位 的二维表
└── HybridReqToTokenPool    # 混合模型(附加 MambaPool)
```

### 2.2 MHA 池的内存布局

`MHATokenToKVPool` 为**每一层**创建两个张量(默认 NHD 布局):

```1990:2001:python/sglang/srt/mem_cache/memory_pool.py
    def _kv_buffer_shapes(self):
        """(k_shape, v_shape)"""
        if self.use_hnd:
            return (
                (self.num_pages, self.head_num, self.page_size, self.head_dim),
                (self.num_pages, self.head_num, self.page_size, self.v_head_dim),
            )
        rows = self.size + self.page_size
        return (
            (rows, self.head_num, self.head_dim),
            (rows, self.head_num, self.v_head_dim),
        )
```

- `size` = `max_total_num_tokens`(总槽位数);`+page_size` 是哨兵行,槽位 0 留给 CUDA Graph padding 的哑写入。
- `k_buffer[l]` 形状 `(size+page_size, num_kv_heads, head_dim)`;GQA 下 `num_kv_heads` 是 KV 头数(远少于 Q 头)。
- `get_key_buffer(layer_id)` / `get_value_buffer(layer_id)` 返回对应层的张量,attention backend 直接拿去做 paged attention。
- `set_kv_buffer(layer, loc, cache_k, cache_v)` 把新算出的 K/V 散射写进 `loc` 指定槽位。

> 为什么「每层一个张量」而不是一整块?因为层与层的访问是串行的,按层取视图对 kernel 友好;且便于按层做量化/offload。

### 2.3 MLA 池:KV 融合

DeepSeek 的 MLA 把 KV 压成低维 latent,`MLATokenToKVPool` 每层只有一个 `kv_buffer`,维度为 `kv_lora_rank + qk_rope_head_dim`:

```3920:3935:python/sglang/srt/mem_cache/memory_pool.py
    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # ...
                self.kv_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, 1, self.kv_cache_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
```

`get_key_buffer` 返回整个 latent;`get_value_buffer` 返回前 `kv_lora_rank` 维。MLA 的 KV 体积远小于 MHA,这正是 DeepSeek 长上下文成本低的原因。

### 2.4 ReqToTokenPool

`ReqToTokenPool` 是一个 `(max_running_requests, max_context_len)` 的 int32 表:第 `req_pool_idx` 行记录该请求每个 token 位置对应的 KV 槽位。`alloc(reqs)` 分配行、`free(req)` 归还行。

---

## 3. 索引层:分配器

槽位的 alloc/free 不在池上,而在 `allocator/`:

- `TokenToKVPoolAllocator`(page_size=1):维护 `free_pages` 张量,`alloc(n)` 切走前 n 个,`free(idx)` 追加回去。
- `PagedTokenToKVPoolAllocator`(page_size>1):按页分配,支持 `alloc_extend` / `alloc_decode`,与 paged-attention 的页表对齐。
- 延迟释放(`free_group`)与周期性合并排序,避免碎片化拖慢分配。

**分配的统一入口**在 `mem_cache/allocation.py`:

- `alloc_for_extend`:prefill 时调用——先 `alloc_req_slots` 拿行,再按「新 token 数」分配槽位,最后把 `prefix_indices`(RadixCache 命中的旧槽位)与新槽位一起写入 `req_to_token` 行。
- `alloc_for_decode`:decode 时每个请求再要 1(或投机解码下若干个)槽位。

---

## 4. RadixCache:前缀共享的灵魂

### 4.1 数据结构

每个节点 `TreeNode`(`radix_cache.py:216`)存:

- `key`:一段 token 序列(`RadixKey`,页对齐);
- `value`:这段序列对应的 **KV 槽位索引张量**(不是 KV 数据!);
- `lock_ref`:引用计数(被活跃请求锁住时不可驱逐);
- `last_access_time`:LRU 依据。

树根为空。所有缓存的前缀都是「根 → 某节点」的路径。

### 4.2 三大操作

**match_prefix —— 查最长命中前缀**

```352:410:python/sglang/srt/mem_cache/radix_cache.py
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if self.disable or len(key) == 0:
            return self._empty_match_result
        key = key.page_aligned(self.page_size)
        ...
        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        return MatchResult(device_indices=value, last_device_node=last_node, ...)
```

沿树逐段比对;若命中止于某节点中间,会 `_split_node` 把节点切成两段。返回的 `device_indices` 就是可以直接复用的 KV 槽位序列——**命中的 token 既不用重算,也不用重新分配显存**。

**insert —— 请求结束时入树**

`cache_finished_req(req)`(`radix_cache.py:434`)把 `(origin_input_ids + output_ids)` 连同其槽位索引插入树;与树中已有部分重叠的槽位会被释放回分配器(去重),`dec_lock_ref` 解锁。

**evict —— 内存不足时逐出**

```562:590:python/sglang/srt/mem_cache/radix_cache.py
    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)
        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)
            self.token_to_kv_pool_allocator.free_segment(x.value, start_pos=0)
            num_evicted += len(x.value)
            self._delete_leaf(x)
            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                ...
                heapq.heappush(eviction_heap, (new_priority, x.parent))
```

从叶子开始按策略(默认 LRU)逐出,父节点变成叶子后也可能被逐出。被逐节点只释放「槽位索引」,KV 数据随之可被覆写。

### 4.3 lock_ref:防止正在用的前缀被逐

请求调度时对 `last_node` 调 `inc_lock_ref`,沿父链把每个祖先的 `lock_ref+1`,这些 token 从「可驱逐」转入「受保护」。请求结束时 `dec_lock_ref` 还原。**这就是「活跃请求的 KV 永不被逐」的保证。**

### 4.4 何时不用 RadixCache

`--disable-radix-cache` 时退化为 `ChunkCache`:无共享,请求结束即整体释放。某些场景(纯吞吐压测、无公共前缀的工作负载)反而更省管理开销。

---

## 5. 显存容量:max_total_num_tokens 是怎么算的

这是适配与调优的关键公式。入口 `KVCacheConfigurator`(`mem_cache/kv_cache_configurator.py`):

1. **测可用显存**:模型加载 + 一次最大 batch 的 profile 前向后,读当前空闲显存。
2. **扣掉运行时 slack**:`pre_model_load_memory * (1 - mem_fraction_static)` —— 给激活、临时缓冲留的空间。
3. **除以每 token 的 KV 字节数(cell size)**:
   - MHA:`num_kv_heads * (head_dim + v_head_dim) * num_layers * dtype_bytes`;
   - MLA:`(kv_lora_rank + qk_rope_head_dim) * num_layers * dtype_bytes`。
4. **页对齐、按 `--max-total-tokens` 封顶、PP 各 rank 取 min**。

```
max_total_num_tokens = (free_mem − slack) // cell_size,再对齐 page_size
```

> **适配提示**:第三方设备要正确实现 `get_device_total_memory` / `get_current_memory_usage` / OOT 的 KV 池工厂,这个公式才能算对(见文档 07)。

---

## 6. 与 Attention Backend 的接口(读路径)

前向时 backend 这样消费内存池(以 FA3 为例):

```1337:1344:python/sglang/srt/layers/attention/flashattention_backend.py
            key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)

            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )
```

页表则来自 `req_to_token_pool.req_to_token[req_pool_indices, :max_seq_len_k]`。**新 K/V 的写路径**是 attention 层内部 `set_kv_buffer(layer, out_cache_loc, k, v)`。

一句话:`req_pool_indices → req_to_token 行 → 槽位序列 → 页表 → 读 k_buffer/v_buffer`;`out_cache_loc` 是本步写入的新槽位。

---

## 7. 进阶变体(了解即可)

| 组件 | 用途 |
|---|---|
| `HiRadixCache` | 分层缓存:GPU → 主机内存 → 存储后端(L3),`--enable-hierarchical-cache` |
| `SWARadixCache` / `SWAKVPool` | 滑动窗口注意力(Gemma 等)的窗口/全量双池 |
| `MambaRadixCache` | 混合线性注意力模型的状态缓存 |
| `radix_cache_cpp.py` | C++ 基数树加速版 |
| `UnifiedKVPool` | 多种子池共享一块物理缓冲 |

---

## 8. 本篇要点回顾

1. 三层分离:RadixCache(逻辑/索引)→ Allocator(槽位)→ KVCache(物理张量);树里存的是**槽位索引**,不是 KV。
2. 前缀命中 = 免计算 + 免分配,这是 SGLang 多轮/共享 prompt 场景快的根本原因。
3. `lock_ref` 保护活跃前缀;驱逐从叶子按 LRU 进行。
4. `max_total_num_tokens` 由显存 profile 自动推算,公式是适配正确性的试金石。
5. Attention backend 通过 `get_kv_buffer` + `req_to_token` 页表读池,通过 `set_kv_buffer` + `out_cache_loc` 写池。

下一篇:[05 · 模型层与如何新增模型](./05-模型层与新增模型.md)。

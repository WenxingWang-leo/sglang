---
title: "SGLang DeepSeek-V4 实现精读"
description: "DeepSeek-V4 在 SGLang 中的运行时路径、模型类层次、DSA/压缩注意力、MoE、KV 池、投机（NextN/MTP/DSpark）与三方设备钩子的实现级精读。"
keywords:
  - deepseek-v4
  - dsv4
  - mhc
  - sparse attention
  - moe
  - dspark
  - code reading
---

本篇是精读系列的 **DeepSeek-V4 专题**（纯 Markdown，可用编辑器 Preview 打开）。对着源码讲四件事：

1. **整条运行时路径**（HTTP → Scheduler → `DeepseekV4ForCausalLM` → sample）
2. **每个算子的具体功能**（§2 按 DecoderLayer 流水线 + kernels 表）
3. **缓存体系**（SWA / C4 / C128 / Indexer / CompressState + Radix 关系，§3）
4. **第三方 Device 接入清单**（照抄 NPU 分流，§6；通用理论见第 6 篇）

配套：[系列总目录](./code_reading_notes_zh.md)、[第 6 篇 第三方硬件接入](./code_reading_hardware_device_zh.md)、操作手册 [Cookbook · DeepSeek-V4](/cookbook/autoregressive/DeepSeek/DeepSeek-V4)。

系列导航：[总目录](./code_reading_notes_zh.md) · [第 1 篇](./code_reading_srt_core_zh.md) · [第 2 篇](./code_reading_deep_dive_zh.md) · [第 3 篇](./code_reading_notes_advanced_zh.md) · [第 4 篇](./code_reading_serving_extensions_zh.md) · [第 5 篇](./code_reading_ecosystem_zh.md) · [第 6 篇](./code_reading_hardware_device_zh.md)

> **判定入口：** `srt/configs/model_config.py::is_deepseek_v4(config)` — HF arch ∈ `{DeepseekV4ForCausalLM, DeepseekV4ForCausalLMNextN, DeepseekV4ForCausalLMDSpark}`。与 V3.2 DSA（`is_deepseek_dsa`，看 `index_topk`）是两条线。

---



## 0. 一张总图：V4 与通用路径差在哪

```text
HTTP / TokenizerManager
        │  GenerateReqInput → TokenizedGenerateReqInput (ZMQ)
        ▼
Scheduler  ── tree_cache (Radix / Unified / HiSparse) ──► SWA + C4 + C128 分配
        │  ScheduleBatch → ModelRunner.forward
        ▼
ModelRunner
  · attention_backend = "dsv4"  → DeepseekV4AttnBackend | HipRadix | Ascend
  · token_to_kv_pool  = DeepSeekV4TokenToKVPool (| DSV4NPU*)
        │
        ▼
DeepseekV4ForCausalLM.forward
  · (可选) DSA prefill CP metadata / indexer reindex
  · DeepseekV4Model：embed → hc_mult 维展开 → DecoderLayer×N → hc_head → RMSNorm
  · LogitsProcessor(+ lm_head) → sample
        │
每层 DeepseekV4DecoderLayer
  hc_pre → MQALayer(QKV/RoPE/Indexer/Compressor/FlashMLA) → hc_post
         → hc_pre → DeepseekV2MoE(is_deepseek_v4=True) → hc_post
```

相对 DeepSeek-V2/V3 的关键差异：


| 维度           | V3 / V3.2                               | V4                                                                                        |
| ------------ | --------------------------------------- | ----------------------------------------------------------------------------------------- |
| Attention 形态 | MLA（低秩 KV）或 DSA indexer + MLA           | **MQA head_dim=512**（nope 448 + rope 64），**非 MLA**；`ModelConfig.attention_arch = MHA`     |
| 稀疏           | DSA `index_topk` + MLA KV               | 层上 `compress_ratios ∈ {0,4,128}`：SWA / C4+Indexer / C128                                  |
| Residual     | 普通 residual                             | **mHC**（manifold-constrained hyper-connections，`hc_mult` 路混合）                             |
| MoE gate     | grouped `noaux_tc`                      | V4：`use_grouped_topk=False` + `scoring_func=sqrtsoftplus`；前 `n_hash_layers` 可用 `HashTopK` |
| KV pool      | `MLATokenToKVPool` / `DSATokenToKVPool` | `DeepSeekV4TokenToKVPool`（SWA + C4 + C128 + indexer + compress state）                     |
| Attn backend | `fa3` / `flashinfer` / `dsa` …          | 强制 `dsv4`（CUDA / HIP / Ascend 三路）                                                         |
| 投机           | EAGLE / NextN                           | CLI 白名单只有 **`EAGLE`（topk=1）** 与 **`DSPARK`**。NextN 是 **draft 模型 arch**（`DeepseekV4ForCausalLMNextN`），通常挂在 `--speculative-algorithm EAGLE` 上，不是第三个算法名 |


---



## 1. 模型文件与类层次



### 1.1 文件地图


| 路径                                                                      | 角色                                                                                                                                                     |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `python/sglang/srt/models/deepseek_v4.py`                               | Target：`MqaAttentionBase` / `MQALayer` / `DeepseekV4DecoderLayer` / `DeepseekV4Model` / `DeepseekV4ForCausalLM`；`EntryClass = [DeepseekV4ForCausalLM]` |
| `python/sglang/srt/models/deepseek_v4_nextn.py`                         | MTP draft：`DeepseekV4ModelNextN` / `DeepseekV4ForCausalLMNextN`；单层 decoder，`compress_ratio_override=0`                                                 |
| `python/sglang/srt/models/deepseek_v4_dspark.py`                        | DSpark draft：`DSparkAttention` / `DSparkV4Stage` / `DeepseekV4ForCausalLMDSpark`                                                                       |
| `python/sglang/srt/configs/deepseek_v4.py`                              | `DeepSeekV4Config` + `try_detect_fp4_experts`                                                                                                          |
| `python/sglang/srt/arg_groups/deepseek_v4_hook.py`                      | 默认值 / MegaMoE token budget / CP 校验                                                                                                                     |
| `python/sglang/srt/arg_groups/overrides.py`                             | `_deepseek_v4_overrides`、`_deepseek_v4_kv_cache_dtype`、`_deepseek_v4_sm120_moe`                                                                        |
| `python/sglang/srt/models/deepseek_common/`                             | 与 V2/V3 共享的 forward helpers、weight loader、AMD fused mHC                                                                                                |
| `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` | ROCm gfx95 fused `hc_post`+`hc_pre`                                                                                                                    |


Registry：`srt/models/registry.py` 扫 `EntryClass`，三个模块分别注册三个 arch。

### 1.2 `DeepSeekV4Config` 关键字段

文件：`srt/configs/deepseek_v4.py`。


| 字段                                                | 默认（示意）                      | 含义                                        |
| ------------------------------------------------- | --------------------------- | ----------------------------------------- |
| `hidden_size`                                     | 4096                        | 隐藏维                                       |
| `num_attention_heads` / `num_key_value_heads`     | 64 / **1**                  | MQA                                       |
| `qk_nope_head_dim` + `qk_rope_head_dim`           | 448 + 64 → **head_dim=512** | 与 FlashMLA sparse 对齐                      |
| `q_lora_rank` / `o_lora_rank` / `o_groups`        | 1024 / 1024 / 8             | Q/O 低秩投影                                  |
| `kv_lora_rank`                                    | 512                         | 配置保留；V4 attention 路径是 MQA 直写 KV，不是 V3 MLA |
| `index_head_dim` / `index_n_heads` / `index_topk` | 128 / 64 / **512**          | C4 Indexer                                |
| `n_routed_experts` / `num_experts_per_tok`        | 256 / 6                     | MoE                                       |
| `scoring_func` / `topk_method`                    | `sqrtsoftplus` / `noaux_tc` | V4 TopK 关掉 grouped                        |
| `n_hash_layers`                                   | 3                           | 前几层 `HashTopK`                            |
| `hc_mult` / `hc_sinkhorn_iters` / `hc_eps`        | 4 / 20 / 1e-6               | mHC                                       |
| `compress_ratios`                                 | `List[int]` 每层 0/4/128      | 注意力压缩比                                    |
| `window_size`                                     | 128                         | SWA 窗（HF 里常作 `sliding_window`）            |
| `compress_rope_theta`                             | 40000                       | 压缩层 RoPE base                             |


`ModelConfig`（`configs/model_config.py` ~867–883）对 V4 强制：

- `attention_arch = AttentionArch.MHA`（不是 MLA）
- 从 HF 读 `compress_ratios`、`index_head_dim`、`sliding_window`
- YaRN：`rope_scaling` 存在时用 `compute_mla_mscale_scaling` 调 `scaling`



### 1.3 类层次（Target）

```text
DeepseekV4ForCausalLM
  ├── DeepseekV4Model
  │     ├── VocabParallelEmbedding (PP first)
  │     ├── DeepseekV4DecoderLayer × N
  │     │     ├── MQALayer (← MqaAttentionBase)
  │     │     │     ├── wqkv_a 或 (wq_a + wkv)     # SGLANG_OPT_FUSE_WQA_WKV
  │     │     │     ├── q_norm / kv_norm (RMSNorm)
  │     │     │     ├── wq_b (ColumnParallel) → n_heads × 512
  │     │     │     ├── wo_a / wo_b (O 低秩，可选 FP8 wo_a)
  │     │     │     ├── Compressor (ratio∈{4,128})
  │     │     │     ├── C4Indexer (ratio==4)
  │     │     │     └── RadixAttention attn_mqa
  │     │     ├── DeepseekV2MoE(is_deepseek_v4=True)
  │     │     ├── input / post_attention RMSNorm
  │     │     └── mHC params (hc_attn_* / hc_ffn_*)
  │     ├── RMSNorm + hc_head_* (PP last)
  │     └── (可选) TBO / DSpark aux capture
  ├── ParallelLMHead
  └── LogitsProcessor
```

`EntryClass = [DeepseekV4ForCausalLM]`（文件末尾）。

### 1.4 Forward 路径（Target）

`DeepseekV4ForCausalLM.forward`（~2586）：

1. 若 `dsa_enable_prefill_cp`：`can_dsa_cp_split` → `prepare_context_parallel_metadata`；round-robin 时对 `core_meta.apply_cp_reindex()` + `init_flashmla_related` + 重建 `indexer_metadata`。
2. `get_attn_tp_context().maybe_input_scattered` 下调用 `self.model.forward`。
3. PP 非 last 直接返回 proxy；否则拆 `(hidden, pre_hc_head)`（及可选 aux），进 `LogitsProcessor(..., hidden_states_before_norm=pre_hc_head)` —— MTP 需要 pre-norm 隐状态。

`DeepseekV4Model.forward`（~2347）：

1. Embed → `unsqueeze(1).repeat(1, hc_mult, 1)`（mHC 多路残差张量）。
2. DP + `moe_a2a=none`：`dp_gather_replicate` 得全局 `input_ids`（hash MoE / TopK 用）。
3. Prefill CP：`cp_split_and_rebuild_data/position` + `cp_round_robin_input_ids`。
4. 清 `forward_batch.freqs_cis_c4/c128` 缓存。
5. 层循环：`DeepseekV4DecoderLayer`；或 `_can_run_tbo` 时走 `_forward_layers_tbo`（prefill-only two-batch-overlap）。
6. Last PP：`hc_head` → `norm`；返回 `(hidden_states, pre_hc_head)`。

`DeepseekV4DecoderLayer.forward`（~1634）：

```text
hc_pre(attn) → [可选 fused rms+fp8] → self_attn
→ hc_post / fused_post_pre → hc_pre(ffn) → MoE → hc_post
```

mHC 实现选型（环境变量门控）：


| 路径                                                     | 门控                                       |
| ------------------------------------------------------ | ---------------------------------------- |
| TileLang `mhc_pre` / `mhc_post` / `mhc_fused_post_pre` | `SGLANG_OPT_USE_TILELANG_MHC_*`          |
| AIter（HIP）                                             | `SGLANG_OPT_USE_AITER_MHC_*`             |
| NPU custom op                                          | `_is_npu` → `npu_hc_pre` / `npu_hc_post` |
| DeepGEMM TF32 prenorm                                  | `SGLANG_OPT_DEEPGEMM_HC_PRENORM`         |
| Torch fallback                                         | `hc_split_sinkhorn` + 手工 mix             |


AMD：`try_fused_hc_post_pre`（`deepseek_common/amd/deepseek_v4_fused_mhc.py`）。

### 1.5 `MQALayer` 注意力核心

**准备 Q/K**（`_forward_prepare` / multi-stream 变体）：

1. `wqkv_a` 或 `wq_a`+`wkv` → q_lora / kv。
2. Q：`q_norm` → `wq_b` → `fused_q_norm_rope`（`kernels/ops/attention/dsv4`）。
3. K：默认 `_compute_kv_to_cache` → `DeepSeekV4TokenToKVPool.set_swa_key_buffer_radix_fused_norm_rope`（norm+RoPE+写 FlashMLA 分页 cache 融合）；DSA CP / unified_kv 等场景保留 bf16 KV 再 `store_cache`。
4. `compress_ratio==4`：`C4Indexer.forward`（写 indexer cache + topk）。
5. `compress_ratio in (4,128)`：`attn_backend.forward_core_compressor`。

**Attention 调用**：`attn_backend.forward(q, k=v, layer=attn_mqa, compress_ratio, attn_sink, save_kv_cache=...)`。K≡V（assert `k is v`）。

**O 投影**：可选逆 RoPE（NPU/ fused）→ `wo_a`（分组低秩，可 FP8 DeepGEMM）→ `wo_b`。

**RoPE**：`rope_type = "deepseek_yarn"`；`precompute_freqs_cis`（`kernels/ops/attention/deepseek_v4_rope.py`）；压缩层用 `compress_rope_theta`。

### 1.6 NextN / MTP

`deepseek_v4_nextn.py`：

- `COMPRESS_RATIO_NEXTN_LAYER = 0`：draft 层不做 C4/C128。
- `DeepseekV4ModelNextN`：`enorm`/`hnorm` + `e_proj`/`h_proj` 把 **target 传来的** `spec_info.hidden_states`**（已是 hc_mult 展平）** 与 token embed 合成 mHC 状态，再进单层 `DeepseekV4DecoderLayer(is_nextn=True)`。
- `DeepseekV4ForCausalLMNextN` 继承 target 的权重 remap / shared expert 逻辑；`load_weights(..., is_nextn=True)`。



### 1.7 DSpark

`deepseek_v4_dspark.py` + `srt/speculative/dspark_components/`：

- `DSparkAttention`：`compress_ratio=0`，SWA 写同一 `DeepSeekV4TokenToKVPool` 的 SWA 区；`RadixAttention` 层 id 走 draft。
- `DSparkV4Stage`：继承 `DeepseekV4DecoderLayer`，换 `_build_self_attn` → `DSparkAttention`。
- `DeepseekV4ForCausalLMDSpark`：多 stage draft + Markov confidence head（`DSparkV4MarkovHead`）+ `StepSampler`。
- Worker：`dspark_worker_v2.py`；`draft_is_deepseek_v4()` 看 draft HF config。
- 0731 checkpoint **同仓 bundling draft**，启动 `--speculative-algorithm DSPARK`，通常不设 `--speculative-draft-model-path`。

架构特点小结：**不是 V3 MLA**；是 **MQA + 分层压缩稀疏（CSA/HCA 风格）+ mHC + MoE（hash/sqrtsoftplus）+ NextN/DSpark**。

---



## 2. 算子全景：每个算子做什么

读法：按「一层 DecoderLayer 的真实调用顺序」扫；每项给出 **输入 → 计算 → 输出 / 写哪里**。路径相对 `python/sglang/`。

### 2.0 一层 DecoderLayer 的算子流水线

```text
hidden [T, hc_mult, H]
  │
  ├─① mHC hc_pre(attn)          # 多路残差混合 → 单路进 attention
  ├─② input RMSNorm             # 可与 FP8 权重量化融合
  ├─③ wqkv_a / (wq_a+wkv)       # Linear：出 q_lora、kv
  ├─④ q_norm → wq_b             # Q 低秩展开 → [T, n_heads, 512]
  ├─⑤ fused_q_norm_rope         # Q：RMSNorm 风格 + YaRN RoPE（nope|rope 分段）
  ├─⑥ fused_k_norm_rope + store # K：norm+RoPE+打包写入 SWA KV 页
  │     └─ K≡V（MQA shared）
  ├─⑦ [ratio==4]  C4Indexer     # 写 IndexerPool + topk indices
  ├─⑧ [ratio∈4,128] Compressor  # SWA → C4/C128 压缩页 + CompressState
  ├─⑨ RadixAttention / FlashMLA # densescWA 或 sparse(C4/C128) attn
  ├─⑩ (可选) inverse RoPE
  ├─⑪ wo_a → wo_b               # O 低秩；wo_a 可走 FP8 DeepGEMM
  ├─⑫ mHC hc_post(attn)         # 注意力输出混回多路残差
  ├─⑬ mHC hc_pre(ffn) + post_attn RMSNorm
  ├─⑭ MoE gate (HashTopK | sqrtsoftplus TopK)
  ├─⑮ Expert GEMM + SiLU×Mul (+ 可选 post-quant)
  ├─⑯ Shared expert + combine / EP A2A
  └─⑰ mHC hc_post(ffn)          # 交给下一层
```

`compress_ratio` 决定 ⑦⑧⑨ 分支：


| `compress_ratio` | Indexer         | Compressor | Attention 读哪                |
| ---------------- | --------------- | ---------- | --------------------------- |
| **0**            | 否               | 否          | **SWA** dense（滑窗内全注意力）      |
| **4**            | **是**（topk≈512） | SWA→C4     | **C4** sparse（按 indexer 选页） |
| **128**          | 否               | SWA→C128   | **C128** sparse             |




### 2.1 Attention backend 选型

注册：`srt/layers/attention/attention_registry.py`

```python
@register_attention_backend("dsv4")
def create_dsv4_backend(runner):
    if _is_npu:   return DeepseekV4AscendAttnBackend(runner)
    elif _is_hip: return DeepseekV4HipRadixBackend(runner)
    else:         return DeepseekV4AttnBackend(runner)
```


| 平台     | 类                             | 文件                                                          |
| ------ | ----------------------------- | ----------------------------------------------------------- |
| CUDA   | `DeepseekV4AttnBackend`       | `srt/layers/attention/deepseek_v4_backend.py`               |
| HIP    | `DeepseekV4HipRadixBackend`   | `.../deepseek_v4_backend_hip_radix.py`                      |
| Ascend | `DeepseekV4AscendAttnBackend` | `srt/hardware_backend/npu/attention/ascend_dsv4_backend.py` |


默认：`attention_backend="dsv4"`，`page_size=256`（NPU 128）。

**Backend 必须实现的契约**（`AttentionBackend` ABC，`base_attn_backend.py`）：


| 方法                                                      | 作用                               |
| ------------------------------------------------------- | -------------------------------- |
| `init_forward_metadata_out_graph(fb, in_capture=False)` | 每 iter 元数据（CPU/动态 shape）；graph 外 |
| `init_forward_metadata_in_graph(fb)`                    | 可录进 CUDA Graph 的静态 GPU op        |
| `forward_extend` / `forward_decode`（及 V4 的统一 `forward`） | 真正算 attention                    |
| `init_cuda_graph_state` / graph replay 钩子               | decode 捕获                        |


V4 额外 mixin：

- `C4IndexerBackendMixin`（`dsv4/indexer.py`）：`forward_c4_indexer`
- `CompressorBackendMixin`（`dsv4/compressor.py`）：`forward_core_compressor` / `forward_compress`

`DeepseekV4AttnBackend.forward`：按层 `compress_ratio` 分发 SWA dense / C4 FlashMLA sparse / C128 sparse；metadata 在 `dsv4/metadata.py`（`DSV4Metadata`、`DSV4AttnMetadata`、verify/decode raw metadata）。

### 2.2 算子表 A：投影 / Norm / RoPE / 写 Cache


| #   | 算子 / API                                                                  | 文件                                          | 功能（输入→输出）                                                 |
| --- | ------------------------------------------------------------------------- | ------------------------------------------- | --------------------------------------------------------- |
| ①   | `mhc_pre` / TileLang·AIter·NPU                                            | `kernels/ops/layernorm/mhc*.py`；NPU custom  | `[T,hc_mult,H]` → 混合后单路 hidden；Sinkhorn 约束流形              |
| ②   | `RMSNorm`                                                                 | `srt/layers/layernorm.py`                   | 标准 RMS；可与后续量化融合                                           |
| ③   | `wqkv_a` 或 `wq_a`+`wkv`                                                   | `MQALayer` Linear                           | hidden → q_lora / kv；`SGLANG_OPT_FUSE_WQA_WKV` 融合成一次 GEMM |
| ④   | `q_norm` + `wq_b`                                                         | RMSNorm + `ColumnParallelLinear`            | q_lora → `[T, n_heads, 512]`（448 nope + 64 rope）          |
| ⑤   | `fused_q_norm_rope`                                                       | `kernels/ops/attention/dsv4`                | 对 Q 做分段 norm + YaRN RoPE                                  |
| ⑥a  | `precompute_freqs_cis`                                                    | `kernels/ops/attention/deepseek_v4_rope.py` | 预计算 YaRN 复数频表（β_fast/slow）；压缩层用 `compress_rope_theta`     |
| ⑥b  | `set_swa_key_buffer_radix_fused_norm_rope` / `fused_k_norm_rope_flashmla` | `DeepSeekV4TokenToKVPool` + dsv4 kernels    | K：norm+RoPE → **NopeFp8RopeBf16Pack** → 写入 SWA 分页 buffer  |
| ⑩   | inverse RoPE（可选）                                                          | NPU/fused 路径                                | O 投影前还原旋转                                                 |
| ⑪   | `wo_a` / `wo_b`                                                           | 分组低秩 Linear；`fp8_wo_a`                      | attn out → hidden；wo_a 可 FP8 DeepGEMM（UE8M0 scale）        |
| ⑫⑰  | `mhc_post` / `mhc_fused_post_pre`                                         | 同 mHC                                       | 单路输出混回 `hc_mult` 路；跨层可融 post+next pre                     |




### 2.3 算子表 B：Indexer / Compressor / Sparse Attn


| #   | 算子 / API                                        | 文件                                                          | 功能                                                                                                                                      |
| --- | ----------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| ⑦   | `C4Indexer.forward`                             | `srt/layers/attention/dsv4/indexer.py`                      | 从 q_lora/hidden 生成 indexer Q/K（`index_head_dim=128`, `index_n_heads=64`），写入 `DeepSeekV4IndexerPool`，再 **topk（默认 512）** 选出本 query 要看的压缩页 |
| ⑦b  | `plan_topk_v2` / `topk_transform_512_v2`        | `kernels/ops/attention/dsv4/topk.py`                        | 规划/变换 topk 索引，供 FlashMLA sparse                                                                                                         |
| ⑦c  | `fp4_indexer`（可选）                               | `kernels/.../dsv4/fp4_indexer.py`                           | FP4 量化 indexer（`--enable-deepseek-v4-fp4-indexer`，SM100/120）                                                                            |
| ⑧   | `Compressor` / `compressor_v2` / `compress_hip` | `srt/layers/attention/dsv4/compressor*.py`                  | 将 SWA 上连续 token **池化/压缩** 进 C4 或 C128 页；更新 `CompressStatePool`（max/sum/kv ring 或 online）                                                |
| ⑧b  | `compress.py` / `fused_compress_triton`         | `kernels/ops/attention/dsv4/`                               | 实际压缩 CUDA/Triton kernel                                                                                                                 |
| ⑧c  | `quant_k_cache` / `dequant_k_cache`             | 同目录                                                         | 把 K 打成/解开 **FP8 nope + BF16 rope + scale** 字节布局                                                                                         |
| ⑧d  | `online_c128_mtp`                               | 同目录                                                         | decode 在线更新 C128；与 MTP 组合需实验开关                                                                                                          |
| ⑧e  | `c128_cleanup`                                  | 同目录                                                         | speculative reject 时清未接受的 C128 draft state                                                                                              |
| ⑨   | FlashMLA sparse / dense                         | backend `forward` + `attn.py` / `sparse_prefill_kernels.py` | Q 对 SWA 或压缩 KV 做 attention；sparse 用 topk 页表；K≡V                                                                                         |
| ⑨b  | `index_buf_accessor`                            | dsv4 kernels                                                | 分页 KV / indexer buffer 的底层读写                                                                                                            |
| ⑨c  | `unified_kv_kernels/`                           | HIP only                                                    | 把 SWA+压缩行拼进统一 bf16 buffer，供 ROCm 路径                                                                                                     |




### 2.4 算子表 C：MoE / EP / Logits


| #   | 算子 / API                                  | 文件                                               | 功能                                                                   |
| --- | ----------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------- |
| ⑭a  | `HashTopK`                                | `srt/layers/moe/hash_topk.py` → `dsv4.hash_topk` | 前 `n_hash_layers` 层用哈希选专家（便宜）；NextN 关闭                               |
| ⑭b  | `TopK(scoring_func=sqrtsoftplus)`         | MoE gate                                         | V4：`use_grouped_topk=False`；按 sqrtsoftplus 分数选 `num_experts_per_tok` |
| ⑮   | Expert GEMM + `silu_and_mul_*_post_quant` | `moe_runner/deep_gemm.py` + `dsv4/moe.py`        | 专家 FFN；可与 FP8/FP4 后量化融合                                              |
| ⑮b  | `mega_moe_pre_dispatch`                   | `dsv4` + `layers/moe/mega_moe.py`                | MegaMoE 预调度 token→rank                                               |
| ⑯   | DeepEP / MegaMoE A2A                      | `moe_a2a_backend`                                | EP all-to-all dispatch/combine                                       |
| —   | Shared expert                             | `DeepseekV2MoE`                                  | 共享专家；可与 routed 融合 per-rank slots                                     |
| —   | `LogitsProcessor`                         | `layers/logits_processor.py`                     | `pre_hc_head`（norm 前）供 MTP；再 lm_head→logits→Sampler                  |




### 2.5 算子表 D：kernels 目录速查

目录：`python/sglang/kernels/ops/attention/dsv4/`


| 模块                                                             | 一句话职责                                        |
| -------------------------------------------------------------- | -------------------------------------------- |
| `attn.py`                                                      | sparse/dense attention 辅助                    |
| `compress.py` / `compress_old.py` / `fused_compress_triton.py` | 压缩实现                                         |
| `topk.py`                                                      | indexer topk + plan                          |
| `quant_k_cache.py` / `dequant_k_cache.py`                      | KV 字节 pack/unpack（DIM_NOPE=448, DIM_ROPE=64） |
| `index_buf_accessor.py`                                        | 分页访问                                         |
| `fp8_wo_a.py` / `elementwise.py` / `gemm.py`                   | O 投影量化与杂项 GEMM                               |
| `moe.py`                                                       | MegaMoE / silu quant                         |
| `online_c128_mtp.py`                                           | online C128 + MTP                            |
| `sparse_prefill_kernels.py`                                    | sparse prefill                               |
| `unified_kv_kernels/`                                          | HIP unified KV                               |
| `fp4_indexer.py`                                               | FP4 indexer                                  |
| `c128_cleanup.py`                                              | draft 清理                                     |
| `rms_normalize_hip.py`                                         | ROCm RMS                                     |


另：`deepseek_v4_rope.py`；AOT `kernels/aot/.../deepseek_v4_topk.cu`；mHC `kernels/ops/layernorm/mhc*`；DSpark `kernels/ops/speculative/dspark/`。

**选型惯例：** 模型 Python → `AttentionBackend` / `MultiPlatformOp` 按 CUDA|HIP|NPU 分流 → JIT/AOT kernel。第三方 device 通常在 **Backend + 若干 MultiPlatformOp** 换血，而不是改 `deepseek_v4.py` 层结构。

---



## 3. KV / Cache 路径



### 3.1 池结构（不是 MLATokenToKVPool）

V4 **不**用 `MLATokenToKVPool` / `DSATokenToKVPool` 作为主池。装配点：`mem_cache/kv_cache_configurator.py::_build_dsv4_kv_pool`。

`DeepSeekV4TokenToKVPool`（`mem_cache/deepseek_v4_memory_pool.py`，继承 `BaseSWAKVPool`）：

```text
DeepSeekV4TokenToKVPool
  ├── swa_kv_pool: DeepSeekV4SingleKVPool     # CUDA page_size=256；NPU=128
  ├── c4_kv_pool:  DeepSeekV4SingleKVPool | HiSparseC4DevicePool
  ├── c128_kv_pool: DeepSeekV4SingleKVPool
  ├── c4_indexer_kv_pool: DeepSeekV4IndexerPool
  ├── CompressStatePool × {c4, c128}          # ring / online 压缩状态
  └── (可选) unified_kv_pool: DeepSeekV4UnifiedKVPool  # HIP Triton unified
```

**单 token 物理布局**（`DeepSeekV4SingleKVPool.create_buffer` 硬断言）：

```text
每 token = 584 bytes（store_dtype=uint8 视角）
  ├── qk_nope   FP8_e4m3 × 448          = 448 B
  ├── qk_rope   BF16 × 64               = 128 B
  ├── nope scales (448/64=7) + pad      =   8 B
  └── 合计 584 B

每 page 再向上对齐到 576 的倍数（FlashMLA 页对齐）：
  bytes_per_page_padded = ceil(page_size * 584 / 576) * 576
```

逻辑上 K≡V（MQA shared KV）；池里只存 **一份** packed K。

NPU：`DSV4NPUTokenToKVPool`（`hardware_backend/npu/dsv4/dsv4_memory_pool.py`）——paged state，非 CUDA ring；`page_size=128`；kv dtype 默认 **BF16**（CUDA 默认 **fp8_e4m3**）；layout 需满足设备 sparse attn 的页格式。

**Indexer 池**（`DeepSeekV4IndexerPool`）：仅 C4 层；与 topk 元数据一起进 `DSV4Metadata.indexer_metadata`。

**CompressStatePool**（`mem_cache/deepseek_v4_compress_state.py`）：非投机 c4/c128 ring≈8/128；投机放大；online c128 → ring_size=1。

### 3.1b 三池与 Radix 的关系（务必分清）

```text
RadixCache / UnifiedRadix（逻辑前缀树）
        │  match_prefix → prefix_indices（token 级 kv_index）
        ▼
ReqToTokenPool[req, pos] = kv_index
        │
        ▼
SWATokenToKVPoolAllocator / DeepSeekV4HiSparse*Allocator
        │  决定哪些 page 属于 SWA / C4 / C128
        ▼
DeepSeekV4TokenToKVPool.*_kv_pool[layer][page]  ← 真正的字节
```

- **Radix 命中的是逻辑 token 前缀**，不是「压缩页编号」。
- 压缩态 / indexer 与 SWA 页生命周期绑定：驱逐时必须同步 recycle compress state。
- HiCache 把 V4 池当作 rank-replicated compressed；HiSparse 可把 C4 放到 device pool 并做索引翻译。



### 3.2 Prefill vs Decode 布局


| 阶段               | 行为                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| Prefill (EXTEND) | 写入 SWA；按层 ratio 跑 Compressor → C4/C128；C4 层跑 Indexer 得 topk；sparse/dense FlashMLA 读压缩 cache                        |
| Decode           | SWA 滑窗推进；online/offline 更新 compress state；C4 topk 更新；C128 online 可用 `SGLANG_OPT_USE_ONLINE_COMPRESS`（ring_size=1）  |
| Spec verify      | `DSV4RawVerifyMetadata`；ragged verify（`SGLANG_RAGGED_VERIFY_MODE`）；拒绝 draft 时 `clear_unaccepted_c128_draft_states` |


`get_compress_state_ring_size(ratio, is_speculative)`：非投机 c4=8 / c128=128；投机放大到 16 / 256；online c128 → 1（与 MTP 默认互斥，除非 `SGLANG_EXPERIMENTAL_ONLINE_C128_MTP`）。

### 3.3 RadixCache 与压缩 KV

- 树节点仍按 **token 前缀** 匹配（通用 `RadixCache` / unified radix）。
- 物理页由 **SWA allocator + 压缩池尺寸** 约束：`SWATokenToKVPoolAllocator` / `DeepSeekV4HiSparseTokenToKVPoolAllocator`（`mem_cache/allocator/swa.py`、`hisparse.py`）。
- HiCache：`CacheController` 把 `DeepSeekV4TokenToKVPool` 视为 **rank-replicated compressed MLA-style**（仅需 rank0 写回 storage，`is_mla_model=True` 语义）。
- HiSparse：C4 落 `HiSparseC4DevicePool`，`full→compressed→device` 索引翻译；`managers/hisparse_coordinator.py`。
- Hybrid assembler：`hybrid_cache/hybrid_pool_assembler.py` 识别 DSV4 组件集合。

**含义：** Radix 命中的是逻辑 token 前缀；压缩态 / indexer 缓冲与 SWA 页生命周期绑定，驱逐时需同步 compress state（NPU：`maybe_evict_dsv4_state_on_swa`）。

### 3.4 投机缓存含义


| 算法              | KV 影响                                                                                                 |
| --------------- | ----------------------------------------------------------------------------------------------------- |
| EAGLE/NextN     | Draft 层 `compress_ratio=0`，主要用 SWA；target verify 写完整压缩路径；`speculative_eagle_topk` **必须为 1**           |
| DSpark          | Draft 写同一 `DeepSeekV4TokenToKVPool` SWA；`write_target_hidden_kv` / `CommitKvProj` 注入；ragged verify 布局 |
| Online C128 MTP | 实验；pending seq lens 缓冲；与 CP / ragged verify 组合有硬限制                                                    |




### 3.5 Indexer cache

`DeepSeekV4IndexerPool`：C4 层专用，head_dim=`index_head_dim`；与 `C4Indexer.forward` / `forward_c4_indexer` 成对。Topk 元数据进 `DSV4Metadata.indexer_metadata`。

---



## 4. 一次 V4 请求的生命周期（DeepSeek 分支标出）

```text
1. HTTP OpenAI/Anthropic
   └─ tool/reasoning parser 可绑 deepseek-v4 / DeepSeekV4Detector

2. TokenizerManager.generate_request
   └─ 常规 tokenize；无 V4 特殊分支（模板检测除外）

3. Scheduler.recv → add waiting queue
   └─ PrefillAdder + Radix match_prefix
   └─ 【V4】SWA/C4/C128 容量与 swa_full_tokens_ratio(默认 0.1) 参与准入
   └─ 【NPU+V4】初始化后 HCCL DP prewarm（scheduler.py ~525）

4. get_next_batch_to_run → ScheduleBatch
   └─ 分配 req_to_token + SWA/compressed pages
   └─ 【V4】HiSparse coordinator（若 enable_hisparse）

5. TpModelWorker / ModelRunner.forward
   └─ 【V4】attn_backend=dsv4；pool=DeepSeekV4TokenToKVPool
   └─ init_forward_metadata（含 indexer / compressor plan）
   └─ 【投机】draft worker：NextN 或 DSpark；target verify metadata

6. DeepseekV4ForCausalLM.forward → layers
   └─ 【V4】mHC × MQA × Indexer/Compressor × MoE(hash/sqrtsoftplus)
   └─ 【V4】可选 TBO prefill；可选 CP round-robin

7. LogitsProcessor → Sampler
   └─ 【V4】pre_hc_head 供 MTP；DSpark confidence / Markov

8. Detokenizer → 流式返回
```

与通用路径相比，**Scheduler/Tokenizer 主体相同**；分叉集中在：**KV 配置器、attention backend、模型 forward、投机 worker、HiCache/HiSparse、CP/MegaMoE 校验**。

---



## 5. CLI / ServerArgs（常用）



### 5.1 自动默认（无需手写）

`apply_deepseek_v4_defaults` + `_deepseek_v4_overrides`：


| 项                         | 默认                                          |
| ------------------------- | ------------------------------------------- |
| `--attention-backend`     | `dsv4`                                      |
| `--page-size`             | 256（NPU 128）                                |
| `--kv-cache-dtype`        | `fp8_e4m3`（NPU `bfloat16`）                  |
| `--swa-full-tokens-ratio` | `0.1`（若用户未改）                                |
| `--max-running-requests`  | 256（若 None）                                 |
| SM120 MoE runner          | `flashinfer_mxfp4`（auto）                    |
| NVFP4 MoE runner          | `flashinfer_trtllm_routed`                  |
| 投机算法白名单                   | `EAGLE` / `DSPARK`；EAGLE 要求 `eagle_topk==1` |




### 5.2 并行与通信

```bash
# 单机 Flash 常见
sglang serve --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 4   # 或 8；H100 常 tp8

# 高并发：TP + DP-Attention
--tp 8 --dp 8 --enable-dp-attention \
  --enable-prefill-delayer --prefill-delayer-max-delay-ms 5000

# Expert parallel + DeepEP
--ep 8 --moe-a2a-backend deepep

# MegaMoE（需 chunked prefill 与 per-rank token 预算）
--moe-a2a-backend megamoe --chunked-prefill-size 4096
# 校验：validate_deepseek_v4_mega_moe_token_budget
# 环境：SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK

# Prefill Context Parallel（仅 interleave；强制若干内部开关）
--enable-prefill-cp --cp-strategy interleave
# validate_deepseek_v4_cp：dp=1、tp<=8、关掉 FLASHMLA sparse prefill 等
```



### 5.3 量化 / Indexer / 投机

```bash
# FP4 MoE + FP8 attn（官方 Instruct 混合盘）
# FP8 整盘：sgl-project/DeepSeek-V4-*-FP8（Hopper/AMD）
# NVFP4：nvidia/DeepSeek-V4-*-NVFP4

--enable-deepseek-v4-fp4-indexer   # 实验 FP4 C4 indexer
--dsa-topk-backend <impl>         # indexer topk 后端

# EAGLE MTP
--speculative-algorithm EAGLE \
  --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4

# DSpark（0731）
--speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5   # 可选；默认读 checkpoint
```



### 5.4 相关环境变量（摘录）


| Env                                     | 作用                                |
| --------------------------------------- | --------------------------------- |
| `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL`    | sparse prefill；ROCm/CP 下 hook 会关掉 |
| `SGLANG_OPT_USE_ONLINE_COMPRESS`        | online C128                       |
| `SGLANG_EXPERIMENTAL_ONLINE_C128_MTP`   | online C128 + MTP                 |
| `SGLANG_OPT_FUSE_WQA_WKV`               | 融合 wqkv_a                         |
| `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP`   | Q/K/Indexer 多流                    |
| `SGLANG_OPT_USE_TILELANG_MHC_*` / AITER | mHC kernel                        |
| `SGLANG_DSV4_MHC_PREWARM`               | load 时预热 mHC                      |
| `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE`      | MegaMoE                           |
| `SGLANG_RAGGED_VERIFY_MODE`             | verify 紧凑布局                       |
| `SGLANG_DSPARK_FAST_KERNEL`             | DSpark fused Q/RoPE               |


完整列表遵循 `environ.Envs` 与 [env-var-conventions](../../../.claude/skills/env-var-conventions/SKILL.md)。

---



## 6. 在 SGLang 中接入第三方 Device 跑 DeepSeek-V4

目标：让 `--attention-backend dsv4` 在你的设备上走出完整 V4 路径（MQA + C4/C128 + MoE + 正确 KV 布局）。通用插件理论见 [第 6 篇](./code_reading_hardware_device_zh.md)；本节是 **V4 专用落地清单**。

### 6.1 你必须交付的能力（按优先级）


| 优先级 | 能力                                                | 不交付的后果                  |
| --- | ------------------------------------------------- | ----------------------- |
| P0  | `dsv4` AttentionBackend（extend/decode + metadata） | 模型无法跑                   |
| P0  | `DeepSeekV4TokenToKVPool` 兼容布局（或设备子类）             | KV 读写错乱 / FlashMLA 对齐失败 |
| P0  | RoPE + 写 SWA cache（fused 或分步）                     | attention 数值错           |
| P0  | MoE gate + grouped expert GEMM + SiLU             | FFN 挂                   |
| P1  | C4 Indexer + topk                                 | `compress_ratio=4` 层不可用 |
| P1  | Compressor c4/c128 + CompressState                | 压缩层不可用 / OOM 策略失效       |
| P1  | TP all-reduce communicator                        | 多卡 TP 不可用               |
| P2  | EP all-to-all（DeepEP 或自研）                         | 大规模 EP 不可用              |
| P2  | FP8（权重/激活/KV nope）                                | 只能 BF16 权重路径，可能与官方盘不兼容  |
| P2  | Graph runner（CUDA Graph 等价）                       | decode 吞吐差              |
| P3  | mHC 加速 / MegaMoE / DSpark / HiSparse              | 功能可先用 PyTorch 慢路径       |




### 6.2 推荐接入形态（照抄 NPU）

NPU 是 in-tree「第三方」样板：不改 `deepseek_v4.py` 主体，只在分流点挂设备实现。

```text
create_dsv4_backend(runner)
  ├─ CUDA → DeepseekV4AttnBackend
  ├─ HIP  → DeepseekV4HipRadixBackend
  └─ NPU  → DeepseekV4AscendAttnBackend   ← 你的设备加一支

_build_dsv4_kv_pool(...)
  ├─ CUDA/HIP → DeepSeekV4TokenToKVPool
  └─ NPU      → DSV4NPUTokenToKVPool       ← 布局不同则子类化

MQALayer / Compressor / Indexer
  └─ MultiPlatformOp 或 backend.forward_* 内 if is_mydevice()
```

对照目录：


| NPU 文件                                                  | 你要仿的职责                                                                                       |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `hardware_backend/npu/attention/ascend_dsv4_backend.py` | 完整 `DeepseekV4AscendAttnBackend`（含 `forward`、`forward_c4_indexer`、`forward_core_compressor`） |
| `hardware_backend/npu/dsv4/dsv4_memory_pool.py`         | 设备 KV / CompressState 布局                                                                     |
| `hardware_backend/npu/dsv4/dsv4_allocator.py`           | 页分配                                                                                          |
| `hardware_backend/npu/dsv4/dsv4_rope.py`                | RoPE 表                                                                                       |
| `hardware_backend/npu/dsv4/dsv4_req_to_token_pool.py`   | req→token 映射特化（若需要）                                                                          |




### 6.3 `mydevice` 最小文件清单

假设设备名 `mydevice`（探测函数 `is_mydevice()`）：

```text
python/sglang/srt/hardware_backend/mydevice/
  ├── __init__.py
  ├── attention/
  │     └── mydevice_dsv4_backend.py   # DeepseekV4MyddeviceAttnBackend
  ├── dsv4/
  │     ├── dsv4_memory_pool.py       # 可选：布局与 CUDA 不同时
  │     ├── dsv4_allocator.py
  │     └── dsv4_rope.py
  ├── communicator.py                 # TP all-reduce
  └── graph_runner.py                 # 可选

# 注册分流（改一处或用 plugin）
srt/layers/attention/attention_registry.py::create_dsv4_backend
  → elif is_mydevice(): return DeepseekV4MyddeviceAttnBackend(runner)

# KV 分流
srt/mem_cache/kv_cache_configurator.py::_build_dsv4_kv_pool
  → is_mydevice() 时构造你的 Pool

# 分布式
srt/distributed/... GroupCoordinator 认 mydevice communicator

# 可选 OOT 包
pyproject：entry_points "sglang.srt.platforms" / "sglang.srt.plugins"
```

**Backend 类最小方法面（对齐 Ascend/CUDA）：**

```python
class DeepseekV4MydeviceAttnBackend(AttentionBackend,
                                    C4IndexerBackendMixin,   # 可先 Python 慢路径
                                    CompressorBackendMixin):
    def __init__(self, runner): ...
    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False): ...
    def init_forward_metadata_in_graph(self, forward_batch): ...
    def forward(self, q, k, v, layer, *, compress_ratio, forward_batch, ...): ...
    # mixin 或自实现：
    def forward_c4_indexer(...): ...
    def forward_core_compressor(...): ...
    def init_cuda_graph_state(self, max_bs, max_num_tokens): ...
```



### 6.4 KV / 数值对齐验收顺序

1. **单层 ratio=0**：只跑 SWA dense，对比 CUDA BF16（或官方）logits / 中间 K pack。
2. **ratio=128**：开 Compressor，检查 C128 页内容与 compress state。
3. **ratio=4**：Indexer topk 集合与 sparse attn 输出。
4. **整网 forward**：`python -m sglang.bench_one_batch --correct --model <V4>`。
5. **多卡 TP**：`all_reduce` 正确性。
6. **EP（可选）**：小 EP 规模 smoke。
7. **投机（可选）**：EAGLE topk=1 或 DSpark。

调试钩子：`kv_canary/pool_patcher/adapters/dsv4.py::attach_dsv4` 可挂 canary 比对池内容。

### 6.5 与 PD / Graph / 量化的交叉点


| 主题         | V4 注意点                                                                                                                               |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| PD disagg  | `disaggregation/prefill.py` & `decode.py` 特判 `DeepSeekV4TokenToKVPool`——传 KV 时要带上 SWA+C4+C128+indexer+compress state，不能只拷 MHA buffer |
| CUDA Graph | Backend 的 out/in-graph metadata 必须可 replay；无 graph 则平台 `support_cuda_graph()→False`                                                  |
| FP8        | CUDA 默认 KV nope FP8；设备若无 FP8，应像 NPU 一样在 overrides 里把 `kv_cache_dtype` 钉成 bf16，并实现匹配的 pack 路径                                         |
| mHC        | 可先 Torch `hc_split_sinkhorn`；再换设备 fused                                                                                              |
| CP         | `validate_deepseek_v4_cp`：策略/并行约束严格；设备侧需支持 metadata reindex                                                                          |




### 6.6 现有平台钩子一览（对照）


| 平台                    | 钩子                                                                                         |
| --------------------- | ------------------------------------------------------------------------------------------ |
| **CUDA**              | `DeepseekV4AttnBackend` + FlashMLA + DeepGEMM / MegaMoE / TileLang mHC                     |
| **HIP**               | `DeepseekV4HipRadixBackend`；`unified_kv`；AIter mHC；默认关 sparse prefill                      |
| **Ascend NPU**        | `DeepseekV4AscendAttnBackend`；`DSV4NPU`* pool/allocator/rope；kv bf16；page 128；HCCL prewarm |
| **XPU**               | mHC 可走 `sgl_kernel`；完整 dsv4 需自补                                                            |
| **KV Canary**         | `attach_dsv4`                                                                              |
| **CP decode attn TP** | `layers/cp/cp_decode_attn_tp.py` 白名单含三个 V4 EntryClass                                      |


扩展新设备时优先挂：`create_dsv4_backend` **分支 → TokenToKVPool 子类 → RoPE/Compressor/Indexer → communicator →（可选）graph runner**。不要从改 `compress_ratios` 语义或绕过 `is_deepseek_v4()` 默认值开始。

---



## 7. 测试 / 文档 / Cookbook 索引



### 7.1 Cookbook & Demo

- `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` — 部署矩阵、MTP、DSpark、MegaMoE、DP-Attn
- `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` + `deepseek-v4-benchmarks.jsx`
- `docs_new/demo/deepseek_v4_flash.ipynb`



### 7.2 测试（节选）


| 区域         | 路径                                                                                                       |
| ---------- | -------------------------------------------------------------------------------------------------------- |
| Attn unit  | `test/registered/attention/unittests/dsv4/`                                                              |
| E2E CUDA   | `test/registered/models_e2e/test_deepseek_v4_flash_fp{4,8}_*.py`、`gb300/`、`cp/`                          |
| AMD        | `test/registered/amd/test_deepseek_v4_{flash,pro}_fp{4,8}*.py`                                           |
| NPU perf   | `test/registered/npu/performance/deepseek_v4_flash/`                                                     |
| Unit       | `test/registered/unit/models/test_deepseek_v4_*.py`、`unit/npu/attention/test_npu_ascend_dsv4_backend.py` |
| Kernels    | `test/registered/kernels/ops/attention/test_deepseek_v4_compress_state_runtime_shapes.py`                |
| DSpark     | `test/registered/spec/dspark/`                                                                           |
| CI recipes | `scripts/ci/slurm/recipes/mi355x-fp{4,8}/dsv4{flash,pro}/`                                               |




### 7.3 改代码落点速查


| 想改…                    | 先打开                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------ |
| 层结构 / mHC / QKV        | `models/deepseek_v4.py`                                                              |
| Sparse attn / metadata | `layers/attention/deepseek_v4_backend.py` + `dsv4/`                                  |
| KV 字节布局 / 池大小          | `mem_cache/deepseek_v4_memory_pool.py` + `kv_cache_configurator._build_dsv4_kv_pool` |
| MoE 路由                 | `models/deepseek_v2.py`（`is_deepseek_v4` 分支）+ `hash_topk.py`                         |
| 启动默认                   | `arg_groups/overrides.py` + `deepseek_v4_hook.py`                                    |
| NextN                  | `models/deepseek_v4_nextn.py`                                                        |
| DSpark                 | `models/deepseek_v4_dspark.py` + `speculative/dspark_components/`                    |
| NPU                    | `hardware_backend/npu/dsv4/` + `ascend_dsv4_backend.py`                              |
| Kernel                 | `kernels/ops/attention/dsv4/`                                                        |


---



## 8. 与 V3.2 DSA 的对照（防混淆）


|                  | V3.2 DSA                 | V4                        |
| ---------------- | ------------------------ | ------------------------- |
| `is_deepseek_*`  | `is_deepseek_dsa`        | `is_deepseek_v4`          |
| KV 主池            | `DSATokenToKVPool` + MLA | `DeepSeekV4TokenToKVPool` |
| Backend 名        | `dsa`                    | `dsv4`                    |
| 注意力              | MLA + indexer topk       | MQA 512 + compress_ratios |
| Residual         | 标准                       | mHC                       |
| `attention_arch` | MLA                      | **MHA**                   |


Indexer / topk 内核目录名仍叫 `dsv4`，部分被 DSA 路径复用（如 `dsa_topk_backend` import `dsv4.topk`）——读代码时以 **模型 arch + pool 类型** 为准，不要只看 kernel 包名。

---

下一篇可回到系列主线：[Mem Cache / Models 精读](./code_reading_deep_dive_zh.md) 或 [投机 / 并行精读](./code_reading_notes_advanced_zh.md)。操作向启动命令以 [Cookbook · DeepSeek-V4](/cookbook/autoregressive/DeepSeek/DeepSeek-V4) 为准。
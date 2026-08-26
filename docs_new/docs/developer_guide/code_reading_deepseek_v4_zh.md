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

本篇是精读系列的 **DeepSeek-V4 专题**，对着源码讲：类层次、forward、压缩 KV、Indexer、MoE/EP、投机与 NPU/HIP 钩子。配套：[系列总目录](./code_reading_notes_zh.md)、[第 2 篇 Cache/Models](./code_reading_deep_dive_zh.md)、[第 3 篇投机/并行](./code_reading_notes_advanced_zh.md)、操作手册 [Cookbook · DeepSeek-V4](/cookbook/autoregressive/DeepSeek/DeepSeek-V4)。

系列导航：[总目录](./code_reading_notes_zh.md) · [第 1 篇](./code_reading_srt_core_zh.md) · [第 2 篇](./code_reading_deep_dive_zh.md) · [第 3 篇](./code_reading_notes_advanced_zh.md) · [第 4 篇](./code_reading_serving_extensions_zh.md) · [第 5 篇](./code_reading_ecosystem_zh.md)

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

| 维度 | V3 / V3.2 | V4 |
|---|---|---|
| Attention 形态 | MLA（低秩 KV）或 DSA indexer + MLA | **MQA head_dim=512**（nope 448 + rope 64），**非 MLA**；`ModelConfig.attention_arch = MHA` |
| 稀疏 | DSA `index_topk` + MLA KV | 层上 `compress_ratios ∈ {0,4,128}`：SWA / C4+Indexer / C128 |
| Residual | 普通 residual | **mHC**（manifold-constrained hyper-connections，`hc_mult` 路混合） |
| MoE gate | grouped `noaux_tc` | V4：`use_grouped_topk=False` + `scoring_func=sqrtsoftplus`；前 `n_hash_layers` 可用 `HashTopK` |
| KV pool | `MLATokenToKVPool` / `DSATokenToKVPool` | **`DeepSeekV4TokenToKVPool`**（SWA + C4 + C128 + indexer + compress state） |
| Attn backend | `fa3` / `flashinfer` / `dsa` … | 强制 **`dsv4`**（CUDA / HIP / Ascend 三路） |
| 投机 | EAGLE / NextN | EAGLE（topk=1）+ **NextN** + **DSpark**（0731） |

---

## 1. 模型文件与类层次

### 1.1 文件地图

| 路径 | 角色 |
|---|---|
| `python/sglang/srt/models/deepseek_v4.py` | Target：`MqaAttentionBase` / `MQALayer` / `DeepseekV4DecoderLayer` / `DeepseekV4Model` / `DeepseekV4ForCausalLM`；`EntryClass = [DeepseekV4ForCausalLM]` |
| `python/sglang/srt/models/deepseek_v4_nextn.py` | MTP draft：`DeepseekV4ModelNextN` / `DeepseekV4ForCausalLMNextN`；单层 decoder，`compress_ratio_override=0` |
| `python/sglang/srt/models/deepseek_v4_dspark.py` | DSpark draft：`DSparkAttention` / `DSparkV4Stage` / `DeepseekV4ForCausalLMDSpark` |
| `python/sglang/srt/configs/deepseek_v4.py` | `DeepSeekV4Config` + `try_detect_fp4_experts` |
| `python/sglang/srt/arg_groups/deepseek_v4_hook.py` | 默认值 / MegaMoE token budget / CP 校验 |
| `python/sglang/srt/arg_groups/overrides.py` | `_deepseek_v4_overrides`、`_deepseek_v4_kv_cache_dtype`、`_deepseek_v4_sm120_moe` |
| `python/sglang/srt/models/deepseek_common/` | 与 V2/V3 共享的 forward helpers、weight loader、AMD fused mHC |
| `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` | ROCm gfx95 fused `hc_post`+`hc_pre` |

Registry：`srt/models/registry.py` 扫 `EntryClass`，三个模块分别注册三个 arch。

### 1.2 `DeepSeekV4Config` 关键字段

文件：`srt/configs/deepseek_v4.py`。

| 字段 | 默认（示意） | 含义 |
|---|---|---|
| `hidden_size` | 4096 | 隐藏维 |
| `num_attention_heads` / `num_key_value_heads` | 64 / **1** | MQA |
| `qk_nope_head_dim` + `qk_rope_head_dim` | 448 + 64 → **head_dim=512** | 与 FlashMLA sparse 对齐 |
| `q_lora_rank` / `o_lora_rank` / `o_groups` | 1024 / 1024 / 8 | Q/O 低秩投影 |
| `kv_lora_rank` | 512 | 配置保留；V4 attention 路径是 MQA 直写 KV，不是 V3 MLA |
| `index_head_dim` / `index_n_heads` / `index_topk` | 128 / 64 / **512** | C4 Indexer |
| `n_routed_experts` / `num_experts_per_tok` | 256 / 6 | MoE |
| `scoring_func` / `topk_method` | `sqrtsoftplus` / `noaux_tc` | V4 TopK 关掉 grouped |
| `n_hash_layers` | 3 | 前几层 `HashTopK` |
| `hc_mult` / `hc_sinkhorn_iters` / `hc_eps` | 4 / 20 / 1e-6 | mHC |
| `compress_ratios` | `List[int]` 每层 0/4/128 | 注意力压缩比 |
| `window_size` | 128 | SWA 窗（HF 里常作 `sliding_window`） |
| `compress_rope_theta` | 40000 | 压缩层 RoPE base |

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

**`DeepseekV4ForCausalLM.forward`**（~2586）：

1. 若 `dsa_enable_prefill_cp`：`can_dsa_cp_split` → `prepare_context_parallel_metadata`；round-robin 时对 `core_meta.apply_cp_reindex()` + `init_flashmla_related` + 重建 `indexer_metadata`。
2. `get_attn_tp_context().maybe_input_scattered` 下调用 `self.model.forward`。
3. PP 非 last 直接返回 proxy；否则拆 `(hidden, pre_hc_head)`（及可选 aux），进 `LogitsProcessor(..., hidden_states_before_norm=pre_hc_head)` —— MTP 需要 pre-norm 隐状态。

**`DeepseekV4Model.forward`**（~2347）：

1. Embed → `unsqueeze(1).repeat(1, hc_mult, 1)`（mHC 多路残差张量）。
2. DP + `moe_a2a=none`：`dp_gather_replicate` 得全局 `input_ids`（hash MoE / TopK 用）。
3. Prefill CP：`cp_split_and_rebuild_data/position` + `cp_round_robin_input_ids`。
4. 清 `forward_batch.freqs_cis_c4/c128` 缓存。
5. 层循环：`DeepseekV4DecoderLayer`；或 `_can_run_tbo` 时走 `_forward_layers_tbo`（prefill-only two-batch-overlap）。
6. Last PP：`hc_head` → `norm`；返回 `(hidden_states, pre_hc_head)`。

**`DeepseekV4DecoderLayer.forward`**（~1634）：

```text
hc_pre(attn) → [可选 fused rms+fp8] → self_attn
→ hc_post / fused_post_pre → hc_pre(ffn) → MoE → hc_post
```

mHC 实现选型（环境变量门控）：

| 路径 | 门控 |
|---|---|
| TileLang `mhc_pre` / `mhc_post` / `mhc_fused_post_pre` | `SGLANG_OPT_USE_TILELANG_MHC_*` |
| AIter（HIP） | `SGLANG_OPT_USE_AITER_MHC_*` |
| NPU custom op | `_is_npu` → `npu_hc_pre` / `npu_hc_post` |
| DeepGEMM TF32 prenorm | `SGLANG_OPT_DEEPGEMM_HC_PRENORM` |
| Torch fallback | `hc_split_sinkhorn` + 手工 mix |

AMD：`try_fused_hc_post_pre`（`deepseek_common/amd/deepseek_v4_fused_mhc.py`）。

### 1.5 `MQALayer` 注意力核心

**准备 Q/K**（`_forward_prepare` / multi-stream 变体）：

1. `wqkv_a` 或 `wq_a`+`wkv` → q_lora / kv。
2. Q：`q_norm` → `wq_b` → `fused_q_norm_rope`（`kernels/ops/attention/dsv4`）。
3. K：默认 **`_compute_kv_to_cache`** → `DeepSeekV4TokenToKVPool.set_swa_key_buffer_radix_fused_norm_rope`（norm+RoPE+写 FlashMLA 分页 cache 融合）；DSA CP / unified_kv 等场景保留 bf16 KV 再 `store_cache`。
4. `compress_ratio==4`：`C4Indexer.forward`（写 indexer cache + topk）。
5. `compress_ratio in (4,128)`：`attn_backend.forward_core_compressor`。

**Attention 调用**：`attn_backend.forward(q, k=v, layer=attn_mqa, compress_ratio, attn_sink, save_kv_cache=...)`。K≡V（assert `k is v`）。

**O 投影**：可选逆 RoPE（NPU/ fused）→ `wo_a`（分组低秩，可 FP8 DeepGEMM）→ `wo_b`。

**RoPE**：`rope_type = "deepseek_yarn"`；`precompute_freqs_cis`（`kernels/ops/attention/deepseek_v4_rope.py`）；压缩层用 `compress_rope_theta`。

### 1.6 NextN / MTP

`deepseek_v4_nextn.py`：

- `COMPRESS_RATIO_NEXTN_LAYER = 0`：draft 层不做 C4/C128。
- `DeepseekV4ModelNextN`：`enorm`/`hnorm` + `e_proj`/`h_proj` 把 **target 传来的 `spec_info.hidden_states`（已是 hc_mult 展平）** 与 token embed 合成 mHC 状态，再进单层 `DeepseekV4DecoderLayer(is_nextn=True)`。
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

## 2. Layers / Ops（按算子族）

### 2.1 Attention backend 选型

`srt/layers/attention/attention_registry.py::create_dsv4_backend`：

| 平台 | 类 | 文件 |
|---|---|---|
| CUDA | `DeepseekV4AttnBackend` | `layers/attention/deepseek_v4_backend.py` |
| HIP/ROCm | `DeepseekV4HipRadixBackend` | `layers/attention/deepseek_v4_backend_hip_radix.py` |
| Ascend NPU | `DeepseekV4AscendAttnBackend` | `hardware_backend/npu/attention/ascend_dsv4_backend.py` |

默认由 `_deepseek_v4_overrides` 设 `attention_backend="dsv4"`（`page_size=256`；NPU=`128` 且 pin prefill/decode backend）。别名 `compressed` → `dsv4`（deprecated）。

**`DeepseekV4AttnBackend` 职责：**

- `init_forward_metadata*`：构造 `DSV4Metadata` / verify / decode metadata（`layers/attention/dsv4/metadata.py`）。
- Mixin：`C4IndexerBackendMixin`（`dsv4/indexer.py`）、`CompressorBackendMixin`（`dsv4/compressor.py`）。
- `forward`：按 `compress_ratio` 选 SWA dense / C4 sparse（FlashMLA）/ C128 sparse；sparse prefill 可走 `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL`。
- MTP：`DeepseekV4MultiStepBackend` 包多步 draft；`OnlineC128MTPController`（`kernels/.../online_c128_mtp.py`）。
- HiSparse：`hisparse_coordinator` 与 C4 device pool 索引翻译。

HIP：`unified_kv` Triton 路径（`kernels/ops/attention/dsv4/unified_kv_kernels/`），2-source prefill 等仅 Hip backend。

### 2.2 Indexer / Compressor

| 组件 | 类 | 计算 |
|---|---|---|
| C4 Indexer | `C4Indexer` | 从 hidden/q_lora 出 indexer Q/K，写 `DeepSeekV4IndexerPool`，`topk`（默认 512）供 sparse attn |
| Compressor | `Compressor` / `compressor_v2` / `compress_hip` | 把 SWA token 池化到 C4/C128；维护 `CompressStatePool` ring / online state |
| TopK backend | `DSATopKBackend`（`dsa/dsa_topk_backend.py`） | 可复用 `dsv4.topk.topk_transform_512_v2` 等 |

FP4 indexer：`--enable-deepseek-v4-fp4-indexer`（需 SM100/SM120）。

### 2.3 MoE / fused experts / EP

复用 **`deepseek_v2.DeepseekV2MoE`**，构造时 `is_deepseek_v4=True`：

- 前 `n_hash_layers`：`HashTopK`（`layers/moe/hash_topk.py` → `kernels/.../dsv4.hash_topk`）；NextN 关掉 hash。
- 其余：`TopK(use_grouped_topk=False, scoring_func=sqrtsoftplus, is_fp4_experts=...)`。
- Shared expert fusion / DeepEP·MegaMoE per-rank shared slots：与 V3 同框架（`has_per_rank_fused_shared_slots`）。
- Runner：`auto` → SM120 `flashinfer_mxfp4`；NVFP4 hybrid → `flashinfer_trtllm_routed`；MegaMoE：`--moe-a2a-backend megamoe` + DeepGEMM kernels（`layers/moe/mega_moe.py` 调 `dsv4.mega_moe_pre_dispatch`；`moe_runner/deep_gemm.py` 调 `silu_and_mul_*_post_quant`）。

DP-Attention + TP-MoE：`layers/dp_attention.py` 注释明确 DP gather 路径给 deepseek_v4。

### 2.4 RoPE / RMSNorm / Linear / Quant

| 算子 | 位置 | 说明 |
|---|---|---|
| YaRN freqs | `kernels/ops/attention/deepseek_v4_rope.py::precompute_freqs_cis` | β_fast/slow + factor |
| fused Q norm+RoPE | `dsv4.fused_q_norm_rope` | MQALayer `_compute_q_b` |
| fused K norm+RoPE+store | `set_swa_key_buffer_radix_fused_norm_rope` / `fused_k_norm_rope_flashmla` | 写 KV |
| NPU RoPE table | `hardware_backend/npu/dsv4/dsv4_rope.py::Dsv4NpuRoPE` | cos/sin cache |
| RMSNorm | `layers/layernorm.RMSNorm`；mHC 内可融合 | |
| Linear | `ReplicatedLinear` / `ColumnParallelLinear` / `RowParallelLinear` | QKV/O；wo_a 可 FP8 |
| FP8 wo_a | `_FP8_WO_A_GEMM` + DeepGEMM scale UE8M0 | `post_load_weights._setup_fp8_wo_a_scales` |
| FP4 experts | `try_detect_fp4_experts` 探 safetensors dtype | U8/I8/F4 vs F8_E4M3 |

### 2.5 Logits

`layers/logits_processor.LogitsProcessor` — V4 传入 `hidden_states_before_norm=pre_hc_head`（或 aux），供 speculative / 分析；采样仍走通用 `SamplingBatchInfo` 路径。

Tool call：`function_call/deepseekv4_detector.py::DeepSeekV4Detector`（继承 V3.2 DSML 标签）。模板：`parser/template_detection.py` rule `deepseek_v4`。

### 2.6 DeepSeek 专用 kernels 目录

`python/sglang/kernels/ops/attention/dsv4/`（核心）：

| 模块 | 用途 |
|---|---|
| `attn.py` | sparse/dense attn helpers |
| `compress.py` / `compress_old.py` / `fused_compress_triton.py` | 压缩 |
| `topk.py` | indexer topk + `plan_topk_v2` |
| `quant_k_cache.py` / `dequant_k_cache.py` | FP8 nope + BF16 rope pack |
| `index_buf_accessor.py` | 分页 KV 读写 |
| `fp8_wo_a.py` / `elementwise.py` / `gemm.py` | 量化与 GEMM |
| `moe.py` | MegaMoE / silu quant |
| `online_c128_mtp.py` | online C128 + MTP |
| `sparse_prefill_kernels.py` | sparse prefill |
| `unified_kv_kernels/` | HIP unified KV |
| `fp4_indexer.py` | FP4 indexer |
| `c128_cleanup.py` | draft reject 清理 |

另：`deepseek_v4_rope.py`；AOT `kernels/aot/csrc/elementwise/deepseek_v4_topk.cu`；MHC：`kernels/ops/layernorm/mhc*.py`；DSpark：`kernels/ops/speculative/dspark/`。

选型惯例：**模型层调 Python API → backend / MultiPlatformOp 按 CUDA/HIP/NPU dispatch → JIT/AOT kernel**。

---

## 3. KV / Cache 路径

### 3.1 池结构（不是 MLATokenToKVPool）

V4 **不**用 `MLATokenToKVPool` / `DSATokenToKVPool` 作为主池。装配点：`mem_cache/kv_cache_configurator.py::_build_dsv4_kv_pool`。

**`DeepSeekV4TokenToKVPool`**（`mem_cache/deepseek_v4_memory_pool.py`，继承 `BaseSWAKVPool`）：

```text
DeepSeekV4TokenToKVPool
  ├── swa_kv_pool: DeepSeekV4SingleKVPool     # 滑窗 token（page 常 128）
  ├── c4_kv_pool:  DeepSeekV4SingleKVPool | HiSparseC4DevicePool
  ├── c128_kv_pool: DeepSeekV4SingleKVPool
  ├── c4_indexer_kv_pool: DeepSeekV4IndexerPool
  ├── CompressStatePool × {c4, c128}          # ring / online 压缩状态
  └── (可选) unified_kv_pool: DeepSeekV4UnifiedKVPool  # HIP Triton unified
```

**单 token 布局**（`DeepSeekV4SingleKVPool` assert）：

```text
qk_nope FP8 (448) + qk_rope BF16 (64×2) + FP8 scales + pad = 584 bytes/token
页按 576 对齐（FlashMLA）
```

NPU：`DSV4NPUTokenToKVPool`（paged state，非 CUDA ring）；`page_size=128`；kv dtype 默认 **BF16**（CUDA 默认 **fp8_e4m3**）。

### 3.2 Prefill vs Decode 布局

| 阶段 | 行为 |
|---|---|
| Prefill (EXTEND) | 写入 SWA；按层 ratio 跑 Compressor → C4/C128；C4 层跑 Indexer 得 topk；sparse/dense FlashMLA 读压缩 cache |
| Decode | SWA 滑窗推进；online/offline 更新 compress state；C4 topk 更新；C128 online 可用 `SGLANG_OPT_USE_ONLINE_COMPRESS`（ring_size=1） |
| Spec verify | `DSV4RawVerifyMetadata`；ragged verify（`SGLANG_RAGGED_VERIFY_MODE`）；拒绝 draft 时 `clear_unaccepted_c128_draft_states` |

`get_compress_state_ring_size(ratio, is_speculative)`：非投机 c4=8 / c128=128；投机放大到 16 / 256；online c128 → 1（与 MTP 默认互斥，除非 `SGLANG_EXPERIMENTAL_ONLINE_C128_MTP`）。

### 3.3 RadixCache 与压缩 KV

- 树节点仍按 **token 前缀** 匹配（通用 `RadixCache` / unified radix）。
- 物理页由 **SWA allocator + 压缩池尺寸** 约束：`SWATokenToKVPoolAllocator` / `DeepSeekV4HiSparseTokenToKVPoolAllocator`（`mem_cache/allocator/swa.py`、`hisparse.py`）。
- HiCache：`CacheController` 把 `DeepSeekV4TokenToKVPool` 视为 **rank-replicated compressed MLA-style**（仅需 rank0 写回 storage，`is_mla_model=True` 语义）。
- HiSparse：C4 落 `HiSparseC4DevicePool`，`full→compressed→device` 索引翻译；`managers/hisparse_coordinator.py`。
- Hybrid assembler：`hybrid_cache/hybrid_pool_assembler.py` 识别 DSV4 组件集合。

**含义：** Radix 命中的是逻辑 token 前缀；压缩态 / indexer 缓冲与 SWA 页生命周期绑定，驱逐时需同步 compress state（NPU：`maybe_evict_dsv4_state_on_swa`）。

### 3.4 投机缓存含义

| 算法 | KV 影响 |
|---|---|
| EAGLE/NextN | Draft 层 `compress_ratio=0`，主要用 SWA；target verify 写完整压缩路径；`speculative_eagle_topk` **必须为 1** |
| DSpark | Draft 写同一 `DeepSeekV4TokenToKVPool` SWA；`write_target_hidden_kv` / `CommitKvProj` 注入；ragged verify 布局 |
| Online C128 MTP | 实验；pending seq lens 缓冲；与 CP / ragged verify 组合有硬限制 |

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

| 项 | 默认 |
|---|---|
| `--attention-backend` | `dsv4` |
| `--page-size` | 256（NPU 128） |
| `--kv-cache-dtype` | `fp8_e4m3`（NPU `bfloat16`） |
| `--swa-full-tokens-ratio` | `0.1`（若用户未改） |
| `--max-running-requests` | 256（若 None） |
| SM120 MoE runner | `flashinfer_mxfp4`（auto） |
| NVFP4 MoE runner | `flashinfer_trtllm_routed` |
| 投机算法白名单 | `EAGLE` / `DSPARK`；EAGLE 要求 `eagle_topk==1` |

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

| Env | 作用 |
|---|---|
| `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL` | sparse prefill；ROCm/CP 下 hook 会关掉 |
| `SGLANG_OPT_USE_ONLINE_COMPRESS` | online C128 |
| `SGLANG_EXPERIMENTAL_ONLINE_C128_MTP` | online C128 + MTP |
| `SGLANG_OPT_FUSE_WQA_WKV` | 融合 wqkv_a |
| `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` | Q/K/Indexer 多流 |
| `SGLANG_OPT_USE_TILELANG_MHC_*` / AITER | mHC kernel |
| `SGLANG_DSV4_MHC_PREWARM` | load 时预热 mHC |
| `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE` | MegaMoE |
| `SGLANG_RAGGED_VERIFY_MODE` | verify 紧凑布局 |
| `SGLANG_DSPARK_FAST_KERNEL` | DSpark fused Q/RoPE |

完整列表遵循 `environ.Envs` 与 [env-var-conventions](../../../.claude/skills/env-var-conventions/SKILL.md)。

---

## 6. 三方设备集成钩子

| 平台 | 钩子 |
|---|---|
| **CUDA** | `DeepseekV4AttnBackend` + FlashMLA + DeepGEMM / MegaMoE / TileLang mHC |
| **HIP/ROCm** | `DeepseekV4HipRadixBackend`；`unified_kv`；AIter mHC/rope；默认关 sparse prefill；`set_force_ck_w8a8` / `set_batched_rope` 在 `DeepseekV4ForCausalLM.__init__` |
| **Ascend NPU** | `DeepseekV4AscendAttnBackend`；`DSV4NPUTokenToKVPool` / `dsv4_allocator` / `dsv4_rope` / `dsv4_req_to_token_pool`；kv dtype bf16；page 128；Scheduler HCCL prewarm；graph runner 认 `is_deepseek_v4` |
| **XPU** | mHC `hc_split_sinkhorn` 走 `sgl_kernel` |
| **KV Canary** | `kv_canary/pool_patcher/adapters/dsv4.py::attach_dsv4` |
| **PD disagg** | `disaggregation/prefill.py` & `decode.py` 特判 `DeepSeekV4TokenToKVPool` |
| **CP decode attn TP** | `layers/cp/cp_decode_attn_tp.py` arch 白名单含三个 V4 EntryClass |

扩展新设备时优先挂：**attention_registry `"dsv4"` 分支 → TokenToKVPool 子类 → RoPE/Compressor MultiPlatformOp →（可选）graph runner**。

---

## 7. 测试 / 文档 / Cookbook 索引

### 7.1 Cookbook & Demo

- `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` — 部署矩阵、MTP、DSpark、MegaMoE、DP-Attn
- `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx` + `deepseek-v4-benchmarks.jsx`
- `docs_new/demo/deepseek_v4_flash.ipynb`

### 7.2 测试（节选）

| 区域 | 路径 |
|---|---|
| Attn unit | `test/registered/attention/unittests/dsv4/` |
| E2E CUDA | `test/registered/models_e2e/test_deepseek_v4_flash_fp{4,8}_*.py`、`gb300/`、`cp/` |
| AMD | `test/registered/amd/test_deepseek_v4_{flash,pro}_fp{4,8}*.py` |
| NPU perf | `test/registered/npu/performance/deepseek_v4_flash/` |
| Unit | `test/registered/unit/models/test_deepseek_v4_*.py`、`unit/npu/attention/test_npu_ascend_dsv4_backend.py` |
| Kernels | `test/registered/kernels/ops/attention/test_deepseek_v4_compress_state_runtime_shapes.py` |
| DSpark | `test/registered/spec/dspark/` |
| CI recipes | `scripts/ci/slurm/recipes/mi355x-fp{4,8}/dsv4{flash,pro}/` |

### 7.3 改代码落点速查

| 想改… | 先打开 |
|---|---|
| 层结构 / mHC / QKV | `models/deepseek_v4.py` |
| Sparse attn / metadata | `layers/attention/deepseek_v4_backend.py` + `dsv4/` |
| KV 字节布局 / 池大小 | `mem_cache/deepseek_v4_memory_pool.py` + `kv_cache_configurator._build_dsv4_kv_pool` |
| MoE 路由 | `models/deepseek_v2.py`（`is_deepseek_v4` 分支）+ `hash_topk.py` |
| 启动默认 | `arg_groups/overrides.py` + `deepseek_v4_hook.py` |
| NextN | `models/deepseek_v4_nextn.py` |
| DSpark | `models/deepseek_v4_dspark.py` + `speculative/dspark_components/` |
| NPU | `hardware_backend/npu/dsv4/` + `ascend_dsv4_backend.py` |
| Kernel | `kernels/ops/attention/dsv4/` |

---

## 8. 与 V3.2 DSA 的对照（防混淆）

| | V3.2 DSA | V4 |
|---|---|---|
| `is_deepseek_*` | `is_deepseek_dsa` | `is_deepseek_v4` |
| KV 主池 | `DSATokenToKVPool` + MLA | `DeepSeekV4TokenToKVPool` |
| Backend 名 | `dsa` | `dsv4` |
| 注意力 | MLA + indexer topk | MQA 512 + compress_ratios |
| Residual | 标准 | mHC |
| `attention_arch` | MLA | **MHA** |

Indexer / topk 内核目录名仍叫 `dsv4`，部分被 DSA 路径复用（如 `dsa_topk_backend` import `dsv4.topk`）——读代码时以 **模型 arch + pool 类型** 为准，不要只看 kernel 包名。

---

下一篇可回到系列主线：[Mem Cache / Models 精读](./code_reading_deep_dive_zh.md) 或 [投机 / 并行精读](./code_reading_notes_advanced_zh.md)。操作向启动命令以 [Cookbook · DeepSeek-V4](/cookbook/autoregressive/DeepSeek/DeepSeek-V4) 为准。

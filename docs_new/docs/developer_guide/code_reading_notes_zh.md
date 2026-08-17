---
title: "SGLang 代码精读系列总目录"
description: "SGLang 全仓库实现级精读系列的入口：五篇深度章节覆盖 SRT 主路径、缓存与模型层、进阶子系统、服务扩展与生态组件；附部署要点与学习路线。"
keywords:
  - sglang
  - code reading
  - architecture
  - developer guide
---

本系列是 **实现级精读**，不是导航清单。每一篇都对着源码讲：类与方法、数据结构、控制流、IPC 契约、不变量与改代码落点。当前合计约 **2900+ 行**，按模块拆成多篇，避免单文件不可读。

> **Note:** 若你只看到过早期「部署与开发」短文：那是系列第 0 篇的雏形。请按下面顺序读 **第 1～5 篇**，那里才是深度解读。

## 系列章节（按推荐顺序）

| 篇 | 文档 | 覆盖模块 | 深度目标 |
|---|---|---|---|
| **1** | [SRT 核心服务路径](./code_reading_srt_core_zh.md) | Launch / Engine / PortArgs·ZMQ / `io_struct` / TokenizerManager / Scheduler（event loop、batch、overlap）/ TpModelWorker·ModelRunner / Detokenizer / RuntimeContext | 能改启动装配、IPC、调度主循环、forward 边界 |
| **2** | [Mem Cache / Models / Layers / Sampling](./code_reading_deep_dive_zh.md) | RadixCache 匹配·split·insert·evict·lock / 页分配器 / HiCache / ModelRegistry·llama 结构·权重加载 / Attention·MoE·Quant / SamplingBatchInfo | 能改前缀缓存、加模型、换 backend、接量化 |
| **3** | [投机 / PD / LoRA / 并行 / Kernel](./code_reading_notes_advanced_zh.md) | Speculative V2 workers / PD 状态机与 KV transfer / LoRA 双进程 / TP·PP·DP·EP / JIT·AOT kernels | 能接投机算法、PD 后端、LoRA slot、并行组 |
| **4** | [服务扩展模块](./code_reading_serving_extensions_zh.md) | OpenAI·Anthropic HTTP / Grammar 约束解码 / Tool calling / VLM 多模态 / Session / Metrics·Trace / 权重热更新·RL / CUDA Graph·torch.compile / PrefillAdder·SchedulePolicy / Hardware backend | 能改 API 适配、结构化输出、VLM、调度策略、编译路径 |
| **5** | [生态组件](./code_reading_ecosystem_zh.md) | Frontend Language（`lang/`）/ sgl-model-gateway / multimodal_gen（Diffusion）/ Test·CI | 能区分 DSL vs SRT、网关路由、扩散运行时、加 CI 测试 |

官方配套（操作手册，非精读）：[Install](/docs/get-started/install)、[Contribution Guide](/docs/developer_guide/contribution_guide)、[Support New Models](/docs/supported-models/support_new_models)、[Server Arguments](/docs/advanced_features/server_arguments)。

## 一张总图：仓库怎么拼起来

```text
                         ┌─────────────────────────┐
                         │  sgl-model-gateway      │  (Rust: LB / PD 路由 / IGW)
                         └────────────┬────────────┘
                                      │ HTTP / gRPC
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
┌─────────────────┐      ┌────────────────────┐      ┌─────────────────────┐
│ SRT HTTP/Engine │      │ SRT Prefill 集群    │      │ SRT Decode 集群      │
│ TokenizerMgr    │      │ Scheduler(P)        │      │ Scheduler(D)         │
│ Detokenizer     │      │ KV send             │──────│ KV recv → decode     │
└────────┬────────┘      └────────────────────┘      └─────────────────────┘
         │ ZMQ + io_struct
         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Scheduler → TpModelWorker → ModelRunner → models/* + RadixAttention      │
│              ↑ LoRA / Spec draft-verify / Grammar mask / MM embed         │
│              ↑ mem_cache (RadixCache + paged KV) + distributed groups     │
└──────────────────────────────────────────────────────────────────────────┘

旁路（不走上述 LLM serving 主路径）：
  lang/            — 可编程 DSL，可后端连 Runtime/Engine
  multimodal_gen/  — 扩散 / 图生视频独立运行时
  kernels/         — JIT/AOT 算子，被 SRT layers 调用
```

## 心智模型（读第 1 篇前先记住）

1. **三进程**：主进程 tokenize + HTTP；每 GPU rank 一个 Scheduler 做 forward；Detokenizer 独立进程。
2. **IPC**：ZMQ + `managers/io_struct.py`（msgspec）；大多模态可走 shm。
3. **RadixAttention / RadixCache**：前缀共享 KV；模型里用 `RadixAttention` 而不是裸 Attention。
4. **配置**：`ServerArgs` 是种子；业务读 `RuntimeContext` namespace bags；环境变量进 `environ.Envs`。

## 部署速查（细节仍以官方 Install 为准）

```bash
# 开发安装
pip install -e "python" && pre-commit install

# 推荐启动
sglang serve --model-path Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 30000

# 多卡 / 多机
sglang serve --model-path MODEL --tp 8
python3 -m sglang.launch_server --model-path MODEL --tp 16 \
  --dist-init-addr HOST:20000 --nnodes 2 --node-rank 0
```

关键 CLI 字段：`--model-path`、`--tp`/`--pp`/`--dp`/`--ep`、`--mem-fraction-static`、`--attention-backend`、`--quantization`、`--speculative-algorithm`、`--enable-lora`、`--disaggregation-mode`。完整列表见 `srt/server_args.py` 与 [Server Arguments](/docs/advanced_features/server_arguments)。

## 学习路线

### 路线 A：只想部署运维

总目录（本文）→ 官方 Install / Multi-node / PD → 第 1 篇 §1 Launch·PortArgs → 第 3 篇 §2 PD（若用分离）→ 第 5 篇 Gateway。

### 路线 B：改 serving 核心（调度 / 缓存 / forward）

第 **1** 篇全文 → 第 **2** 篇 Mem cache + Models → 第 **4** 篇 PrefillAdder / CUDA Graph → 本地改一行日志跟 rid。

### 路线 C：加模型 / 量化 / attention

第 **2** 篇 Models + Layers + Quant → [Support New Models](/docs/supported-models/support_new_models) → `bench_one_batch --correct`。

### 路线 D：投机 / PD / LoRA / 多卡

第 **1** 篇 Scheduler·Worker → 第 **3** 篇全文 → 第 **4** 篇权重热更新（RL）。

### 路线 E：API / 结构化输出 / VLM

第 **4** 篇 §1–4 → 第 **1** 篇 TokenizerManager 多模态段。

## 开发约定（改代码前）

| 主题 | 读 |
|---|---|
| RuntimeContext / bags / override | `.claude/skills/sglang-runtime-context/SKILL.md` |
| Scheduler / TokenizerManager / ModelRunner 风格 | `.claude/skills/large-class-style/SKILL.md` |
| `SGLANG_*` 环境变量 | `.claude/skills/env-var-conventions/SKILL.md` |
| 写测试 / CI suite | `test/README.md`、`.claude/skills/write-sglang-test/SKILL.md` |
| JIT / AOT kernel | `.claude/skills/add-jit-kernel`、`add-sgl-kernel` |

## 符号速查（跨篇）

| 符号 | 文件 | 篇 |
|---|---|---|
| `prepare_server_args` / `PortArgs` | `srt/server_args.py` | 1 |
| `Engine._launch_subprocesses` | `srt/entrypoints/engine.py` | 1 |
| `TokenizerManager.generate_request` | `srt/managers/tokenizer_manager.py` | 1 |
| `Scheduler.event_loop_*` / `run_batch` | `srt/managers/scheduler.py` | 1 |
| `ModelRunner.forward` | `srt/model_executor/model_runner.py` | 1 |
| `RadixCache.match_prefix` | `srt/mem_cache/radix_cache.py` | 2 |
| `EntryClass` / `ModelRegistry` | `srt/models/registry.py` | 2 |
| `EAGLEWorkerV2` | `srt/speculative/eagle_worker_v2.py` | 3 |
| `PrefillAdder.add_one_req` | `srt/managers/schedule_policy.py` | 4 |
| `GrammarManager` | `srt/constrained/grammar_manager.py` | 4 |
| `OpenAIServingChat` | `srt/entrypoints/openai/serving_chat.py` | 4 |
| `@sgl.function` / `StreamExecutor` | `lang/api.py`、`lang/interpreter.py` | 5 |

下一篇请直接打开：[SRT 核心服务路径精读](./code_reading_srt_core_zh.md)。

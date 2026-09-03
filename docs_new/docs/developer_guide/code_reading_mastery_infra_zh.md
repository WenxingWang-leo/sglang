---
title: "SGLang 精通路线（已有模型/设备适配经验）"
description: "给已经在第三方设备上适配过模型的 LLM infra 工程师：补齐调度、缓存契约、投机/PD、运行时配置，用可自检的掌握标准积累，而不是再读一遍硬件接入。"
keywords:
  - sglang
  - mastery
  - infra
  - scheduler
  - radix cache
---

读者假设：**已经在 SGLang 上把多个模型跑到第三方 device**——熟悉 `models/*`、`RadixAttention`、attention backend、KV pool 布局、quant/MoE 分流、communicator。本页不再讲「怎么注册一个 backend」，只讲你通常还没被适配工作逼到、却决定「能不能称为精通 SRT」的那一层。

配套精读：[总目录](./code_reading_notes_zh.md) · [第 1 篇 SRT 核心](./code_reading_srt_core_zh.md) · [第 2 篇 Cache](./code_reading_deep_dive_zh.md) · [第 3 篇进阶](./code_reading_notes_advanced_zh.md) · [第 4 篇 PrefillAdder/Graph](./code_reading_serving_extensions_zh.md) · [V4 专题](./code_reading_deepseek_v4_zh.md)

---

## 0. 你现在大概在哪

设备适配工程师常见能力剖面：

| 已经很熟 | 往往只有「用过 / 绕过」 |
|---|---|
| 模型 `forward`、层结构、权重加载 | TokenizerManager / `io_struct` / 多进程 ZMQ |
| AttentionBackend + 设备 kernel | Scheduler 组 batch、overlap、抢占 |
| KV 字节布局、page、allocator 接口 | Radix `lock_ref` / chunked insert / 与准入的耦合 |
| TP all-reduce、有时 EP A2A | DP-attn、PD 状态机、投机 V2 接缝 |
| `is_*()` 分流、NPU/自研 pool | RuntimeContext bags / `override` / 角色 publish |
| `bench_one_batch --correct` | `PrefillAdder`、grammar、权重热更新、gateway |

精通 SRT **不是**再适配第 N 个模型，而是能在**不改模型文件**的情况下，解释并改动：一条请求如何被准入、如何复用前缀、如何和 GPU overlap、投机/PD 如何换掉 `model_worker`、配置如何进各进程。

---

## 1. 精通的操作定义（自检，不要只「读过」）

对每个条目，合上文档能讲清「数据从哪来、写到哪、错了会怎样」。

**P0（没有这些，只是设备专家，不是 SRT 专家）**

1. 画出本机 `tp=4` 的**进程数**和 ZMQ 拓扑；指出谁 bind、谁 connect；非 0 rank 如何拿到请求。
2. 从 `GenerateReqInput` 追到 `TokenizedGenerateReqInput` → `Req` → `ScheduleBatch` → `ForwardBatch`，每个结构多了什么字段。
3. 默写 `event_loop_overlap`：收包、组 batch、`run_batch`、`result_queue`、何时不能 overlap。
4. 解释 `match_prefix` + `lock_ref` + `cache_unfinished_req` / `cache_finished_req`；chunked prefill 为什么 `chunked=True` 不抬 hit。
5. 指出 `PrefillAdder.add_one_req` 用哪些预算（token、SWA、page 对齐、`new_token_ratio`）拒绝或切 chunk。
6. 说明 `ServerArgs` 为何不能当运行时真相；举一个必须 `get_context().override` 的例子。

**P1（serving 系统级）**

7. 投机：Scheduler 如何把 `model_worker` 换成 `*WorkerV2`；一步 decode 的 draft/verify/extend；`spec_info` 如何活到下一 iter。
8. PD：Prefill/Decode 队列与 bootstrap room；KV 之外还传哪些 `StateType`；Decode 为何能 `is_prebuilt` 跳过 prefill compute。
9. CUDA Graph：out-graph vs in-graph metadata；你设备上 `support_cuda_graph=False` 时吞吐差在哪一层。
10. DP-attn + TP-MoE：哪些 tensor 在 DP gather、哪些仍 TP；V4 hash MoE 为什么要全局 `input_ids`。

**P2（按你业务选，不必全做）**

11. LoRA 双进程：Registry vs Manager，slot 换页。
12. Grammar：异步编译 → vocab mask 与 overlap 的冲突。
13. 权重热更新四条 RPC（disk/distributed/tensor/ipc）与 RL 框架怎么挂。
14. Gateway cache-aware 路由 vs 单实例 LPM：两层分别解决什么。

---

## 2. 推荐学习顺序（针对你的背景）

**不要从第 6 篇 / 硬件接入再读一遍。** 那是你已经交付过的层。按下面顺序对着**源码**读精读篇，每阶段用 §1 的自检收口。

### 阶段 A — 控制面与进程（补最大盲区）

精读：[第 1 篇](./code_reading_srt_core_zh.md) §1–3、§7

源码必跟：

| 文件 | 跟什么 |
|---|---|
| `srt/entrypoints/engine.py` `_launch_subprocesses` | 装配顺序、`node_rank>=1`、DP controller 条件 |
| `srt/server_args.py` `PortArgs.init_new` | ipc vs tcp |
| `srt/managers/io_struct.py` | `GenerateReqInput` → `Tokenized*` → `BatchTokenIDOutput` → `BatchStrOutput` |
| `srt/managers/tokenizer_manager.py` | `generate_request` / `_send_one_request` / `handle_loop` |
| `srt/managers/scheduler_components/request_receiver.py` | rank0 ZMQ + 组内广播 |
| `srt/runtime_context.py` + skill `sglang-runtime-context` | `publish(role=)`、bags、`override` |

动手（用你已有设备或 CPU/小模型即可）：在 tokenizer 发送和 scheduler `handle_generate_request` 各打 `rid`，确认贯通；故意 `tp=2` 看只有一个进程打到 ZMQ recv。

### 阶段 B — 调度心脏

精读：第 1 篇 §4–5；第 4 篇 PrefillAdder / SchedulePolicy

源码必跟：

| 文件 | 跟什么 |
|---|---|
| `scheduler.py` `dispatch_event_loop` / `event_loop_overlap` / `run_batch` | 主循环全部分支，不要只看 normal |
| `schedule_batch.py` `Req`、`ScheduleBatch.merge` / `filter` | running vs waiting |
| `schedule_policy.py` `calc_priority`、`PrefillAdder.add_one_req` | LPM/FCFS、预算、chunk、preempt |

自检：给一个「waiting 里 3 条、一条超长 prefix hit、一条 ignore_eos」的情景，口头走完本 iter 谁进 batch。

### 阶段 C — 缓存契约（你懂 pool，补树与准入）

精读：[第 2 篇](./code_reading_deep_dive_zh.md) §1（Radix / allocator / registry）

源码必跟：`radix_cache.py` 的 `match_prefix`、`_split_node`、`cache_*_req`、`evict`；`allocator/paged.py` `alloc_extend`；`kv_cache_builder.py` + `registry.py` 选树。

把你设备上的 pool **画进这张图**：Radix 逻辑 index → `ReqToTokenPool` → 你的 allocator → 你的 KV bytes。能指出驱逐时哪些设备状态必须一起回收（V4 就是 compress state）。

V4 设备同事加读：[V4 专题](./code_reading_deepseek_v4_zh.md) §2–3、§6。

### 阶段 D — 换掉 worker 的两条大路

精读：[第 3 篇](./code_reading_notes_advanced_zh.md) §1–2、§4

- 投机：`spec_info.py` `create_worker` → `eagle_worker_v2.forward_batch_generation` → Scheduler `maybe_init_draft_worker`
- PD：`disaggregation/prefill.py`、`decode.py` 文件头状态机 + `utils.get_kv_class`
- 并行：`parallel_state.initialize_model_parallel` + `data_parallel_controller.py` 前部 LB

若你们线上没有 PD/投机，也建议读完接口；这是和「只跑 eager forward」的分水岭。

### 阶段 E — 配置纪律与上游贡献

- `.claude/skills/sglang-runtime-context/SKILL.md`
- `.claude/skills/large-class-style/SKILL.md`（三大类怎么加代码）
- `.claude/skills/env-var-conventions/SKILL.md`
- 第 5 篇 CI 节 + `test/README.md`：你的设备测试如何 `register_*_ci` 而不污染 CUDA suite

精通的外部标志之一：能给 `sgl-project/sglang` 提**非设备特化**的 PR（调度 bug、文档、单测、Radix 边角），说明你已离开「fork 里堆 `if is_mydevice`」。

---

## 3. 刻意练习（比再读文档更接近精通）

按难度排，做 2～3 道即可，不必全做。

1. **跟 rid**：overlap 开/关各抓一条 timeline（tokenize → 入队 → prefix hit 长度 → `run_batch` → detokenize）。
2. **解释一次 OOM/拒入**：用 `PrefillAdder` 预算和你的 pool 尺寸，算出为什么第 N 个并发进不去。
3. **假想改调度**：例如「同 prefix 的请求优先连 batch」——只改 `schedule_policy.py`，不动模型。
4. **投机纸上推演**：`topk=1` EAGLE 一步，哪些 KV 页是 draft 的、verify reject 时谁负责 free（对照 V4 `c128_cleanup` 若相关）。
5. **配置审计**：在你的设备代码里搜 `os.getenv("SGLANG_")` 和直接读 `server_args.xxx` 做决策的点，列该迁到 `Envs` / bags 的清单。
6. **upstream 级单测**：给 `radix_cache` 或 `PrefillAdder` 补一个你适配时踩过的边角（page 不对齐、extra_key、SWA evict）。

---

## 4. 和本系列其它篇怎么配合

| 你的目标 | 读 | 可跳过或略读 |
|---|---|---|
| 把 SRT 吃成「能改调度的人」 | 本文 A–C + 第 1、2、4 篇相关节 | 第 6 篇硬件清单、第 5 篇 lang/diffusion |
| 吃到投机/PD 与集群 | 再加阶段 D + 第 3 篇 | Cookbook 部署矩阵当附录 |
| V4 压缩栈精通 | V4 专题 §2–3 + 你现有 pool 实现对照 | 再抄一遍 NPU 目录结构 |
| 给社区贡献通用代码 | 阶段 E + large-class / runtime-context skill | — |

第 6 篇只在你要**对照官方插件模型和 NPU 差在哪一层**时回看，不要当精通主线。

---

## 5. 时间感（技术深度，不是日历）

对已经能独立交付设备模型的人，阶段 A+B 是质变（控制面 + 调度）；C 把你已有的 KV 知识接到 Radix 不变量上；D 是第二条职业曲线。读精读篇是索引，**跟源码 + 自检 + 一两道刻意练习**才是积累。

行号会变，以符号为准。

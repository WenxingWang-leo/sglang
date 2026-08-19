# SGLang 深度解读系列(中文)

> 面向两类读者:**① 想给 SGLang 贡献代码的开发者;② 要做第三方设备(NPU/XPU/自研芯片)适配的工程师。**
> 所有文档基于当前 main 分支源码(`python/sglang/srt/`)逐文件核实编写。
> 建议按编号顺序阅读;赶时间可直接看下方「速通路线」。

---

## 文档目录

| # | 文档 | 内容 | 适配相关度 |
|---|---|---|---|
| 01 | [架构总览与启动流程](./01-架构总览与启动流程.md) | 进程拓扑、模块地图、Engine 启动时序、ServerArgs | ★★ |
| 02 | [请求生命周期与数据流](./02-请求生命周期与数据流.md) | HTTP→分词→调度→前向→回传;Req/ScheduleBatch/ForwardMode | ★★ |
| 03 | [调度器与连续批处理](./03-调度器与连续批处理.md) | 事件循环(normal/overlap)、拼批、PrefillAdder 预算、new_token_ratio 反馈环、retract | ★ |
| 04 | [内存池与 RadixCache](./04-内存池与RadixCache.md) | KV 三层抽象、基数树、驱逐、容量测算公式 | ★★★ |
| 05 | [模型层与新增模型](./05-模型层与新增模型.md) | 模型积木、EntryClass 注册、load_weights、CUDA Graph 机制、实操清单 | ★★★ |
| 06 | [Attention Backend](./06-Attention-Backend.md) | 后端契约、注册选择、为新硬件写后端 | ★★★ |
| 07 | [设备抽象与第三方硬件适配](./07-设备抽象与第三方硬件适配.md) | **重点**:platforms/ 插件机制、分发路径、M0–M4 路线图 | ★★★ |
| 08 | [采样与 Logits 处理](./08-采样与Logits处理.md) | lm_head 词表并行、惩罚、采样后端扩展点 | ★★ |
| 09 | [分布式推理](./09-分布式推理.md) | TP/PP/DP attention/EP、通信组、GroupCoordinator | ★★ |
| 10 | [开发调试与贡献](./10-开发调试与贡献.md) | 环境、pre-commit、CI 测试注册、调试工具箱、PR 流程 | ★★ |

---

## 核心心智模型(一图流)

```
            主进程                        Scheduler 进程 (×TP)                  Detokenizer 进程
┌────────────────────────┐      ┌────────────────────────────────┐      ┌──────────────────┐
│ HTTP + TokenizerManager │ ZMQ │ 事件循环                        │ ZMQ  │ 增量解码          │
│  分词 / 参数校验 / 路由   │────▶│  waiting_queue ─▶ PrefillAdder │────▶ │ token ids → text │
└────────────────────────┘      │        │                       │      └──────────────────┘
          ▲                     │        ▼                       │               │
          └─────────────────────│  ScheduleBatch ─▶ ModelRunner  │◀──────────────┘
                                │        │        ┌─────────────┐│
                                │        │        │ models/     ││   ┌─ platforms/(接口)
                                │        ▼        │ layers/     │◀┼───┤
                                │  ForwardBatch   │  └ attention││   └─ hardware_backend/(实现)
                                │        │        └──────┬──────┘│
                                │        ▼               ▼       │
                                │  Sampler ◀── KV 池 + RadixCache │
                                └────────────────────────────────┘
```

**三条主线**:

1. **控制流**:HTTP → TokenizerManager → Scheduler(拼批)→ ModelRunner → Sampler → Detokenizer → HTTP;
2. **内存流**:Allocator 发槽位 → KV 池存数值 → RadixCache 存「序列→槽位」索引 → attention backend 按页表读写;
3. **设备流**:`current_platform` 单例(身份/显存/同步)+ 工厂方法(attention/graph/KV 池/量化)+ 算子分发(MultiPlatformOp / is_*())。

---

## 速通路线

### 路线 A:内核开发者(改调度/缓存/采样)

```
01 → 02 → 03 → 04 → 08 → 10
重点精读:03(调度循环)、04(RadixCache)、10(测试规范)
```

### 路线 B:第三方设备适配工程师

```
01 → 07 → 06 → 05 → 04 → 09 → 08 → 10
重点精读:07(全文背下来)、06(写后端)、05(模型层差异)
配套官方文档:docs_new/docs/hardware-platforms/plugin.mdx
参考实现:hardware_backend/npu/(最全)、hardware_backend/xpu/、musa/
```

### 路线 C:模型支持工程师(新增/修复模型)

```
01 → 05 → 06 → 04 → 10
重点精读:05(load_weights)、06(RadixAttention 与后端关系)
```

---

## 关键源码路标速查

| 想看什么 | 去哪 |
|---|---|
| 进程如何拉起 | `srt/entrypoints/engine.py::_launch_subprocesses` |
| 主事件循环 | `srt/managers/scheduler.py::event_loop_normal / event_loop_overlap` |
| 请求状态机 | `srt/managers/schedule_batch.py::Req` |
| 拼批预算 | `srt/managers/schedule_policy.py::PrefillAdder` |
| 准入悲观度调节 | `srt/managers/scheduler_components/new_token_ratio_tracker.py` |
| CUDA Graph 分桶/回放 | `srt/model_executor/runner/{base,decode}_cuda_graph_runner.py` |
| 前向批 | `srt/model_executor/forward_batch_info.py::ForwardBatch / ForwardMode` |
| KV 池 | `srt/mem_cache/memory_pool.py`、`srt/mem_cache/allocator/` |
| 基数树 | `srt/mem_cache/radix_cache.py::RadixCache` |
| 模型模板 | `srt/models/llama.py`、`srt/models/registry.py` |
| attention 契约 | `srt/layers/attention/base_attn_backend.py`、`attention_registry.py` |
| 平台抽象 | `srt/platforms/`(device_mixin / interface / __init__) |
| 树内设备实现 | `srt/hardware_backend/{npu,xpu,musa,cpu,gpu,mlx}/` |
| TP 通信 | `srt/distributed/parallel_state.py`、`srt/layers/linear.py` |
| 采样 | `srt/layers/sampler.py`、`srt/sampling/sampling_batch_info.py` |
| 参数中枢 | `srt/server_args.py` |
| 环境变量 | `srt/environ.py` |

---

## 学习建议

1. **边读边跑**:准备一张卡,用 Qwen2-0.5B/Llama-3.2-1B 之类小模型起服务,对照日志验证文档中的每个论断。
2. **善用 git log/blame**:`platforms/`、`hardware_backend/`、attention 契约迭代快,遇到与文档不符处先看最近 commit。
3. **带着问题读源码**:比如「一个新请求命中 800 token 前缀,省下了哪些计算与内存?」——能自己答出这类问题即算过关。
4. **适配从 torch_native 开始**:任何设备的第一个里程碑都是 `--attention-backend torch_native` 出正确结果。
5. **盯紧官方插件文档**:`docs_new/docs/hardware-platforms/plugin.mdx` 是 OOT 适配的权威契约,与本系列 07 篇互为参照。

祝开发顺利。

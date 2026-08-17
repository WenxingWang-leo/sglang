---
title: "SGLang SRT 核心服务路径精读（Engine / Managers）"
description: "面向二次开发的 SRT 核心路径深读：Launch/Engine、io_struct、TokenizerManager、Scheduler、TpModelWorker/ModelRunner、Detokenizer、RuntimeContext。含类方法、数据结构、控制流、IPC 与设计意图。"
keywords:
  - sglang
  - SRT
  - TokenizerManager
  - Scheduler
  - ModelRunner
  - code reading
---

本页是精读系列第 **1** 篇，覆盖 SRT 主 serving 路径的实现级解读。目标读者：要改调度 / IPC / 启动装配 / forward 边界的工程师。路径相对仓库根；线号以当前 `main` 附近代码为准，读时请对照源文件。

系列导航：[总目录](./code_reading_notes_zh.md) · [第 2 篇 Cache/Models](./code_reading_deep_dive_zh.md) · [第 3 篇进阶](./code_reading_notes_advanced_zh.md) · [第 4 篇服务扩展](./code_reading_serving_extensions_zh.md) · [第 5 篇生态](./code_reading_ecosystem_zh.md)

## 0. 总览：一条请求怎么穿过三个进程

```text
Client HTTP / Engine.generate
        │
        ▼
┌─────────────────────────────────────────┐
│ 主进程 (node_rank==0)                    │
│  FastAPI / Engine                        │
│  TokenizerManager                        │
│    rid_to_state[rid] = ReqState          │
│    tokenize → TokenizedGenerateReqInput  │
│    ZMQ PUSH → scheduler_input_ipc_name   │
│    await ReqState.event ← BatchStrOutput│
└──────────────────┬──────────────────────┘
                   │ msgspec / pickle IPC
                   ▼
┌─────────────────────────────────────────┐
│ Scheduler 子进程 (每 TP/PP rank 一个)     │
│  recv → handle_generate_request → Req   │
│  waiting_queue / running_batch           │
│  get_next_batch_to_run → ScheduleBatch   │
│  run_batch → TpModelWorker → ModelRunner │
│  process_batch_result → BatchTokenIDOut  │
│  ZMQ PUSH → detokenizer_ipc_name         │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ DetokenizerManager 子进程                │
│  BatchTokenIDOutput → 增量 decode        │
│  BatchStrOutput → ZMQ PUSH tokenizer_ipc │
└─────────────────────────────────────────┘
```

设计要点：

1. **主进程不做 GPU forward**；GPU 独占在 Scheduler 子进程。
2. IPC 载荷定义在 `python/sglang/srt/managers/io_struct.py`，默认 **msgspec msgpack**（`msgpack_encode` / `sock_send`）；多模态等不透明字段用 `PickleWrapper`。
3. HTTP 与 Python `Engine` API **共用** `_launch_subprocesses` + `TokenizerManager.generate_request`；差异只在入口层。

---

## 1. Launch & Engine

### 1.1 入口分层

| 入口 | 文件 | 行为 |
|---|---|---|
| `sglang serve ...` | `python/sglang/cli/serve.py` | 解析 `--model-type` / 位置参数；LLM 走 `prepare_server_args` → `launch_server.run_server`；Diffusion 另路 |
| `python -m sglang.launch_server` | `python/sglang/launch_server.py` | 兼容入口；同样 `prepare_server_args` → `run_server` |
| `sgl.Engine(**kwargs)` | `srt/entrypoints/engine.py` | 进程内 API；默认 `log_level=error`；**禁止** `SGLANG_RUST_SERVER` |

`run_server(server_args)` 分支（`launch_server.py`）：

```python
def run_server(server_args):
    if server_args.encoder_only: ...          # Encode server
    elif server_args.smg_grpc_mode: ...       # Legacy SMG gRPC
    elif server_args.use_ray: ...             # Ray HTTP
    else:
        from sglang.srt.entrypoints.http_server import launch_server
        launch_server(server_args)            # 默认 HTTP
```

### 1.2 ServerArgs 如何准备

```python
# server_args.py ~9174
def prepare_server_args(argv: List[str]) -> ServerArgs:
    parser = argparse.ArgumentParser(prog="sglang serve")
    ServerArgs.add_cli_args(parser)
    if "--config" in argv:
        argv = ConfigArgumentMerger(parser).merge_config_with_args(argv)
    raw_args = parser.parse_args(argv)
    _apply_fuseep_mode_env_compat(raw_args, argv)
    return ServerArgs.from_cli_args(raw_args)
```

要点：

- `ServerArgs` 是巨大 dataclass（`server_args.py`），字段带 `NS("path")` 元数据，用于投影到 RuntimeContext 的 namespace bags。
- `__post_init__` 做大量校验与默认值折叠（例如 `--grpc-mode` → `smg_grpc_mode`）。
- `Engine(**kwargs)` 可直接 `ServerArgs(**kwargs)`，或传入已构造的 `server_args=`。

### 1.3 PortArgs / ZMQ 拓扑

```python
# server_args.py ~9217
@dataclasses.dataclass
class PortArgs:
    tokenizer_ipc_name: str          # Detokenizer → Tokenizer (PULL bind)
    scheduler_input_ipc_name: str    # Tokenizer → Scheduler (PUSH / PULL)
    detokenizer_ipc_name: str        # Scheduler → Detokenizer
    nccl_port: int
    rpc_ipc_name: str                # Engine ↔ Scheduler RPC (DEALER)
    metrics_ipc_name: str
    tokenizer_worker_ipc_name: Optional[str]  # multi-tokenizer router
    decoupled_spec_ipc_config: Optional[DecoupledSpecIpcConfig]
    load_collector_ipc_name: str = ""
    instance_id: str = ""
```

`PortArgs.init_new(server_args)`：

| 模式 | 地址形态 |
|---|---|
| 默认（非 DP-attn） | `ipc://<NamedTemporaryFile>`（本机 Unix socket） |
| `enable_dp_attention` | `tcp://host:port`，端口从 `dist_init_addr` 或 `port + ZMQ_TCP_PORT_DELTA(=233)` 派生 |

典型套接字方向（单 tokenizer / 单 detokenizer）：

```text
TokenizerManager.send_to_scheduler  PUSH → scheduler_input_ipc_name ← PULL Scheduler
Scheduler.send_to_detokenizer       PUSH → detokenizer_ipc_name    ← PULL Detokenizer
Detokenizer.send_to_tokenizer       PUSH → tokenizer_ipc_name      ← PULL TokenizerManager
Engine.send_to_rpc                  DEALER → rpc_ipc_name          ← Scheduler RPC handler
```

Multi-tokenizer：`tokenizer_worker_num > 1` 时，worker PUSH 到 `tokenizer_worker_ipc_name`，由 `MultiTokenizerRouter` 再转发；返回路径靠 `http_worker_ipc` 字段路由回具体 worker。

Multi-detokenizer：`detokenizer_worker_num > 1` 时，`_launch_detokenizer_subprocesses` 建 router 占用原 `detokenizer_ipc_name`，各 worker 用私有 IPC。

### 1.4 `_launch_subprocesses` 装配顺序

`Engine._launch_subprocesses`（`engine.py` ~993）是 HTTP 与 Engine 共用的装配根：

```text
configure_logger / _set_envs_and_config / check_server_args
→ PortArgs.init_new
→ (可选) EngineInfoBootstrapServer / weight_cache daemon
→ _launch_scheduler_processes
→ (node_rank>=1) wait + dummy health / block；不启 tokenizer
→ (SGLANG_RUST_SERVER) 只等 scheduler，不启 Python detokenizer/tokenizer
→ _launch_detokenizer_subprocesses
→ init_tokenizer_manager 或 MultiTokenizerRouter
→ wait_for_ready；把 max_req_input_len 写回 tokenizer_manager
→ SubprocessWatchdog.start
```

`_launch_scheduler_processes`（~820）：

- **无 DP controller**（`dp_size==1` 且非 elastic scale）：对本节点每个 `(pp_rank, tp_rank)` 起一个 `mp.Process(target=run_scheduler_process, ...)`，用 `mp.Pipe` 收 `get_init_info()`。
- **有 DP**：起 `run_data_parallel_controller_process`，由它再拉起各 DP scheduler。

`run_scheduler_process`（`scheduler.py` ~4808）：

```python
publish(server_args, role="scheduler")   # 必须在 Scheduler.__init__ 前
scheduler = Scheduler(...)
pipe_writer.send(scheduler.get_init_info())
scheduler.run_event_loop()               # 阻塞至 ShutdownReq
```

`run_detokenizer_process`：**不** `publish`；只读构造函数传入的 `server_args`。

### 1.5 HTTP `launch_server` vs `Engine` API

**HTTP**（`http_server.py` ~2674）：

```python
def launch_server(server_args, ...):
    (tokenizer_manager, template_manager, port_args,
     scheduler_init_result, subprocess_watchdog, _) = Engine._launch_subprocesses(...)
    if envs.SGLANG_RUST_SERVER.get():
        # Rust 接管 API/tokenize/detokenize；主进程只 warmup + block
        ...
    else:
        _setup_and_run_http_server(...)  # FastAPI + Granian/uvicorn
```

`lifespan` 把 `TokenizerManager` / `TemplateManager` 装进 `_GlobalState`，并构造 OpenAI / Anthropic / Ollama serving handlers。

**原生 `/generate`**（~841）直接调 manager：

```python
async def generate_request(obj: GenerateReqInput, request: Request):
    if obj.stream:
        async def stream_results():
            async for out in _global_state.tokenizer_manager.generate_request(obj, request):
                yield b"data: " + dumps_json(out) + b"\n\n"
            yield b"data: [DONE]\n\n"
        return StreamingResponse(..., background=tokenizer_manager.create_abort_task(obj))
    else:
        ret = await tokenizer_manager.generate_request(obj, request).__anext__()
        return orjson_response(ret)
```

OpenAI 路由（`/v1/chat/completions` 等）先经 `OpenAIServing*` 转成 `GenerateReqInput`，再进同一 `generate_request`。

**Engine API**（`engine.py`）：

```python
def generate(self, prompt=..., sampling_params=..., stream=False, ...) -> Dict | Iterator:
    obj = GenerateReqInput(...)
    # loop.run_until_complete / async generator over tokenizer_manager.generate_request
```

对比：

| | HTTP | Engine |
|---|---|---|
| 主进程 HTTP 框架 | 有 | 无 |
| TokenizerManager | 有 | 有 |
| 默认日志 | CLI log_level | `error` |
| Rust server | 可 | 禁止 |
| 调用方 | curl / SDK | Python 同进程 |

### 1.6 HTTP 路由 → manager 映射（高频）

| 路由 | 落到 |
|---|---|
| `POST /generate` | `TokenizerManager.generate_request` |
| `POST /encode` `/classify` | 同上（`EmbeddingReqInput`） |
| `/v1/chat/completions` 等 | `OpenAIServing*` → `generate_request` |
| `/abort_request` | `abort_request` → `AbortReq` IPC |
| `/flush_cache` | communicator → Scheduler |
| `/load_lora_adapter` | LoRA registry + Scheduler |
| `/get_server_info` `/health*` | manager / scheduler 元信息 |
| 权重更新系列 | TokenizerManager → Scheduler `WeightUpdater` |

---

## 2. io_struct：IPC 消息契约

文件：`python/sglang/srt/managers/io_struct.py`（~2384 行）。

### 2.1 基类与序列化

```python
class BaseReq(msgspec.Struct, tag=True, kw_only=True, array_like=True):
    rid: Optional[str] = None
    http_worker_ipc: Optional[str] = None   # multi-tokenizer 回程路由

class BaseBatchReq(msgspec.Struct, tag=True, kw_only=True, array_like=True):
    rids: Optional[List[str]] = None
    http_worker_ipcs: Optional[List[Optional[str]]] = None

class PickleWrapper(msgspec.Struct, ...):
    data: bytes   # mm_inputs / time_stats 等不透明载荷
```

- `msgpack_encode` / `msgpack_decode` + `sock_send` / `sock_recv` 是线格式入口。
- `GenerateReqInput` / `EmbeddingReqInput` 是 **dataclass**（FastAPI / Pydantic 友好），**不**直接过 ZMQ；tokenize 后变成 `Tokenized*` msgspec 结构。

### 2.2 请求侧关键类型

**`GenerateReqInput`**（HTTP/Engine 入口，~160）：`text` / `input_ids` / `image_data` / `sampling_params` / `stream` / `lora_path` / `rid` / PD `bootstrap_*` / `routed_dp_rank` / `session_params` / `mm_hashes` 等。`normalize_batch_and_arguments()` 统一 batch / 单请求形状并生成 `rid`。

**`TokenizedGenerateReqInput(BaseReq)`**（~835）——真正发给 Scheduler：

| 字段 | 含义 |
|---|---|
| `input_ids` | `array` of int |
| `mm_inputs` | `PickleWrapper` → `MultimodalProcessorOutput` |
| `sampling_params` | 已解析的 `SamplingParams` |
| `return_logprob` / `logprob_start_len` / `top_logprobs_num` | logprob 控制 |
| `stream` | 是否流式（Scheduler 侧决定何时 flush） |
| `lora_id` | 已由 TokenizerManager resolve 的 uid |
| `session_params` / `session_id` | 多轮 |
| `bootstrap_*` | PD disagg |
| `http_worker_ipc` | 多 tokenizer 回程 |
| `time_stats` | 观测时间戳（pickle） |

**`BatchTokenizedGenerateReqInput`**：`batch: List[TokenizedGenerateReqInput]`，路由存在 **每条** `batch[i].http_worker_ipc` 上，不在顶层 `http_worker_ipcs`。

Embedding 对称：`EmbeddingReqInput` → `TokenizedEmbeddingReqInput` / `BatchTokenizedEmbeddingReqInput`。

控制面：`AbortReq`、`FlushCacheReqInput`、`LoadLoRAAdapterReqInput`、`OpenSessionReqInput`、`ProfileReq`、`RpcReqInput`、`PauseGenerationReqInput` 等。

### 2.3 响应侧与流式跨进程

```text
Scheduler 产出 BatchTokenIDOutput  (token ids + meta，仍未成完整字符串)
    → Detokenizer → BatchStrOutput (output_strs 增量 / 全量)
    → TokenizerManager._handle_batch_output
    → ReqState.out_list.append(dict); ReqState.event.set()
    → _wait_one_response yield 给 HTTP SSE / Engine
```

**`BatchTokenIDOutput`**（~1293）：并行数组，按 index 对齐 `rids`。含 `finished_reasons`、`decoded_texts`（增量用）、`decode_ids` / `read_offsets`、token 计数、各类 logprobs、`routed_experts`（tensor）、`time_stats` 等。

**`BatchStrOutput`**（~1393）：detokenize 后的 `output_strs`；`routed_experts` 已变 base64 str。

**`BatchEmbeddingOutput`**：`embeddings`；可 stacked 以减 pickle。

流式语义：

- Scheduler 按 `stream_interval` 周期性把未完成请求打进 `BatchTokenIDOutput`。
- Detokenizer 用 `DecodeStatus`（`rid → decode_ids / read_offset / decoded_text`）做 **增量 decode**：只 decode `decode_ids[read_offset:]` 相对 surrogate 前缀的差。
- TokenizerManager：`incremental_streaming_output` 时每个 chunk 是 delta；否则中间 chunk 的 `text` 可先置 `None`，yield 前再 `state.get_text()`，避免 O(n²) 字符串重建。多 chunk 积压时 `_coalesce_streaming_chunks` 合并。

`skip_tokenizer_init`：Scheduler 可直接把 `BatchTokenIDOutput` 推回 TokenizerManager，跳过 Detokenizer。

---

## 3. TokenizerManager

文件：`tokenizer_manager.py` + mixin：

- `TokenizerControlMixin` — pause / communicators 等控制面
- `TokenizerManagerScoreMixin` — scoring / rerank

### 3.1 初始化编排（`__init__` ~363）

```text
set_global_server_args_for_tokenizer → publish(role="tokenizer")
init_model_config
init_tokenizer_and_processor      # tokenizer / mm_processor / async batch tokenizer
init_ipc_channels                 # PULL tokenizer_ipc; PUSH scheduler_input
init_running_status               # rid_to_state, ServerStatus
init_request_logging_and_dumping
init_weight_update
init_lora                         # LoRA registry
init_disaggregation               # PD bootstrap 等
init_metric_collector_watchdog
init_request_dispatcher           # TypeBasedDispatcher for 非 batch 输出
```

`ReqState`（~182）：每请求状态机——`out_list`、`finished`、`event: asyncio.Event`、累积 `output_ids` / logprobs / `text_chunks`、`time_stats`。

### 3.2 `generate_request` 逐步

```python
async def generate_request(self, obj, request=None):
    self.auto_create_handle_loop()          # 确保 handle_loop 在跑
    obj.normalize_batch_and_arguments()
    self._set_default_priority(obj)
    # 校验 routed_dp_rank
    self._init_req_state(obj, request)      # rid → ReqState
    try:
        # EPD language_only 时先发 encode
        await self.is_pause_cond.wait_for(lambda: not self.is_pause)
        async with self.model_update_lock.reader_lock:
            await self._validate_and_resolve_lora(obj)
            if obj.is_single:
                tokenized = await self._tokenize_one_request(obj)
                self._send_one_request(tokenized)
                async for r in self._wait_one_response(obj, request):
                    yield r
            else:
                async for r in self._handle_batch_request(obj, request):
                    yield r
    except Exception:
        self._discard_pending_req_states(obj)  # 防 rid_to_state 泄漏
        raise
```

**Tokenize**（`_tokenize_one_request` ~925）：

1. `input_embeds` / `input_ids` / `text` 三选一；text 走 `_tokenize_texts`（可 async dynamic batch）。
2. 多模态：`mm_processor` 处理 image/video/audio → `mm_inputs`，并可能扩展 `input_ids`（placeholder）。
3. `_validate_one_request`：长度 vs `context_len` / `max_req_input_len`、vocab、mm limits。
4. `_create_tokenized_object` → `TokenizedGenerateReqInput`（含 `SamplingParams` 解析）。

**发送**（`_send_one_request` ~1475）：

```python
tokenized_obj.time_stats.set_api_server_dispatch_time()
tokenized_obj = wrap_shm_features(tokenized_obj)   # 大 mm tensor 可走 shm
tokenized_obj.wrap_pickle_fields()
self._dispatch_to_scheduler(tokenized_obj)         # sock_send PUSH
```

**等待**（`_wait_one_response` ~1590）：

- 循环 `await state.event.wait()`；超时检查 HTTP disconnect → `abort_request`。
- 排空 `out_list`；流式 yield；结束时处理 abort finish_reason。

**匹配回 HTTP**：`handle_loop`（~1992）从 detokenizer PULL；`_handle_batch_output` 按 `recv_obj.rids[i]` 找 `rid_to_state`，组装 `{"text", "meta_info", "output_ids", ...}`，`state.event.set()`。流式 SSE 的消费者就是卡住的 `_wait_one_response`。

### 3.3 Abort / LoRA / 多模态要点

**Abort**（`abort_request` ~1819）：构造 `AbortReq(rid=..., abort_all=...)` PUSH 给 Scheduler；HTTP 断连时 `create_abort_task` 在 `StreamingResponse` background 里触发。

**LoRA**：

- `_validate_and_resolve_lora`：未开 `--enable-lora` 直接报错。
- `_resolve_lora_path`：registry 查找 / 热加载未注册 adapter → 填 `lora_id`。
- HTTP `/load_lora_adapter` 走 registry + 转发 Scheduler `load_lora_adapter`。

**多模态**：主进程预处理；Scheduler 侧 `_process_and_broadcast_mm_inputs` 在 TP 间广播。`language_only` + EPD 时 `_handle_epd_disaggregation_encode_request` 先把 encode 派到 encoder。

### 3.4 `handle_loop` 与控制面回包

Batch 输出走 `_handle_batch_output`；其余走 `_result_dispatcher`：`AbortReq`、`OpenSessionReqOutput`、权重更新结果、`ActiveRanksOutput`、`ElasticScaleUpdateReq` 等。

---

## 4. Scheduler

文件：`scheduler.py`（~4897）+ `schedule_batch.py` + `schedule_policy.py` + `scheduler_components/`。

Mixin：`SchedulerDisaggregationDecode/PrefillMixin`、`SchedulerMultiplexMixin`、`SchedulerPPMixin`、`SchedulerDllmMixin`、`SchedulerMlxOverlapMixin`。

### 4.1 `__init__` 编排（~369）

按依赖顺序（与 large-class-style 一致：编排根）：

```text
init_soft_watchdog
解析 server_args / ParallelState
init_model_config
init_metrics_collector
init_ipc_channels / init_idle_sleeper
init_zbal_on_npu / (pdmux)
init_tokenizer
init_moe_gemm_config / init_mamba_backend
init_model_worker          # TpModelWorker + 可选 draft worker
kv_cache_builder.build_kv_cache → req_to_token_pool, allocator, tree_cache
init_hisparse_coordinator / decode_offload...
init_running_status        # waiting_queue, running_batch, last_batch
init_chunked_prefill
init_schedule_policy       # SchedulePolicy + PrefillAdder 参数
init_disaggregation / init_overlap (future_map, streams)
init_request_dispatcher
init_profiler / weight_updater / lora_* / grammar_manager
init_request_receiver / dp_attn_adapter / pool_stats_observer
init_output_streamer / batch_result_processor
...
```

`enable_overlap = not disable_overlap_schedule`（非 MLX）。Overlap 时 CPU 调度与 GPU forward 用不同 CUDA stream（`schedule_stream` vs `forward_stream`），靠 WAR barrier / future_map 同步。

### 4.2 Event loop：`normal` vs `overlap`

`run_event_loop` → `dispatch_event_loop(self)` 选择具体循环。

#### `event_loop_normal`（~1632）

```python
while True:
    recv_reqs = self.request_receiver.recv_requests()
    self.process_input_requests(recv_reqs)
    plan = self.get_next_batch_to_run(self.running_batch, self.last_batch)
    self.running_batch = plan.running_batch
    batch = plan.batch_to_run
    if batch:
        result = self.run_batch(batch)
        self.process_batch_result(batch, result)   # 同步：forward 完立刻处理
    else:
        self.on_idle()
    self.last_batch = batch
```

#### `event_loop_overlap`（~1666）

```python
result_queue = deque()   # (ScheduleBatch copy, result)

while True:
    recv + process_input
    plan = get_next_batch_to_run(...)
    batch = plan.batch_to_run

    if is_disable_overlap_for_batch(batch, last_batch):
        pop_and_process()   # 先把上一批结果处理完（例如连续 prefill 保 TTFT）

    if batch:
        batch_result = self.run_batch(batch)       # 在 forward_stream 上 launch
        self._apply_war_barrier()
        result_queue.append((batch.copy(), batch_result))

    if last_batch and not disable_overlap:
        pop_and_process()                          # 处理「上一批」结果，与本批 GPU 重叠
    elif batch is None:
        on_idle()

    if is_generation:
        self.launch_batch_sample_if_needed(...)    # grammar 等依赖上批结果的 sampling

    self.last_batch = batch
```

设计意图：

- **Overlap**：把 `process_batch_result`（CPU：更新 Req、写 radix、组 IPC）叠在下一 iter 的 GPU forward 上。
- **连续 prefill 禁用 overlap**（`SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`）：优先第一批 TTFT。
- **Grammar sync**：投机 + grammar 时可能强制同步，避免 bitmask 用到未 advance 的 FSM。
- **WAR barrier**：防止 schedule_stream 写 unified memory pool 与上一 forward 读冲突。

### 4.3 入队：`handle_generate_request`

~2247：`TokenizedGenerateReqInput` → `Req(...)`：

1. Session / radix-native session 分支。
2. 构造 `Req`（origin_input_ids、sampling、lora、bootstrap、http_worker_ipc…）。
3. 多模态：`_process_and_broadcast_mm_inputs`。
4. Grammar 入队或直接 `_add_request_to_queue` → `waiting_queue`。
5. Prefill 路径上 `tree_cache.match_prefix` 等由后续 PrefillAdder 触发。

`waiting_queue: List[Req]`；`running_batch: ScheduleBatch`（当前在飞的 decode/混合批）。

### 4.4 `get_next_batch_to_run`（~2872）

伪代码级行为：

```text
处理 chunked abort / waiting&running timeout
若 last_batch 是 extend：
    filter 掉 chunked 未完成 req
    merge 进 running_batch          # prefill 完成后进入 decode 集合
若有新 prefill（get_new_batch_prefill）：
    优先返回 prefill batch
否则：
    update_running_batch(running)   # 分配 decode out_cache_loc、组 decode batch
    返回 decode batch 或 None
DP-attn / ngram embedding 适配
return NextBatchPlan(batch_to_run, running_batch)
```

**Prefill 选择**（`_get_new_batch_prefill_raw` ~3035）：

1. Grammar ready → 入 waiting。
2. HiCache 事件检查。
3. `batch_is_full` 或空队列 → 无新 prefill（除非有 `chunked_req`）。
4. `policy.calc_priority(waiting_queue, running_batch)`（LPM / FCFS / LOF / ROUTING_KEY…）。
5. `PrefillAdder` 按 KV 预算、`chunked_prefill_size`、`max_prefill_tokens`、优先级抢占，从 waiting 取 req：
   - `tree_cache` 前缀命中 → 少算 token、复用 KV slot。
   - 不够预算 → chunked prefill（`chunked_req` 挂起）。
6. 组成 `ScheduleBatch`（`forward_mode=EXTEND` 或 `MIXED`）。

**Decode**：`update_running_batch` 检查 finished / abort / retract（KV 不够时回退 waiting）、为每个 req 分配下一 token 的 `out_cache_loc`。

Continuous batching 本质：**每个 step 可插入新 prefill；running 中未完成 decode 与新来请求共享 GPU iter**。

### 4.5 `run_batch` / `process_batch_result`

**`run_batch`**（~3470）：

- Generation + overlap：在 `forward_stream` 上 `resolve_forward_inputs` → `model_worker.forward_batch_generation` → `future_map.publish` → 异步 `copy_to_cpu`。
- Generation + 非 overlap：同步 forward + sample + `update_cache_from_scheduler`。
- Embedding：`forward_batch_embedding`。
- Speculative：走 `model_worker`（常为 Eagle worker），`spec_info` 带回下一 draft。

**`process_batch_result`**（~3756）：

```python
if decode:  batch_result_processor.process_batch_result_decode
elif extend: process_batch_result_prefill / disagg / dllm
elif prebuilt / idle: ...
# 然后 metrics、清 mm_inputs、health check 信号
```

Decode 路径典型动作：把 `next_token_ids` 写入各 `Req.output_ids`；检查 stop / length；更新 radix cache；按 stream 间隔经 `output_streamer` 发 `BatchTokenIDOutput`。

### 4.6 与 tree_cache / memory pools

| 对象 | 角色 |
|---|---|
| `req_to_token_pool` | req slot → token 位置表 |
| `token_to_kv_pool_allocator` | 物理 KV page / slot 分配 |
| `tree_cache`（RadixCache 等） | 前缀树；match / insert / evict；HiCache 时分层 |

PrefillAdder 用 `tree_cache` 算 `num_matched_prefix_tokens`；完成 prefill/decode 后 `process_batch_result_*` 把新 KV 挂回树。OOM 时 `retract` 释放 running req 的 KV，请求回到 waiting。

### 4.7 `schedule_batch` 核心结构

**`Req`**（~767）：单请求全生命周期状态——`origin_input_ids`、`output_ids`、`extend_range`、`prefix_indices`、KV 记账、`finished_reason`、grammar、lora、stream 标志、`http_worker_ipc`。

**`ScheduleBatch`**（~1919）：一批要 forward 的 reqs + GPU tensor（`input_ids`、`req_pool_indices`、`seq_lens`、`out_cache_loc`、`sampling_info`、`forward_mode`、`tree_cache` 引用…）。`ForwardBatch.init_new(batch, model_runner)` 从这里借/派生 forward 输入。

**`NextBatchPlan`**：`batch_to_run` + 更新后的 `running_batch`。

### 4.8 `scheduler_components/`（职责切分）

| 组件 | 职责 |
|---|---|
| `request_receiver.py` | ZMQ 收包 / DP 广播 |
| `output_streamer.py` / `output_sender.py` | 组 BatchTokenIDOutput 并发送 |
| `batch_result_processor.py` | prefill/decode 结果处理 |
| `dp_attn.py` | DP attention MLP sync / idle batch |
| `weight_updater.py` | 热更新权重 |
| `grammar_manager`（scheduler 内） | 约束解码 FSM |
| `metrics_reporter.py` / `profiler_manager.py` | 观测 |
| `ipc_channels.py` | socket 封装 |

改调度逻辑时：优先落在这些 collaborator，而不是把算法塞进 `scheduler.py` 编排层。

---

## 5. TpModelWorker 与 ModelRunner

### 5.1 TpModelWorker（`tp_worker.py`）

```python
class TpModelWorker(BaseTpWorker):
    def __init__(self, server_args, gpu_id, ps, nccl_port, is_draft_worker=False, ...):
        self._init_model_config()     # ModelConfig.from_server_args；draft 用 draft path
        self._init_model_runner()    # 构造 ModelRunner（内部 load 权重 + dist init）
        # 可选 multi-layer eagle runners / dllm
        # tokenizer（非 draft 且未 skip）
        # broadcast random_seed across TP
```

Scheduler 的 `init_model_worker` 之后：`alloc_memory_pool` → `init_attention_backends` → `init_cuda_graphs`（顺序由 scheduler 编排；KV 池由 `kv_cache_builder` 与 runner 协同）。

**`forward_batch_generation`**（~533）：

```python
forward_batch = ForwardBatch.init_new(batch, self.model_runner, ...)
out = self.model_runner.forward(forward_batch, pp_proxy_tensors=...)
# last PP rank:
batch_result.next_token_ids = self.model_runner.sample(logits_output, forward_batch)
# overlap + grammar: 可设 delay_sample_func，留给 launch_batch_sample_if_needed
return GenerationBatchResult(...)
```

非 last PP rank：返回 `pp_hidden_states_proxy_tensors`，不做 sample。

### 5.2 ForwardBatch（`forward_batch_info.py`）

**`ForwardMode`**：`EXTEND`（prefill）、`DECODE`、`MIXED`、`IDLE`、`TARGET_VERIFY`、`DRAFT_EXTEND_V2`、`PREBUILT`、`SPLIT_PREFILL`、`DLLM_EXTEND`。`is_extend()` / `is_decode()` / `is_cuda_graph()` 驱动分支。

**`ForwardBatch` 关键字段**（~412）：

| 字段 | 含义 |
|---|---|
| `forward_mode` | 本 iter 模式 |
| `input_ids` | token ids（decode 时常为上一输出） |
| `req_pool_indices` | req → token pool 行 |
| `seq_lens` / `seq_lens_sum` | 序列长度 |
| `out_cache_loc` | 本步要写的 KV 槽位 |
| `positions` | RoPE 位置（`init_new` 内计算） |
| `extend_num_tokens` / `extend_*` | prefill 专用切分信息 |
| `sampling_info` | `SamplingBatchInfo`（温度、top-k、grammar…） |
| `mm_inputs` | 多模态 |
| `spec_info` | 投机解码 |
| `lora_ids` | 本批 LoRA |
| `capture_hidden_mode` | 是否抓 hidden |

`ForwardBatch.init_new(batch, model_runner, ...)`（~703）：从 `ScheduleBatch` 组装/借用张量，算 positions，处理 DP padding 前置数据等。

### 5.3 ModelRunner（`model_runner.py`，frozen 编排文件）

**构造 / `initialize`**：

```text
set device → Mooncake TE → init_torch_distributed → forward_stream
→ initialize()：
    load_model()
    maybe LoRA / EPLB / elastic EP / …
    （内存池 / attn backend / cuda graph 多由 Scheduler 稍后显式调用）
```

注意：`ModelRunner` 按约定是 **orchestration-only**；领域逻辑应在 collaborator。读 `forward` 时把它当装配 + 分发。

**`forward` → `_forward_raw`**（~1312 / ~1456）：

```text
1. 若 decode 且 decode_cuda_graph_runner.can_run_graph → execute(graph) 直接返回
2. _prepare_eager_forward_batch（DP pad / attn-tp）
3. deferred mamba COW/clear
4. split_prefill / prefill_cuda_graph / 否则 eager:
     attn_backend.init_forward_metadata
     model.forward(input_ids, positions, forward_batch, ...)
5. logits 经 LogitsProcessor；返回 ModelRunnerOutput(logits_output, can_run_graph)
```

**CUDA Graph**：

- `init_cuda_graphs` → `init_decode_cuda_graph` / `init_prefill_cuda_graph`。
- Decode 热路径：固定 shape 的 capture/replay；大幅降 launch overhead。
- Prefill graph：piecewise / 条件启用（`can_run_graph`）。

**Sampling**（`sample` ~1571）：在 logits 上施加温度 / top-k / top-p / min-p / grammar bitmask / custom logit processor，采样 `next_token_ids`。Overlap 模式下可能延迟到 `launch_batch_sample_if_needed`，以便等上一批 grammar 状态。

---

## 6. DetokenizerManager

文件：`detokenizer_manager.py`（~537）。**不 publish RuntimeContext**。

### 6.1 初始化与循环

```python
def __init__(...):
    init_ipc_channels   # PULL detokenizer_ipc; PUSH tokenizer_ipc (单 worker)
    init_tokenizer
    init_running_status # DecodeStatus LimitedCapacityDict
    init_request_dispatcher

def event_loop(self):
    while True:
        recv_obj = sock_recv(self.recv_from_scheduler)
        output = self._request_dispatcher(recv_obj)
        if output is not None:
            sock_send(self.send_to_tokenizer, output)
```

Multi-tokenizer：`multi_http_worker_event_loop`，按 `http_worker_ipc` 分发回各 worker。

### 6.2 Token → 字符串

`handle_batch_token_id_out`：

1. 对每个 rid 维护 `DecodeStatus`（累计 `decode_ids`、`surr_offset` / `read_offset`、decoded text）。
2. `_grouped_batch_decode`：按 `(skip_special_tokens, spaces_between_special_tokens)` 分组 `batch_decode`。
3. 增量文本 = decode(read 段) 相对 decode(surr 段) 的差；处理 UTF-8 残缺字符。
4. `trim_matched_stop`：按 finish_reason 裁 stop str/token。
5. 组装 `BatchStrOutput`（`output_strs` 为本步增量或最终串，取决于协议字段填充方式）。

Embedding：`handle_batch_embedding_out` 原样转发。

容量：`LimitedCapacityDict` 防止 rid 泄漏撑爆内存（完成的状态可淘汰）。

---

## 7. RuntimeContext / ServerArgs / Environ：配置如何进各进程

### 7.1 谁 publish 什么

| 进程 | `publish(..., role=)` | 读配置方式 |
|---|---|---|
| Scheduler | `scheduler`（`run_scheduler_process`） | `get_schedule()` / `get_exec()` / … bags；**权威** |
| TokenizerManager | `tokenizer`（`set_global_server_args_for_tokenizer`） | 业务多用 **`self.server_args`**（多 Engine 同进程） |
| Detokenizer | **不 publish** | 仅构造函数 `server_args` |
| DP controller | `dp_controller` | 受限 namespace（enforce 时仅 `exec`） |
| Launcher / HTTP 装配 | 可能 `launcher` 再被 tokenizer 覆盖 | last-publish-wins |

`publish`（`runtime_context.py` ~1265）：把 `ServerArgs` 投影进 namespace bags；`ServerArgs` 实例保持 **pristine snapshot**。运行期变更只能 `get_context().override(source, **fields)`，**不写回** ServerArgs。

### 7.2 Namespace bags

访问器：`get_exec()`、`get_memory()`、`get_schedule()`、`get_model()`、`get_spec()`、`get_serving()`、`get_observability()`、`get_disagg()`、`get_lora()`、`get_mm()`、`get_device()`、`get_parallel()`。

例：`get_schedule().max_running_requests`、`get_exec().moe.moe_a2a_backend`。

`SGLANG_ROLE_NAMESPACES=record|enforce`：审计/限制某 role 可读的 namespace。

### 7.3 Environ

`python/sglang/srt/environ.py`：`envs.SGLANG_*` 类型化环境变量（`EnvField`）。影响调度行为的例子：

- `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`
- `SGLANG_SCHEDULER_MAX_RECV_PER_POLL`
- `SGLANG_RUST_SERVER`
- `SGLANG_ROLE_NAMESPACES`
- `SGLANG_ENABLE_WAR_BARRIER` / `SGLANG_FORCE_COARSE_WAR_BARRIER`

约定见 `.claude/skills/env-var-conventions/SKILL.md`：新变量加到 `environ.py`，命名 `SGLANG_*`。

### 7.4 配置流小结

```text
CLI / kwargs
  → prepare_server_args / ServerArgs(**kwargs)     # 含 __post_init__
  → 复制进各子进程 (mp spawn/fork 参数)
  → 各进程 publish(role) 投影 bags
  → Scheduler 热路径读 get_*() bags
  → Tokenizer / HTTP / Engine 入口读 self.server_args
  → Detokenizer 只读传入的 server_args
  → 运行期调参：get_context().override(...) 或 SetInternalState IPC
```

Draft worker：**不要** publish，以免覆盖 target 的 bags；draft 用自己的 `server_args` 深拷贝字段（attention_backend 等）。

---

## 8. 改代码时的落点清单

| 你想改… | 优先文件 |
|---|---|
| 启动参数 / 端口 | `server_args.py` `PortArgs`、`prepare_server_args` |
| 子进程装配 | `engine.py` `_launch_*` |
| HTTP 路由 | `http_server.py`；协议适配 `entrypoints/openai/` |
| IPC 字段 | `io_struct.py`（两端一起改 + encode hook） |
| Tokenize / 流式回包 | `tokenizer_manager.py` |
| 调度策略 / continuous batching | `schedule_policy.py` `PrefillAdder`、`get_next_batch_to_run`（详见 [服务扩展精读 §9](/docs/developer_guide/code_reading_serving_extensions_zh#9-continuous-batching--schedule-policy-内部)） |
| OpenAI/Anthropic / Grammar / VLM / Session | [服务扩展模块精读](./code_reading_serving_extensions_zh.md) |
| Overlap / stream 同步 | `event_loop_overlap`、`future_map`、`overlap_utils.py` |
| 结果处理 / 输出频率 | `scheduler_components/batch_result_processor.py`、`output_streamer.py` |
| Forward / graph / sample | `model_runner.py` 编排 + 其 collaborator；`tp_worker.py` |
| KV / 前缀缓存 | `mem_cache/`、`kv_cache_builder`、`Req`/`ScheduleBatch` |
| 配置读取方式 | `runtime_context.py`；新代码用 bags，勿扩 legacy shim |

三大类风格：改 `Scheduler` / `TokenizerManager` / `ModelRunner` 前读 `.claude/skills/large-class-style/SKILL.md`。配置体系读 `.claude/skills/sglang-runtime-context/SKILL.md`。

---

## 9. 一条请求的时序（对照调试）

```text
t0  HTTP /generate 或 Engine.generate
t1  TokenizerManager._init_req_state；tokenize；PUSH TokenizedGenerateReqInput
t2  Scheduler.request_receiver 收到；handle_generate_request → waiting_queue
t3  get_next_batch_to_run 选中 → ScheduleBatch(EXTEND)
t4  run_batch → ForwardBatch → ModelRunner.forward (prefill)
t5  process_batch_result_prefill；merge 进 running_batch；可能已 stream 首包
t6  后续 iter：DECODE(+CUDA graph)；每 stream_interval 发 BatchTokenIDOutput
t7  Detokenizer 增量 decode → BatchStrOutput
t8  TokenizerManager._handle_batch_output → event.set → SSE/Engine yield
t9  finish_reason 非空 → 清 rid_to_state；HTTP [DONE]
```

调试断点建议：`generate_request` 入口、`handle_generate_request`、`get_next_batch_to_run` 返回值、`run_batch`、`_handle_batch_output`。

---

## 10. 与概览笔记的关系

- 部署 / 仓库地图 / 开发工作流：见 [SGLang 代码精读笔记（部署与开发）](./code_reading_notes_zh.md)。
- 本页专注 **可修改级别的行为与契约**；PD / 投机 / HiCache / Elastic EP 的专门路径在对应 mixin 与 `srt/disaggregation/`、`srt/speculative/`、`srt/mem_cache/` 中继续下钻。

---
title: "SGLang 生态组件精读（Frontend Language / Gateway / Diffusion / CI）"
description: "实现级精读：lang DSL 解释器与后端、sgl-model-gateway 控制面与路由策略、multimodal_gen 扩散运行时、Test/CI 扩展方式——与 LLM SRT 主路径的边界。"
keywords:
  - sglang
  - frontend language
  - sgl-model-gateway
  - multimodal_gen
  - diffusion
  - CI
  - code reading
---

本篇是精读系列第 **5** 篇，覆盖不在「单机 LLM Scheduler forward」主链上、但生产与开发高频碰到的生态组件。配套：[系列总目录](./code_reading_notes_zh.md)、[SRT 核心](./code_reading_srt_core_zh.md)、[进阶模块](./code_reading_notes_advanced_zh.md)。

## 边界先划清

| 组件 | 是不是 LLM SRT 主路径 | 典型用途 |
|---|---|---|
| `python/sglang/srt/` | **是** | 本地 GPU serving |
| `python/sglang/lang/` | 否（可调用 SRT） | 可编程生成 DSL / 多后端 |
| `sgl-model-gateway/` | 否（站在 SRT 前） | 集群路由、PD、LB、IGW |
| `python/sglang/multimodal_gen/` | 否 | 扩散 / 图生视频 |
| `test/` + `python/sglang/test/` | 开发基础设施 | CI / 单测 / e2e |

把 `lang` 里的 `gen()` 和 SRT 的 `Scheduler.run_batch` 当成同一层，是最常见的读码误区。

---

## 1. Frontend Language（`python/sglang/lang/`）

### 1.1 定位

`lang` 提供 **Python 嵌入式 DSL**：用 `@sgl.function` 写多步生成程序（含 `gen` / `select` / 角色块 / 多模态占位），由 **解释器** 把 IR 变成对某个 **Backend** 的调用。Backend 可以是：

- 远端 OpenAI / Anthropic / Vertex 等（`lang/backend/`）
- 本机 SRT：`sgl.Runtime(...)`（HTTP 连已启动的 server）或 `sgl.Engine(...)`（直接 `srt.entrypoints.engine.Engine`）

它 **不包含** KV cache、TP、RadixCache、cuda graph；那些全在 SRT。

### 1.2 关键文件与类型

| 文件 | 职责 |
|---|---|
| `lang/api.py` | 对外 API：`function`、`gen`、`select`、`Runtime`、`Engine`、`set_default_backend` |
| `lang/ir.py` | IR 节点：`SglFunction`、`SglGen`、`SglSelect`、`SglRoleBegin/End`、`SglImage`/`SglVideo`、`SglExprList`… |
| `lang/interpreter.py` | `run_program` / `StreamExecutor` / `ProgramState`：执行 IR |
| `lang/backend/base_backend.py` | Backend 抽象（`generate`、`select`、`flush_cache`…） |
| `lang/backend/runtime_endpoint.py` | HTTP 连 SRT `/generate` |
| `lang/backend/openai.py` 等 | 第三方 API |
| `lang/choices.py` | `select` 的打分 / 采样策略 |
| `lang/tracer.py` | 可选追踪 |

### 1.3 执行流（实现级）

```text
@sgl.function
def prog(s, question):
    s += "Q: " + question + "\nA:"
    s += sgl.gen("answer", max_tokens=64)

prog.run(backend, question="...")   # 或 prog.batch / stream
```

1. `@sgl.function` → `SglFunction(func, ...)`，保存 Python 可调用对象与 bind 参数。
2. `run_program`（`interpreter.py`）：
   - 解析 `backend`（若带 `.endpoint` 则取 endpoint）
   - 构造 `StreamExecutor(backend, kwargs, sampling, stream=...)`
   - 构造 `ProgramState(stream_executor)`
   - 在线程（或同步）里跑 `program.func(state, *args, **kwargs)`
3. 用户代码里 `s += "text"` / `s += sgl.gen(...)` 把 IR 追加进 executor 的表达式流。
4. `StreamExecutor` 遇到 `SglGen`：把当前前缀文本 + sampling 参数交给 `backend.generate(...)`；遇到 `SglSelect` 走 `backend.select` / choices 方法。
5. `state.ret_value` / `state["answer"]` 从变量绑定取回生成结果；`stream_executor.end()` / `sync()` 收尾。

**与 SRT 的接缝：**

- `api.Engine` **直接 import** `sglang.srt.entrypoints.engine.Engine`——同进程，走 TokenizerManager，无 HTTP。
- `api.Runtime` → `runtime_endpoint.Runtime`：对已启动的 `sglang serve` 发 HTTP，走完整三进程路径。

读码顺序：`api.py` → `ir.py`（看有哪些节点）→ `interpreter.StreamExecutor` 里对 `SglGen`/`SglSelect` 的分支 → 选一个 `backend/*.py` 看如何拼 HTTP body。

### 1.4 何时改 lang vs 改 SRT

| 需求 | 改哪里 |
|---|---|
| 新的 DSL 原语（新 IR 节点） | `lang/ir.py` + `interpreter.py` |
| 新的远端厂商后端 | `lang/backend/` |
| 更快的前缀缓存 / 调度 | **SRT**，不是 lang |
| OpenAI 兼容 HTTP 语义 | `srt/entrypoints/openai/`（系列第 4 篇） |

---

## 2. sgl-model-gateway（`sgl-model-gateway/`）

### 2.1 定位

Rust 实现的 **控制面 + 数据面路由**。它 **不跑** Transformer forward；真正算力仍在 SRT worker。Gateway 负责：

- Worker 注册、健康检查、能力发现
- 按策略把请求打到 regular / prefill / decode worker
- OpenAI 兼容 API 聚合、可选 gRPC 管线（Rust tokenizer / reasoning / tool parser）
- 重试、熔断、限流、队列、可观测性
- IGW：多模型多 router（`--enable-igw`）

官方用户文档：[SGLang Model Gateway](/docs/advanced_features/sgl_model_gateway)。

### 2.2 目录与入口

| 路径 | 职责 |
|---|---|
| `src/main.rs` | CLI：策略、PD、IGW、端口等 → `RouterConfig` |
| `src/server.rs` | HTTP 服务装配 |
| `src/app_context.rs` | 全局上下文（registry、policies、resilience） |
| `src/routers/` | HTTP / PD / gRPC / OpenAI 等 router 实现 |
| `src/policies/` | `random` / `round_robin` / `cache_aware` / `power_of_two` / `bucket`… |
| `src/core/` | Worker、registry、job queue |
| `src/service_discovery.rs` | K8s 等发现 |
| `src/observability/` | metrics / tracing |
| `bindings/python/` | Python 包封装（maturin） |

### 2.3 控制面 vs 数据面

```text
Control plane:
  Worker Manager 校验并注册 worker
  Job Queue 串行化 add/remove
  Health checker + load monitor → 熔断器 / 策略权重
  (可选) K8s service discovery

Data plane:
  Client → Gateway HTTP/gRPC
       → Policy.select(worker)
       → Regular HTTP router  或  PD router (P 再 D)  或  gRPC pipeline
       → SRT worker(s)
```

**PD 模式**：CLI 可分别指定 `--prefill-policy` / `--decode-policy`（如 prefill 用 `cache_aware`、decode 用 `power_of_two`）。Gateway 处理 bootstrap 端口与请求亲和；**KV 字节**仍由 SRT `disaggregation` 后端（mooncake/nixl/…）在 P/D 之间直传（见系列第 3 篇）。

**cache_aware**：在 `policies/cache_aware.rs`；按前缀亲和把请求打到更可能命中 RadixCache 的 worker，并可 mesh sync。这与 SRT 内部 `SchedulePolicy` 的 LPM **不是同一层**——一个是集群路由，一个是单实例 waiting_queue 排序。

### 2.4 与 SRT 的分工清单

| 问题 | Gateway | SRT |
|---|---|---|
| 哪个 worker 接请求 | 是 | 否 |
| Prefill/Decode 角色选择 | 路由 | `--disaggregation-mode` 进程角色 |
| Tokenize（HTTP 路径） | 可选（gRPC Rust 路径） | TokenizerManager（默认 HTTP） |
| KV 分配 / Radix 树 | 否 | 是 |
| 熔断 / 限流 | 是 | 基本无（有 max_running_requests） |

读码顺序：`main.rs` CLI → `policies/mod.rs` → `routers/` 下 PD router → 对照 SRT `disaggregation/utils.py`。

---

## 3. Multimodal_gen（`python/sglang/multimodal_gen/`）

### 3.1 与 VLM 路径的区别

| | `srt/multimodal` + encode server | `multimodal_gen` |
|---|---|---|
| 任务 | VLM：图像→embedding→LLM token | 扩散：噪声→图像/视频 |
| 调度 | 进 LLM Scheduler / RadixCache | 自有 diffusion runtime |
| 安装 | 默认 SRT | 常需 `sglang[diffusion]` |
| 文档 | 系列第 4 篇 VLM 节 | 本系列 + `docs_new/.../sglang-diffusion/` |

### 3.2 目录结构（读码地图）

| 路径 | 职责 |
|---|---|
| `registry.py` | 扩散模型注册 |
| `runtime/` | 采样管线、调度、（可选）diffusion disaggregation transport |
| `configs/` | 模型/采样配置 |
| `csrc/` | 扩散相关 CUDA / 注意力扩展 |
| `apps/` | WebUI、ComfyUI 插件等 |
| `envs.py` | 扩散专用环境变量 |

入口形态（概念上）：

```python
from sglang.multimodal_gen import DiffGenerator  # 以实际导出为准
gen = DiffGenerator.from_pretrained(model_path)
gen.generate(...)
```

CLI / OpenAI 兼容扩散 API 见 `docs_new/docs/sglang-diffusion/`。

### 3.3 实现阅读顺序

1. `multimodal_gen/README.md` + `registry.py`（模型如何挂上）
2. `runtime/` 里 generator / pipeline 主类（噪声步循环、CFG、scheduler）
3. 若做多卡：`runtime` 内 parallel / disaggregation 与 SRT PD **独立**——不要复用 `srt/disaggregation` 的假设
4. 性能：`docs_new/docs/sglang-diffusion/performance-optimization.mdx`、`attention_backends.mdx`

改扩散模型权重加载或采样步，动 `multimodal_gen`；改 LLaVA 式对话，动 `srt/models/*_vl.py` + `srt/multimodal/processors/`。

---

## 4. Test / CI（如何扩展）

权威：`test/README.md`、`.claude/skills/write-sglang-test/SKILL.md`、`.claude/skills/ci-workflow-guide/SKILL.md`。

### 4.1 布局

| 路径 | 用途 |
|---|---|
| `test/registered/<area>/` | **CI 自动发现**；目录即领域（`disaggregation/`、`lora/`、`unit/`…） |
| `test/registered/unit/` | 无服务器单测，镜像 `srt/` 树 |
| `test/manual/` | 本地/特殊硬件，不进默认 suite |
| `test/run_suite.py` | 收集、过滤、LPT 分区、执行 |
| `python/sglang/test/` | 夹具、`CustomTestCase`、`ci_register`、runners |
| `.github/workflows/pr-test.yml` | Stage A → B → C；kernel 与 multimodal_gen 并行轨 |

### 4.2 注册契约（必须字面量）

```python
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")
```

`run_suite.py` 用 **AST** 抽 `est_time` / `stage` / `runner_config`，不能写成变量。

文件末尾：

```python
if __name__ == "__main__":
    unittest.main()
# 或 pytest.main([__file__]) —— 不要抢先改 sys.argv
```

### 4.3 选 suite

| 需求 | Suite |
|---|---|
| 无 GPU | `base-a-test-cpu` |
| 普通单卡 | `base-b-test-1-gpu-small` |
| 大显存 / Hopper | `base-b-test-1-gpu-large` |
| JIT kernel | `base-b-kernel-unit-test-1-gpu-large` |
| 多卡 | `base-b-test-2-gpu-*` / `base-c-test-*` |
| 长时 / 实验 | `nightly-*` |

### 4.4 本地命令

```bash
pytest test/registered/unit/mem_cache/ -v
python3 test/registered/core/test_srt_endpoint.py
python3 test/run_suite.py --hw cuda --suite base-b-test-1-gpu-small
ONLY_RUN=Org/Model python3 -m unittest test_generation_models.TestGenerationModels.test_others
```

### 4.5 与精读系列的对应

| 你改的模块 | 测试落点示例 |
|---|---|
| RadixCache / allocator | `test/registered/unit/mem_cache/` |
| Scheduler 策略 | `test/registered/` 下 scheduling / server e2e |
| 投机 | `test/registered/` speculative 相关 |
| PD | `test/registered/disaggregation/` |
| LoRA | `test/registered/lora/` |
| 新模型 | `test/registered/models/test_generation_models.py` |
| JIT kernel | `test/registered/jit/` |
| Diffusion | multimodal_gen 独立 CI 轨 |

---

## 5. 推荐精读顺序（本篇）

1. `lang/api.py` + `interpreter.run_program` / `StreamExecutor` 对 `SglGen` 分支  
2. `lang/backend/runtime_endpoint.py`（看如何打到 SRT `/generate`）  
3. `sgl-model-gateway/README.md` + `src/policies/mod.rs` + PD router  
4. `multimodal_gen/README.md` + `registry.py` + `runtime/` 主循环  
5. `test/README.md` + 仿写一个 `register_cuda_ci` 单测  

读完系列五篇后，你应能指出：任意功能落在 **Gateway / lang / SRT / multimodal_gen** 哪一层，并打开对应文件改到正确的类与方法。

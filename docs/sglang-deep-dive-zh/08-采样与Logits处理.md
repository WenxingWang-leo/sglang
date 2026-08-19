# 08 · 采样管道与 Logits 处理

> 本篇追踪「hidden states → 下一个 token」的完整链路:词表并行 lm_head、温度/惩罚/约束,
> 以及可插拔的采样后端。适配新设备时,采样是继 attention 之后第二个必须啃下的性能点。

---

## 1. 管道总览

```
模型最后一层 hidden_states
        │
        ▼
LogitsProcessor                # 选位置 + 词表并行 lm_head + TP all-gather
        │  next_token_logits [num_seqs, vocab]
        ▼
SamplingBatchInfo.apply_logits_bias   # 惩罚(frequency/presence/repetition/min_new_tokens)
        │                             # + grammar mask(约束解码)+ logit_bias
        ▼
Sampler                        # 温度 → softmax → top-k/top-p/min-p → 采样
        │  next_token_ids [num_seqs]
        ▼
回 scheduler,追加进 Req.output_ids
```

四个主角:`SamplingParams`(用户参数)、`SamplingBatchInfo`(批级张量化)、`LogitsProcessor`(logits 生产)、`Sampler`(token 产出)。

---

## 2. SamplingParams:用户意图的载体

`sampling/sampling_params.py` 中定义为 `msgspec.Struct`(零开销序列化,跨进程友好)。高频字段:

| 类别 | 字段 |
|---|---|
| 长度 | `max_new_tokens`(默认 128)、`min_new_tokens`、`ignore_eos` |
| 停止 | `stop`(字符串)、`stop_token_ids`、`stop_regex` |
| 随机性 | `temperature`(1.0)、`top_p`、`top_k`、`min_p`、`sampling_seed` |
| 惩罚 | `frequency_penalty`、`presence_penalty`、`repetition_penalty` |
| 约束解码 | `json_schema`、`regex`、`ebnf`、`structural_tag` |
| 其它 | `n`(多候选)、`logit_bias`、`custom_params` |

`normalize()` 做预处理(停止串去重、归一化标记等)。**temperature == 0 等价贪心**,走独立的快路径。

---

## 3. SamplingBatchInfo:批级张量化

`sampling/sampling_batch_info.py`。调度批(`ScheduleBatch`)在 `run_batch` 前生成它,把每个请求的标量参数拼成 GPU 张量:

- 张量:`temperatures / top_ps / top_ks / min_ps`;
- 快速路径标记:`is_all_greedy`(全贪心)、`need_top_p_sampling` 等——让 Sampler 能整批走最快分支;
- 约束解码:`grammars` + `grammar_mask`(XGrammar 生成的词表掩码);
- 惩罚:`BatchedPenalizerOrchestrator` 统一管理 frequency/presence/repetition/min_new_tokens 四种惩罚器,`apply_logits_bias(logits)` 按「加性惩罚 → 缩放惩罚 → grammar mask → logit_bias」顺序应用;
- `from_schedule_batch()`:在 pinned memory 里组 CPU 张量再异步 H2D,避免阻塞;
- `filter_batch` / `merge_batch`:与调度批同步增删。

---

## 4. LogitsProcessor:从 hidden 到 logits

`layers/logits_processor.py`。三个关键步骤:

1. **选位置(`_get_pruned_states`)**:决定「hidden_states 里哪些行需要过 lm_head」。这一步是 prefill 阶段的重要省算点,但**分支比想象中多**:
   - **decode / target_verify / draft_extend_v2**:每行本来就对应一个待采样位置,`pruned_states = hidden_states` 原样透传,不裁剪;
   - **extend 且不要 input logprob**(最常见的 prefill 路径):只取每个序列的**最后一个**位置——用 `torch.cumsum(extend_seq_lens) - 1` 算出各序列末尾的偏移。一个 4096 token 的 prompt 只需过 1 行 lm_head 而不是 4096 行,这在大词表模型上省下的算力和显存非常可观;
   - **extend 且要 input logprob**(`return_logprob=True` 且请求了 prompt logprobs):中间位置的 logits **有人消费**了,必须保留相应区间,不能只留末尾。这也是为什么开启 prompt logprobs 会显著增加 prefill 显存峰值和耗时。

   > 换句话说,「只算最后一个 token」是默认优化而非恒定行为,取决于 `logits_metadata.extend_return_logprob`。为新硬件实现相关算子时,两条路径都要覆盖测试。
2. **词表并行 lm_head(`_get_logits`)**:`lm_head` 是 `VocabParallelEmbedding` 的线性形态——vocab 维按 TP 切分,每 rank 只算自己那段,然后 **tensor-parallel all-gather** 拼出全词表 logits。这是 TP 下模型主体(每 block 的两次 all-reduce)之外最主要的集合通信;此外采样阶段在特定条件下还有一次 token id 对齐的 all_reduce(见 §5.4)。
3. **精度处理**:默认把 logits 升到 fp32 再交给采样(数值稳定性);支持 logit softcapping(Gemma 系)。

产出 `LogitsProcessorOutput`:`next_token_logits [num_seqs, vocab]` + logprob 相关字段。

> 适配注意:词表并行 all-gather 依赖 TP 通信组正确初始化(文档 09);DP attention 下 lm_head 有专门的 `enable_dp_lm_head` 路径。

---

## 5. Sampler:把概率变成 token

`layers/sampler.py`,`Sampler.forward` 的分支逻辑:

### 5.1 预处理

- 自定义 logit processor(用户注册的钩子);
- NaN 清洗(防御异常输入)。

### 5.2 贪心快路径

`is_all_greedy` 时直接 `torch.argmax`(ROCm 上可换 aiter 的 `greedy_sample`),跳过 softmax——生产中最常见的路径,务必优先优化。

### 5.3 随机采样路径

1. `logits.div_(temperatures)` → 原地 softmax → probs;
2. 按 `--sampling-backend` 选实现:
   - **flashinfer**:`top_k_renorm_prob` / `top_p_renorm_prob`(sgl_kernel)+ `min_p_sampling_from_probs` / `top_k_top_p_sampling_from_probs`(flashinfer);
   - **pytorch**:纯 PyTorch 回退 `top_k_top_p_min_p_sampling_from_probs_torch`;
   - **ascend**:昇腾专用(树内示例);
3. 无 top-k/p/min-p 的简单场景走 `sampling_from_probs_torch`;
4. 需要**确定性**时(seed 固定)用 `multinomial_with_seed`:`@torch.compile` 的 Gumbel-max trick + murmur hash,不依赖设备 RNG 状态。

### 5.4 跨 TP rank 一致性

各 rank 各自采样,理论上应采出同一 token(输入相同)。若启用 grammar 或设置了 `SYNC_TOKEN_IDS_ACROSS_TP`,会用 `all_reduce(MIN)` 显式对齐采样结果,防止 rank 间分叉(分叉 = TP 死锁/错乱)。

### 5.5 Logprobs

需要 logprobs 时,由 `OutputLogprobProcessor`(`layers/logprob_processor.py`)计算 top-k logprobs 与所选 token 的 logprob,随结果回传(OpenAI 接口的 `logprobs` 参数)。

### 5.6 扩展点

```python
# layers/sampler.py
def register_sampler_backend(backend: str, factory: Callable[[], "Sampler"]) -> None:
    """注册自定义采样后端。注意:factory 返回的是整个 Sampler 实例(子类),
    会自动把 backend 名加入 SAMPLING_BACKEND_CHOICES(即 --sampling-backend 白名单)。"""
```

也就是说,新设备接入采样内核 = **子类化 `Sampler` + 注册工厂**,CLI 白名单自动扩展,无需改 `server_args.py`。

**适配提示**:新设备先让 `--sampling-backend pytorch` 跑通(纯 torch 算子,任何设备都支持),性能达标后再注册原生后端子类。

---

## 6. 约束解码(结构化输出)简述

`constrained/` 集成 XGrammar:用户给 `json_schema/regex/ebnf` 时,每步生成前由 grammar 引擎产出 **vocab mask**,经 `SamplingBatchInfo.apply_logits_bias` 打进 logits(非法 token 置 -inf)。它是「批级状态」,与请求生命周期绑定,overlap 模式下有专门的状态同步考虑。

---

## 7. 本篇要点回顾

1. 链路:LogitsProcessor(选位置 + 词表并行 lm_head + all-gather)→ 惩罚/grammar → Sampler。
2. 贪心走 argmax 快路径;随机采样按 backend 分发,pytorch 后端是适配保底。
3. `SamplingBatchInfo` 是批级张量化与惩罚/掩码的执行者。
4. TP 下 lm_head 有 all-gather;采样结果必要时跨 rank 对齐。
5. 采样后端可通过 `register_sampler_backend` 扩展,是第三方适配的正式入口。

下一篇:[09 · 分布式推理 TP/PP/DP/EP](./09-分布式推理.md)。

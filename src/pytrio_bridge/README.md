# PyTRIO Bridge

把 [PyTRIO](https://docs.pytrio.com) 接到 Polar 上，用远程 LoRA 训练跑 agentic RL。
**不修改 Polar 的任何代码**，只增加这一层。

> 文档：<https://docs.pytrio.com>　（`docs.pytrio.cn` 会 301 重定向到 `.com`）

## 为什么需要两件套

Polar 的 gateway 只认一种推理后端：

```
POST {base_url}/v1/chat/completions      （只发这一种，`stream` 恒为 false）
GET  {base_url}/health
GET  {base_url}/v1/models
```

而且它需要的响应形状 PyTRIO 的官方 OpenAI 端点给不了：

- 逐 token 的 `token_ids`
- 逐 token 的 `logprobs.content[]`
- prompt 的 `input_token_ids`

所以必须有**一个本地 shim** 把 `SamplingClient.sample_async()` 包装成这个形状。

```
┌─────────────────────────────────────────────────────────────┐
│ polar serve_rollout  (:8080)                                │
│   └─ polar serve_gateway  (:8100)                           │
│        │  POST /v1/chat/completions                         │
│        ▼                                                    │
│  shim  (:9000, 同进程 uvicorn)                              │
│        │  sample_async()                                    │
│        ▼                                                    │
│  PyTRIO  （远程采样 / 训练）                                 │
│        ▲                                                    │
│        │  save_weights_and_get_sampling_client()            │
│  train.py 训练循环  ── 更新 shim 的 sampler ─────────────────┘
└─────────────────────────────────────────────────────────────┘
```

**shim 与 trainer 跑在同一个进程**（uvicorn 在后台线程），这样
`SamplingClient` 对象可以直接换，不用跨进程序列化。

## 文件结构

```
src/pytrio_bridge/
├── README.md       本文
├── __init__.py     空包声明
├── config.py       配置 + topology 渲染                 ~60 行
├── submit.py       提交 Polar task、轮询结果            ~70 行
├── advantage.py    LOO advantage                        ~40 行
├── adapter.py      Trace → trio.Datum                   ~80 行
├── shim.py         OpenAI 兼容转发层                    ~200 行
└── train.py        训练循环（入口）                     ~120 行
```

合计约 **570 行**，对比 `slime_bridge` 的 2429 行。

| 文件 | 职责 | 对标 slime_bridge |
|---|---|---|
| `config.py` | 唯一的配置来源；渲染 topology 指向 shim | `config.py`（删掉模板引擎） |
| `submit.py` | Polar HTTP 客户端 | `rollout.py` 的提交路径 |
| `advantage.py` | 算 advantage | `reward_post_process.py` |
| `adapter.py` | 造训练样本 | `adapter.py` |
| `shim.py` | **Polar 的推理后端** | 无（Slime 用 SGLang router） |
| `train.py` | 主循环 | `rollout.py` 的入口 + Slime 的训练循环 |

---

## 各文件要做什么

### `config.py`

```python
BASE_MODEL = "Qwen/Qwen3.5-4B:peft:256k"

@dataclass(frozen=True)
class BridgeConfig:
    rollout_server_url: str            # http://127.0.0.1:8080
    task_template: dict                # Polar TaskRequest，必须含 agent
    shim_host: str = "127.0.0.1"
    shim_port: int = 9000
    model_served: str = BASE_MODEL
    base_model: str = BASE_MODEL
    group_size: int = 8                # 每个 prompt 采几次
    max_concurrency: int = 4           # 同时在飞的 task 数
    reward_key: str = "score"

def render_topology(template_path: str, cfg: BridgeConfig) -> dict:
    """只改三个字段：model_served / inference.engine / inference.base_url"""
```

要点：

- **`model_served == base_model == BASE_MODEL`**，用一个常量加一条 assert 保证
- `inference.engine` 固定 `"sglang"`（`vllm` 分支会多做一层 `reasoning_content` 重命名，无收益）
- `render_topology` 返回 dict，由启动脚本写到临时文件
- 不需要模板引擎：task_template 直接当 JSON 用

### `submit.py`

```python
async def submit_task(client, url, payload) -> str
async def wait_task(client, url, task_id) -> TaskResult
```

要点：

- `payload` 就是 Polar 的 `TaskRequest`：`task_id / instruction / num_samples / agent / runtime / evaluator`
- **`num_samples = group_size`**：一次提交一个 prompt，Polar 起 N 个 session
- `task_id` 必须含 step 和 prompt 序号，避免跨步撞车
- **第一版用同步轮询**（0.5s 间隔），不做 callback listener

### `advantage.py`

```python
def compute_advantages(traces: list[TraceRecord]) -> list[float]:
    """输入一个 prompt group 的所有 trace，输出与之等长的 advantage。"""
```

三步，照 `reward_post_process.py:63-86` 的逻辑：

1. 按 `(session_id)` 聚合，session 内 reward 取平均
2. leave-one-out baseline：其他 session 均值的均值
3. `advantage[i] = reward[i] - baseline[session(i)]`

边界：失败 session 的 advantage 置 0；组里只有一条 session 时 baseline 为 0。

### `adapter.py`

```python
def trace_to_datum(trace, advantage) -> trio.Datum
```

```python
tokens = trace.prompt_ids + trace.response_ids
obs_len = len(trace.prompt_ids) - 1

datum = trio.Datum(
    model_input=trio.ModelInput.from_ints(tokens[:-1]),
    loss_fn_inputs={
        "target_tokens": [0]*obs_len + trace.response_ids,
        "logprobs":      [0.0]*obs_len + trace.response_logprobs,
        "advantages":    [0.0]*obs_len + [adv if m else 0.0 for m in trace.loss_mask],
    },
)
```

要点：

- **三个数组必须与 `model_input` 等长**（右移一位）。错位不会报错，只会静默训坏 —— 这一处必须 `assert`
- 工具返回/environment 段靠 `loss_mask=0` 屏蔽，advantage 也要在这里填 0
- 一条 trace 一个 Datum。**多 trace 的 session 不拼接**，因为 chain 1 的 logprob 是在 chain 1 自己的 prompt 下采的

### `shim.py`

```python
class SamplerHolder:
    sampler: trio.SamplingClient
    tokenizer: Any
    lock: asyncio.Lock

app = FastAPI()   # /v1/chat/completions, /v1/models, /health
```

**这是最容易写错的文件。** 处理一个请求的流程：

1. **本地渲染 chat template**

   ```python
   kwargs = request.get("chat_template_kwargs") or {}
   ids = tok.apply_chat_template(
       messages, tools=tools, add_generation_prompt=True, **kwargs
   )
   ```

   `chat_template_kwargs` 里的 `enable_thinking=False` 是 Polar 强制塞进来的
   （`transform/base.py` 对名字含 `qwen3.5` 的模型生效）。**必须透传**，否则
   模板走 `<think>\n` 分支，下一轮 prompt 就不是上一轮的 token 前缀，
   `prefix_merging` 会把每条 trace 拆成一轮。

2. **调采样**

   ```python
   resp = await sampler.sample_async(
       prompt=trio.ModelInput.from_ints(ids),
       num_samples=1,
       sampling_params=trio.SamplingParams(
           max_tokens=request["max_tokens"],
           temperature=request.get("temperature", 1.0),
           top_p=request.get("top_p", 1.0),
           stop=request.get("stop"),
       ),
   )
   ```

3. **把 XML 工具调用解析成结构化 `tool_calls`**

   Qwen3.5 输出：

   ```
   <tool_call>
   <function=read_file>
   <parameter=path>
   foo.py
   </parameter>
   </function>
   </tool_call>
   ```

   Polar **不做这个解析**（全仓库没有 `<tool_call>` 的解析器），
   而 OpenAI 格式的 harness（codex、opencode）只认 `message.tool_calls`。
   **shim 必须自己解析**，并把 `finish_reason` 设成 `"tool_calls"`。

4. **返回 canonical 形状**

   ```python
   {
     "choices": [{
       "index": 0,
       "message": {"role": "assistant", "content": text, "tool_calls": [...]},
       "finish_reason": "tool_calls" | "length" | "stop",
       "input_token_ids": ids,                 # prompt
       "token_ids": seq.tokens,                # response
       "logprobs": {"content": [
           {"token": ..., "token_id": ..., "logprob": ...}, ...
       ]},
     }]
   }
   ```

5. **`finish_reason` 映射**

   | 条件 | finish_reason |
   |---|---|
   | 解析出工具调用 | `tool_calls` |
   | `seq.stop_reason == "length"` | `length` |
   | 其余 | `stop` |

其他要求：

- 不用鉴权（gateway 不发任何 header）
- 不用实现流式（gateway 的 `InferenceClient.completion()` 强制 `stream=False`）
- `/v1/models` 返回 `model_served`
- 权重刷新：`train.py` 调 `save_weights_and_get_sampling_client_async()` 后，
  在同一个 event loop 上用 `asyncio.run_coroutine_threadsafe` 换掉 `sampler`

### `train.py`

```python
async def main():
    start_shim()                                   # uvicorn 后台线程

    svc = trio.ServiceClient()
    trainer = await svc.create_training_client_async(base_model=BASE_MODEL)
    sampler = await svc.create_sampling_client_async(base_model=BASE_MODEL)
    shim.install(sampler)

    for step in range(num_steps):
        groups = await rollout(step)               # 提交 + 等结果
        data = [trace_to_datum(t, a) for ...]      # advantage → Datum
        await (await trainer.forward_backward_async(data, loss_fn="importance_sampling"))
        await (await trainer.optim_step_async(trio.AdamParams(learning_rate=lr)))

        sampler = await trainer.save_weights_and_get_sampling_client_async()
        await shim.refresh(sampler)
```

---

## 开发顺序（三步，每步都能独立验证）

### 第 1 步：只写 shim，验证轨迹

不接 PyTRIO 训练，固定一个 sampler，让 Polar 跑一个 task：

```bash
polar serve_rollout -c topology.yaml
polar serve_gateway -c topology.yaml --node-id localhost-node-01
polar submit examples/calculator/task.json -c topology.yaml
```

**验收标准**：`prefix_merging` 产出的 trace 里，`loss_mask` 是 `001111000111` 这种三明治结构，
而不是每条 trace 只有一轮（后者说明 `enable_thinking` 没透传）。

这一步**不需要容器、不需要 GPU**。

### 第 2 步：写 config + submit + adapter

用一个假 reward（比如 `len(response_ids) > 0`）跑通"提交 → 收结果 → 造 Datum"，
先不训练。

**验收标准**：`Datum` 三个数组长度一致，`target_tokens` 和 `response_ids` 对得上。

### 第 3 步：接训练循环

加 `forward_backward` / `optim_step` / `save_weights_and_get_sampling_client`。

**验收标准**：loss 不炸、不变 NaN、权重刷新后采样内容跟着变。

---

## 必须遵守的六条契约

1. **`model_served` == `base_model` == `Qwen/Qwen3.5-4B:peft:256k`**
   写成别的：上下文退化成 64k，或 Polar 不强制 `enable_thinking=False`
2. **chat template 在本地渲染，`chat_template_kwargs` 原样透传**
3. **XML 工具调用必须由 shim 解析成 `tool_calls`**，Polar 不做这件事
4. **`Datum` 三个数组靠右移对齐，长度严格相等**
5. **多 trace 的 session 不拼接**，每条 trace 独立成一个 Datum
6. **同一个 session 的多条 trace，梯度权重要归一化**
   PyTRIO 没有 Slime 的 `group_id` 归并；不处理的话，compact 过的 session 权重会翻倍

---

## 不要抄的东西

`slime_bridge` 里有大量为 Slime 接口打的补丁，**全部不要带过来**：

| slime_bridge 的写法 | pytrio_bridge 的做法 |
|---|---|
| `_extract_reward` 五级 fallback | `float(trace.reward or 0.0)` |
| `_build_dummy_sample` 占位样本 | 直接跳过该 trace |
| `try/except ImportError` 延迟 import | 直接 `import pytrio` |
| `**kwargs` 吞参数 | 明确签名 |
| callback listener + 准入队列 + staleness | 同步轮询 |
| 模板引擎（`{dotted.path}`） | 直接当 JSON |
| `key_by_sample` 这类只写不读的索引 | 不写 |

**原则：能 `assert` 的地方就 `assert`，不要写兜底分支。** 静默兜底比报错难查得多。

**唯一的例外是长度校验**——`len(logprobs) != len(response_ids)` 这类检查必须保留，
因为它防的是"不报错的错误"，不是"报错的错误"。

---

## 参考文档

| 主题 | 链接 |
|---|---|
| 快速开始 | <https://docs.pytrio.com/docs/content/get-started/quick-start/content.md> |
| 推理 / 采样（`SamplingClient`、`SamplingParams`） | <https://docs.pytrio.com/docs/content/guide/sample/content.md> |
| 训练（`TrainingClient`、`forward_backward`） | <https://docs.pytrio.com/docs/content/guide/train/content.md> |
| 保存权重（`save_weights_and_get_sampling_client`） | <https://docs.pytrio.com/docs/content/guide/save/content.md> |
| 损失函数（`importance_sampling`） | <https://docs.pytrio.com/docs/content/guide/loss_fn/content.md> |
| 自定义损失函数（用于 group 梯度归一化） | <https://docs.pytrio.com/docs/content/guide/custom_loss/content.md> |

站点根路径 <https://docs.pytrio.com> 会跳到「什么是 PyTRIO」；页面 404 时先看
<https://docs.pytrio.com/llms.txt>。

## 待验证

| 项 | 状态 |
|---|---|
| 权重刷新延迟（`save_weights_and_get_sampling_client` 耗时） | 未测 |
| 高并发下 PyTRIO 的限流行为（探针只测到 16 路） | 未测 |
| 长 session 撞 262144 上限时丢样本的比例 | 未测 |
| 同进程 uvicorn 与训练线程的 GIL 争抢 | 未测 |
| `SamplingClient` 在跨 event loop 使用是否安全 | 未测 |

## 前置改动

`pyproject.toml` 的 `[tool.setuptools.packages.find]` 目前只包含 `polar*` 和
`slime_bridge*`，需要加上 `pytrio_bridge*`：

```toml
include = ["polar*", "slime_bridge*", "pytrio_bridge*"]
```

另外本机（macOS arm64）没有容器运行时，Polar 起不了 runtime。
跑通第 1 步需要一台有 Docker/Apptainer 的机器，或者先用 `session_completed`
这类不依赖容器的 evaluator 做纯接口验证。

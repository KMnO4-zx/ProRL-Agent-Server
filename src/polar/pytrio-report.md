# Polar × PyTRIO 改造方案

> 2026-11-25。本文取代 `src/pytrio_bridge/README.md`。
> 文中的代码：只有 `template.py` 的三个函数在文本层跑过（§6.2），其余都是**示意，没有运行过**
> （这台 Mac 没有容器运行时，也没装 PyTRIO SDK）。
>
> 标注含义：
> **读码** = 从本仓库源码读出来的；**源码** = 从 PyTRIO / HuggingFace / uvicorn 的官方文档或源码读出来的；
> **探针** = 之前实际调用 PyTRIO 测到的；**设计** = 我的方案，没有验证过。

## 0. 一句话

`rollout / gateway / runtime / trajectory / agent` 保留（rollout 加两个小接口，gateway 删掉 `engine.py`），删掉"多后端"那一层。
新增 `polar/pytrio/`（6 个文件，估计 400 行）：**同一个进程、同一个事件循环里，同时跑采样 shim 和训练循环**。
Slime、sglang / vllm、platform 面板全部删除。

---

## 1. 目标与范围

要做成的样子：

- clone → `uv pip install -e .` → 在仓库里跑（和现在 `README.md:33-34` 的做法一样，只是不再装 vllm / sglang）。
- PyTRIO 是唯一的采样和训练后端。
- harness 不改：claude code / codex / pi 照常跑，看到的还是 OpenAI / Anthropic / Gemini 接口。
- 把代码写对就行，不为"万一出错"多写分支。只在结构保证不了的地方加 assert，新增的 `polar/pytrio/` 里一共 5 条（§7）。

第一版不做：异步 / partial rollout、thinking 模式、micro-batch、自定义 loss、多模型。

---

## 2. 现状：哪些留、哪些删、哪些加

| 目录 | 行数 | 处置 | 原因 |
|---|---:|---|---|
| `src/polar/rollout/` | 1578 | 留，加两个小接口 | 只认 gateway 的 HTTP 接口，和训练后端无关 |
| `src/polar/gateway/` | 6468 | 留，删 `engine.py` | 协议转换、起容器、调 evaluator 都还要 |
| `src/polar/runtime/` | 795 | 留 | 一个 session 一个容器，和训练后端无关 |
| `src/polar/trajectory/` | 2133 | 留 | `prefix_merging` 产出的 `Trace` 正好能转成 `Datum` |
| `src/polar/agent/` | 1651 | 留，按需删 presets | harness 适配 |
| `src/polar/config/` | 238 | 留，删 `engine` 字段 | |
| `src/polar/platform/` | 1470 | 删，先搬走 `events.py` | 只读面板 |
| `src/slime_bridge/` | 2357 | 删 | 被 `polar/pytrio/` 取代 |
| `src/pytrio_bridge/` | 375 | 删 | 被本文取代 |
| `src/polar/pytrio/` | 0 → ~400 | 新增 | |

行数是读码数的；`src/` 下一共约 1.7 万行。

---

## 3. 改完长什么样

```
              +------------------------- polar train (one process, one event loop) ------------------------+
              |                                                                                                   |
 (container)  |   rollout:8080 ---> gateway:8100 ---POST /v1/chat/completions---> shim:9000                      |
   harness ---|-->^                      |                                                  |  sample_async(num_samples=1)  |
              |   |                      +---- SessionResult(trajectory.traces) <--+       v                  |
              |   |                                                                   |   PyTRIO (remote)         |
              |   |                                                                   |      ^                    |
              |   |   train loop: Trace[] --> Datum[] --> forward_backward_async --> optim_step_async            |
              |   |                  ^                                                  |                          |
              |   |                  +---- save_weights_and_get_sampling_client_async --> shim.install()          |
              +---------------------------------------------------------------------------------------------------+
```

一个 step 的顺序（**设计**，完全同步）：

1. `save_weights_and_get_sampling_client_async()` 拿到当前权重的 sampler，装进 shim。
2. 提交 `tasks_per_step` 个 `TaskRequest`，每个 `num_samples = group_size`。
3. rollout 把每个任务拆成 N 个 session 分给 gateway；gateway 起容器；harness 每次调模型，都被转成 OpenAI 格式打到 shim。
4. shim 本地渲染 prompt → 采样 → 解析工具调用 → 返回带 `token_ids` 和 `logprobs.content[]` 的响应。
5. gateway 把这个 session 的所有调用用 `prefix_merging` 拼成 `Trace[]`，再让 evaluator 打分写进 `Trace.reward`（`gateway/node.py:700-738`）。
6. 训练循环取回 `TaskResult`：只留成功的 session，算 advantage，一条 trace 一个 `Datum`。
7. `forward_backward_async` → `optim_step_async`，回到第 1 步。

**为什么同步**：一个 step 里所有采样用的都是同一份权重，不存在过期样本；换 sampler 的时候没有自己的 session 在飞，不需要锁，也不需要 `/admin/inference/pause`。

**为什么 shim 放在训练进程里**：gateway 继续只管协议，不认识 PyTRIO；sampler 是个对象，换个变量就完成权重同步。
以后要横向扩展可以改成"每个 gateway 机器一个 shim"，代价是每步要存一个有名字的 checkpoint 再通知各机加载（PyTRIO 有 `create_sampling_client(base_model, model_path=...)`）。第一版不做。

---

## 4. 目标文件结构

```
src/polar/
├── __init__.py
├── _imports.py
├── events.py                 ← 从 platform/events.py 搬来（109 行，只依赖标准库）
├── cli.py                    serve_rollout / serve_gateway / submit / status / train（删 dashboard）
├── config/topology.py        删 inference.engine
├── agent/                    不动
├── runtime/                  不动
├── rollout/                  加 progress / delete 两个接口
├── gateway/                  删 engine.py；proxy.py、server.py 各改两行
├── trajectory/               不动
└── pytrio/                   新增
    ├── __init__.py
    ├── config.py             ~35 行   TrainConfig + load_config
    ├── template.py           ~95 行   parameter_types / parse_tool_calls / render_prompt
    ├── shim.py               ~100 行  三个路由 + install()
    ├── rollout.py            ~25 行   run_task()
    ├── data.py               ~50 行   build_group() / trace_to_datum()
    └── train.py              ~110 行  main() / train()
```

行数是估计。对比：`slime_bridge` 是 2357 行。

---

## 5. 写代码的规矩

就一条：**把代码写对就行**，不为"万一出错"多写分支。

### 不写的

| 不写什么 | 仓库里的反面例子 | 换成 |
|---|---|---|
| 多级 fallback | `slime_bridge/reward.py:17-30`：dict 里找 key → 找 `score` → 找第一个数字 → 0.0 | 直接读 `trace.reward` |
| 缺数据就补 0 | `slime_bridge/adapter.py:273-279`、`:303-308` | assert 它存在 |
| 占位样本 | `slime_bridge/adapter.py:75`、`:180` 的 `_build_dummy_sample` | 这个 session 不要了 |
| try/except 包住 import | `slime_bridge/adapter.py:317-325` | 直接 `import pytrio` |
| `**kwargs` 吞参数 | `slime_bridge/reward.py:8` | 明确签名 |
| 嵌套 `getattr(args, "a", getattr(args, "b", 默认值))` | `slime_bridge/reward.py:11` | frozen dataclass，多一个键就报错 |
| 两条路取结果，失败再回落 | callback 发失败后回落轮询（`rollout/manager.py:176-182`） | 只用轮询 |

### 允许的

- 协议里本来就可选的字段给默认值：`body.get("temperature", 1.0)`。1.0 是 OpenAI 协议规定的默认值，不是猜的。
- 按明确的规则丢数据，并且计数、打日志：失败的 session、奖励全相同的组。
- 模型输出不合规时原样交给环境：参数 JSON 写坏了，就让工具去报错，模型从 reward 里学到。

### 不 assert 的地方（靠结构保证）

| 不 assert 什么 | 为什么不用 |
|---|---|
| `model_served == base_model` | shim 不看 `model_served`，它只是给 harness 看的名字 |
| `num_samples == group_size` | 训练循环自己填的 |
| `loss_mask`、`response_logprobs` 和 `response_ids` 等长 | Pydantic 已经校验（`trajectory/models.py:103-123`），取回结果时再校验一次。空的 `loss_mask` 它放过去，由 §7 第 3 条的 assert 抓住 |
| `task_id` 不重复 | `step00003-001` 这种格式，按构造唯一 |

---

## 6. 各文件设计

### 6.1 `pytrio/config.py`

```python
"""训练配置。YAML 里多写一个键会直接报错。"""
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainConfig:
    base_model: str                 # PyTRIO 模型 id，用 service.get_supported_models() 里的写法
    tasks_file: str                 # 一行一个 TaskRequest（不含 task_id / num_samples）
    rollout_url: str = "http://127.0.0.1:8080"
    shim_host: str = "0.0.0.0"
    shim_port: int = 9000
    lora_rank: int = 32
    group_size: int = 8             # 一个任务起几个 session
    tasks_per_step: int = 4         # 一步采几个任务
    max_steps: int = 100
    start_step: int = 0
    learning_rate: float = 1e-5
    save_every: int = 10            # 每几步存一次完整 checkpoint
    max_tokens: int = 16384         # 请求没带 max_tokens 时用
    enable_thinking: bool = False
    resume_from: str | None = None  # save_state() 返回的 path；None 表示从头训


def load_config(path: str) -> TrainConfig:
    return TrainConfig(**yaml.safe_load(Path(path).read_text()))
```

- `TrainConfig(**dict)`：YAML 里多一个键 → `TypeError`。
- `base_model` 的默认建议：`Qwen/Qwen3.5-4B:peft:256k`。**探针**：256k 上下文只有这个 id 有，不带 `:peft:256k` 的 `Qwen/Qwen3.5-4B` 上限是 65536。
- `tasks_file`：一行一个 JSON，就是 `TaskRequest` 去掉 `task_id` 和 `num_samples`。`builder` 要写 `prefix_merging`（模型默认值是 `per_request`）。

### 6.2 `pytrio/template.py`

三个纯函数：`parameter_types`（从 `tools` 里抠出参数类型）、`parse_tool_calls`（采样文本 → tool_calls）、`render_prompt`（messages → prompt token ids）。

```python
"""Qwen3.5 chat template 的两个方向。解析规则是照着模板反向写的
（HF 上 Qwen/Qwen3.5-4B 的 chat_template.jinja）：
- L82        所有 content 渲染前都 trim
- L101       最后一条 user 消息之后的 assistant，渲染成
             "<think>\n" + reasoning + "\n</think>\n\n" + content + 工具调用
- L111-117   第一个工具调用：有正文时前面接 "\n\n"；之后的工具调用之间用 "\n"
- L121-124   每个参数渲染成 "<parameter=名>\n" + 值 + "\n</parameter>\n"
- L122       参数值是 dict / list 用 json.dumps（transformers 的 tojson），其它类型用 str()
- L149-153   enable_thinking=False 时，生成提示词以 "<think>\n\n</think>\n\n" 结尾
"""
import json
import re
from uuid import uuid4

_CALL = re.compile(r"<tool_call>\n<function=(?P<name>[^>\n]+)>\n(?P<body>.*?)</function>\n</tool_call>", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=(?P<name>[^>\n]+)>\n(?P<value>.*?)</parameter>", re.DOTALL)
_KEYWORD = {"True": True, "False": False, "None": None}


def parameter_types(tools: list[dict]) -> dict[str, dict[str, str | None]]:
    """tools → {工具名: {参数名: JSON Schema 里的 type}}。"""
    return {
        tool["function"]["name"]: {
            name: prop.get("type")
            for name, prop in (tool["function"].get("parameters") or {}).get("properties", {}).items()
        }
        for tool in tools
    }


def parse_tool_calls(text: str, types: dict[str, dict[str, str | None]]) -> tuple[str, list[dict]]:
    """采样出来的文本 → (正文, OpenAI 格式的 tool_calls)。"""
    calls = []
    for call in _CALL.finditer(text):
        kind = types.get(call["name"], {})
        # value 末尾多出来的那个 "\n" 是格式自带的，去掉；值自己的换行保留
        arguments = {name: _typed(value[:-1], kind.get(name))
                     for name, value in _PARAMETER.findall(call["body"])}
        calls.append({"id": f"call_{uuid4().hex[:16]}", "type": "function",
                      "function": {"name": call["name"], "arguments": json.dumps(arguments)}})
    return _CALL.sub("", text).strip(), calls


def _typed(text: str, kind: str | None) -> object:
    """把渲染出来的参数值还原成 JSON 类型。

    模板对 dict / list 用 json.dumps，其它类型用 str()。同一段文本是"字符串"还是"JSON"，
    模板里看不出来（content 是 '{"a": 1}' 的字符串参数，和对象参数，渲染出来一样），
    只能靠工具的 schema 区分。
    """
    if kind in ("object", "array"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text          # 模型写坏了 JSON：原样交给工具，让工具去报错
    if kind != "string":         # 数字 / 布尔 / null，以及 schema 没写类型的参数
        if re.fullmatch(r"-?\d+(\.\d+)?", text):
            for cast in (float, int):
                value = cast(text)
                if str(value) == text:      # 只接受能原样渲染回去的类型
                    return value
        return _KEYWORD.get(text, text)
    return text


def render_prompt(tokenizer, messages: list[dict], tools: list[dict] | None,
                  template_kwargs: dict, add_generation_prompt: bool = True) -> list[int]:
    """messages → prompt 的 token ids。history 里的 arguments 是 JSON 字符串，模板要求 dict。"""
    history = []
    for message in messages:
        calls = message.get("tool_calls")
        if calls:
            message = {**message, "tool_calls": [
                {**call, "function": {**call["function"],
                                      "arguments": json.loads(call["function"]["arguments"])}}
                for call in calls]}
        history.append(message)
    return tokenizer.apply_chat_template(history, tools=tools, add_generation_prompt=add_generation_prompt,
                                         return_dict=False, **template_kwargs)
```

要点（都是**读码 / 源码**，除非另注）：

- **`return_dict=False` 要显式写**。这个参数的默认值各个版本不一样：`main` 分支默认 `True`，返回的是 `BatchEncoding`；v4.57.1 默认 `False`，返回 `list[int]`。探针（2026-11-24）里 PyTRIO 给的 tokenizer 默认返回的就是 `BatchEncoding`。装到的会是哪个版本现在不确定，显式写 `False` 就都是 `list[int]`。
- **history 里的 `arguments` 必须转回 dict**。模板对它做 `| items`，传 JSON 字符串会 `TypeError`（**探针**）。而 harness 发回来的一定是字符串：比如 Anthropic 这条路在 `transform/anthropic.py:523` 就是 `json.dumps(block["input"])`。
- **响应里的 `tool_calls` 用 OpenAI 标准形状**（`content` 是字符串、`arguments` 是 JSON 字符串）。anthropic 的 transform 就是按这个读的（`transform/anthropic.py:424-438`）；openai_responses 两种都认（`transform/openai_responses.py:196-201`）；google 的没细看。
- **参数值的类型要查工具的 schema**。模板渲染 dict / list 用 `json.dumps`，渲染其它类型用 `str()`，所以一个值是"内容恰好是 JSON 的字符串参数"还是"对象参数"，从文本上分不出来。比如 `write_file(content='{"a": 1}')`，`content` 是字符串。`parameter_types` 从请求的 `tools` 里取出每个参数的 JSON Schema `type`，`_typed` 按它还原。

#### 为什么必须"逐 token 还原"

训练序列（builder 产出）= 每一轮**模型实际采样出来的 token** + 轮与轮之间**模板渲染出来的 token**（工具结果、生成提示词）。
而模型生成第 k+1 轮时，看到的上文是"把第 k 轮的响应**解析成结构化消息、再渲染一遍**"的结果。

这两份上文只在**解析后再渲染 == 原始采样文本**时才逐 token 相同。
不相同时不报错，拼链也不断，只是第 k+1 轮的旧 logprob 是在另一份上文下算出来的，重要性比值里混进一点和策略更新无关的偏差。

thinking 关掉时，生成提示词以 `<think>\n\n</think>\n\n` 结尾，而历史里的 assistant 渲染成 `<think>\n` + 空 + `\n</think>\n\n` + 正文，两边正好接上。所以只要正文和工具调用能渲染回去，就逐 token 相等。

上面的解析规则就是为此定的：

1. 正文 `strip()`：模板对所有 content 都 trim。
2. 工具调用之间的 `\n\n` / `\n` 是模板自己加的，解析时直接丢掉。
3. 数字 / 布尔 / null 只还原成 `str(v) == 原文` 的类型：`5` → `int`，`1.5` → `float`，`True / False / None` → 对应类型。`01`、`1e5` 保持字符串，因为它们渲染回去和原文不一样。
4. 值末尾的换行要保留：`<parameter=c>\nabc\n\n</parameter>` 的值是 `"abc\n"`。
5. **dict / list 只有在模型写的 JSON 正好是 `json.dumps(v, ensure_ascii=False)` 的输出时才还原**：一行，分隔符是 `", "` 和 `": "`，数字写成 `1.0` 而不是 `1`。键序不是问题（解析后保留原顺序），非 ASCII 也不是问题（transformers 的 `tojson` 不转义）。模型写成多行或紧凑格式就对不上。这是已知偏差，靠第 1 步的还原率去量。

**这套规则在文本层验证过**（2026-11-25，本机）：用 Jinja 直接渲染 HF 上的真实模板，配置照 transformers 的来（`trim_blocks / lstrip_blocks` 打开，`tojson` 换成 `json.dumps(x, ensure_ascii=False)`；它另外加的两个 Jinja 扩展 `AssistantTracker` 和 `loopcontrols` 没加，这个模板里没有 `{% generation %}`、`{% break %}`，用不到），走一遍"采样文本 → 解析 → 当作 harness 回传的历史 → 再渲染"，和"生成提示词 + 采样文本 + `<|im_end|>`"逐字比对。
13 个用例里 12 个逐字相等，覆盖：纯文本、字符串、结尾带换行的字符串、内容是 JSON 的字符串、整数 / 小数 / 布尔 / null、数组、单行对象、中文和特殊字符、正文加工具调用、一次两个工具调用、多轮。对不上的只有多行 JSON。
**没有过 tokenizer**，token 级是否相等要在第 1 步实测。

**怎么量**（第 1 步验收时临时加，量完删掉）：

```python
if sequence.stop_reason == "stop":
    replay = await asyncio.to_thread(
        template.render_prompt, tokenizer, [*body["messages"], message], body.get("tools"),
        {**kwargs, "add_generation_prompt": False})
    exact = (replay[: len(prompt_ids)] == prompt_ids
             and replay[len(prompt_ids): len(prompt_ids) + len(tokens)] == tokens)
```

**量出来不是 100% 怎么办**：下一步让 shim 自己维护每条链的 token 序列，采样时直接用原始 token 拼 prompt（builder 的 `_slice_interstitial` 就是这个思路，`prefix_merging.py:266-296`）。第一版不做。

**一个已知的限制**（**读码**推导，没实测）：session 中途如果出现一条新的 `user` 消息（不是工具结果），模板会把之前所有 assistant 轮的 `<think>` 块去掉重渲染（模板 L67-104 的 `last_query_index`），上文从那里开始变，builder 会另起一条链（`prefix_merging.py:174-177`）。
Polar 现在的任务都是一个 session 一条指令，碰不到。

### 6.3 `pytrio/shim.py`

```python
"""冒充 OpenAI chat completions 接口，gateway 的 inference.base_url 指到这里。

gateway 只发三种请求：POST /v1/chat/completions、GET /v1/models、GET /health（gateway/proxy.py）。
和 train.py 同进程、同事件循环：换 sampler 就是改一个模块级变量，不需要锁。
"""
import asyncio
import time
from uuid import uuid4

import fastapi
import pytrio as trio

from polar.pytrio import template

app = fastapi.FastAPI()
tokenizer = None
sampler = None
max_tokens = 0
template_kwargs = {}


def configure(tok, default_max_tokens: int, enable_thinking: bool) -> None:
    global tokenizer, max_tokens, template_kwargs
    tokenizer = tok
    max_tokens = default_max_tokens
    template_kwargs = {"enable_thinking": enable_thinking}


def install(new_sampler) -> None:
    global sampler
    sampler = new_sampler


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"object": "model", "id": sampler.model_id}]}


@app.post("/v1/chat/completions")
async def chat_completions(body: dict):
    kwargs = {**template_kwargs, **body.get("chat_template_kwargs", {})}
    # 建议加：thinking 开着时训练上文和采样上文对不上，第一版不支持
    assert kwargs["enable_thinking"] is False
    # 建议加：只返回一个 choice，harness 要了两个就是静默少给
    assert body.get("n", 1) == 1

    prompt_ids = await asyncio.to_thread(
        template.render_prompt, tokenizer, body["messages"], body.get("tools"), kwargs)
    response = await sampler.sample_async(
        prompt=trio.ModelInput.from_ints(prompt_ids),
        num_samples=1,
        sampling_params=trio.SamplingParams(
            max_tokens=body.get("max_tokens", max_tokens),
            temperature=body.get("temperature", 1.0),
            top_p=body.get("top_p", 1.0),
            top_k=body.get("top_k", -1),
            stop=body.get("stop"),
        ),
    )
    sequence = response.sequences[0]
    tokens, logprobs = sequence.tokens, sequence.logprobs
    assert len(tokens) == len(logprobs)

    types = template.parameter_types(body.get("tools") or [])
    content, tool_calls = template.parse_tool_calls(
        tokenizer.decode(tokens, skip_special_tokens=True), types)
    if sequence.stop_reason == "length":
        finish_reason = "length"
    else:
        finish_reason = "tool_calls" if tool_calls else "stop"
    message = {"role": "assistant", "content": content, "tool_calls": tool_calls}
    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body["model"],
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": message,
            "input_token_ids": prompt_ids,
            "token_ids": tokens,
            "logprobs": {"content": [
                {"token": token, "token_id": token_id, "logprob": logprob}
                for token, token_id, logprob
                in zip(tokenizer.convert_ids_to_tokens(tokens), tokens, logprobs)
            ]},
        }],
        "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(tokens),
                  "total_tokens": len(prompt_ids) + len(tokens)},
    }
```

响应里每个字段是给谁看的（**读码**）：

| 字段 | 谁读 | 注意 |
|---|---|---|
| `choices[0].input_token_ids` | `Trace.prompt_ids` | |
| `choices[0].token_ids` | `Trace.response_ids` | 原始采样 token，**包含结尾的 `<|im_end|>`** |
| `choices[0].logprobs.content[].{token_id, logprob}` | `Trace.response_logprobs` | `token_id` 序列必须和 `token_ids` **完全相等**，否则不报错、直接变成 `None`（`trajectory/builder/record_utils.py:82-107`） |
| `choices[0].finish_reason` | 结束符的自动探测（`prefix_merging.py:248-264`） | 只认 `stop / tool_calls / stop_sequence`，所以映射成这三个加 `length` |
| `choices[0].message` | 转回 harness 自己的协议 | |
| `usage` | 转给 harness | |

其它要点：

- **不做流式**。gateway 向上游发请求时固定 `stream=False`（`gateway/proxy.py:114-136`），流式是它自己合成的。
- **渲染放线程里**。长上下文的 Jinja 渲染 + 分词是几十到几百毫秒的 CPU，放事件循环上会卡住别的请求。
- **`enable_thinking`**：gateway 对名字含 `qwen3.5` 的模型本来就会塞 `enable_thinking=False`（`gateway/transform/base.py:116-121`，注释写的原因是 Qwen3.5 会把工具调用写进 thinking 里）。shim 自己再默认一次，这样 `model_served` 叫什么都行。那段 gateway 代码不用动。
- **没有透传的请求参数**：`logit_bias`、`response_format`、`tool_choice`、各种 penalty、`min_p`（PyTRIO 的 `SamplingParams` 没有），以及 `seed`（PyTRIO 有，但训练时不需要固定种子）。现在的 harness 不发这些；以后哪个 harness 发了，行为会悄悄不一样，到时候再加 assert。

### 6.4 `pytrio/rollout.py`

```python
"""向 rollout server 提交任务，等它跑完，取回结果。"""
import asyncio

import httpx

from polar.rollout.models import TaskResult


async def run_task(client: httpx.AsyncClient, request: dict) -> TaskResult:
    task_id = request["task_id"]
    await client.post("/rollout/task/submit", json=request)
    while (await client.get(f"/rollout/task/{task_id}/progress")).json()["status"] == "running":
        await asyncio.sleep(2.0)
    result = TaskResult.model_validate((await client.get(f"/rollout/task/{task_id}")).json())
    await client.delete(f"/rollout/task/{task_id}")
    return result
```

需要 rollout 加两个接口（见 §6.7）。原因（**读码**）：

- 现有的 `GET /rollout/task/{id}` 每次都返回**所有已完成 session 的完整轨迹**（`rollout/manager.py:260-272`）。长上下文的 session 一个就有几 MB，每 2 秒取一次，一个 step 下来是几个 GB 的序列化。
- `RolloutManager._tasks` 只增不删（`rollout/manager.py:106`、`:127`）。几千个 step 的训练会把 rollout 进程的内存吃光。

### 6.5 `pytrio/data.py`

```python
"""SessionResult → Datum。一条 trace 一个 Datum；失败的 session 整个丢掉。"""
import numpy as np
import pytrio as trio

from polar.rollout.models import SessionResult, SessionStatus
from polar.trajectory.models import Trace


def build_group(results: list[SessionResult]) -> list[trio.Datum]:
    """一个任务的 num_samples 个 session → Datum 列表。奖励全相同的组返回空列表。"""
    kept = [r for r in results
            if r.status == SessionStatus.COMPLETED
            and r.trajectory.status == "COMPLETED"
            and r.trajectory.traces]
    rewards = [sum(trace.reward for trace in r.trajectory.traces) / len(r.trajectory.traces) for r in kept]
    if len(set(rewards)) <= 1:
        return []
    baseline = sum(rewards) / len(rewards)
    return [trace_to_datum(trace, (reward - baseline) / len(r.trajectory.traces))
            for r, reward in zip(kept, rewards) for trace in r.trajectory.traces]


def trace_to_datum(trace: Trace, advantage: float) -> trio.Datum:
    assert trace.response_logprobs is not None, "trace 没有逐 token 的 logprob，训练会用错的比值"
    tokens = trace.prompt_ids + trace.response_ids
    logprobs = [0.0] * len(trace.prompt_ids) + trace.response_logprobs
    advantages = [0.0] * len(trace.prompt_ids) + [advantage if mask else 0.0 for mask in trace.loss_mask]
    assert len(tokens) == len(logprobs) == len(advantages), "Datum 三个数组没有对齐"
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": np.asarray(tokens[1:], dtype=np.int64),
            "logprobs": np.asarray(logprobs[1:], dtype=np.float32),
            "advantages": np.asarray(advantages[1:], dtype=np.float32),
        },
    )
```

规则（**设计**，依据在右列）：

| 规则 | 依据 |
|---|---|
| 一条 trace 一个 `Datum`，不把一个 session 的多条 trace 拼起来 | 每条 trace 的 logprob 是在它自己的 prompt 下采的 |
| prompt 和工具结果部分的 `logprobs`、`advantages` 填 0（`target_tokens` 保留真实 token，反正 advantage 是 0） | PyTRIO 文档：不参与训练的位置把 `advantages` 设为 0，三个数组长度和 `model_input` 一致 |
| 三个数组一起右移一位 | 同上；`model_input` 比 `tokens` 少最后一个 |
| 组 = 一个任务的 `num_samples` 个 session；`advantage = reward - 组内平均` | 和 PyTRIO 的 GRPO 示例一致（Slime 那边用的是 leave-one-out，这里不沿用） |
| 失败的 session（`TIMEOUT / ERROR`，或者轨迹里一条 trace 都没有）整个丢掉：不在数据里，也不在平均里 | `slime_bridge/adapter.py:252-260` 把它们标成 ABORTED / FAILED，`slime_bridge/reward_post_process.py:64-67` 把它们从 baseline 里排除 |
| 奖励全相同的组跳过 | advantage 全 0，没有梯度，白算一次前反向 |
| 同一个 session 的每条 trace 除以这个 session 的 trace 数 | `importance_sampling` 在一条 Datum 内是 `.sum()`（PyTRIO 文档）。不除的话 trace 多的 session 权重就大 |
| session 的 reward = 它的 trace reward 的平均 | evaluator 把 `outcome_reward` 广播到每条 trace（`gateway/node.py:724-729`）；写成平均是为了以后支持按 trace 打分 |
| 被 `max_tokens` 截断的 trace 照常训练 | 它的 token 和 logprob 都是真的 |
| reward 是 `None` 不 assert | `sum()` 遇到 `None` 自己会抛 `TypeError`，不会静默 |

关于"除以 trace 数"：如果 PyTRIO 在一次 `forward_backward` 里是按 token 平均而不是按样本平均，这个除法只是近似。见 §11 第 1 条。

### 6.6 `pytrio/train.py`

```python
"""训练主循环：polar train -c train.yaml"""
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import httpx
import pytrio as trio
import uvicorn

from polar.pytrio import data, shim
from polar.pytrio.config import load_config
from polar.pytrio.rollout import run_task

logger = logging.getLogger("polar.pytrio")


class Server(uvicorn.Server):
    def handle_exit(self, sig, frame) -> None:
        raise KeyboardInterrupt      # 不走 uvicorn 的优雅退出，Ctrl-C 和普通脚本一样


def make_request(tasks: list[dict], step: int, index: int, cfg) -> dict:
    task = dict(tasks[(step * cfg.tasks_per_step + index) % len(tasks)])
    task.update(task_id=f"step{step:05d}-{index:03d}", num_samples=cfg.group_size)
    return task


async def train(trainer, cfg) -> None:
    server = Server(uvicorn.Config(shim.app, host=cfg.shim_host, port=cfg.shim_port, log_level="warning"))
    asyncio.create_task(server.serve())

    lines = Path(cfg.tasks_file).read_text().splitlines()
    tasks = [json.loads(line) for line in lines if line.strip()]
    adam = trio.AdamParams(learning_rate=cfg.learning_rate)

    async with httpx.AsyncClient(base_url=cfg.rollout_url, timeout=60.0) as client:
        for step in range(cfg.start_step, cfg.max_steps):
            started = time.monotonic()
            shim.install(await trainer.save_weights_and_get_sampling_client_async())
            results = await asyncio.gather(*(
                run_task(client, make_request(tasks, step, index, cfg))
                for index in range(cfg.tasks_per_step)))
            rollout_done = time.monotonic()

            groups = [data.build_group(result.results) for result in results]
            datums = [datum for group in groups for datum in group]
            if datums:                # 整步都失败、或奖励全相同时，这一步不更新
                fwd_bwd = await trainer.forward_backward_async(datums, loss_fn="importance_sampling")
                optim = await trainer.optim_step_async(adam)
                metrics = (await fwd_bwd).metrics
                await optim
            else:
                metrics = {}

            record = {"step": step, **summarize(results, groups), **metrics,
                      "rollout_s": rollout_done - started, "train_s": time.monotonic() - rollout_done}
            with Path("metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            logger.info("step %d %s", step, record)

            if (step + 1) % cfg.save_every == 0:
                await (await trainer.save_state_async(name=f"step-{step + 1}"))

    await (await trainer.save_weights_for_sampler_async(name="final"))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(sys.argv[1])
    service = trio.ServiceClient()
    if cfg.resume_from:
        trainer = service.create_training_client_from_state_with_optimizer(cfg.resume_from)
    else:
        trainer = service.create_lora_training_client(base_model=cfg.base_model, rank=cfg.lora_rank)
    shim.configure(trainer.get_tokenizer(), cfg.max_tokens, cfg.enable_thinking)
    asyncio.run(train(trainer, cfg))
```

- `summarize(results, groups)`（略）：数成功 / 超时 / 出错的 session、被跳过的组、平均 reward、`Datum` 个数、可训练 token 数，以及从 `trajectory.metadata["reconstruction_stats"]` 汇总出来的合并率（`prefix_merging.py:121-124`）。每步追加一行到 `metrics.jsonl`。
- **同步的 client 在事件循环外建**：`ServiceClient()` 的构造函数会登录、建连接、拉模型列表（PyTRIO 文档）。
- **uvicorn 当成事件循环上的一个 task**。`uvicorn.Server.serve()` 是协程（uvicorn 0.44.0 `server.py:77`）。它在主线程会接管 SIGINT / SIGTERM（`server.py:322-329`），所以子类里把 `handle_exit` 改成直接抛 `KeyboardInterrupt`。
- **`forward_backward_async` 和 `optim_step_async` 要 await 两次**；`save_*_async` 同理；`save_weights_and_get_sampling_client_async` 只 await 一次（PyTRIO 文档）。
- **损失先用 `importance_sampling`**，和 PyTRIO 的 GRPO / Search-R1 示例一致。它没有裁剪；同步采样时比值一开始就是 1，够用。想要裁剪，换成 `loss_fn="ppo"` 即可，`Datum` 不用改。
- **权重每步同步一次**。并发量由 gateway 的 `max_run_workers` 决定（topology，默认 2）。

### 6.7 对现有代码的改动

**`rollout/server.py` + `rollout/manager.py`** —— 加两个接口，约 20 行：

- `GET /rollout/task/{task_id}/progress` → `{"task_id", "status", "total_sessions", "completed_sessions"}`。`status` 的取值和现在一样是 `running / completed / failed`。
- `DELETE /rollout/task/{task_id}` —— 只清内存里的记录。落盘的 `task_<id>/ses_<sid>.json` 不动。

**`gateway/`** —— 删 `engine.py`（208 行），然后：

| 位置 | 改动 |
|---|---|
| `gateway/server.py:21`、`:84` | `InferenceClient(node.inference_base_url)`，不再传 engine |
| `gateway/proxy.py:72`、`:123`、`:136`、`:170` | 去掉 `engine` 参数和 `prepare_request / normalize_response` 两个调用 |

`transform/` 一行不动。

**`config/topology.py`** —— 删 `engine` 字段（`:23`）和 `engine` 属性（`:76-77`）；`cli.py:263` 的标签去掉 engine。
`_StrictModel` 是 `extra="forbid"`（`topology.py:16-19`），所以旧的 topology 文件里留着 `engine:` 会直接报错。

topology 只需要改这两行（**设计**，只列改动处）：

```yaml
gateway:
  nodes:
    - model_served: Qwen/Qwen3.5-4B:peft:256k    # 只是给 harness 看的名字
      inference:
        base_url: http://127.0.0.1:9000           # 训练进程里的 shim
```

**`platform/events.py` → `polar/events.py`** —— 文件只依赖标准库（`platform/events.py:8-16`），搬之前有 4 处引用要改：
`rollout/server.py:14`、`rollout/manager.py:15`、`rollout/pipeline.py:15`、`gateway/server.py:43`。

**`cli.py`** —— 删 dashboard（`:18-19`、`:41`、`:122`），加 `train` 子命令。

**`pyproject.toml`**：

- 依赖加 `pytrio`、`numpy`；`transformers` 如果不是 `pytrio` 带进来的也要加。
- `:54` 的 `include` 改成 `["polar*"]`。
- `:56-61` 的 package-data 去掉 `platform/web/dist/**/*` 和 `slime_bridge`。
- `:29-30` 的 `swebench` extra 跟着 evaluator 一起决定去留。
- `:75`、`:92` 是 pytest 和 ruff 排除列表里的 `slime`（旁边还有 `Megatron-LM`），一并清掉。

**`README.md`** —— `:44`、`:49` 的 vllm / sglang 安装、`:59` 的面板构建、`:95` 的 `polar dashboard` 都要改；再补一段训练的跑法。

---

## 7. 断言清单

一共 5 条，前 3 条必加，后 2 条建议加。每条都对应一种"不报错的错误"。

| # | 位置 | 断言 | 不断言会怎样 |
|---|---|---|---|
| 1 | `shim.py` | `len(tokens) == len(logprobs)` | 两个列表 zip 之后变短，`token_ids` 和 `logprobs.content[].token_id` 不再相等，builder 不报错地把 `response_logprobs` 置成 `None` |
| 2 | `data.py` | `trace.response_logprobs is not None` | 如果兜底成 0.0，重要性比值变成 `exp(logp_new − 0)`，梯度方向整个是错的，loss 曲线上看不出来 |
| 3 | `data.py` | `tokens / logprobs / advantages` 三个等长 | 右移错位后服务端怎么处理不知道；如果被对齐或截断，就是静默训坏。`loss_mask` 为空的 trace 也靠它抓住（Pydantic 对空列表不校验长度，`trajectory/models.py:116`） |
| 4 | `shim.py` | `enable_thinking is False` | thinking 开着时训练上文和采样上文对不上（§6.2），训练照常进行，只是数据慢慢变偏 |
| 5 | `shim.py` | `n == 1` | harness 要 2 个 choice 只拿到 1 个，没有任何地方报错 |

**不放在运行时的检查，放测试里**：

- `tests/pytrio/test_template.py`：采样文本 → 解析 → 当作回传的历史 → 再渲染。文本层不需要模型：用一个假 tokenizer，直接用 Jinja 渲染真实模板（模板文件在 `tests/fixtures/` 存一份，或者测试时从 HF 拉），比对字符串。token 层的用例需要真实 tokenizer，标 `integration`。
- `tests/pytrio/test_data.py`：手造的 `Trace` → `Datum`，检查右移后的三个数组；失败 session 被丢；奖励全相同的组返回空；多 trace 的 session 权重是 1/n。
- `tests/pytrio/test_shim.py`：塞一个假的 sampler（`sample_async` 返回写死的 `sequences`），用 `fastapi.testclient` 打三个接口，检查响应形状和 `finish_reason` 映射。

测试里的假对象是测试设施，不算兜底代码。

---

## 8. 删除清单

### 一定删

| 删什么 | 行数 | 说明 |
|---|---:|---|
| `src/slime_bridge/` | 2357 | 整个目录 |
| `tests/slime_bridge/` | 6 个文件 | |
| `src/pytrio_bridge/README.md` | 375 | 被本文取代 |
| `src/polar/gateway/engine.py` | 208 | sglang / vllm 两个 engine，加一个基类 |
| `src/polar/platform/` | 1470 | 先搬 `events.py` |
| `web/` | 26 个文件 | 面板前端 |
| `tests/platform/` | 3 个文件 | |
| `tests/gateway/test_engine.py` | | |
| `tests/trajectory/test_engine_trajectory_equivalence.py` | | 它比较的是两个 engine 下的 builder 输出 |
| `examples/swegym_slime_grpo/` | | Slime + Megatron 的启动脚本 |
| `scripts/patch/patch_slime_router_tokens.sh` | | 给 Slime router 打的补丁 |
| `examples/*/topology.sgl.yaml`、`topology.vllm.yaml` | | 换成一个指向 shim 的 `topology.yaml` |

要跟着改的测试：`tests/gateway/test_proxy.py`、`tests/config/test_topology.py`（都有 `engine` 参数）。

### 看情况删

| 删什么 | 行数 | 条件 |
|---|---:|---|
| `runtime/apptainer.py` | 205 | 只用 Docker 的话 |
| `trajectory/builder/per_request.py` + `tests/trajectory/test_per_request_builder.py` | 51 | 5 个 example 都用 `prefix_merging` |
| `trajectory/evaluator/swebench_harness.py` | 164 | 不跑 SWE-bench / SWE-Gym 的话（它依赖 `swebench` extra） |
| `trajectory/evaluator/harbor_rubric.py` | 426 | 5 个 example 都没用它 |
| `agent/presets/` 里不用的 harness | 共 1450 | 每个是一个独立文件，加 `factory.py` 里一行 |
| `scripts/monitor_wandb_gpu.py` | | 没读过；看名字是监控本地 GPU 的 |

### 死代码（上一轮读代码记下的，删之前 grep 一下引用）

`rollout/pipeline.py:367`、`:385` 两个没人调的函数；`gateway/dispatcher.py` 的 `_STOP`；`RuntimeSpec.storage_mb`；从来没被传过的 `session_base_dir`。

### 顺手发现的问题（不阻塞，训练前要知道）

1. **`allow_internet: false` 现在不能用**：Docker 那边加了 `--network none`（`runtime/docker.py:50-51`），而 agent 要通过 `public_url` 访问 gateway（`gateway/node.py:769`），结果 agent 连不上模型。没有 example 开这个选项。要么删掉这个选项，要么改成真正的网络隔离。
2. **eval 容器的目录嵌在 agent 能看到的目录里**：`session_dir/eval_runtime`（`gateway/node.py:425`），而 `session_dir` 整个挂进了 agent 容器的 `/polar/session`。agent 理论上能提前看到评测环境。
3. **超时或失败的 session 仍然会跑 evaluator**（`gateway/node.py:559-624`）。训练这边按状态把它们丢掉了，所以不影响训练，但会多花评测的时间。

---

## 9. 开发顺序与验收

### 第 0 步：环境

需要一台有 Docker 的 Linux 机器（这台 Mac 跑不了容器）。`uv pip install -e .`，`trio login`。

### 第 1 步：只写 `template.py` + `shim.py`，验证轨迹

不训练，用一个固定的 base 模型 sampler 把 shim 单独起起来：

```python
# 临时脚本，只用于这一步
import uvicorn, pytrio as trio
from polar.pytrio import shim

service = trio.ServiceClient()
sampler = service.create_sampling_client(base_model="Qwen/Qwen3.5-4B:peft:256k")
shim.configure(sampler.get_tokenizer(), 16384, False)
shim.install(sampler)
uvicorn.run(shim.app, host="0.0.0.0", port=9000)
```

topology 的 `inference.base_url` 指向 `:9000`，照常起 rollout + gateway，用 `polar submit` 跑一个带工具调用的 example。

验收：

- `test_template.py` 全过。
- 真实采样几十次工具调用，§6.2 的临时对比代码量出来的还原率是 100%（标量参数）。
- session 状态是 COMPLETED，harness 真的执行了工具调用（session 结果的 `response_messages` 里有工具返回）。
- `trajectory.metadata` 里 `trace_count` 明显小于 `record_count`（多轮被拼成了一条链），`chains_reconstructed_truncated` 是 0。

这一步不需要训练，也不需要 GPU。

### 第 2 步：`config.py` + `rollout.py` + `data.py`

不训练。提交任务、取结果、造出 `Datum`，把长度和内容打印出来核对。

验收：`test_data.py` 全过；真实任务跑出来的 `Datum` 里，`target_tokens[len(prompt)-1:]` 就是 `response_ids`，`advantages` 非零的位置和 `loss_mask` 一致。

### 第 3 步：`train.py`

1 个任务 × 4 个 session × 3 个 step。

验收：`metrics.jsonl` 里 `loss_sum / loss_mean / token_count` 是有限值；3 步之后 `save_state` 的 checkpoint 存在；训练后再采样，输出和训练前不一样。

### 最后：删除 + 改文档

按 §8 删，跑一遍剩下的测试，改 `README.md`。

---

## 10. 已验证的结论从哪来

| 结论 | 来源 |
|---|---|
| gateway 只发三种上游请求，`stream` 固定为 False | 读码：`gateway/proxy.py:114-136` |
| `token_ids` 和 `logprobs.content[].token_id` 不等时 logprob 静默变 `None` | 读码：`trajectory/builder/record_utils.py:82-107` |
| builder 用原始 response + 渲染出来的 interstitial 拼链；拼链只比较两次请求的 prompt | 读码：`prefix_merging.py:134-242`、`:266-296`、`:339-363` |
| evaluator 把 reward 写进 `Trace.reward` | 读码：`gateway/node.py:700-738` |
| PyTRIO 的 API 签名 | 源码：PyTRIO 文档，2026-11-25 |
| `importance_sampling` 在一条 Datum 内是 `.sum()`；指标有 `loss_sum / loss_mean / token_count` | 源码：PyTRIO 文档 /guide/loss_fn、/guide/train |
| Qwen3.5 模板的渲染规则 | 源码：HF `Qwen/Qwen3.5-4B` 的 `chat_template.jinja`，2026-11-25 拉取 |
| `apply_chat_template` 的 `return_dict` 默认值：`main` 分支是 `True`（返回 `BatchEncoding`），v4.57.1 是 `False` | 源码：transformers `main` 分支 `tokenization_utils_base.py:3054`、`:3235-3237`；`v4.57.1` 同文件 `:1559`（2026-11-25 拉取） |
| transformers 渲染 chat template 时覆盖了 `tojson`：`json.dumps(x, ensure_ascii=False)`，不排序键、不做 HTML 转义 | 源码：transformers `main` 分支 `utils/chat_template_utils.py:623-626`、`:662-667` |
| 解析 → 当作回传历史 → 再渲染，在真实模板上逐字相等（13 个用例里 12 个，只有多行 JSON 对不上） | 本机实跑：Jinja 3.1.6 + HF 的 `chat_template.jinja`，文本层，没过 tokenizer |
| `uvicorn.Server.serve()` 是协程；主线程上接管信号 | 源码：uvicorn 0.44.0 `server.py:77`、`:322-329` |
| `<|im_end|>` = 248046，带 logprob 一起返回；`stop_reason` 是 `stop / length`；turn N 的 prompt 是 turn N+1 的 token 前缀；渲染出的 assistant 正文 == 采样 token | 探针：pytrio 0.2.10，2026-11-24 |
| 256k 上下文只有 `Qwen/Qwen3.5-4B:peft:256k` 有 | 探针：同上 |

---

## 11. 未验证的

PyTRIO 这边：

1. **一次 `forward_backward` 里多条 Datum 之间是求和、按样本平均，还是按 token 平均。** 文档只给了单条 Datum 内的 `.sum()`。它决定学习率的尺度，也决定"除以 trace 数"是精确的还是近似的。第 3 步用两条已知 advantage 的 Datum 对着 `metrics` 算一遍就知道。
2. 单次请求的大小上限。探针看到训练侧 `max_batch_size=1024`、`max_tokens_per_request=327680`，没测一个 step 几十条长序列的总 token 量。
3. `save_weights_and_get_sampling_client_async()` 的耗时。每个 step 一次，直接算进训练时间里。
4. 几十个 session 同时采样时的限流行为。探针只测到 16 路短请求。
5. `prompt + max_tokens` 超过 262144 时是报错还是截断。
6. `pip install pytrio` 是否自带 `transformers`（本地渲染 chat template 要用）。
模板 / shim 这边：

7. 模型实际写出的 dict / list 是不是 `json.dumps(v, ensure_ascii=False)` 这个格式。规则在文本层验证过，模型的实际输出要采样看。
8. 模型把布尔写成 `True` 还是 `true`。写成 `true` 时还原是精确的，但 harness 拿到的是字符串 `"true"`，严格的工具会报类型错误。
9. token 级的还原。文本层相等了，BPE 在拼接处会不会合并出不同的 token，要第 1 步实测。
10. 真实 harness 会不会改写 `arguments`（换键序、换空白）。标量值不受影响，dict / list 值会受影响。用第 1 步的还原率确认。

Polar 这边：

11. uvicorn 当 task 跑在训练循环同一个事件循环上，按源码可行，没跑过。
12. 各 harness 对"`content` 是空字符串 + `tool_calls`"的接受度。anthropic 和 openai_responses 两个 transform 读码确认过，google 的没看。
13. "thinking 开着拼链不断"是读 builder 和模板推出来的，没实测。
14. 这台 Mac 上没有容器运行时、没装 PyTRIO SDK。文中的代码只有 `template.py` 的三个函数在文本层跑过，其余一行都没跑过。

---

## 附 A. PyTRIO API 速查（文档，2026-11-25）

```text
service = trio.ServiceClient(api_key=None)                 # 同步；构造时登录、建连接、拉模型列表
service.get_supported_models() -> list[str]
service.create_lora_training_client(base_model, rank=32, seed=None,
                                    train_mlp=True, train_attn=True, train_unembed=True)
service.create_sampling_client(base_model, model_path=None)
service.create_training_client_from_state_with_optimizer(path)

trainer.get_tokenizer()                                    # transformers / modelscope 的 AutoTokenizer
await trainer.save_weights_and_get_sampling_client_async()      # await 一次 → SamplingClient（匿名临时存档）
fwd = await trainer.forward_backward_async(data, loss_fn="importance_sampling");  await fwd
opt = await trainer.optim_step_async(trio.AdamParams(learning_rate=1e-5));         await opt
fut = await trainer.save_state_async(name="step-10");        await fut   # 权重 + 优化器状态
fut = await trainer.save_weights_for_sampler_async(name="final"); await fut  # .path / .model / .size
fwd.metrics                                                                  # loss_sum / loss_mean / token_count

await sampler.sample_async(prompt=trio.ModelInput.from_ints(ids), num_samples=1,
                           sampling_params=trio.SamplingParams(
                               max_tokens=None, seed=None, stop=None,
                               temperature=1, top_k=-1, top_p=1),
                           return_text=True)                 # await 一次 → .sequences[]
sequence.tokens / .logprobs / .text / .stop_reason
```

- 也有 `_async` 版本的 `create_lora_training_client`、`create_sampling_client`。
- 内置损失：`cross_entropy`、`importance_sampling`、`ppo`（默认裁剪 0.2，可用 `loss_fn_config` 改）、`cispo`、`dro`。RL 的 `Datum.loss_fn_inputs` 都是 `target_tokens / logprobs / advantages`。
- client 没有 `close()`。
- 文档：`https://docs.pytrio.com/docs/content/<路由>/content.md`，路由有 `api/ServiceClient`、`api/TrainingClient`、`api/SamplingClient`、`api/SamplingParams`、`guide/train`、`guide/loss_fn`。

## 附 B. 和旧 README 的差别

| 旧 README（`src/pytrio_bridge/README.md`） | 本文 | 原因 |
|---|---|---|
| `svc.create_training_client_async(base_model=…)` | `service.create_lora_training_client(base_model=…, rank=…)` | 旧写法的方法名在官方文档里不存在 |
| shim 跑在后台线程的 uvicorn 里，用 `run_coroutine_threadsafe` 换 sampler | 同一个事件循环 | 不用赌 `SamplingClient` 能不能跨 event loop；旧 README"未测"里的 GIL 争抢和跨 loop 两项就没有了 |
| `model_served == base_model`，配一条 assert | 不需要 | shim 不看 `model_served`，thinking 开关是 shim 自己的配置 |
| 启动时渲染 topology 写临时文件 | 静态 topology，只改 `base_url` | |
| "不透传 `enable_thinking` 会让 `prefix_merging` 把每条 trace 拆成一轮" | 这个说法不对 | 拼链只比较两次请求的 prompt（`prefix_merging.py:174-177`），不看采样出来的 response。真正的问题是训练上文和采样上文不一致（§6.2） |
| "失败 session 的 advantage 置 0" | 失败 session 整个丢掉，也不进平均 | 和 `slime_bridge/reward_post_process.py:64-67` 一致 |
| 轮询 `GET /rollout/task/{id}` | 轮询进度接口，结束后取一次 | 那个接口每次都带完整轨迹 |
| 位置 `src/pytrio_bridge/` | `src/polar/pytrio/` | 项目级改造 |

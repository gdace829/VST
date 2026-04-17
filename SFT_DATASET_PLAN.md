# SFT Dataset Construction Plan

## Goal

先做一个最小可训练的闭环，验证下面这条链路能不能跑通：

`raw video + QA -> teacher rollout -> SFT trajectory jsonl -> trainable dataset`

这一步的目标不是一开始就做最强 teacher，而是先把问题定义真正落成可训练数据。

---

## Overall Pipeline

完整训练闭环按三阶段理解：

1. `teacher data construction` 先不用模型自己学，先用一个离线 teacher 规则/工具流程，把原始视频和 QA 变成可监督的训练标签
2. `SFT warm-up`
3. `RL refinement`

当前第一优先级是只做前两步中的最小版本：

- 先用一个简化 teacher 把长视频转成 write-time trajectories
- 再让现有 SFT 训练框架能够读取这些 trajectories

---

## Teacher construction philosophy

这里建议直接借 AutoTool 的高层训练哲学，但把对象从 `tool selection` 改成 `memory writing`。

AutoTool 的核心做法不是先训练一个 teacher policy，而是：

1. 先拿一条已有 trajectory 或可构造的 decision context
2. 用一个强 expert model 离线补出中间 supervision
3. 再用 judge / filter 做质量控制
4. 最后把这些中间监督插回 trajectory，形成可用于 SFT 的训练数据

对应到你这里，可以改写成：

1. 先把长视频切成 causal clips，并给出对应的 query / answer / evidence 信息
2. 用一个离线 teacher 为每个 clip 生成：
   - utility 估计
   - evidence type
   - teacher action
   - write content
3. 再用 judge / filter 去过滤明显错误或不一致的标签
4. 最后按时间顺序 rollout，形成 write-time trajectories

因此，你的 `teacher data construction` 可以有两版：

- `v0 rule-based teacher`
  - 直接用 overlap / heuristic / temporal grounding 规则生成 action 和 content
  - 用来先跑通最小闭环

- `v1 strong-model teacher`
  - 用一个强 VLM / reasoning model 为每个 clip 生成更高质量的 action + write_content
  - 再用 judge 做过滤与一致性修正
  - 这版更接近 AutoTool 的数据构造风格

建议顺序：

- 先做 `v0 rule-based teacher`，验证 schema 和 pipeline
- 再升级到 `v1 strong-model teacher`

---

## Combined SFT Design: VST + DeepEyesV2 + AutoTool

你的 SFT 可以明确设计成一种 `tool-call-style memory-state trajectory SFT`。它综合三类已有工作的训练思想，但监督目标换成 `memory_call`：

- `VST`
  - 借它的 `sequential clip-level supervision`
  - 长视频不是一次性监督最终答案，而是切成 causal clips，并按时间展开成多轮中间监督

- `DeepEyesV2`
  - 借它的 `visual observation as later context`
  - 工具生成的视觉证据可以作为后续上下文里的外部 observation
  - 在你这里，对应为 `write_visual -> visual memory slot`

- `AutoTool`
  - 借它的 `cheap vs expensive action prior`
  - 先用 teacher 教模型什么时候省预算、什么时候花预算
  - 在你这里，对应为 `cheap: skip/write_summary` 和 `expensive: write_visual/write_structured`

因此你的每一步 SFT 不是：

```text
clip -> caption
clip -> thought
```

而是：

```text
clip_k + external_memory_state_k + budget_k
    -> memory_call_k
    -> external_memory_state_{k+1}
```

其中 assistant 只监督 `memory_call_k`。`external_memory_state_{k+1}` 由外部 memory updater 执行 `memory_call_k` 后得到，并被序列化进下一轮输入。

### One-turn SFT format

推荐把每个 step 写成一轮 memory-operation dialogue：

```text
User:
Current streaming clip:
<video>

External hierarchical memory state:
[summary_memory]
...

[visual_memory]
- slot_id: v_001
  time_span: 0.0-4.0s
  image_ref: memory/video_a/v_001.jpg
  text_anchor: A man places a silver key in the drawer.

[structured_memory]
- event: person puts key into drawer
  time_span: 0.0-4.0s

Budget state:
storage_left: 73
retrieval_left: 12

Instruction:
Emit exactly one memory_call for the current clip.
```

Assistant target:

```text
<memory_call>
{
  "effort": "expensive",
  "action": "write_visual",
  "memory_level": "visual",
  "content": {
    "time_span": [8.0, 12.0],
    "text_anchor": "The label on the box reads 'fragile'.",
    "visual_target": "keyframe"
  },
  "cost": 3
}
</memory_call>
```

### DeepEyes-style visual memory

DeepEyesV2 的视觉中间结果更像：

```text
tool output image -> append to later context
```

你这里应该改成：

```text
write_visual -> create visual memory slot -> serialize/retrieve later
```

第一版推荐 `text-ref visual memory`，即只把视觉记忆写成可检索的 slot 引用：

```json
{
  "slot_id": "v_0012",
  "type": "visual",
  "time_span": [48.0, 52.0],
  "image_ref": "memory/video_a/v_0012.jpg",
  "text_anchor": "A woman puts a small silver key into the top drawer.",
  "cost": 3
}
```

后续 v1 再升级到 `image-backed visual memory`：

```text
[visual_memory]
- slot_id: v_0012
  image: <image>
  time_span: 48.0-52.0s
  text_anchor: A woman puts a small silver key into the top drawer.
```

注意区别：

- DeepEyesV2 的 visual observation 通常服务于当前已知 query
- 你的 visual memory slot 是在 future query unknown 时提前保存的视觉证据

### External memory updater

assistant 只输出 `memory_call`，不负责重写完整 memory state。外部 updater 执行：

```text
skip:
  no memory update

write_summary:
  append content to summary_memory

write_visual:
  extract/store keyframe or crop
  create visual memory slot
  append slot to visual_memory

write_structured:
  create event/fact slot
  append slot to structured_memory
```

同时扣预算：

```text
skip: 0
write_summary: 1
write_structured: 2
write_visual: 3
```

这和工具调用 SFT 的结构同构：

| Tool-use SFT | Ours |
|---|---|
| tool call | memory_call |
| tool executor | memory updater |
| tool observation | updated hierarchical memory state |
| tool cost | storage / retrieval cost |
| solve known query | prepare for unknown future query |

---

### Strong-model teacher template

如果后面要走 AutoTool 风格的强 teacher，这一层建议拆成三步：

1. `expert generation`
   - 用强模型根据当前 clip、局部上下文、已有 query/evidence 信息生成：
     - `teacher_effort`
     - `teacher_action`
     - `write_content`

2. `judge filtering`
   - 用 judge 或规则过滤掉：
     - action 和 evidence type 明显不匹配
     - write_content 与 clip 内容不一致
     - 预算层级不合理的样本

3. `trajectory review`
   - 在 video-level 再检查一次前后时序一致性：
     - 是否前后重复写同样内容
     - 是否该 structured 的地方被 summary 掉
     - 是否 memory 更新逻辑前后冲突

这样最终得到的数据更像：

`teacher-labeled and filtered write trajectory`

而不是生硬的逐 clip 独立标签。

---

## Phase 1: Minimal SFT Loop

### Step 1. Define the final training schema

先把最终的监督样本格式定死。建议先用 step-level JSONL：

```json
{
  "video_id": "xxx",
  "step": 3,
  "clip": {
    "start": 12.0,
    "end": 16.0,
    "video": "path/or/frame_list"
  },
  "memory_state": {
    "summary_history": "...",
    "structured_history": []
  },
  "budget_state": {
    "storage_left": 72
  },
  "teacher_effort": "cheap",
  "teacher_action": "write_summary",
  "assistant_target": "<memory_call>{...}</memory_call>",
  "write_content": "The person enters the kitchen and opens the fridge."
}
```

其中：

- `teacher_effort` 是可选辅助字段
- `teacher_action` 是核心监督字段
- `assistant_target` 是最终 SFT 监督文本，推荐使用 `<memory_call>{...}</memory_call>`
- `write_content` 是非 `skip` 情况下的目标写入内容，可以作为构造 `assistant_target` 的中间字段

如果后面不想保留 `teacher_effort`，也可以去掉；但第一版建议保留，便于做 `effort-first` teacher 对比。

---

### Step 2. Pick a very small debug subset

先不要上全量数据。建议只选：

- `20 ~ 50` 个视频
- 每个视频带 query / answer
- 如果有 evidence span 或 temporal grounding 更好

目标：

- 先验证 builder 和 dataset schema
- 降低 debug 成本

---

### Step 3. Build a minimal clip segmenter

先实现最稳的 clip 切分器，而不是最复杂的切分器。

输入：

- video path
- clip duration，例如 `4s`

输出：

- `clip_1, clip_2, ..., clip_K`

第一版只做：

- fixed-length causal clips

先不要做：

- scene segmentation
- dynamic event segmentation

目的：

- 给 teacher rollout 一个明确、稳定的 step 单位

---

### Step 4. Build a minimal utility estimator

第一版只做最小 utility teacher，不做复杂 graph。

输入：

- 视频 clips
- 训练集里的 query / answer
- evidence span（如果有）

输出：

- 每个 clip 的 `utility_score`
- 粗 evidence type

最小 evidence type 建议：

- `context`
- `detail`
- `event_fact`

最小规则：

- 如果有 evidence span：
  - clip 和 evidence overlap 越多，utility 越高
- 如果没有：
  - 用简单 temporal grounding 或 teacher VLM 找 supporting clips

目的：

- 先回答“值不值得记”
- 再回答“更像哪种 memory form”

---

### Step 5. Build a minimal teacher policy

第一版 teacher policy 先保持规则化，不要过复杂。

建议规则：

- `low utility -> skip`
- `medium utility -> write_summary`
- `high utility + detail -> write_visual`
- `high utility + event_fact -> write_structured`

如果保留 effort：

- `skip / write_summary -> cheap`
- `write_visual / write_structured -> expensive`

注意：

- `teacher_effort` 只是 teacher-side factorization
- 最终模型的主监督还是 `teacher_action`

---

### Step 6. Build minimal memory-call synthesis

动作确定后，再生成 `write_content` 和最终 assistant target。

建议第一版直接让 assistant target 使用可解析的 `<memory_call>` 格式，但 content 内部可以先保持简单。

- `skip -> action=skip, content=""`
- `write_summary -> 一句低成本摘要`
- `write_visual -> visual slot text-ref`
- `write_structured -> 一句结构化事件/事实描述`

例如：

```json
{
  "effort": "expensive",
  "action": "write_visual",
  "memory_level": "visual",
  "content": {
    "time_span": [12.0, 16.0],
    "text_anchor": "red cup appears on table",
    "visual_target": "keyframe"
  },
  "cost": 3
}
```

目标：

- 先让 SFT 有清晰 target
- 后续 online rollout / RL 可以直接 parse `memory_call`

---

### Step 6.5. Define a concrete v0 rule-based teacher

为了避免第一版 teacher 过于抽象，建议把 `v0 rule-based teacher` 明确定义成一个规则表。

#### v0 teacher input

每一步至少输入：

- `clip_start, clip_end`
- `clip_caption`
- `keyframe_path` 或 `keyframe_desc`
- `novelty_score`
- `utility_score`
- `evidence_type in {context, detail, event_fact}`
- `memory_state`
- `budget_state`

#### v0 teacher effort rule

建议先用一个简单阈值版：

- `utility < tau_low -> cheap`
- `utility > tau_high -> expensive`
- `tau_low <= utility <= tau_high`
  - 如果 `novelty_score` 低 -> cheap
  - 如果 `novelty_score` 高 -> expensive

注意：

- `teacher_effort` 是中间 supervision，不是最终必须保留给 policy 的显式动作
- 这一步主要是为了 SFT 稳定，而不是为了定义新 action space

#### v0 teacher action rule

建议第一版规则写死成下面这样：

| 条件 | teacher_action |
|---|---|
| `utility` 很低 且 `novelty` 很低 | `skip` |
| `effort = cheap` | `write_summary` |
| `effort = expensive` 且 `evidence_type = detail` | `write_visual` |
| `effort = expensive` 且 `evidence_type = event_fact` | `write_structured` |
| `effort = expensive` 且 `evidence_type = context` | `write_summary` |

#### v0 teacher content rule

第一版建议全部输出成可读文本，避免一开始就设计复杂 slot schema。

- `skip`
  - `write_content = ""`

- `write_summary`
  - 模板：
    ```text
    [summary] <one-sentence clip summary>
    ```

- `write_visual`
  - 模板：
    ```text
    <memory_call>
    {
      "effort": "expensive",
      "action": "write_visual",
      "memory_level": "visual",
      "content": {
        "time_span": [start, end],
        "text_anchor": "<fine-grained visual fact>",
        "visual_target": "keyframe"
      },
      "cost": 3
    }
    </memory_call>
    ```

- `write_structured`
  - 模板：
    ```text
    <memory_call>
    {
      "effort": "expensive",
      "action": "write_structured",
      "memory_level": "structured",
      "content": {
        "event": "<event description>",
        "state_change": "<optional>"
      },
      "cost": 2
    }
    </memory_call>
    ```

#### v0 teacher filter rule

为了减少明显噪声，第一版先加三个简单过滤：

- 如果 `teacher_action = skip`，则 `write_content` 必须为空
- 如果 `teacher_action = write_visual`，则 `memory_call.content.text_anchor` 里必须包含具体视觉实体或属性词
- 如果 `teacher_action = write_structured`，则 `write_content` 里必须包含事件、状态变化或关系变化

这一步不需要复杂 judge，先用规则保证标签可读、可训练。

---

### Step 6.6. Define a concrete v1 strong-model teacher

当 `v0 teacher` 跑通后，再升级成 AutoTool 风格的 `v1 strong-model teacher`。

#### v1 teacher roles

建议拆成三个角色：

1. `expert model`
   - 负责生成 action 和 write content
2. `judge model`
   - 负责过滤明显错误或不一致结果
3. `trajectory reviewer`
   - 负责 video-level 一致性检查

#### v1 teacher input package

给 expert model 的输入建议统一成：

```json
{
  "video_id": "...",
  "step": 3,
  "clip_window": {
    "start": 12.0,
    "end": 16.0
  },
  "clip_caption": "...",
  "keyframe_description": "...",
  "memory_state": {
    "summary_history": "...",
    "structured_history": []
  },
  "budget_state": {
    "storage_left": 72
  },
  "query_support_summary": {
    "utility_score": 0.81,
    "evidence_type": "detail",
    "supporting_queries": [
      "What color is the mug?",
      "What object is on the table?"
    ]
  }
}
```

注意：

- `supporting_queries` 只在 teacher 阶段可见
- 它们不能直接出现在最终 online policy 输入里

#### v1 expert prompt template

建议 expert model 的目标是直接输出结构化字段：

```text
You are an offline teacher for write-time memory policy supervision.

Given:
- the current video clip description
- the current memory state
- the remaining storage budget
- the future-query support summary of this clip

Decide:
1. whether this clip deserves cheap or expensive memory effort
2. which memory action to take: skip / write_summary / write_visual / write_structured
3. what content should be written

Rules:
- future queries are available only to you as the offline teacher
- the online policy will not see any future query
- prefer cheaper actions when the clip has low future utility
- use write_visual when the clip's value comes from fine-grained visual details
- use write_structured when the clip's value comes from events, state changes, or stable facts

Output JSON only:
{
  "teacher_effort": "...",
  "teacher_action": "...",
  "write_content": "..."
}
```

#### v1 judge prompt template

judge 只做验收，不重新生成。

```text
You are a judge for teacher-generated write-time memory labels.

Check whether:
1. the chosen action matches the clip's future utility
2. the action matches the declared evidence type
3. the write content is faithful to the clip description
4. the content is not redundant with the current memory state

Return JSON:
{
  "valid": true/false,
  "error_type": "...",
  "revision_hint": "..."
}
```

#### v1 trajectory reviewer

trajectory reviewer 在 video-level 再检查：

- 前后是否重复写同一内容
- 是否同一事实被先 summary 又反复 structured
- 是否 budget 明显不合理地被过早耗尽
- 是否关键 clip 被持续误标成 skip

第一版 reviewer 可以先做规则版，不一定非要再调用大模型。

---

### Step 6.7. Define the minimal state update rule

rollout 时必须明确 `memory_state` 怎么更新，不然 trajectory 很难稳定。

建议第一版先用最小更新规则：

- 如果 `skip`
  - memory 不更新
  - budget 不变

- 如果 `write_summary`
  - 把 `write_content` 追加到 `summary_history`
  - `storage_left -= c_summary`

- 如果 `write_visual`
  - 把 `write_content` 追加到 `summary_history`
  - 另外记录一个 `visual_refs` 列表
  - `storage_left -= c_visual`

- 如果 `write_structured`
  - 把 `write_content` 追加到 `structured_history`
  - `storage_left -= c_structured`

建议成本先写死，例如：

```text
c_skip = 0
c_summary = 1
c_visual = 3
c_structured = 2
```

先不要一开始就设计复杂 learned cost。

---

### Step 6.8. Define concrete files and functions

为了让后续实现更顺，第一版可以直接按下面拆函数：

#### `segment.py`

- `segment_video(video_path, clip_seconds) -> list[Clip]`

#### `utility_estimator.py`

- `find_supporting_clips(sample) -> dict[query_id, list[clip_id]]`
- `aggregate_clip_utility(sample, clip_ids) -> dict[clip_id, UtilityInfo]`

#### `teacher_policy.py`

- `assign_teacher_effort(utility_info, novelty_score) -> str`
- `assign_teacher_action(utility_info, effort) -> str`
- `synthesize_write_content(action, clip_info, utility_info) -> str`

#### `rollout.py`

- `init_memory_state() -> dict`
- `init_budget_state(total_budget) -> dict`
- `apply_action(memory_state, budget_state, action, write_content) -> tuple[memory_state, budget_state]`
- `rollout_video(sample, teacher) -> list[dict]`

#### `export.py`

- `export_step_jsonl(records, path)`
- `export_video_json(records, path)`

---

### Step 7. Roll out teacher over time

这是 Phase 1 最关键的一步。

这不是独立 clip 分类，而是 state-dependent rollout：

1. 读当前 clip
2. 构造当前 state
3. teacher 给出 `effort / action / write_content`
4. 更新 memory state
5. 扣 budget
6. 进入下一步

最终得到：

```text
(s1, e1, a1, w1)
(s2, e2, a2, w2)
...
```

这里的 state 至少包含：

- `current clip`
- `running memory`
- `remaining budget`

---

### Step 8. Export step-level JSONL

建议第一版先导出 step-level JSONL，而不是 video-level nested trajectory。

理由：

- 更容易 debug
- 更容易接进当前训练框架
- 更容易做样本级检查

---

### Step 9. Add a new trajectory dataset reader

不要硬改当前的 `streamingDataset` 去兼容新范式。

新增：

- `streaming_vlm/data/write_traj_dataset.py`

职责：

- 读取新 jsonl
- 把 `state` 拼成 model input
- 把 `teacher_action + write_content` 拼成 target output

这一层做的是：

`(state, action, write_content) -> tokenized SFT sample`

---

### Step 10. Run a small SFT sanity check

先不追求效果，只验证：

- 数据能读
- loss 正常下降
- 模型能输出像样的 action + write content

如果这一步能通，说明第一阶段已经从“想法”变成“代码上有训练对象”。

---

## Phase 2: Improve teacher quality

等 Phase 1 跑通后，再做更强 teacher。

### Step 11. Add query-marginalized utility

从“单 query 支撑关系”升级到：

- 多 query 聚合
- query 权重
- delayed evidence 加权

目标：

- 得到更稳定的 clip-level future utility

### Step 12. Add graph as teacher-side auxiliary structure

graph 只作为：

- 离线 label refinement 工具

不要把它写成：

- online policy input
- inference-time explicit memory component

graph 的主要用途：

- 帮助区分 `summary` vs `structured`
- 提高 `write_structured` 正样本质量

### Step 13. Compare effort-first teacher vs flat teacher

做一个小对比：

- `flat 4-way teacher`
- `effort-first teacher`

验证 `teacher_effort` 是否真的帮助稳定 warm-up。

---

## Phase 3: RL refinement

等有一个能工作的 SFT writer，再上 RL。

### Step 14. Define online rollout environment

状态：

- current clip
- current memory
- current budget

动作：

- `skip`
- `write_summary`
- `write_visual`
- `write_structured`

### Step 15. Warm-start RL from SFT checkpoint

先用最干净的 reward：

```math
R = R_{qa} - \lambda_s C_{store} - \lambda_r C_{ret}
```

先不要一开始就加太多 auxiliary reward。

### Step 16. Compare against heuristic / SFT-only baselines

至少比较：

- heuristic writer
- SFT-only writer
- SFT + RL writer

目标：

- 证明 RL 学到的是更好的 budget tradeoff
- 而不是简单地“多存一点”

---

## Recommended File Layout

建议目录先这样拆：

```text
VST-SFT/
├── train.py
├── streaming_vlm/
│   ├── data/
│   │   ├── lmm_dataset.py
│   │   └── write_traj_dataset.py
│   ├── sft_builder/
│   │   ├── segment.py
│   │   ├── utility_estimator.py
│   │   ├── teacher_policy.py
│   │   ├── rollout.py
│   │   └── export.py
```

后面如果需要更复杂 teacher，再加：

- `graph_builder.py`
- `evidence_mining.py`

---

## What to reuse from current VST-SFT

当前代码可复用的部分：

- `train.py` 训练入口
- 模型加载和冻结视觉塔逻辑
- processor 使用方式
- 视频读取和部分预处理逻辑

当前代码不满足、需要新增的部分：

- teacher data construction
- query-marginalized utility estimation
- teacher rollout
- `state -> action -> write_content` dataset schema

---

## Immediate TODO

如果只压成最短 TODO，当前最该先做三件事：

1. 定义 `write trajectory jsonl` schema
2. 写最小 `teacher rollout builder`
3. 写新的 `write_traj_dataset.py`

这三步做完，方法才真正从“想法”变成“有训练对象”。

---

## One-line Summary

先做一个最小可训练的 teacher-rolled SFT trajectory pipeline，把 `write-time memory decision` 真正落成数据；等这条链路跑通，再上更强 teacher 和 RL。

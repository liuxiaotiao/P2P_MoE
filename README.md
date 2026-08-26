# p2pmoe —— P2P MoE 双段模型放置 serving 框架

对应技术方案 v24.0《P2P MoE 双段模型放置优化 — 分散环境完整方案》。

交付内容：**离线规划器**（II.1–II.7）+ **分散网络模拟器** + **在线运行时**
（II.5 的完整协议 + toy MoE 执行层）+ **多机部署**（节点 agent / 真实探测 / 控制器）。

```bash
pip3 install -r requirements.txt          # 只有 numpy —— 控制机与 toy 模型路径
pip3 install -r requirements-node.txt     # + torch/safetensors（跑真实模型的节点）
pip3 install -r requirements-dev.txt      # + pytest（只有开发机）

python examples/model_fit.py --nodes 15 --mem-gb 24   # 选模型：算哪个 MoE 跑得起来
python examples/appendix_c.py                    # 附录 C 的 24 节点算例（只算内存）
python examples/deployment.py --json plan.json   # 带专家身份：逐节点加载清单 + 组合矩阵
python examples/e2e.py                           # 单进程 fork + 人造延迟：验证协议时序
python examples/serving.py                       # 服务循环：并发请求、排队、自动配对、回池
python examples/multinode_local.py               # 真实子进程 + 真实 socket：验证部署路径
python -m pytest -q tests/                       # 343 项测试
```

---

## 为什么规划器不依赖任何推理框架

规划器的输入只有三张表：**节点内存、逐对延迟实测、逐层内存形状**。
它不 import torch，也不知道 MoE 是什么。这带来两个好处：

* 框架选型（HF / vLLM / 原生 torch）不阻塞这一层的开发与验证；
* 真实部署时，把 `SimNetwork` 换成打真实探测包的实现即可，规划器代码一行不动
  —— `NetworkOracle` 只有一个方法 `probe(a, b, k) -> Probe(p50, p95)`。

---

## 代码地图

| 模块 | 对应文档 | 内容 |
|---|---|---|
| `planner/types.py` | 第〇部分、I.2 | Node / ModelSpec / TaskProfile / SegmentSpec / Segment / Objective / PlannerConfig |
| `planner/memory.py` | I.2.2、III.5.1–5.4 | KV 与权重内存公式、跳数整数下界、可行性必要条件、L₀ 选取 |
| `planner/network.py` | 第〇部分、I.2.3、II.1 | 分位数探测、缓存、尾闸/抖动闸、端点接入画像 |
| `planner/capacity.py` | II.7、III.6.1、III.8.5 | 分档估上限（含段形态枚举与精确位配平）、最大余额法配额、公平比 |
| `planner/solver.py` | II.1 | `deploy_path` beam 求解器（跳数优先、单节点成段优先、稳定性一等项） |
| `planner/tighten.py` | II.2、III.4 | 间隙检测去慢尾、`tighten_lex` 字典序收紧（两头都拆、整轮回退） |
| `planner/common_band.py` | II.3、III.8.1 | 公共中值域滑窗（精确最优）、扫 W 取拐点、异类入口诊断 |
| `planner/loop_trim.py` | II.4 Step 6、III.8.3 | 回环画像、升序裁剪、备胎池 |
| `planner/hf_config.py` | — | HF config.json → ModelSpec；细粒度判据（模型适不适合本方案） |
| `planner/experts.py` | I.1.1、II.5、III.7.3–7.4 | 逐层驻留专家集（身份）、前段并集、可检性矩阵 q(u,û)、池合并信号 |
| `planner/static_pairing.py` | — | **静态简化模式**：前后段配对离线定死（贪心最小化组合延迟） |
| `planner/manifest.py` | II.4、III.8.2 | 逐节点逐层加载清单、前后段组合矩阵、七项一致性校验 |
| `planner/pipeline.py` | II.4 | 六步流水线编排、两条反馈口、终审与账本 |
| `sim/network.py` | II.3.1 | 按「出口接入 + 骨干 + 入口接入」建模的可复现分散网络 |
| `sim/replay.py` | I.1.1 | 回放语料的合成：逐 task 逐层专家激活质量画像 |
| `sim/scenario.py` | 附录 C | 24 节点池、三 task 画像、p(L₀) 曲线 |
| `runtime/torch_model.py` | I.1.1、II.5 | **Qwen3-MoE 执行层**：GQA + RoPE + QK-norm、只驻留子集专家、drop-expert。需 torch |
| `runtime/qwen3_next.py` | I.1.1、II.5 | **Qwen3-Next 执行层**：混合注意力（Gated DeltaNet + 标准）、共享专家、零中心 norm、部分旋转 RoPE |
| `runtime/weights.py` | I.1.1 | safetensors 选择性加载：按 (层, 专家 id) 过滤 key，只打开相关分片 |
| `runtime/model.py` | I.1.1、II.5 | toy MoE：`PartialExpertMoEBlock` 只驻留子集专家、KV cache、miss 统计、drop-expert 重归一 |
| `runtime/corpus.py` | I.1.1、II.5 | 回放语料、用全专家模型统计激活画像、**实测**告警基线与分类器参考 |
| `runtime/profile.py` | I.1.1 | **激活画像**：逐层路由质量统计 → 驻留专家集（采/存/用） |
| `runtime/text.py` | — | **文本进出（控制机侧）**：tokenizer、增量解码、chat template、停止条件 |
| `runtime/identify.py` | II.5 | 直方图分类器、置信三区 |
| `runtime/wire.py` | II.6、第〇部分 | 消息编解码、保温长连接、延迟抖动注入 |
| `runtime/node.py` | II.5 | 节点 agent 进程：加载自己那份、段内转发、跨接口出段 |
| `runtime/coordinator.py` | II.5 | 盲绑 pop、派发、通道二检出、换绑、进程集群启动器 |
| `deploy/agent.py` | — | 节点 agent 守护进程（两阶段启动） |
| `deploy/probe.py` | II.3.1 | 真实网络探测：由节点自己发起的逐对测量 |
| `deploy/control.py` | II.4、II.5 | 控制器：发现 → 探测 → 规划 → 下发 → 服务 |
| `deploy/relay.py` | — | **中继**：节点之间没有直连时的接线员（握手后纯搬字节） |
| `deploy/fetch.py` | — | **只拉本机要的那部分权重**：safetensors 逐张量 HTTP Range 下载 |
| `deploy/manual.py` | — | **手动放置**：布局文件 → 部署清单 + 连线表，只校验不优化 |
| `deploy/run.py` | — | **按给定布局跑**：连上 → 预检 → 下发 → 服务，不探测不规划 |
| `deploy/launch.py` | — | 批量拉起/查看/停止 agent（ssh），systemd unit 模板 |

测试只覆盖**有严格证明**的命题（§III.9 的「严格证明」清单）。启发式部分不写断言
—— 给启发式写断言等于把调参结果冻进测试。

执行层是例外，它有一条更硬的判据：`tests/test_reference_parity.py` 拿
**transformers 官方的 Qwen3-MoE** 当参考，同一份权重同一个输入逐元素比对
（prefill、decode、逐层中间结果、分段跑 vs 整段跑），最大误差 ~1e-6。
自己手写 attention/RoPE/QK-norm/路由的代价就是这个 —— 错了不会抛异常，
只会让输出「看起来像模型很笨」，而合成权重的输出本来就是乱码，肉眼分辨不了。

---

## 最短路径：布局你写，我只管连上跑

不想让规划器决定怎么分？写个布局文件，直接跑：

```json
{
  "model_dir": "/data/qwen3-30b-a3b",
  "l0": 6,
  "channels": [
    {"front": "n1",         "back": ["n2", "n3", "n4"]},
    {"front": ["n5", "n6"], "back": ["n7", "n8", "n9"]}
  ]
}
```

```bash
# ① offline：布局 → 清单（不连机器、不要权重）
python3 -m p2pmoe.deploy.run --spec deploy.json --plan-only --save-plan plan.json

# ② 各节点只拉自己那部分权重
python3 -m p2pmoe.deploy.launch fetch --hosts hosts.txt --plan plan.json \
        --repo Qwen/Qwen3-30B-A3B --out /data/qwen3-part

# ③ 建链 + 推理
python3 -m p2pmoe.deploy.run --spec deploy.json \
    --agents "$(python3 -m p2pmoe.deploy.launch agents --hosts hosts.txt)" \
    --advertise 10.0.0.1 --model-dir /data/qwen3-part \
    --chat --prompt "用一句话解释 MoE 的稀疏激活"
```

```
[1/4] 布局：2 条通道，用 9 台机器，L₀=6（前段 1..6，后段 7..48）
      ch0 [general]  前段 n1:1-6 → 后段 n2:7-20 n3:21-34 n4:35-48
      专家：全装（无 drop-expert 近似，输出与单机逐位一致）
[2/4] 预检   内存：最紧的是 n2（要 15.2 / 有 15.0 GB）   checkpoint：9 台都读得到
[3/4] 下发   n2  back/Bgeneral0  层 7–20  从 checkpoint 读了 29.2%（5/16 个分片）
[4/4] req0  F0×Bgeneral0  停于 eos  37 token  首token 812ms  «MoE 每个 token 只激活…»
```

层怎么切按机器数均分；要精确控制就把层区间写出来：

```json
{"front": [{"node": "n1", "layers": [1, 4]}, {"node": "n2", "layers": [5, 6]}],
 "back":  [{"node": "n3", "layers": [7, 30]}, {"node": "n4", "layers": [31, 48]}]}
```

本地先演练一遍（起 N 个 agent 进程，不需要真机）：

```bash
python examples/manual_deploy.py --channels 2 --front 1 --back 2
python examples/manual_deploy.py --spec deploy.json --real --chat --prompt "你好"
```

### 支持哪些模型

| | 层结构 | 专家 | 细粒度比 | 状态 |
|---|---|---|---|---|
| **Qwen3-MoE**（30B-A3B 等） | 48 层全标准注意力 | 128,top-8 | 16 | ✓ 与官方逐元素一致 |
| **Qwen3-Next**（80B-A3B） | **36 层 DeltaNet + 12 层标准** | **512,top-10** + 共享专家 | **51** | ✓ 与官方逐元素一致 |

`--model-dir` 指过去就行 —— 节点按 checkpoint 自报的 `model_type` 分派，不用额外开关。

**Qwen3-Next 更适合本方案**：细粒度比 51 意味着单 token 只碰 2% 的专家（Qwen3-30B 是
6.2%），「后段只驻留子集」的收益大得多。而且 3/4 的层是 Gated DeltaNet，
它的状态**定长**、不随上下文增长 —— KV 内存远小于同规模的标准 MoE。

它也多了三处能悄悄算错的地方，都由逐元素比对钉住（写的时候三处都踩了）：
**零中心 RMSNorm**（乘 `1+w` 而非 `w`，权重初始化为 0）、**部分旋转 RoPE**
（head_dim 256 里只有 64 维带位置信息）、**同一模型两种 norm 约定**
（`RMSNorm` 用 `1+w`、`RMSNormGated` 用 `w`）。三者都不会报错，只会让输出
「看起来像模型很笨」。

**共享专家不参与裁剪** —— 它对每个 token 都激活，是那一层的固定成分而非某个
task 的驻留集，承载该层的节点必须装它（好在只有 6.3MB，与单个路由专家同量级）。

### 权重也只下需要的那部分

选择性加载已经知道自己要哪些 key 了。把同一份 key 集合往上游推一层，就变成
「只下这些字节」—— safetensors 的文件头里有每个张量的 `data_offsets`，
HF 的 CDN 支持 HTTP Range：

```bash
# 每台节点上（--node 是它自己的名字，--out 各机同一个路径）
python3 -m p2pmoe.deploy.fetch --repo Qwen/Qwen3-30B-A3B \
        --plan plan.json --node n3 --out /data/qwen3-part

# 或者让控制机把 15 台一起推起来（各节点直连上游，并行下）
python3 -m p2pmoe.deploy.launch fetch --hosts hosts.txt --plan plan.json \
        --repo Qwen/Qwen3-30B-A3B --out /data/qwen3-part
```

15 台跑 Qwen3-30B-A3B（3 条通道 × 5 台），后段全装专家时：

| | 下载量 |
|---|---|
| 每台都下全量 | 916 GB |
| **只下本机需要的** | **179 GB**（省 80%） |
| 后段再按画像装 20% 专家 | 约 45 GB |

产出的是一个**更小但完全合法**的 checkpoint 目录，`--model-dir` 指过去即可，
节点侧代码一行不用改。张量与上游**逐位一致**（测试里逐个比对过 —— 搬错一个偏移
不会抛异常，只会让权重变成噪声，而噪声权重照样出 token，肉眼看不出来）。

`--mode shard` 是兜底：只下含目标 key 的整个分片。省得少，但不依赖 Range。
上游无视 Range 时脚本会**明确报出来**而不是将就 —— 将就的话「省下载」根本没发生，
拼出来的文件还是错的。

### 后段只装 hot expert 损失多大

drop-expert 被文档标注为「运维近似，非无损」—— 但从来没量过有多不无损。
现在可以量：

```bash
python examples/drop_expert_impact.py --model-dir /data/qwen3-30b-a3b \
    --profile prof.json --coverage 0.9 0.95 0.99
```

```
  覆盖率   后段每层    miss率    丢门控        KL   top1一致   生成分叉于
   0.90    5.4/8    16.7%    0.066   0.0286      75%      第 1 步
   0.95    6.2/8     9.4%    0.036   0.0144      81%      第 1 步
   0.99    7.4/8     0.0%    0.000   0.0000     100%        未分叉
```

三个指标量的是不同的事：**miss 率**是通道二的观测量，但丢掉的可能是门控权重
0.02 的那个；**丢门控质量**更诚实；**KL** 才是输出分布的实际距离。

### 缺专家时怎么补救：三种策略

`--miss-policy` 三选一，缺失发生时才有区别：

| | 做法 | 缺失时用几个专家 |
|---|---|---|
| `drop`（默认，文档 II.5） | 跳过缺失的，剩下的**重归一** | < k，最坏 0 个（只走残差） |
| `drop_noscale` | 跳过缺失的，**不重归一** | 同上，但贡献按丢掉的质量成比例缩小 |
| `local_topk` | 把路由概率**限制到驻留集**再取 top-k | 恒为 k，不会退化成零 |

**没有缺失时三者完全等价** —— 全量 top-k 都在驻留集里，限制后再取选出的是同一批。
所以「驻留集覆盖实际路由时逐位一致」这条性质在三种策略下都成立（有测试钉住）。

实测（合成权重，`--policy` 横向对比）：

```
覆盖率   后段每层   miss率   KL·drop  KL·drop_noscale  KL·local_topk   最好的
 0.70    3.2/8   42.7%    0.1007           0.0686         0.1160  drop_noscale
 0.85    5.0/8   22.9%    0.0317           0.0184         0.0342  drop_noscale
 0.95    6.2/8    6.2%    0.0144           0.0080         0.0143  drop_noscale
```

**`drop_noscale` 全程最好**，KL 低 1.5–2 倍。直觉是：重归一等于宣称「路由本来
就只想要剩下这几个」，而那不是事实 —— 它把一个低概率专家的输出放大到权重 1。
不重归一则让这一层的 FFN 贡献按丢掉的门控质量成比例缩小，缺得越多越接近
「只走残差」，退化是平滑的。

**但这个结论很可能被随机权重误导**：合成模型里各专家的输出是互不相关的随机方向，
「用错专家」等于加一个随机向量，而「什么都不加」离均值更近。真模型的专家彼此
相关（都在做合理的事），替补一个次优专家大概率比什么都不做强 —— 也就是说
`local_topk` 在真权重上会比这里好看。**默认保持 `drop` 是为了忠于文档；
真上线前拿这个脚本在真权重上测一遍再定。**

**「生成分叉于」这一列最该看。** prefill 偏差小推不出生成没问题 —— decode 的
误差会沿步累积，而且被扰动的 hidden state 会选出不同的专家，越走越偏。
不过它也最脆：top-1 与 top-2 间距小时再小的扰动都会翻。所以**看 KL 判断质量，
看分叉判断可复现性**。

（上表出自合成权重，路由接近均匀、hot expert 概念退化，只用来验工具。
真模型上分布集中得多，同样覆盖率只要 10–20% 的专家。）

### 后段只装该装的专家

默认全装（无近似）。要让后段真的只驻留 n_{u,l} 个专家，得先知道**哪些** ——
这个知识只能来自真实数据上的真实路由。跑两轮就有了：

```bash
# 第一轮：全装跑一遍，顺便把路由统计下来
python3 -m p2pmoe.deploy.run --spec deploy.json --agents ... \
        --prompt "$(cat 这个task的真实请求.txt)" --profile-out prof.json

# 第二轮：按画像只装子集
python3 -m p2pmoe.deploy.run --spec deploy.json --agents ... \
        --profile prof.json --coverage 0.95
```

```
第一轮  n2  back/Bgeneral0  层 7–20   1792 个专家  从 checkpoint 读了 29.2%
        画像已存 prof.json
          @ 覆盖率 95%  general: 42 层，每层 14–31 个（层均 21.4/128，约 17%）

第二轮  专家（画像 @ 覆盖率 95%）：general: 42 层，每层 14–31 个（层均 21.4/128）
        n2  back/Bgeneral0  层 7–20    300 个专家  从 checkpoint 读了 6.1%
```

**为什么第一轮必须全装**：只驻留子集时输出被 drop-expert 近似带偏，后面几层的
路由就不是真实路由了 —— 采到的画像会把自己的错误固化下来。两个参数一起给会直接报错。

**为什么采得到**：路由是全量的，即使专家不是。`mlp.gate.weight` 每层都完整加载
（它比一个专家还小），所以 `MoEStats.hist` 记的是 top-k 在**全部** E 个专家上的
质量分布 —— 哪怕本地只驻留了 3 个。否则画像只会确认已有的选择，
「换一批专家会不会更好」这个问题就问不出来。

**为什么存质量而不是 id**：覆盖率（0.9？0.95？0.99）是个内存换质量的部署期权衡。
存分布的话换阈值不用重新采样；存 id 就把这个选择冻死在采样那一刻了。

逐 task 分开：多条通道写不同的 `"task"`，把各自的请求打进去，画像就是逐 task 的
S_{u,l}。前段不参与 —— 它是 task 无关的（I.1.1），装全集。

**它不做什么，说清楚**：不探测、不估容量、不选 L₀、不建段、不算公共中值域、
不做回环裁剪。所以方案文档的那些结论（零后悔、任意组合、延迟均匀）在这条路上
**不成立** —— 它们的前提是「离线把组合极差压到抖动量级以下」，而这里没人去压。
这条路只保证一件事：**按你说的装上、连上、能出 token。**

它仍然会拦住的事（都在下发之前，一次报完）：层中间漏了没人算、段内层区间不连续、
最后几层没人管、一台机器被两条段用（I.2.2 排他，否则两条请求会互相污染 KV）、
各通道 L₀ 不一致、专家 id 越界、内存明显不够、节点读不到 checkpoint。
这些错误的症状都很隐蔽 —— 层没接上不会报错，只会让输出静默变成垃圾。

想要规划器那一套，走 [`deploy/control.py`](DEPLOY.md)。两条路产出的是同一种
`DeploymentManifest`，节点侧完全一样。

---

## 两个能力，两个入口

**「每个节点部署特定 layer、每个 layer 指定的专家」** → `NodePlan.layers`

```
节点    角色        段      层区间   专家数   权重GB  KV GB    合计  端点
g3    back:X     BX0     6–11      48   13.74   0.40   14.14  head
g1    back:X     BX0    12–32     158   45.39   1.41   46.80  tail
v3    front      F0      1–5       97   26.84   0.34   27.18  head/tail
...
layer  1: 18 个专家 [0, 4, 6, 10, 12, 16, 19, 22, 24, 27, 32, 35, 45, 49, 53, 54, 57, 61]
```

给出的是专家 **id 列表**而不是个数 —— 配合 safetensors 的逐张量 mmap，可以直接
翻译成「只打开这些 key」。专家身份同时解锁了三件只有基数做不了的事：前段并集的
逐层真实规模、命题 III.7.3 的可检性 q(u,û)、推论 III.7.4 的池合并信号。

**「前后 cluster 可以任意组合」** → `DeploymentManifest.pairings`

```
组合矩阵 (前段 × 后段 → 单 token p50 ms)
            BX0      BX1      BY0      BZ0
  F0       99.4    102.9    104.0     99.5
  F1      101.3    102.5    108.3    101.2
  ...      5 × 4 = 20 组全部可用，p50 99.4–112.0ms
```

这不是事后罗列而是被逐条校验过的：任何一对的正向接口若落在公共带
`[w_lo, w_hi]` 之外，或任一接口抖动超 J_cap，校验就不通过。**这正是整套方案的
价值所在** —— 公共中值域（II.3）保证每个 (f,b) 的正向接口都在同一条窄带里，
回环裁剪（Step 6）保证每条前段对全体后段的回环都小，于是在线才敢盲绑：弹一条
前段，事后无论识别成哪个 task，都能配上任意一条该池的后段而不后悔
（定理 III.3.1、推论 III.3.2）。

七项校验：排他 · 内存含 KV · 层区间连续完整 · 并集支配 · 后段驻留集不多不少 ·
组合矩阵完整 · 逐对过闸。测试里对每一项都注入了违规样本，确认校验器抓得住。

---

## 日常怎么用：服务循环

离线规划的产物是一个**可用池** —— 若干条前段、若干条后段（按 task 分池）。
之后就只有一个循环，没有别的：

```
请求到达 → 从前段池 pop 一条（盲绑，不比不挑）
        → 前段跑完，tail(f) 本地识别出 task
        → 从该 task 的后段池 pop 一条 → 两段自动建链
        → decode 绕环直到完成
        → 两段各自回池 → 等下一条请求
```

**池子空了不是错误，是排队**（II.5「池满 → 有界等待」）。前面哪条一完成，
队首立刻被接走，不用等整批。

```bash
python examples/serving.py --requests 12 --concurrency 8
```

```
可用池: 前段 4 条；后段 X:2  Y:1  Z:1
  → 同时最多服务 4 条请求；第 5 条起排队

打入 r00–r07（8 条）→ 空闲前段 0，排队 4

请求   task  前段   后段    前段排队   后段排队   首token
r00   X    F0    BX1        0ms      0ms     58ms
r03   X    F3    BX0        0ms      0ms     68ms
r04   Y    F0    BY0      148ms      5ms    179ms   ← F0 跑完 r00 后接走队首
r07   Y    F2    BY0      169ms     89ms    280ms   ← Y 池只有 1 条，还等了后段

出现过的组合: 10 种
  F0  配过 ['BX0', 'BX1', 'BY0']
  F3  配过 ['BX0', 'BX1', 'BZ0']
→ 4/4 条前段在不同请求里配了不同的后段

最终池深: free_fronts 4, free_backs {X:2, Y:1, Z:1}   ← 全部归位
```

真机上同一件事：`control --requests 12 --concurrency 8`，跑完会打出配对历史。

**并发度 = 前段条数。** 15 台节点建出 5 条前段就只能同时服务 5 条 —— 这是
排他独占（I.2.4，放弃 batching 换确定性延迟）的直接后果，不是实现限制。
批处理已列入 [TODO.md](TODO.md)，那里记了它要动哪些文件、会破坏哪几条定理。

---

## 文本进出

请求进来是一句话，出去是一句话；节点那边自始至终只见 token id。

```python
from p2pmoe.runtime.text import TextIO

textio = TextIO.from_model_dir("/path/to/Qwen3-30B-A3B", chat=True)
coord  = Coordinator(manifest, ..., textio=textio,
                     on_text=lambda rec, delta: print(delta, end=""))
rec = coord.submit("r0", text="用一句话解释 MoE 的稀疏激活")
rec.done.wait()
print(rec.text, rec.stop_reason)     # → "..." "eos"
```

命令行：

```bash
python examples/static_qwen.py --chat --prompt "你好" --prompt "hello"
python -m p2pmoe.deploy.control --agents ... --model-dir /path/to/model --chat        --prompt "用一句话解释 MoE"
```

三件容易被做错、这里都做了的事：

**增量解码不能逐 token 拼。** byte-level BPE 的一个 token 可以是半个 UTF-8 字符
（汉字 3 字节，常被切成 2+1）。单独解那半个字节得到的是 `�`，而且**错误不可逆**
—— 后面补上的字节救不回来。正确做法是整段重解取增量，结尾是 `�` 就先不吐字。

**停止 token 要读 `generation_config.json`，不是 `config.json`。** 两个文件里的
`eos_token_id` 可以不同：Qwen3 的 config 是 151643 `<|endoftext|>`，
generation_config 是 151645 `<|im_end|>`。只读前者的话，对话模型永远等不到那个
它其实已经吐出来的结束符。

**指令模型必须套 chat template。** 不套是 completion 语义，输出看起来像模型坏了。
模板就在 `tokenizer_config.json` 里，用 jinja2 渲染 —— transformers 底下做的
也是同一件事。

依赖因此分成三份，互不包含：

| 文件 | 内容 | 谁装 |
|---|---|---|
| `requirements.txt` | numpy | 两边 |
| `requirements-control.txt` | + tokenizers、jinja2 | 控制机 |
| `requirements-node.txt` | + torch、safetensors | 推理节点 |

**控制机不碰张量，节点不碰文本。** 不是洁癖 —— 15 台节点上少一套分词器、
控制机上少几 GB 的 CUDA 依赖，`test_requirements.py` 两条对称的测试守着它。

---

## 简化版：把链路在部署时定死

如果暂时不想要「到达时 task 未知 → 在线识别 → 任意组合」这一整套，可以走
**静态模式**：每条前段的 task 在部署时给定，前后段怎么连也在离线算好、随配置
下发。请求自报 task，在线就只剩「按 task 取一条通道」。

```bash
python examples/static_qwen.py                          # 合成的微型 Qwen3-MoE
python examples/static_qwen.py --preset qwen3-30b-a3b   # 只规划，不要权重
python examples/multinode_local.py --nodes 15 --static  # 15 进程演练
python -m p2pmoe.deploy.control --agents ... --static   # 真机
```

连接默认自动配对（贪心最小化组合延迟）。要自己指定：先 `--save-plan` +
`--save-wiring` 存下来，改完再 `--load-plan` + `--wiring` 喂回去 —— 载清单是
必须的，因为放置依赖不可复现的延迟实测，不固定住的话段 id 每次都可能变。
完整步骤见 [DEPLOY.md](DEPLOY.md) 第 5 步。

|  | 主线 | 静态 |
|---|---|---|
| 前段驻留 | 并集 ∪_u S_{u,l} | **全部专家** |
| task 来源 | 前段 tail 在线识别 | 请求给定 |
| 配对 | 到达时 pop，任意组合 | 离线定死 |
| 在线控制面 | 识别 → 要后段（一个 RTT） | 无 |
| 换绑 | 通道二触发 | 不存在 |
| 公共中值域 | 必需（它保证任意组合成立） | 退化成「选中的那几对齐就行」|

**换来的**：少一个控制面 RTT，少一整套分类器 / 检出 / 换绑，好调试；前段不需要
激活画像就能装，将来加 task 也不用重装前段。

**放弃的**：I.1.1 的「到达时 task 未知」这个前提本身；负载不能在通道之间流动
（某个 task 变热只能重新下发配对）；备胎顶替要改配置而不是自动发生。

**代价看得见** —— `--preset qwen3-30b-a3b` 会把两种口径并排打出来：

```
L₀ = 6（前段 1..6，后段 7..48）
前段驻留 7.5GB（全集），可单节点承载的机器 15/15 台
── 对照：若前段只装并集（87/128 个/层），L₀ 可到 6、前段 5.2GB、通道上限 7

5 条静态通道，组合 p50 75–86ms（极差 11ms）
  ch0  X  F0 × BX0   正向 29.1ms  回环 29.1ms  组合 75.0ms
  ch2  Y  F4 × BY0   正向 33.2ms  回环 35.7ms  组合 85.7ms
```

真正吃内存的那部分 —— 逐节点逐层放置、每层只驻留指定专家、段内流水、绕环
decode、drop-expert、有界等待 —— 一条没少。

---

## 在多台真实机器上运行

> 15 台跑 Qwen3-30B-A3B 的**逐条指令**见 [RUNBOOK.md](RUNBOOK.md) —— 从裸机到第一个
> token，每条都标了在哪台机器上执行。下面是解释与其它选项。

每台节点起一个 agent，控制机跑一次控制器。就两条命令。

**节点之间不需要 SSH** —— 数据面是节点直接互发的 TCP（agent 端口，默认 9101）。
`ssh` 只出现在 `deploy/launch.py` 里，那是批量起停的便利脚本；不配 SSH 就用
`--dry-run` 把命令拿去各机自己跑，或者走 systemd。

**但节点之间要两两可达。** 都在 NAT 后面、连不通的话，用 VPN overlay
（WireGuard/Tailscale，恢复真直连），或者跑一个中继：

```bash
python3 -m p2pmoe.deploy.relay --bind 0.0.0.0:9200        # 有公网入站的那台
python3 -m p2pmoe.deploy.agent --id n3 --relay HOST:9200  # 节点：不监听端口
python3 -m p2pmoe.deploy.run  --spec deploy.json --relay HOST:9200 ...
```

中继是接线员不是代理 —— 握手完只搬字节，协议与节点逻辑一行不用改。
代价是每跳绕一圈、逐 token 延迟大致翻倍，且它是带宽瓶颈与单点。
连通性要求见 [DEPLOY.md](DEPLOY.md#需要哪些连通性先确认这个)。

```bash
# 每台节点（10.0.0.11、10.0.0.12、…）
python -m p2pmoe.deploy.agent --id v1 --bind 0.0.0.0:9101 --mem-mb 47000

# 控制机
python -m p2pmoe.deploy.control \
    --agents v1=10.0.0.11:9101,v2=10.0.0.12:9101,g1=10.0.0.21:9101 \
    --advertise 10.0.0.5 --requests 4
```

**上真机的操作手册在 [DEPLOY.md](DEPLOY.md)** —— 从「先量延迟决定要不要往下走」
开始，到常见问题与这一轮验证不了什么。

### N 台机器怎么起

**agent 不需要同时启动**，也不需要手敲 N 次。它们是常驻守护进程，先后起来都行 ——
控制器只要求「跑规划的那一刻它们都在」。写个 hosts 文件批量拉：

```
# hosts.txt   节点id  地址              [附加参数]
v1            10.0.0.11
v2            10.0.0.12         --mem-mb 32000
g1            10.0.0.21:9102    --mem-mb 47000
```

```bash
python -m p2pmoe.deploy.launch start  --hosts hosts.txt   # ssh 并发拉起
python -m p2pmoe.deploy.launch status --hosts hosts.txt   # 看谁在线
python -m p2pmoe.deploy.launch probe  --hosts hosts.txt   # 逐对延迟矩阵 + 适用性判定
python -m p2pmoe.deploy.launch agents --hosts hosts.txt   # 打印 --agents 串
python -m p2pmoe.deploy.launch stop   --hosts hosts.txt
```

生产环境用 systemd 更稳（开机自启、崩溃重启、日志轮转，`launch` 这些都不管）：

```bash
python -m p2pmoe.deploy.launch systemd --id v1 --workdir /opt/p2pmoe > p2pmoe-agent.service
```

agent 是**无状态**的：崩溃重启后回到未配置态，等控制器重新下发即可 —— 所以
`Restart=always` 是安全的。

**没起来的节点不会拖累其余节点。** 控制器在采集能力那步就把连不上的剔除，
继续用剩下的规划。分散环境下这是常态，不是异常。同理，探测时 A 连不上 B
（NAT / 防火墙 / 单向可达）会被记成不可达，规划自然绕开那条链路。

**规划不会用满所有节点。** 15 台的实测：

| | |
|---|---|
| 池子 | 15 台，其中 3 台劣质接入 |
| 实际用上 | 11 台（5 条前段 + 1 条备胎 + 5 条后段） |
| 未使用 | 4 台 —— **恰好包含全部 3 台劣质接入** |

劣质接入节点被自然排除，不需要人工标记：它们既进不了公共带（做不了前段出口），
也当不了后段入口（端点项会惩罚它们）。未被使用的节点就在那儿闲着 —— 它们是
churn 时的天然扩容池，重跑一次控制器就能把它们纳入新方案。
（`test_deploy.py::test_multinode_15_nodes_end_to_end` 把这条固定下来了）

控制器做五件事，每件都对应文档里的一段：

| 步骤 | 做什么 | 对应 |
|---|---|---|
| 1 | 连每个 agent 问内存/算力 → 拼出规划器的 Node 表 | II.7.1 分档估算的输入 |
| 2 | 下发探测指令，**由节点自己**量逐对 p50/p95 | II.3.1 |
| 3 | `planner.plan()` 全流程 | II.4 六步 |
| 4 | 每个节点只收到属于自己的那份 NodePlan（层 + 专家 id） | II.5「在线零计算」 |
| 5 | 协调器盲绑派发，节点之间自己转发 | II.5 |

**三处只在多机场景下才存在的设计**：

**agent 必须两阶段启动。** 起 agent 时它还不知道自己该装哪几层，因为规划要先量到
全网的逐对延迟，而量延迟又要求 agent 已经在跑。先起服务、后收清单是唯一能解开这个
循环的顺序。未配置态下 agent 只响应三类消息：`capabilities`（报能力）、`echo`
（给别人当靶子）、`probe`（去量指定对端）。

**探测必须由节点自己发起。** 逐对代价里最大的一项是两端的接入段（II.3.1），从控制器
去 ping 两台机器，量到的是控制器自己的接入质量。所以是控制器下发指令、节点执行、
结果回传。这也是真机部署绕不开 agent 的根本原因 —— 没有 agent 就没有「从 A 出发」的
测量点。

**方向不对称观测不到。** 文档把正向接口 ŵ(t,v) 与回环 d_loop(v,t) 当作两个独立的有向
量，但应用层只能量 RTT，取 RTT/2 作单向估计 —— 没有时钟同步就拆不开这两个方向。
非对称带宽的接入（常见于家宽）会让其中一向被低估、另一向被高估。真正逐向的观测只能
来自 II.6 的在线仪表：decode 每 token 实付的 w 与 d_loop。`deploy/probe.py` 里对此有
完整说明，`--asymmetric-probe` 可以逐向分别探测，但拿到的仍是同一个数。

### 先在一台机器上演练

```bash
python examples/multinode_local.py --nodes 8 --requests 3
```

用 `subprocess` 拉起 8 个**独立的 agent 进程**，然后跑真正的控制器。流程与真机一字不差，
只是地址都指向 127.0.0.1：

```
[1/5] 采集节点能力（8 台）
[2/5] 逐对探测（由节点自己发起，k=8）  28 对，用时 1.4s，28 次 RPC
[3/5] L₀=6，配额 {X:1, Y:1, Z:1}，前段 3 条，组合矩阵 9 组，清单校验 通过
[4/5] n2  front/F0   层 [1..6]  110 个专家  驻留 23.54MB（全装要 39.67MB）
      n4  back:X/BX0 层 [7,8]    23 个专家  驻留  5.16MB（全装要 13.22MB）
[5/5] req0 真实 X → 识别 X ✓  首token 101ms  逐token p50 93ms
      汇总：识别 3/3，逐 token p50 100ms
```

有一处需要说明：这个脚本默认给每个 agent 配了一个**模拟的出口接入段延迟**
（`--access-ms`）。不加的话本地 socket 只有几十微秒，「网络主导」这个前提不成立 ——
公共带的宽度 W = η·median T 会小到零点几毫秒，凑不出人口，规划会（正确地）失败。
真机部署不要用这个参数，网络自己会给。加了之后逐对 RTT ≈ access(a) + access(b)，
正是 II.3.1 的加法结构，而且**探测量到的就是它**，不是另开一套账。

`e2e.py` 与 `multinode_local.py` 验证的是不同的东西：前者是单进程 fork + 人造延迟，
验证协议在分散环境的**时序**下成立；后者是真实子进程 + 真实 socket + 真实 RPC，
验证**部署路径**本身。

---

## 在线运行时跑通了什么

`examples/e2e.py` 起真实进程（一个 manifest 节点一个进程），本地 socket 通信，
发送前按规划期实测的 (p50, jitter) 注入延迟。一条请求的完整生命周期：

```
▶ r-inject  真实 task = X  [注入误绑到 Y]  → 识别 X [正确]  换绑 1 次
    t+   0.0ms  到达 → 盲绑 F3（队列头，未做任何比较）
    t+  11.0ms  sml2 本地识别: X (c=1.00, commit 区)
    t+  11.0ms  ⚠ 故障注入：无视识别结果，强行绑到 Y 池
    t+  28.3ms  首 token（后段 prefill 完成）
    t+ 191.2ms  通道二报警: 滑窗 miss 率 60.0% > 基线 13.4% × 3
    t+ 191.2ms  换绑决策: Y → X（前段 KV 保留，只重算后段 prefill）
    t+ 191.4ms  换绑: pop(X 池) → BX1（原 BY0）
    t+ 191.6ms  big4 释放后段 KV（成功）
    t+ 286.1ms  完成 12 个 token
```

盲绑是字面意义的：`pop()` 就是队列头，不比较任何延迟。识别在 tail(f) **本地**做，
直方图捎带在 hidden state 后传、不额外付 RTT。换绑时前段 KV 一层都不重算 ——
测试 `test_rebind_keeps_front_kv_and_drops_back_kv` 直接验证了命题 III.7.1。

干净请求识别 3/3、零误报；注入的误绑在 ~6 个 decode token 后被通道二发现并纠正。

---

## 与文档的偏差

实现过程中发现了若干处文档需要修正或补全的地方。每一处在代码里都有就地说明，
这里汇总。**没有一处是为了让算例好看而改的**——每一处都能给出机制层面的理由。

### 一、数值层面：算例 C.1 的两处断言过强

**1. 16GB 档「整档归零、废料率 40%」不成立。**
文档的理由是「后段两节点形态的任一位 ✗（最小位 > 23GB）」，这隐含了两节点
平分层区间。但切分不必平分：Y 后段 49.1GB 可以由 A40 承 25 层（45.4GB）+
一台 15GB 节点承剩下 2 层（3.6GB）。故 16GB 档能承担 Y/Z 后段形态里的小位。
实际废料率是 0%，不是 40%。
（`capacity.py` / `test_appendix_c.py::test_c1_tier_zeroing_diverges_from_document`）

**2. `N_max = 5` 与「精确位配平」不是一回事。**
按活跃供给 ÷ 单通道需求算出的是 7（内存上界），按节点位做精确整数配平算出的
是 5。文档把两者混为一谈。本实现两个都算、都报，并用**更紧的那个**做 θ 的基数
—— θ 的职责本来就是吸收上界与可达值之间的间隙，基数越紧、θ 越站得住。

### 二、模型层面：III.5.2 的跳数下界不含供给约束

`hops_min` 以 `max_v(M_v − 预留)` 为分母，等于假设最大内存档供给无限。用它单独
做 L₀ 准则会选错：算例 C 的池子里 L₀=7 时 Y 池后段恰好缩进单张 A40（加权总跳数
3.00→2.70，看起来更优），但同时前段也被迫只能用 A40，4 张 A40 要同时供前段和
X/Y 后段，**通道数从 5 塌到 3**。

本实现把容量估算并入 L₀ 准则（字典序：通道数 → 加权跳数 → 识别准确率），
在算例 C 上复现出文档的 L₀=5。`memory.py::L0Candidate.n_channels`

### 三、目标函数层面：Step 3 对接口完全无知

I.2.1 的延迟模型是 `T = T_F(f) + w(f,b) + T_B(b) + d_loop(b,f)`。文档的 Step 3
只优化 `T_B`，把 `w` 与 `d_loop` 留给后面的步骤。但 **`w` 依赖 `head(b)`、
`d_loop` 依赖 `tail(b)`** —— Step 3 在自由构造后段时其实已经把这两项的一半定死了，
却没把它们计入代价。

后果实测可见：入口若落在接入差的节点上，它的 in-access 是一个公共加项，会同时
抬高**所有**候选出口对该入口的 ŵ，公共带被整体压缩，Step 4 无论怎么滑窗都凑不出
人口。文档给的补救是 Step 4→3 的异类入口诊断，但那一次只能换一个入口。

补正：后段目标函数加端点项 `δ̂_in(head) + δ̂_out(tail)`，δ̂ 是实测中位数（不引入任何
结构假设）。权重取 1.0 不是调参 —— head 的 in-access 被每个组合的每个 token 各付
一次，它进总延迟的系数就是 1。置 `cfg.endpoint_w = 0` 可退回文档的原始目标函数。
`types.py::Objective.endpoint_w`

### 四、闸门层面：回环是整条通路上唯一不过闸的一跳

I.2.3 的尾闸写的是「逐跳 ŵ95 − ŵ50 ≤ J_cap」，逐跳自然包含回环。但 II.3 的闸门
只作用在正向接口上，II.4 Step 6 只按 `D(f)` 升序排、全程没提闸门。实测中回环链路
是终审「最大单跳抖动超限」的主要来源。补正见 `loop_trim.py::loop_profile`。

### 五、测量层面：k ≥ 8 对 p95 不够

文档对采样统一写「k ≥ 8」。但 p50 与 p95 对样本量的需求差一个量级：中位数在 k=8
时已经相当稳，而 p95 在 k=8 时几乎就是「8 个样本的最大值」—— 对指数尾它的期望是
`scale·H₈ ≈ 2.72·scale`，真值是 `scale·ln20 ≈ 3.00·scale`，**系统性偏低且方差极大**。

后果可复现：用 k=8 过闸放行的链路，到终审用 k=16 复测就超 J_cap。这是测量方法的
问题，不是放置的问题。本实现把闸门采样分离出来（`cfg.k_gate = 32`），p50 仍用
`k_probe = 8`，探测预算只在闸门上多花。`types.py::PlannerConfig.k_gate`

### 六、影子租金层面：III.8.4 的 R_F 选法在本场景下反向

III.8.4 说 `R_F` 按 `(M_v, 稳定性)` 取前 N 名（大内存优先），论证前提是「前段候选
几乎全落在排序前列」。这在前段驻留量最重时成立。但前段只需装 L₀ 层并集，后段要装
`(L−L₀)` 层专用集 —— L₀ 小时后段的单节点需求反而**大于**前段（算例 C：前段
28.0GB，X 后段的大位要 47GB）。按 M_v 降序预留会把最稀缺的大节点划给用不上那么多
内存的前段，逼得后段多一跳。

默认改为「在够装下前段的节点里取最小的 N 台」，把大节点留给真正需要的一侧。
`rent_policy="largest"` 可切回文档口径。`pipeline.py::select_rent_nodes`

另外补了一条**稀缺档保护租金**：II.4 的影子租金只保护前段，但逐条顺序建段还有第二个
抢占问题 —— 先建的段会顺手占掉某个稀缺档，而它本可以不占。判据是纯组合的、不含调参
（「某档对别的 spec 不可避免、而对本 spec 可避免」）。`pipeline.py::scarcity_rent_nodes`

### 七、算例 C.0 与 C.5 的专家重叠度自相矛盾

C.0 说三个 task 的驻留集是 8 / 6 / 7 个专家、**并集 20** —— 20 ≈ 21，意味着两两
几乎不重叠。C.5 又说误绑后「后段滑窗 miss 率 19%（X 池基线 3%）」，即
q(X, Y) ≈ 0.19。这两个数字不能同时成立：驻留集几乎不重叠时，误绑等于把激活质量
几乎全部丢到集合外，q 应该接近 1 而不是 0.19。

用 `sim/replay.py` 扫共享核大小可以量化（覆盖率 0.97，规模 8/6/7 固定）：

| 共享核 | 并集规模（层均） | q(X,Y) | q(Y,X) |
|---|---|---|---|
| 0 | 20.7 | 1.000 | 1.000 |
| 1 | 18.7 | 0.630 | 0.553 |
| 3 | 14.7 | 0.251 | 0.173 |
| 4 | 12.7 | 0.159 | 0.097 |

要让 q 落到 0.19，并集得是 **13–14**，不是 20。这不是无关紧要的口径差：并集 20 时
前段每层 5.53GB、L₀=5 的前段共 28.0GB；并集 13.7 时每层 3.83GB、前段共 19.5GB ——
后者能让更多节点承载前段，Step 4 的候选出口人口、Step 5 的超建量都会跟着变。
建议核对回放语料的实际重叠度后，把 C.0 与 C.5 的数字统一。
（`test_experts_manifest.py::test_q_is_monotone_in_overlap`）

### 八、II.5 的「基线 = 1−覆盖率」偏低 3–6 倍，会让绑对的池持续误报

这条是把在线协议真跑起来才暴露的，而且后果很实际：**通道二会对绑对的池报警，
触发无谓换绑**，每次换绑要重算一遍后段 prefill。

实测（`examples/e2e.py`）：名义基线 3.9–4.4%，实测 13.4–31.1%，差 3–6 倍。三条原因叠加：

1. **口径差一个 k**。覆盖率是**质量**口径（激活质量有多少落在驻留集内），miss 是
   **事件**口径（该 token 的 top-k 里有没有缺的）。top-k 给了 k 次落空机会。
2. **前段的 miss 沿层放大**。前段驻留的是并集，但并集同样按覆盖率截取，它自己也
   在做 drop-expert 重归一。送给后段的隐状态已经偏离画像统计时的分布，后段路由跟着
   漂。实测：前段换成全专家时后段 miss 从 19% 掉到 2.8%，差的就是这一项。
3. **decode 会漂出回放分布**。生成的 token 不再来自语料，隐状态逐步漂移。只用
   prefill 校准的基线仍然偏低。

正确做法是**在与在线完全一致的配置和轨迹上实测基线**：前段=并集、后段=该 task
驻留集、跑 prefill + decode、且只统计 decode 段（因为滑窗看的就是 decode）。
`runtime/corpus.py::measure_baseline_miss`

同一条教训还适用于识别：分类器的参考直方图也必须在「前段只装并集」的配置下测，
拿全专家画像当参考会掉准确率（本 demo 实测 1/3 → 3/3）。`measure_front_refs`

### 九、流程层面：三条缺失的护栏

* **反馈口会震荡。** 异类入口诊断直接取第一名会把「涨了 1」当成异类 —— 实测中它把
  一条好后段拆成三跳的怪物，下一轮再换回来。加了判据（跃升 ≥ 2 且明显高于第二名）
  与整轮回退制。`common_band.py::pick_outlier`
* **单条池的段没人管。** II.2.1 的间隙检测需要 ≥3 条同池段才跑得起来。加了跳数质量
  下限：建段时就地对照 III.5.2 的整数下界，超出 slack 即当作建不出、记入公平比缺口
  （否则会出现比下界多 4 跳、延迟 3 倍的段混进池子）。`cfg.max_hops_slack`
* **「降条数」这条迭代口是空的。** II.4 写了「某池建不出目标条数 → 回 Step 2 调配额
  或降该池条数并记入公平比缺口」，但没说什么时候触发。本实现把它接到公共带人口不足
  上：每条后段都往公共带加一个约束，入口少一个带就宽松一分，所以人口凑不够时正确
  动作是先降条数、而不是先放宽 η。

---

## 池群鲁棒性扫描

同一算例换 16 个网络随机种子，规划全部跑通，终审通过 8/16。**未通过的都不是崩溃，
而是带诊断的结论**：

| 失败类型 | 出现次数 | 含义与出路 |
|---|---|---|
| 某 task 一条后段都建不出 | 6 | 抖动闸切碎链路图后，最后建的池只剩劣质节点。出路：池合并（推论 III.7.4）或扩容 |
| 相对均匀性 > η | 3 | 池群本身撑不住 12% 的目标。出路：放宽 η、或域分裂（II.6） |
| 回环抖动闸后前段不够 | 1 | 抛 `PlanningError`，日志给出出路 |

这个扫描本身是交付物的一部分：它说明**算例 C 是一个偏乐观的样本**，同样的池子换个
网络实现有一半概率达不到 η = 12%。文档 §III.9「不声称」里的「非平稳网络下的均匀性
保持」在这里有了量化的注脚。

---

## 下一步

待办清单在 **[TODO.md](TODO.md)** —— 每条写了要动哪些文件、会破坏什么既有性质。
优先级最高的两条：

1. **真实模型执行层** —— 把 `PartialExpertMoEBlock` 换成 torch/HF 版本接真实 MoE
   权重。接口契约已定死（`forward(x, cache) -> (y, MoEStats)`），测试原样适用。
2. **权重分发** —— toy 模型的权重由种子派生，节点间零传输；真实模型要解决
   「每个节点只拿自己那几层的那些专家」的分发。

**批处理明确推迟**（TODO.md P1 有完整分析）：它不是加个 feature，是把方案的核心
取舍翻过来 —— 会同时影响确定性延迟、零后悔定理、命题 III.3.3 的论证与 III.6.2 的
无量纲比。重拾时应先回到 I.2.3 重估均匀性可达性。

# 15 台真实节点的操作手册

> **只想要能直接贴的命令** → [RUNBOOK.md](RUNBOOK.md)。本文解释每一步为什么这么做。

> 两条路，按需要选：
>
> * **管路验证**（不给 `--model-dir`）—— toy MoE，节点只要 numpy。验的是部署
>   路径、网络探测、放置规划、在线协议。跑得快，不需要下载权重。
> * **真模型**（给 `--model-dir`）—— 真 checkpoint、选择性加载、文本进出。
>   节点要装 torch + safetensors，而且**每台机器上都得有权重**（见第 1 步）。
>
> 下面每一步都会标出两条路的差异。先跑通管路再上真模型，出问题时好定位。

---

## 最短路径：offline 算好 → 只拉自己那份 → 建链推理

四步，中间不需要规划器，也不需要谁下 61GB：

```bash
# ① offline：写布局 → 出清单（不连机器、不要权重）
python3 -m p2pmoe.deploy.run --spec deploy.json --plan-only --save-plan plan.json

# ② 各节点只拉自己那部分权重（并行，各自直连上游）
python3 -m p2pmoe.deploy.launch fetch --hosts hosts.txt --plan plan.json \
        --repo Qwen/Qwen3-30B-A3B --out /data/qwen3-part

# ③ 起 agent（常驻，先后起来都行）
python3 -m p2pmoe.deploy.launch start --hosts hosts.txt --workdir /opt/p2pmoe

# ④ 建链 + 推理
python3 -m p2pmoe.deploy.run --spec deploy.json \
        --agents "$(python3 -m p2pmoe.deploy.launch agents --hosts hosts.txt)" \
        --advertise <控制机IP> --model-dir /data/qwen3-part --device cuda:0 \
        --chat --prompt "用一句话解释 MoE 的稀疏激活" --tokens 64 --once
```

`--model-dir` 对 15 台是**同一个值** —— 每台机器上那个目录里装的正好是它自己要的
那些张量。多个 agent 挤在一台机器上演练时，用 `{node}` 占位
（`--model-dir /data/parts/{node}`），节点会代入自己的名字。

下面是这四步的细节与其它选项。

---

## 只想先跑起来？跳过规划

如果这一轮的目标只是「前后段连上、能出 token」，不需要下面那套探测与规划。
写个布局文件说清楚哪台装哪几层、哪条前段连哪条后段，一条命令：

```bash
cat > deploy.json <<'JSON'
{
  "model_dir": "/data/qwen3-30b-a3b",
  "l0": 6,
  "channels": [
    {"front": "n1",  "back": ["n2",  "n3",  "n4"]},
    {"front": "n5",  "back": ["n6",  "n7",  "n8"]},
    {"front": "n9",  "back": ["n10", "n11", "n12"]}
  ]
}
JSON

python3 -m p2pmoe.deploy.run --spec deploy.json \
    --agents "$(python3 -m p2pmoe.deploy.launch agents --hosts hosts.txt)" \
    --advertise <控制机IP> --device cuda:0 \
    --chat --prompt "用一句话解释 MoE 的稀疏激活" --tokens 64 --once
```

前置条件只有第 1 步那些（装依赖、同步权重、放行端口、起 agent），第 0 / 3 / 5 步
都不需要。15 台机器上面这个布局用了 12 台，剩下 3 台闲着 —— 会在日志里点名。

**代价**：不探测就没有延迟画像，各通道的快慢可能差很多，方案文档里那些以
「组合极差被压到抖动量级以下」为前提的结论（零后悔、任意组合、延迟均匀）
都不成立。要那些就走下面的完整流程。

**后段默认全装专家**（无 drop-expert 近似）。要让它只装该装的那些，跑两轮：

```bash
# 第一轮：全装，顺便采路由统计
python3 -m p2pmoe.deploy.run --spec deploy.json --agents ... \
        --prompt "真实请求 1" --prompt "真实请求 2" --profile-out prof.json

# 第二轮：按画像只装子集
python3 -m p2pmoe.deploy.run --spec deploy.json --agents ... \
        --profile prof.json --coverage 0.95
```

第一轮**必须**全装 —— 只驻留子集时输出被近似带偏，后面几层采到的就不是真实路由。
两个参数一起给会直接报错。多个 task 就给各通道写不同的 `"task"`，把各自的请求
打进去，画像自然是逐 task 的。

细节（三种写法、逐层指定专家、画像怎么采、能拦住哪些错）见
`p2pmoe/deploy/manual.py` 与 `p2pmoe/runtime/profile.py` 的文件头。

---

## 第 0 步：先量延迟，再决定要不要往下走

整套方案的前提是「网络是主项」——文档 I.2.1 说分散环境下网络占单 token 延迟约九成。
**如果你的 15 台在同一个机房同一台交换机后面，这个前提不成立**，均匀性机制没有对象，
规划会在公共中值域那步（正确地）失败。

所以第一件事是量清楚：

```bash
# 在控制机上
cat > hosts.txt <<'EOF'
# 节点id   地址              [附加参数]
n1         10.0.0.11
n2         10.0.0.12
...
n15        10.0.0.25
EOF

python3 -m p2pmoe.deploy.launch start --hosts hosts.txt --workdir /opt/p2pmoe
sleep 5
python3 -m p2pmoe.deploy.launch probe --hosts hosts.txt
```

输出与判定：

```
28 对，可达 28 对
  单向 p50     13.14 –   25.45 ms（中位 15.24）
  抖动 p95−p50  0.53 –    7.95 ms（中位 3.45）

  接入质量（对全网的 p50 中位，越小越好）：
    n8   14.30 ms
    ...
    n5   23.74 ms        ← 接入差的节点会自己浮出来

✓ 中位 15ms，符合分散环境的设定 —— 网络是主项，这套方法有对象。
```

| 中位单向 p50 | 判定 | 怎么办 |
|---|---|---|
| **≥ 5ms** | ✓ 符合设定 | 往下走 |
| 1–5ms | △ 收益有限 | 能跑，但跳数优先与均匀性收紧的杠杆都变小了 |
| **< 1ms** | ⚠ LAN 量级 | 这套方法没有对象。想验证部署路径，给 agent 加 `--access-ms` 模拟接入段 |

命令的退出码就是判定：0 可用、2 是 LAN 量级。

---

## 需要哪些连通性（先确认这个）

| 方向 | 协议/端口 | 必须吗 | 用来做什么 |
|---|---|---|---|
| **节点 ↔ 节点**（两两） | TCP 9101 | **必须** | 数据面：段内转发、跨接口出段、绕环 decode、逐对探测 |
| 控制机 → 节点 | TCP 9101 | **必须** | 采集能力、下发清单、打请求 |
| 节点 → 控制机 | TCP（协调器端口，随机） | **必须** | 上报 token、识别结果、错误 |
| 节点 → 上游 | HTTPS 443 | 拉权重时要 | `fetch` 只拉本机那部分 |
| 控制机 → 节点 | **SSH 22** | **不必须** | 只有 `launch.py` 用它批量起停/拉权重 |

**节点之间不需要 SSH。** 数据面是节点直接互发的 TCP，走的就是 agent 那个端口。
`ssh` 在整个仓库里只出现在 `deploy/launch.py` 一个文件里，那是个批量操作的便利脚本，
框架本身不认识它。

不配 SSH 的话，把命令拿去各机自己跑就行，效果一模一样：

```bash
# 打印各节点该执行的命令，自己贴过去（或交给 ansible / salt / k8s）
python3 -m p2pmoe.deploy.launch start --hosts hosts.txt --workdir /opt/p2pmoe --dry-run
python3 -m p2pmoe.deploy.launch fetch --hosts hosts.txt --plan plan.json \
        --repo Qwen/Qwen3-30B-A3B --out /data/qwen3-part --dry-run
```

生产环境本来也该用 systemd 而不是 ssh 拉起（开机自启、崩溃重启、日志轮转）：

```bash
python3 -m p2pmoe.deploy.launch systemd --id n1 --workdir /opt/p2pmoe \
    > /etc/systemd/system/p2pmoe-agent.service
```

**「节点两两可达」这条最容易踩坑**：数据面是节点直接互发，控制机不在中间。
只开「控制机 → 各节点」是不够的。`launch probe` 会把不可达的对报出来。
NAT / 防火墙导致部分不通是分散环境的常态 —— 规划器会自然绕开，
但如果绝大多数对都不通，说明放行没配对。

### 如果节点之间根本连不通

家宽 NAT 后面的机器只有出站没有入站，两台之间压根建不了连接。两条路：

**① VPN overlay（推荐）** —— WireGuard / Tailscale 之类，给每台一个虚拟 IP，
节点之间恢复**真正的直连**。框架这边什么都不用改，`--bind` 到虚拟网卡即可。
延迟只多一层封装。

**② 中继（零配置兜底）** —— 找一台有公网入站的机器（控制机通常就是）跑中继，
所有节点只需要**一条出站连接**：

```bash
# 中继机
python3 -m p2pmoe.deploy.relay --bind 0.0.0.0:9200

# 每个节点：不再监听任何端口
python3 -m p2pmoe.deploy.agent --id n3 --relay relay.example.com:9200

# 控制机：加同一个 --relay
python3 -m p2pmoe.deploy.run --spec deploy.json --relay relay.example.com:9200 ...
python3 -m p2pmoe.deploy.control --agents ... --relay relay.example.com:9200
```

中继是**接线员不是代理**：握手完只搬字节，不认识 p2pmoe 的消息格式。
所以协议、节点逻辑一行都不用改 —— 对它们来说那就是一条普通的 TCP 流。

代价说清楚：

* **每一跳变成两段**，正向接口、回环、段内转发全要绕一圈，逐 token 延迟大致翻倍；
* **中继是带宽瓶颈与单点**，所有 hidden state 都过它。15 台还行，再大要按段分流；
* **探测量到的是「经中继的往返」**，不是两台之间的真实延迟。规划据此做的放置仍然
  自洽（它优化的就是实付延迟），但别拿这些数字推断链路质量。

本地演练一遍（起中继 + 6 个不监听端口的 agent）：

```bash
python examples/manual_deploy.py --channels 2 --front 1 --back 2 --relay
# 中继在 127.0.0.1:37387；agent 将不监听任何端口
# req0  F0×Bgeneral0  6 token  首token 20ms
# 中继：接通 28 次，搬运 0.03MB，仍挂起 {'n1': 8, 'n2': 8, ...}
```

---

## 第 1 步：准备（一次性）

每台节点需要：

* **Python ≥ 3.10**。依赖按机器角色分，两侧互不包含：

  ```bash
  # 15 台节点
  pip3 install -r requirements.txt          # 管路验证：numpy，就这一个
  pip3 install -r requirements-node.txt     # 真模型：再加 torch + safetensors

  # 控制机
  pip3 install -r requirements.txt          # 管路验证
  pip3 install -r requirements-control.txt  # 文本进出：再加 tokenizers + jinja2
  ```

  **控制机不装 torch，节点不装 tokenizers。** 节点自始至终只见 token id；
  文本在控制机的入口编码、出口解码。
* **真模型还要：每台节点本地有权重。** `--model-dir` 是**各节点上的本地路径**。
  三种办法，按省下载排序：

  ```bash
  # ① 只拉本机要的那部分（推荐）—— 15 台合计 179GB 而不是 916GB
  #    先要有一份部署清单：deploy.run --save-plan plan.json 或 control --save-plan
  python3 -m p2pmoe.deploy.launch fetch --hosts hosts.txt --plan plan.json \
          --repo Qwen/Qwen3-30B-A3B --out /data/qwen3-part
  #    各机同一个路径，因为节点 id 只决定「拉什么」，不决定「放哪」

  # ② 挂共享存储（NFS/Lustre），各机路径一致，一份就够
  # ③ 每台都下全量（最简单也最贵：61GB × 15）
  huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir /data/qwen3-30b-a3b
  ```

  ① 需要先有清单，而清单**不需要权重也不需要机器在线**就能出：

  ```bash
  python3 -m p2pmoe.deploy.run --spec deploy.json --plan-only --save-plan plan.json
  ```

  它只把布局翻译成逐节点逐层的加载指令，顺便把布局校验一遍。控制机上要能读到
  模型的 `config.json`（几 KB）—— 拉一个就够，不用拉权重。

  控制机也要能读到 `config.json` 与 `tokenizer.json`（只读这两个，不读权重）。
  路径不一致或漏了机器，预检会在探测**之前**点名报出来。
* **代码**。`rsync -a p2p-framework/ node:/opt/p2pmoe/` 或 git clone。不用 `pip install -e .`
* **端口放行**。默认 9101，节点之间要**两两可达**（见上一节的连通性表）
* **SSH 是可选的** —— 只有 `launch` 的批量起停/拉权重用它。配了就免密（`BatchMode=yes`），
  不配就用 `--dry-run` 把命令拿去各机自己跑，或者走 systemd。

---

## 第 2 步：起 agent

**不需要同时启动。** agent 是常驻守护进程，先后起来都行；控制器只要求「跑规划的
那一刻它们都在」。没起来的会在采集能力那步被剔除，不影响其余节点。

```bash
python3 -m p2pmoe.deploy.launch start  --hosts hosts.txt --workdir /opt/p2pmoe
python3 -m p2pmoe.deploy.launch status --hosts hosts.txt
```

```
  n1   10.0.0.11:9101   ✓ 空闲  内存 32000MB  基准 0.412ms
  ...
  15/15 台在线
```

生产环境用 systemd（开机自启、崩溃重启、日志轮转，`launch` 这些都不管）：

```bash
python3 -m p2pmoe.deploy.launch systemd --id n1 --workdir /opt/p2pmoe \
    > /etc/systemd/system/p2pmoe-agent.service
systemctl daemon-reload && systemctl enable --now p2pmoe-agent
```

agent 是**无状态**的：崩溃重启后回到未配置态，等控制器重新下发即可 ——
所以 `Restart=always` 是安全的。

---

## 第 3 步：跑控制器

**管路验证**（toy 模型）：

```bash
python3 -m p2pmoe.deploy.control \
    --agents "$(python3 -m p2pmoe.deploy.launch agents --hosts hosts.txt)" \
    --advertise <控制机对节点可见的IP> \
    --mem-cap-mb 26 \
    --requests 5 --tokens 12 --once
```

**真模型 + 静态链路**（15 台，Qwen3-30B-A3B）：

```bash
python3 -m p2pmoe.deploy.control \
    --agents "$(python3 -m p2pmoe.deploy.launch agents --hosts hosts.txt)" \
    --advertise <控制机对节点可见的IP> \
    --model-dir /data/qwen3-30b-a3b \
    --static --tasks general --ctx 2048 --device cuda:0 \
    --save-plan plan.json --save-wiring wiring.json \
    --chat --prompt "用一句话解释 MoE 的稀疏激活" \
    --requests 3 --tokens 64 --once
```

三个参数必须显式给：

**`--advertise`** —— 控制机的 IP，节点靠它回连上报。控制机多网卡或在 NAT 后面时，
自动推断会猜错。

**`--mem-cap-mb 26`** —— **这是 toy 模型的产物，不是配置项**。整个 toy 模型才几十 MB，
你的节点报几十 GB 的话一台就装得下整条通道，规划会退化成「全放一台、零跳」——
部署路径照样验证得了，但看不到分段、跳数、公共带这些真正的机制。压到 26MB 左右，
前段（6 层并集）刚好占满一台，后段落到别的机器上，链路就活了。
不给这个参数控制器会警告并建议一个值。接真实 MoE 后删掉它。

**`--once`** —— 跑完这批请求就退出。不加会常驻（agent 不受影响）。

**`--tasks general`** —— 首次跑真权重时后段默认**全装**专家（`--resident-frac 1.0`），
各 task 装的东西完全一样、彼此无差别，用一个 task 就够。等有了真实激活画像
（`--profile`）再按 task 分池才有意义。

**`--resident-frac` 与 `--profile`** —— 这两个决定后段每层装多少专家：

| 配置 | 后段驻留 | 输出质量 |
|---|---|---|
| 默认（`--resident-frac 1.0`） | 全部专家 | 与单机参考实现**逐位一致**，无近似 |
| `--profile p.json` | 画像给的子集 | 有 drop-expert 近似，miss 率由覆盖率决定 |
| `--resident-frac 0.3` 且无 profile | 按 id 硬取前 30% | **基本是废的**，只用来压内存看放置 |

第三行不是能力，是一个明确标注的坑：驻留集不按激活质量选就等于随机丢专家。
控制器会为此打一条 warning。画像怎么产出见 TODO.md（还没接真模型）。

**`--model-dir` / `--prompt` / `--chat`**（可选）—— 走**文本**进出而不是 token id。
`--model-dir` 指向 checkpoint 目录，控制机从里面读 `tokenizer.json`；
指令模型要加 `--chat` 套对话模板（不加是 completion 语义，输出会像坏了）。

```bash
pip3 install -r requirements-control.txt     # 控制机：tokenizers + jinja2
python3 -m p2pmoe.deploy.control ... --model-dir /path/to/Qwen3-30B-A3B --chat \
        --prompt "用一句话解释 MoE 的稀疏激活"
```

**节点不用装 tokenizers。** 它们只见 token id；控制机下发的停止条件也只是几个
整数（`stop_ids`）。节点装 `requirements-node.txt`，控制机装
`requirements-control.txt`，两份互不包含。

**`--static`**（可选）—— 走简化模式：前段装**全部**专家，前后段配对在探测完之后
离线算好、随配置一起下发。在线不识别、不派发、不换绑，请求自报 task。
第 3/5 步会多打一张链路表：

```
静态链路：3 条静态通道，组合 p50 32–37ms（极差 5ms）
  ch0  X      F0(n3) → BX0(n5)  组合 p50 32ms
  ch1  Y      F2(n1) → BY0(n8)  组合 p50 35ms
```

用它的判断标准很简单：**如果每条请求的 task 在进系统之前就已经知道**
（比如按端点/队列分流），静态模式省掉的那一整套在线机制本来就用不上。
反之若 task 要靠内容判断，就得走主线。取舍的完整说明见
`p2pmoe/planner/static_pairing.py` 与 `examples/static_qwen.py` 的文件头。

**`--concurrency N`**（可选，默认 1）—— 同时打入多少条。超过池子容量的会排队，
前面一完成就自动接走队首。设成大于前段条数才看得到排队行为。

---

## 第 4 步：看输出

```
[1/5] 采集节点能力（15 台）
[2/5] 逐对探测（由节点自己发起，k=8）  105 对，用时 4.2s
[3/5] L₀=6，配额 {X:3, Y:1, Z:1}，前段 5 条，组合矩阵 25 组，清单校验 通过
[4/5] n5   front/F0      层 [1..6]  110 个专家  驻留 23.54MB（全装要 39.67MB）
      n15  back:X/BX0    层 [7,8]    23 个专家  驻留  5.16MB（全装要 13.22MB）
      ...
[5/5] req0  真实 X → 识别 X ✓  换绑 0  首token 95ms  逐token p50 100ms
      汇总：识别 5/5，逐 token p50 100ms，换绑 0 次
```

几个该看的地方：

* **配对历史** —— 跑完会列出每条请求用了哪对段。同一条前段在不同请求里配不同的
  后段，就是「跑完回池、下次重新配」在起作用
* **「驻留 X（全装要 Y）」** —— 选择性加载真的生效了，节点上只有清单点名的专家
* **不是所有节点都会被用上**。15 台的实测：用 11 台（5 前段 + 1 备胎 + 5 后段），
  剩下 4 台闲着。**接入差的节点会被自然排除**，不用人工标记
* **清单校验 通过** —— 七项一致性都过了（排他、内存含 KV、层区间连续、并集支配、
  后段驻留集不多不少、组合矩阵完整、逐对过闸）
* `--save-plan plan.json` 可以把完整清单存下来看

真模型那条路的输出多两行：

```
      预检：15 台节点能不能读到 /data/qwen3-30b-a3b  全部就绪
[4/5] n5   front/F0   层 1–6   768 个专家  驻留  7523.4MB（全装  7523.4MB）
                                          从 checkpoint 读了 12.3%（4/16 个分片）
      n15  back/Bg0   层 7–20  ...        从 checkpoint 读了 21.1%（5/16 个分片）
[5/5] req0  F0×Bg0  停于 eos  37 token  首token 812ms  «MoE 每个 token 只激活…»
```

**「从 checkpoint 读了 12.3%（4/16 个分片）」是选择性加载的直接证据** ——
61GB 的权重里，这台只打开了它那几层那些专家的 key，连不相关的分片文件都没碰。

---

## 第 5 步：把连接定死（可选）

`--static` 默认自动配对（贪心最小化组合延迟）。要**自己指定**哪条前段连哪条后段，
分三步：

**1. 先跑一次，把放置和连接都存下来**

```bash
python3 -m p2pmoe.deploy.control ... --static \
        --save-plan plan.json --save-wiring wiring.json --once
```

**2. 改 `wiring.json`**。`front` / `back` 既接受段 id（`F0`、`Bgeneral2`），也接受
**节点 id**（`n5`）—— 后者更好用，段 id 是每次规划现编的，机器名是你自己起的：

```json
{"pairs": [
  {"front": "n5",  "back": "n6"},
  {"front": "n4",  "back": "n7"},
  {"front": "n1",  "back": "n10"}
]}
```

**3. 载入清单 + 指定连接重跑**

```bash
python3 -m p2pmoe.deploy.control ... --static \
        --load-plan plan.json --wiring wiring.json --once
```

```
[2/5] 跳过探测（--load-plan）
[3/5] 载入清单 plan.json：L₀=6，12 个节点，12 条段
      静态链路：按 wiring.json 指定的 5 条
        ch0  general  F0('n5',) → Bgeneral3('n6',)  组合 34ms（清单值）
```

**为什么必须配 `--load-plan`。** 规划的输入里有一项不可复现：逐对延迟实测。
同一批机器换个时间跑，探测值会变，段的构成与 id 编号都可能跟着变 —— 于是
「F0 连 BX1」在下一次规划里可能指到别的东西上。载清单把放置固定住，指定才有
稳定的所指。代价是这份清单反映的是**当时**的网络与机器集合：换机器、链路劣化，
就得去掉 `--load-plan` 重跑规划。

指定会被校验，写错当场报错而不是默默跑歪：一条前段不能连两条后段（I.2.4 排他
独占）、一条后段不能被两条前段共用、`task` 写的和后段实际装的对不上、名字既不是
段 id 也不是本次用到的节点 —— 每一条都会指出可用的名字。

---

## 常见问题

**预检失败：节点读不到 checkpoint**

`--model-dir` 是**各节点上的本地路径**，控制机读得到不代表节点读得到。用
`launch fetch` 让各节点自己拉本机需要的那部分，或挂共享存储，或加
`--skip-model-check` 自担风险。预检故意放在探测**之前**：不然故障要等到下发
那一刻才爆，前面几分钟的探测已经白跑了。

**节点加载报「缺 N 个 key」**

`fetch` 拉的是**按当时的驻留集**那一份。改了画像、改了覆盖率、重新分层之后
驻留集就变了，得重拉：

```bash
python3 -m p2pmoe.deploy.launch fetch --hosts hosts.txt --plan 新的plan.json ...
```

**`fetch` 报「不支持 Range 请求」**

镜像把整个文件发回来了 —— 那样「省下载」根本没发生，而按区间切出来的还会是错的，
所以脚本拒绝将就。换 `--mode shard`（只下整分片，省得少但不依赖 Range），
或换一个支持 Range 的源。

**节点加载报「没装 torch」**

节点装的是 `requirements.txt`（只有 numpy）。真模型要 `requirements-node.txt`。
反过来控制机报「没装 tokenizers」则是要 `requirements-control.txt`。

**规划失败：「公共中值域人口 N < 后段条数 M」**
最常见的原因是延迟太均匀（回第 0 步）或池子太小。日志里会给三条出路：放宽 η、
降条数、域分裂。也可以直接 `--eta 0.25` 放宽相对均匀性目标试试。

**某台节点连不上**
控制器在采集能力那步就剔除，继续用剩下的规划。`launch status` 看是哪台。

**探测显示大量不可达**
节点之间没有两两放行。数据面是节点直接互发的，只放行控制机到节点不够。

**逐 token 延迟远高于探测出的延迟**
正常。一个 token 要绕完整一圈：前段各跳 + 正向接口 + 后段各跳 + 回环。
`--save-plan` 出来的 `pairings` 里有每个组合的分解。

**想重新规划**
再跑一次控制器就行，agent 不用重启——它会用新清单覆盖旧的。
节点池变了（加机器、换网络）之后就该重跑。

---

## 这一轮验证不了什么

* **真实模型质量**。toy MoE 的权重是随机初始化的，输出没有意义。
* **真实显存压力**。toy 模型几十 MB，真实 MoE 是几十 GB，权重分发是个独立问题
  （现在权重由种子确定性派生，所以节点之间不用传一个字节）。
* **churn 下的维护**。控制器现在只在采集能力时剔除连不上的节点，运行中掉线没有
  重构逻辑（文档 II.6 的三层维护还没实现）。
* **通道一（SPRT）**。只实现了通道二（后段 miss 率检出）。
* **批处理**。并发度 = 前段条数，第 N+1 条请求排队。明确推迟，见 [TODO.md](TODO.md)。

完整待办与优先级见 [TODO.md](TODO.md)。

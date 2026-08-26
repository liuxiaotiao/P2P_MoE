# 15 节点部署 Qwen3-Next-80B-A3B：完整运行流程

从零到拿到测量结果。每一步都标了**在哪台机器上跑**。

设计与取舍在 [DEPLOY.md](DEPLOY.md)，算法在 `task.docx`。这里只讲怎么跑。

---

## 0. 这套东西在跑什么

模型按层切成两段：

- **前段** 层 1–11（L₀=11），**任务无关**，装并集专家。任何请求先进这里。
- **后段** 层 12–48，**任务相关**，每个 task 一套，只装该 task 的热专家。

请求随机落到任意一条前段 → 前段一边算一边**在线识别**它是 mbpp 还是 gsm8k → 识别定了就派发到对应的后段 → 后段算完出 token。

当前部署（`task/plan_deploy.json`，用真实激活数据 + 真实拓扑离线算出来的）：

| | |
|---|---|
| L₀ | 11（拐点，p ≥ 97.69%） |
| 前段 | 3 条 + 1 条备胎 |
| 后段 | mbpp 2 条、gsm8k 1 条 |
| 到达比 | mbpp : gsm8k = 5 : 3 |
| 覆盖率 | 0.70 —— mbpp 层均 121/512 专家，gsm8k 层均 98/512 |
| 负载 / 产能 | 0.94×（基本均衡） |
| 总拉取 | 141 GB（全模型 × 15 台 = 2400 GB，省 94%） |

---

## 1. 角色分工

**控制机**（1 台）——可以是这 15 台里的任意一台，也可以是你的笔记本。

只有它有 `deploy_15.sh`、`plan_deploy.json`、`hosts.txt`。它做：读清单、采集节点能力、探测两两延迟、算配对、把 prompt 编码成 token id 发出去、收 token 解回文本、统计时序。

**它不装 torch** —— 一个张量都不碰，只要 `numpy + tokenizers + jinja2`。这不是省事：控制机装了 torch 就会有人图方便让它"顺手算一层"，那条路一开，控制机就成了隐形的第 16 个计算节点，时序测量立刻失真。`test_requirements.py` 和 `test_heavy_deps_stay_in_the_execution_layer` 守着这条线。

**节点**（15 台）——装 torch，跑矩阵乘。**不需要有 `deploy_15.sh`**。

### 网络要求

| 方向 | 端口 | 必须吗 |
|---|---|---|
| **节点 ↔ 节点（两两）** | TCP 9101 | **必须** —— 数据面是节点直接互发，控制机不在中间 |
| 控制机 ↔ 节点 | TCP 9101 + 协调器随机端口 | **必须** |
| 节点 → huggingface.co | TCP 443 | 拉权重时 |
| 控制机 → 节点 SSH | 22 | **不必须**，见 §3b |

节点之间**不需要 SSH**。都在 NAT 后面两两连不通的话见 §9。

---

## 2. 控制机：一次性准备

```bash
cd /path/to/p2p-framework
pip3 install -r requirements-control.txt      # numpy + tokenizers + jinja2
./deploy_15.sh meta                            # 取 config + tokenizer（约 10MB）
```

`meta` 那一步容易被跳过。控制机与节点**共用 `--model-dir` 这一个路径**，但要的东西
不一样：节点要权重（几十 GB），控制机一个张量都不碰 —— 只用 `config.json` 算模型
规格、用 tokenizer 把 prompt 编成 id 再把 id 解回文本。

而权重是**各节点自己拉的，不经过控制机**，所以控制机上那个目录默认是空的。
症状是 `serve` 一上来就报 `/data/xxx/config.json 不存在`。

写 `task/hosts.txt`：

```
# 节点 id  地址:端口
N01  192.168.1.11:9101
N02  192.168.1.12:9101
...
N15  192.168.1.25:9101
```

节点 id 必须和 `plan_deploy.json` 里的对上——清单按 id 认人，不按 IP。

设三个环境变量：

```bash
export ADVERTISE=192.168.1.100        # 控制机对节点可见的 IP —— 必填
export SSH_USER=ubuntu                # ssh 用户名，root 就不用设
export SSH_OPTS="-i ~/.ssh/pool.pem"  # 私钥/端口，没有就不设

export WORKDIR=/home/ubuntu/P2P_MoE            # 各节点上代码放哪儿
export WEIGHTS=$WORKDIR/weights                # 权重放哪儿（15 台路径要一致）
export NODE_PY=/home/ubuntu/anaconda3/envs/moe/bin/python   # 节点上带 torch 的解释器
```

### 权重别放 `/data`

`/data`、`/opt` 这类系统目录通常要 root，而这套东西不该需要 root。默认放在
`$WORKDIR/weights`。

放在代码目录里有个**会静默删掉 141GB 的陷阱**：`sync` 用 `rsync --delete`，
目的端有、源端没有的东西会被删 —— 而权重恰恰是各节点自己拉的、源端根本没有。
所以 `sync` 会自动把它排除（`bootstrap` 打包时同理），并在跑之前告诉你：

```
  权重在 /home/ubuntu/P2P_MoE/weights（代码目录里面）—— rsync 会排除 weights/
```

`test_weights_survive_sync.py` 盯着这个排除项，顺带盯着「别为了保权重把
`--delete` 也废掉」—— 旧代码留在节点上会跑出错误结果。

### `NODE_PY` 为什么要写绝对路径

torch 装在 conda 环境里的话，**裸 `python3` 是看不见它的**。而 `ssh host '...'` 起的是
非交互 shell，不会执行 conda init —— `conda activate moe` 在那里不存在，`python3`
指的是系统的那个。所以要直接指到环境里的解释器。

不知道路径就跑 `./deploy_15.sh doctor`，它会在各节点上找出来并打印该填什么。

**`NODE_PY` 和控制机的 `PY` 是两回事** —— 控制机不装 torch（见 §01）。

`ADVERTISE` 是必填的：节点建链时要回连控制机，得知道往哪连。写 `127.0.0.1` 会让 15 台各自连自己。

想确认它对不对：在**任意一台节点**上 `curl -v telnet://$ADVERTISE:9300` 应该能连上（先起 `bootstrap`）。

---

## 3a. 有 SSH：一条命令

```bash
./deploy_15.sh all
```

等价于依次跑 `sync → check → fetch → start → serve`。

**第一次建议分开跑**，因为 `fetch` 是 141 GB，视带宽可能要几十分钟：

```bash
./deploy_15.sh sync      # rsync 代码到 15 台 + 各自 pip install
./deploy_15.sh check     # ssh 可达 → agent 在线 → 两两可达探测
./deploy_15.sh fetch     # 先 dry-run 给你看会下多少，按 y 才真拉
./deploy_15.sh start     # 起 15 个 agent
./deploy_15.sh measure   # 建链 + 打请求 + 逐请求时序
./deploy_15.sh report    # 出表
```

### SSH 必须免密

脚本用 `ssh -o BatchMode=yes`，**密码提示会直接把它挂住**。没配的话：

```bash
ssh-keygen -t ed25519 -N ''
for i in $(seq 11 25); do ssh-copy-id ubuntu@192.168.1.$i; done
```

`check` 的第 1 步会**并行**探这 15 台（10 秒出结果），哪台不通会点名。

---

## 3b. 没有 SSH：HTTP 引导

只有引导这一步依赖 ssh —— **agent 起来之后整个控制面都是 TCP**。ssh 在这套代码里从来只是批量便利，不是协议依赖。

**控制机**（终端 A，起着别关）：

```bash
ADVERTISE=192.168.1.100 ./deploy_15.sh bootstrap
```

它打包代码、起一个 HTTP 服务（默认 9300）。

**控制机**（终端 B）：

```bash
./deploy_15.sh cmds          # 15 台各自的命令全打出来
./deploy_15.sh cmds N07      # 只看某一台
```

输出形如：

```
┌─ N07  192.168.1.17:9101
│  段 F0（front），段内第 2/3
│  层 2–10（9 层），要拉 11.9 GB，占显存约 12.0 GB
└─

  mkdir -p /home/ubuntu/P2P_MoE && cd /home/ubuntu/P2P_MoE \
    && curl -fsSL http://192.168.1.100:9300/p2pmoe.tar.gz | tar xz \
    && curl -fsSL http://192.168.1.100:9300/plan.json -o /tmp/plan.json \
    && $NODE_PY -m pip install -q -r requirements-node.txt \
    && $NODE_PY -m p2pmoe.deploy.fetch --plan /tmp/plan.json --node N07 \
         --repo Qwen/Qwen3-Next-80B-A3B-Instruct --out /data/qwen3-next-part \
    && setsid nohup $NODE_PY -m p2pmoe.deploy.agent --id N07 --bind 0.0.0.0:9101 \
         > /tmp/agent-N07.log 2>&1 &
```

**每台节点**：把属于它的那一段粘进终端。15 台**互不依赖，可以同时跑**。

一条命令四件事：拉代码 → 装依赖 → **只拉本机那份权重** → 起 agent。

15 台都起来后，控制机 Ctrl-C 关掉 HTTP，然后：

```bash
./deploy_15.sh check
./deploy_15.sh measure
./deploy_15.sh report
```

**两条路可以混用**：ssh 通的那几台走 `sync`，剩下的手工粘 `cmds` 的命令，结果完全一样。

---

## 4. 每台节点具体拿到什么

| 节点 | 段 | 位置 | 层 | 拉取 |
|---|---|---|---|---|
| N01 | F0 前段 | 1/3 段首 | 1–1 | 1.7 GB |
| N07 | F0 前段 | 2/3 | 2–10 | 11.9 GB |
| N10 | F0 前段 | 3/3 段尾 | 11–11 | 0.9 GB |
| N11 | F1 前段 | 1/1 独占 | 1–11 | 14.5 GB |
| N09 | F2 前段 | 1/1 独占 | 1–11 | 14.5 GB |
| N13 | F-standby0 备胎 | 1/4 段首 | 1–8 | 11.6 GB |
| N14 | F-standby0 备胎 | 2/4 | 9–9 | 1.1 GB |
| N15 | F-standby0 备胎 | 3/4 | 10–10 | 0.9 GB |
| N12 | F-standby0 备胎 | 4/4 段尾 | 11–11 | 0.9 GB |
| N03 | Bmbpp0 后段 | 1/2 段首 | 12–20 | 7.3 GB |
| N04 | Bmbpp0 后段 | 2/2 段尾 | 21–48 | 21.8 GB |
| N05 | Bmbpp1 后段 | 1/2 段首 | 12–20 | 7.3 GB |
| N02 | Bmbpp1 后段 | 2/2 段尾 | 21–48 | 21.8 GB |
| N06 | Bgsm8k0 后段 | 1/2 段首 | 12–14 | 1.6 GB |
| N08 | Bgsm8k0 后段 | 2/2 段尾 | 15–48 | 22.4 GB |

节点本身**不做任何决策**。它不知道自己是前段还是后段——这些全在 `plan.json` 里，离线就算好了。`fetch --node N07` 打开清单找到 N07 那条，只拉这一条列出的张量；`agent --id N07` 起进程监听 9101，报出自己是 N07。剩下的等控制机下发。

所以 15 条命令长得一模一样，唯一的差别是把 `N01`…`N15` 填进去两处。

**备胎链路**（N12–N15）平时不接流量，只在某条前段掉了才顶上。想省 14.5 GB 可以先不部署这 4 台——代价是池子没有余量，掉一条前段就直接降容量。

### 节点自查

```bash
tail -5 /tmp/agent-N07.log      # 应该只有一行 listening on 0.0.0.0:9101
ss -lntp | grep 9101
```

统一验在控制机做，不用逐台看：`./deploy_15.sh check`。

---

## 5. 测不同的 request

### 测试集：`task/cases.txt`

一行一条 prompt。制表符前是**真实 task**：

```
# 制表符前是真实 task，用来核对在线识别对不对
mbpp	Write a python function to reverse a linked list.
gsm8k	Natalia sold clips to 48 friends in April, and half as many in May...
```

标注**只用来评分，不影响派发**——请求发到哪条后段靠前段自己识别，那才是被测的东西。不写标注也行，按轮转指派真实 task。`#` 开头与空行跳过。

制表符前的字段只有**在 task 列表里认识**时才当成标注，否则整行都是 prompt——代码类 prompt 里制表符是缩进，不该被误读（`test_prompts_file.py` 守着这条）。

`--requests` 比条数多就循环重跑，少就只跑前几条。

### 跑

```bash
REQUESTS=40 TOKENS=64 RESULTS=results/cov70.json ./deploy_15.sh measure
./deploy_15.sh report results/cov70.json
```

`measure` 默认 `WARMUP=2`：先跑 2 条**不计入结果**的请求。torch 首次前向要选 kernel、分配显存池、把权重从 mmap 拉进页缓存——这些只发生一次，混进 p50 会把结果拖歪。

### 看

```
req    真实   识别      通道              总ms   排队  首tok  /tok   算力
req0   mbpp  mbpp  ✓  F0×Bmbpp0          33     0    15    3.5   41%
req1   gsm8k gsm8k ✓  F1×Bgsm8k0         33     0    14    3.1   34%
──────────────────────────────────────────────────────────────
  40 条：总时延 p50 25ms，逐 token p50 2.8ms，算力使用率均值 42.1%，识别 98%，换绑 1 次

  时延构成（均值）
    计算              10ms  41.0%  ████████████████
    排队               0ms   0.0%
    网络+协议+调度      15ms  59.0%  ████████████████████████

  各节点算力占比（占总时延，均值）
    N07   F0      层 2–10   121 专家  30.6%  ██████████████████
    N04   Bmbpp0  层 21–48  121 专家  13.1%  ████████
```

### 怎么读这张表

**「网络+协议+调度」是反推的上界，不是测量值。** 15 台没有时钟同步，跨机的绝对时刻拼不到一条轴上，所以它 = 总时延 − 各节点计算 − 排队。三段相加恒等于总时延，这张表不会骗人，但那一栏里混着协议开销和调度延迟，不能当纯网络时间读。

**算力使用率的分母是墙钟，不是「节点数 × 墙钟」。** 这条路上的节点是流水线，同一时刻只有一个在算（段内逐跳、跨段绕环）。所以这个比值的含义是「这条请求的生命周期里，有多少时间真的在做矩阵乘」——40% 不代表浪费了 60% 的卡。

**算力使用率低时先看各节点占比**，是不是某一跳独吞。层数分得不均（比如 N04 拿了 28 层而 N03 只有 9 层）会让流水线卡在长的那一段。

**识别准确率与换绑次数**是动态模式独有的。换绑 = 前段先识别错了、miss 报警后改派到另一个后段。换绑多说明分类器在这批 prompt 上不灵，可以调 `--eta` 或换更长的观测窗。

### 落盘的 JSON

`results/run.json` 里逐请求都有：时序四段拆解、每个 token 的耗时、各节点计算时长与发出字节、识别置信度与判定区、生成的完整文本。测完有一份可复算的底稿，不用回头翻滚屏日志。

---

## 5b. 上游拉不动怎么办

两种失败，含义完全不同：

| 症状 | 含义 |
|---|---|
| `SSL: UNEXPECTED_EOF_WHILE_READING`<br>`Connection reset` | 链路在传输中途被掐断 —— 几 KB 的文件过得去，11MB 的 `tokenizer.json` 过不去 |
| `RangeNotSupported` | 源不支持 Range 请求 —— 「只拉自己那份张量」这件事做不成 |

### 掐断

`Source.read` **会按已收字节续传** —— 断在 8MB 就从 8MB 接着要，不是从头重来。
每次只肯传几百字节的链路也能爬完（实测 137 字节一段，256KB 用 1869 次请求拼齐，
逐字节一致）。

续传之后还是失败，说明每次都断在很早。**这通常不是 HF 的问题**，先复现一次：

```bash
curl -v -o /dev/null https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/resolve/main/config.json
```

三个常见原因：

- **TLS 检查型代理 / 防火墙**掐长连接。查 `$https_proxy` / `$HTTPS_PROXY`，
  或者让 IT 放行 `huggingface.co` 与 `cdn-lfs.huggingface.co`。
- **MTU 黑洞**（小包过、大包丢，VPN / 隧道上常见）。试
  `sudo ip link set dev <网卡> mtu 1400` 再跑。
- **出口 CDN 抖动**。换个时间，或 `--endpoint` 指到别的镜像。

私有/受限仓库另说 —— 那要 `HF_TOKEN`（`hf auth login` 之后在
`~/.cache/huggingface/token`）。

### 绕过去：下一次全量，局域网分发

在**一台**能稳定联网的机器上拉一份完整 checkpoint：

```bash
hf download Qwen/Qwen3-Next-80B-A3B-Instruct --local-dir ./ckpt
```

`hf download`（`pip install -U huggingface_hub`）比 git 稳：**能断点续传**，
断了重跑接着下，不用从头来。在会掐断的链路上这是决定性的差别。

用 git 的话记住 `GIT_LFS_SKIP_SMUDGE=1` **只做了一半**：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
cd Qwen3-Next-80B-A3B-Instruct
git lfs pull          # ← 这一步才是真正下权重
```

只跑第一条的话，41 个 `.safetensors` 都在、名字对、数量对，但**每个只有 130 字节
的指针文本**。`serve-weights` 和 `fetch` 都会认出来并点名（否则错会以
`头长 1936026161 不合理` 的形式出现在解析文件头那一步，离真正的原因隔了三层）。

起一个**支持 Range 的**权重源：

```bash
bash ./deploy_15.sh serve-weights ./ckpt
```

另开终端：

```bash
SRC_URL=http://<那台的IP>:9400 bash ./deploy_15.sh fetch
```

合计仍然只拉 141GB，只是源站换成了自己人，走局域网。

**不能用 `python -m http.server`** —— 实测它无视 Range 头，返回 200 加整个文件。
那样每台会拿到整个分片（`fetch` 认得出来并报 `RangeNotSupported`，不会将就 ——
按区间长度去切一个整文件会得到一份内容错位的权重）。

有 NFS / 共享盘就连服务都不用起：

```bash
SRC_DIR=/mnt/shared/ckpt bash ./deploy_15.sh fetch
```

各节点直接 seek 本地文件，连 HTTP 都省了。

---

## 6. 横向对比

```bash
# 换覆盖率
for c in 0.70 0.80 0.90; do
  COVERAGE=$c RESULTS=results/cov$c.json ./deploy_15.sh measure
done

# 换 miss 策略
./deploy_15.sh measure --miss-policy drop_noscale
./deploy_15.sh measure --miss-policy local_topk
```

`serve` / `measure` 后面多写的参数**原样透传给 `control.py`**，argparse 取最后一次
出现，所以透传的值会盖掉脚本里设的默认值：

```bash
./deploy_15.sh measure --coverage 0.60 --tasks mbpp=4,gsm8k=4

# 换并发
./deploy_15.sh measure --concurrency 1     # 每条 path 只跑一个 request
```

三种 miss 策略——后段没装全被路由到的专家时怎么办：

| | 做法 |
|---|---|
| `drop` | 丢掉缺的，门控权重重归一 |
| `drop_noscale` | 丢掉缺的，**不**重归一（实测更好） |
| `local_topk` | 在驻留集里重新取 top-k |

**改 `COVERAGE` 是真的改了后段驻留集**，每轮会重新装载专家，那几分钟躲不掉。只改 `cases.txt` 和 `REQUESTS` 的话重跑就快得多。

---

## 7. 改了代码之后

`start` 只是 `cd $WORKDIR && python -m ...`，**它不会推代码**：

```bash
./deploy_15.sh stop && ./deploy_15.sh sync && ./deploy_15.sh start
```

漏了 `sync` 的话跑的还是旧代码，**而且不会报错**——这是最容易踩的坑。

没 ssh 的话重跑 `bootstrap` + 各节点重粘一次命令（权重已经在本地，`fetch` 会跳过已有的）。

---

## 8. 常见问题

**`PermissionError: /data`** —— 系统目录要 root。`export WEIGHTS=$WORKDIR/weights`。

**`xxx/config.json 不存在`** —— 控制机也要这个目录，但只要里面的
`config.json` 与 tokenizer（约 10MB），**不要权重**。跑 `./deploy_15.sh meta`。

**某个 task 分到 0 条后段** —— 规划器按内存与跳数约束分**整数条**通道，配额不够时
会给某个 task 分到 0。这是离散配平的正常结果。`serve` 现在会在开跑前拦住并点名。
降 `--coverage`、调 `--tasks` 的到达比、或加节点。注意覆盖率与通道数**不是单调的**
（段的组成是离散的、跳数是整数），0.60 未必比 0.70 建得多 —— 挨个试比推理快。

**`✗ 必须设 ADVERTISE`** —— 见 §2。

**`sync` 报「同一个目录」** —— 控制机自己就是这台，而且就坐在 `$WORKDIR` 里。跳过 rsync 是对的（对着自己跑 `--delete` 会咬人），依赖照装。

**`规划失败: 公共中值域人口 0`** —— 节点之间延迟差异太小（比如全在一台机器上模拟），公共中值域退化。真机上不会遇到；本地演练加 `--mem-cap-mb`。

**`没有 L₀ 同时满足 p ≥ 0.8…`** —— 内存上限太紧，前段单节点装不下。真机上 24 GB 卡装 L₀=11 的前段（14.5 GB）是够的。

**agent 起来就死（`start` 报 ✓ 但 `check` 全部不可达）** —— PID 只说明 fork 成功了。
跑 `./deploy_15.sh doctor` 逐台体检。三种典型：

- **日志不存在** → `cd $WORKDIR` 就失败了，代码不在那儿。`./deploy_15.sh sync`
- **`No module named 'p2pmoe'`** → 目录在但代码没进去。同上
- **`No module named 'torch'`** → `NODE_PY` 指错了解释器。`doctor` 会找出正确的路径

`doctor` 里 ssh 失败与远端某项缺失是**分开报**的 —— 它们看起来一样，修法相反。

**`节点上报的错误`** —— 通常是权重缺张量。让那台重跑一次 `fetch`（它会跳过已有的，只补缺的）。

**请求超时** —— `report` 里看 `missing_traces`：哪台没上报埋点。没上报的节点，它的计算时间会被算进「网络」那一栏。

---

## 9. 都在 NAT 后面

节点之间两两连不通时，起一台中继：

```bash
# 一台双方都能连到的机器上
python3 -m p2pmoe.deploy.relay --bind 0.0.0.0:9200
```

各节点的 agent 和控制机的 control 都加 `--relay <中继IP>:9200`。

中继握手完就**只搬字节，不认识 p2pmoe 的协议**——上层完全分辨不出中间隔了一台机器（`test_relay.py` 测的就是这件事）。代价是每跳绕一圈，逐 token 延迟大致翻倍。

---

## 附：命令速查

| 命令 | 在哪跑 | 做什么 |
|---|---|---|
| `./deploy_15.sh sync` | 控制机 | rsync 代码 + 装依赖到 15 台（要 ssh） |
| `./deploy_15.sh bootstrap` | 控制机 | 起 HTTP 服务发代码（不要 ssh） |
| `./deploy_15.sh cmds [节点]` | 控制机 | 打印每台该跑的命令 |
| `./deploy_15.sh check` | 控制机 | ssh 可达 + agent 在线 + 两两可达 |
| `./deploy_15.sh fetch` | 控制机 | 各节点只拉自己那份权重 |
| `./deploy_15.sh meta` | 控制机 | 取 config + tokenizer（10MB，不含权重） |
| `./deploy_15.sh serve-weights <目录>` | 有全量的那台 | 起一个支持 Range 的局域网权重源 |
| `./deploy_15.sh start` | 控制机 | 起 15 个 agent |
| `./deploy_15.sh serve` | 控制机 | 建链 + 打请求 |
| `./deploy_15.sh measure` | 控制机 | 同上 + 预热 + 逐请求时序 |
| `./deploy_15.sh report [文件]` | 控制机 | 把结果 JSON 读成表 |
| `./deploy_15.sh logs [节点]` | 控制机 | 拉 agent 日志 |
| `./deploy_15.sh doctor [节点]` | 控制机 | 逐台体检：代码/torch/权重/端口/日志 |
| `./deploy_15.sh stop` | 控制机 | 停 15 个 agent |

配置项（环境变量覆盖，或改脚本顶部）：

```
HOSTS=task/hosts.txt          PLAN=task/plan_deploy.json
PROFILE=task/profile_real.json  REPO=Qwen/Qwen3-Next-80B-A3B-Instruct
WEIGHTS=$WORKDIR/weights        WORKDIR=/home/ubuntu/P2P_MoE
SRC_DIR=  SRC_URL=  MODE=slice  FETCH_TIMEOUT=14400  WSRC_PORT=9400
NODE_PY=<节点上带 torch 的解释器>  CONDA_ENV=moe
ADVERTISE=<必填>               DEVICE=cuda:0
COVERAGE=0.70                  TASKS=mbpp=5,gsm8k=3
TOKENS=64  REQUESTS=20  WARMUP=2
PROMPTS=task/cases.txt         RESULTS=results/run.json
SSH_USER=  SSH_OPTS=  BOOT_PORT=9300  HF_ENDPOINT=
```

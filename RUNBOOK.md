# 15 节点部署 Qwen3-30B-A3B：全部指令

照着从上往下执行。每条都标了**在哪台机器上跑**。
解释性内容在 [DEPLOY.md](DEPLOY.md)，这里只有命令。

约定：15 台节点叫 `n1`…`n15`，控制机另算（可以是其中一台，也可以是第 16 台）。

---

## 0. 先确认连通性

| 方向 | 端口 | 必须吗 |
|---|---|---|
| **节点 ↔ 节点（两两）** | TCP 9101 | **必须** |
| 控制机 ↔ 节点 | TCP 9101 + 协调器随机端口 | **必须** |
| 节点 → huggingface.co | TCP 443 | 拉权重时 |
| 控制机 → 节点 SSH | 22 | 不必须（见 §2b） |

节点之间**不需要 SSH**。但两两 TCP 必须通 —— 数据面是节点直接互发，控制机不在中间。
都在 NAT 后面连不通的话见 §7。

---

## 1. 每台节点：装依赖 + 放代码

```bash
# —— 在 n1…n15 每台上 ——
pip3 install -r requirements-node.txt      # numpy + torch + safetensors
mkdir -p /opt/p2pmoe && cd /opt/p2pmoe     # 代码放这里
# 代码用 rsync 或 git clone 弄过去，不需要 pip install -e .
```

```bash
# —— 控制机 ——
pip3 install -r requirements-control.txt   # numpy + tokenizers + jinja2
cd /opt/p2pmoe
cat > hosts.txt <<'EOF'
n1   10.0.0.11:9101
n2   10.0.0.12:9101
n3   10.0.0.13:9101
n4   10.0.0.14:9101
n5   10.0.0.15:9101
n6   10.0.0.16:9101
n7   10.0.0.17:9101
n8   10.0.0.18:9101
n9   10.0.0.19:9101
n10  10.0.0.20:9101
n11  10.0.0.21:9101
n12  10.0.0.22:9101
n13  10.0.0.23:9101
n14  10.0.0.24:9101
n15  10.0.0.25:9101
EOF
```

---

## 2a. 起 agent（控制机能 SSH 到节点时）

```bash
# —— 控制机 ——
python3 -m p2pmoe.deploy.launch start  --hosts hosts.txt --workdir /opt/p2pmoe
python3 -m p2pmoe.deploy.launch status --hosts hosts.txt      # 应显示 15/15 在线
python3 -m p2pmoe.deploy.launch probe  --hosts hosts.txt --k 8  # 两两可达性
```

## 2b. 起 agent（没有 SSH）

```bash
# —— 控制机：打印各节点该跑的命令，自己贴过去 ——
python3 -m p2pmoe.deploy.launch start --hosts hosts.txt --workdir /opt/p2pmoe --dry-run
```

或者做成 systemd（推荐，开机自启 + 崩溃重启）：

```bash
# —— 控制机：生成 unit（每台改 --id）——
python3 -m p2pmoe.deploy.launch systemd --id n1 --workdir /opt/p2pmoe --user p2pmoe

# —— 每台节点 ——
# 把上面的输出写到 /etc/systemd/system/p2pmoe-agent.service，然后：
systemctl daemon-reload && systemctl enable --now p2pmoe-agent
```

agent 是**无状态**的，崩溃重启后回到未配置态等控制器重新下发，所以 `Restart=always` 安全。

---

## 3. 第一轮：2 条通道全装，采激活画像

> 为什么不能直接上 4 条：4 前段只剩 11 台分后段，全装每台要 26GB，放不下。
> 而采画像**必须**全装 —— 只驻留子集时输出被 drop-expert 带偏，采到的不是真实路由。

```bash
# —— 控制机 ——
cat > profile.json <<'EOF'
{
  "model_dir": "/data/qwen3-part",
  "l0": 6,
  "channels": [
    {"front": "n1", "back": ["n2",  "n3",  "n4",  "n5",  "n6",  "n7"]},
    {"front": "n8", "back": ["n9", "n10", "n11", "n12", "n13", "n14", "n15"]}
  ]
}
EOF

# 3.1 布局 → 清单（不连机器、不要权重）
python3 -m p2pmoe.deploy.run --spec profile.json --plan-only --save-plan p1.json

# 3.2 各节点只拉自己那部分权重（并行，各自直连 HF）
python3 -m p2pmoe.deploy.launch fetch --hosts hosts.txt --plan p1.json \
        --repo Qwen/Qwen3-30B-A3B --out /data/qwen3-part
#   先看会下多少：加 --dry-run
#   国内镜像：加 --endpoint https://hf-mirror.com
#   镜像不支持 Range：加 --mode shard（省得少但不依赖 Range）

# 3.3 跑真实请求，采画像
python3 -m p2pmoe.deploy.run --spec profile.json \
    --agents "$(python3 -m p2pmoe.deploy.launch agents --hosts hosts.txt)" \
    --advertise <控制机对节点可见的IP> --device cuda:0 --ctx 2048 \
    --chat --prompt "<你的真实请求 1>" --prompt "<你的真实请求 2>" \
    --tokens 128 --profile-out prof.json --once
```

画像存在 `prof.json`。**采样量越大越稳** —— 多给几条 `--prompt`，或多跑几次
（每次换 `--profile-out`，事后合并）。

---

## 4. 挑覆盖率

```bash
# —— 控制机（需要 pip3 install transformers）——
python examples/drop_expert_impact.py --model-dir /data/qwen3-part \
    --profile prof.json --coverage 0.90 0.95 0.99 \
    --policy drop drop_noscale local_topk
```

看 **KL** 挑覆盖率（不是看 miss 率）。同时会告诉你三种 `--miss-policy` 哪个好。

---

## 5. 第二轮：4 前段 + 4 后段，正式部署

```bash
# —— 控制机 ——
cat > deploy.json <<'EOF'
{
  "model_dir": "/data/qwen3-part",
  "l0": 6,
  "channels": [
    {"front": "n1",  "back": ["n2",  "n3",  "n4"]},
    {"front": "n5",  "back": ["n6",  "n7",  "n8"]},
    {"front": "n9",  "back": ["n10", "n11", "n12"]},
    {"front": "n13", "back": ["n14", "n15"]}
  ]
}
EOF

# 5.1 新清单
python3 -m p2pmoe.deploy.run --spec deploy.json --plan-only --save-plan p2.json

# 5.2 重拉权重（驻留集变了，不重拉会报「缺 N 个 key」）
python3 -m p2pmoe.deploy.launch fetch --hosts hosts.txt --plan p2.json \
        --repo Qwen/Qwen3-30B-A3B --out /data/qwen3-part

# 5.3 上线
python3 -m p2pmoe.deploy.run --spec deploy.json \
    --agents "$(python3 -m p2pmoe.deploy.launch agents --hosts hosts.txt)" \
    --advertise <控制机IP> --device cuda:0 --ctx 2048 \
    --profile prof.json --coverage 0.95 --miss-policy drop \
    --chat --prompt "用一句话解释 MoE 的稀疏激活" --tokens 64 --once
```

去掉 `--once` 就常驻。

---

## 6. 量时延与算力使用率

```bash
python3 -m p2pmoe.deploy.run --spec deploy.json \
    --agents "$(python3 -m p2pmoe.deploy.launch agents --hosts hosts.txt)" \
    --advertise <控制机IP> --device cuda:0 --ctx 2048 \
    --profile prof.json --coverage 0.95 \
    --chat --prompt "用一句话解释 MoE 的稀疏激活" --tokens 64 \
    --warmup 2 --timing --requests 1 --once
```

**`--warmup 2` 不是可选的**：torch 首次前向含 kernel 选择与惰性初始化，
不预热的话首请求量到的大半是冷启动开销。

---

## 7. 节点之间连不通时（都在 NAT 后面）

```bash
# —— 一台有公网入站的机器（控制机通常就是）——
python3 -m p2pmoe.deploy.relay --bind 0.0.0.0:9200

# —— 每台节点：不监听端口，改成挂到中继上 ——
python3 -m p2pmoe.deploy.agent --id n3 --relay <中继IP>:9200

# —— 控制机：所有 run/control 命令加同一个 --relay ——
python3 -m p2pmoe.deploy.run --spec deploy.json --relay <中继IP>:9200 ...
```

代价：每跳绕一圈，逐 token 延迟大致翻倍；中继是带宽瓶颈与单点。
**能上 WireGuard / Tailscale 就上**，那样恢复真直连，框架什么都不用改。

---

## 8. 停

```bash
python3 -m p2pmoe.deploy.launch stop --hosts hosts.txt        # 有 SSH
python3 -m p2pmoe.deploy.launch stop --hosts hosts.txt --dry-run   # 没 SSH，打印命令
```

---

## 内存账（Qwen3-30B-A3B，ctx=2048，bf16）

全模型 61.0GB；每层 attention 38MB + 128 × 专家 9.4MB。

| 布局 | 前段/台 | 后段最大/台 | 合计下载 |
|---|---|---|---|
| 2 通道（后段 6/7 台）全装 | 7.5GB | **8.7GB** ✓ | 239GB |
| 3 通道（后段各 4 台）全装 | 7.5GB | **13.7GB** ✓ | 239GB |
| 4 通道（后段各 3/2 台）全装 | 7.5GB | **26.2GB** ✗ | 239GB |
| 4 通道，后段 20% 专家 | 7.5GB | **6.0GB** ✓ | 78GB |

前段装全部 128 个专家（task 无关，L₀=6）；后段按画像裁。
**4 通道必须裁后段** —— 这就是为什么要先用 2 通道采画像。

---

## 常见故障

| 症状 | 原因 | 处置 |
|---|---|---|
| 预检「读不到 checkpoint」 | `--model-dir` 是**各节点本地路径**，控制机能读不代表节点能 | 跑 §3.2 的 fetch，或挂共享存储 |
| 加载报「缺 N 个 key」 | 驻留集变了但没重拉权重 | 用新 plan 重跑 fetch |
| fetch 报「不支持 Range 请求」 | 镜像把整文件发回来了 | 换 `--mode shard`，或换支持 Range 的源 |
| 「没装 torch」 | 节点装成了 `requirements.txt` | 装 `requirements-node.txt` |
| 「没装 tokenizers」 | 控制机少装 | 装 `requirements-control.txt` |
| probe 大面积不可达 | 节点之间没放行 | 放行 9101 两两；实在不通走 §7 中继 |
| 首 token 特别慢 | torch 冷启动 | 加 `--warmup 2` |

#!/usr/bin/env bash
# ============================================================================
# 在 15 个真实节点上部署 Qwen3-Next-80B-A3B
#
#   ./deploy_15.sh sync       # 有 ssh：推代码 + 装依赖到 15 台
#   ./deploy_15.sh bootstrap  # 没 ssh：起个 HTTP 服务，打印每台该跑的一条命令
#   ./deploy_15.sh cmds       # 逐台打印它自己该跑的命令（层区间/拉取量各不同）
#   ./deploy_15.sh check      # 只检查连通性与依赖，不动任何东西
#   ./deploy_15.sh fetch      # 各节点只拉自己那份权重（合计 141GB，不是 2400GB）
#   ./deploy_15.sh start      # 起 agent
#   ./deploy_15.sh serve      # 建链 + 打请求（动态模式：随机到达、在线识别）
#   ./deploy_15.sh measure    # 同上，外加逐请求时序与算力使用率
#   ./deploy_15.sh stop
#   ./deploy_15.sh all        # sync → check → fetch → start → serve
#
# 全部命令都在**控制机这一台**上跑。控制机可以是这 15 台里的任意一台，也可以是
# 第 16 台 —— 它只需要能 ssh 到各节点、且各节点能回连它（--advertise）。
#
# 但代码与依赖**不会自己长到节点上**：launch start 只是 `cd $WORKDIR && python -m …`。
# 所以第一次部署、或改了代码之后，都要先 sync。
#
# 放置来自 plan_deploy.json（离线用真实激活数据 + 真实拓扑算出来的）：
#   L₀=11，前段 3 条 + 备胎 1 条，后段 mbpp 2 条 + gsm8k 1 条
#   到达比 mbpp:gsm8k = 5:3，负载/产能 0.94×（基本均衡）
#   覆盖率 0.70：mbpp 层均 121/512、gsm8k 层均 98/512
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ---- 改这里 ----------------------------------------------------------------
HOSTS=${HOSTS:-task/hosts.txt}                 # 节点 id → IP:端口
PLAN=${PLAN:-task/plan_deploy.json}            # 离线算好的部署清单
PROFILE=${PROFILE:-task/profile_real.json}     # 激活画像（动态模式的分类器要它）
REPO=${REPO:-Qwen/Qwen3-Next-80B-A3B-Instruct}
WEIGHTS=${WEIGHTS:-/data/qwen3-next-part}      # 各节点上的本地路径（**各机相同**）
WORKDIR=${WORKDIR:-/opt/p2pmoe}                # 各节点上的代码目录
ADVERTISE=${ADVERTISE:-}                       # 控制机对节点可见的 IP —— 必填
DEVICE=${DEVICE:-cuda:0}
COVERAGE=${COVERAGE:-0.70}
TASKS=${TASKS:-mbpp=5,gsm8k=3}
TOKENS=${TOKENS:-64}
REQUESTS=${REQUESTS:-20}
# ---------------------------------------------------------------------------

PY=${PY:-python3}
# ssh 相关：SSH_USER=ubuntu 或 SSH_OPTS="-i ~/.ssh/pool.pem -p 2222"
SSHARG=()
[ -n "${SSH_USER:-}" ] && SSHARG+=(--user "$SSH_USER")
[ -n "${SSH_OPTS:-}" ] && SSHARG+=(--ssh "ssh -o BatchMode=yes -o ConnectTimeout=10 $SSH_OPTS")
AGENTS() { $PY -m p2pmoe.deploy.launch agents --hosts "$HOSTS"; }
say() { printf '\n\033[1m══ %s\033[0m\n' "$*"; }
need() { [ -f "$1" ] || { echo "缺文件：$1"; exit 1; }; }

# 没有 ssh 时的引导：把代码打包、用 HTTP 发出去，每台节点自己拉。
#
# 为什么只有这一步需要特殊处理：**agent 起来之后整个控制面都是 TCP** ——
# 采集能力、下发清单、打请求、收上报，全走 9101 与协调器端口。ssh 从头到尾
# 只被 launch 的批量脚本用，而那是便利，不是依赖。
#
# 节点要能连到控制机的 $BOOT_PORT（它们本来就要能回连协调器，所以这个前提已经有了）。
cmd_bootstrap() {
  local port=${BOOT_PORT:-9300}
  local ip=${ADVERTISE:-<控制机IP>}
  local tmp; tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  say "B. 引导（无 ssh）"
  tar czf "$tmp/p2pmoe.tar.gz" \
      --exclude '__pycache__' --exclude '.git' --exclude '*.pyc' \
      --exclude '.pytest_cache' --exclude '.*-logs' --exclude 'task/*.csv' .
  cp "$PLAN" "$tmp/plan.json"
  echo "  代码包 $(du -h "$tmp/p2pmoe.tar.gz" | cut -f1)，清单 $(du -h "$tmp/plan.json" | cut -f1)"
  echo
  echo "  ── 在**每台节点**上跑这一条（把 <ID> 换成该机的节点 id）─────────────"
  echo "  ── 每台各自的层区间与拉取量不同，逐台展开见：./deploy_15.sh cmds ──"
  cat <<EOS

  ID=<ID>   # N01 … N15
  mkdir -p $WORKDIR && cd $WORKDIR \\
    && curl -fsSL http://$ip:$port/p2pmoe.tar.gz | tar xz \\
    && curl -fsSL http://$ip:$port/plan.json -o /tmp/plan.json \\
    && $PY -m pip install -q -r requirements-node.txt \\
    && $PY -m p2pmoe.deploy.fetch --plan /tmp/plan.json --node \$ID \\
         --repo $REPO --out $WEIGHTS \\
    && setsid nohup $PY -m p2pmoe.deploy.agent --id \$ID --bind 0.0.0.0:9101 \\
         > /tmp/agent-\$ID.log 2>&1 &

EOS
  echo "  ──────────────────────────────────────────────────────────────────"
  echo "  一条命令四件事：拉代码 → 装依赖 → **只拉本机那份权重** → 起 agent。"
  echo "  15 台各自跑完之后，回控制机："
  echo "      ./deploy_15.sh check     # 确认 15/15 在线、两两可达"
  echo "      ./deploy_15.sh serve     # 之后全程 TCP，不再需要 ssh"
  echo
  echo "  HTTP 服务在 $ip:$port —— **Ctrl-C 结束**（等 15 台都拉完再关）"
  cd "$tmp" && exec $PY -m http.server "$port"
}

# 打印每台节点各自要跑的命令。每台干的事不一样 —— 层区间、拉多少权重、
# 在段里排第几 —— 所以这里按清单逐台展开，而不是给一条通用命令让人自己填。
#
#   ./deploy_15.sh cmds        # 15 台全打出来
#   ./deploy_15.sh cmds N07    # 只看某一台
cmd_cmds() {
  need "$PLAN"; need "$HOSTS"
  local only=${1:-}
  local port=${BOOT_PORT:-9300}
  local ip=${ADVERTISE:-<控制机IP>}
  $PY - "$PLAN" "$HOSTS" "$only" "$ip" "$port" "$WORKDIR" "$WEIGHTS" "$REPO" "$PY" <<'PYEOF'
import json, sys
plan, hostsf, only, ip, port, workdir, weights, repo, py = sys.argv[1:10]
m = json.load(open(plan))

addr = {}
for line in open(hostsf, encoding="utf-8"):
    line = line.split("#")[0].split()
    if len(line) >= 2:
        addr[line[0]] = line[1]

where = {}
for sid, sg in m["segments"].items():
    for i, n in enumerate(sg["nodes"]):
        where[n] = (sid, i, len(sg["nodes"]), sg["role"], sg.get("task"))

B, R, D = "\033[1m", "\033[0m", "\033[2m"
nodes = sorted(m["nodes"], key=lambda x: x["node"])
if only:
    nodes = [n for n in nodes if n["node"] == only]
    if not nodes:
        print(f"清单里没有 {only}"); raise SystemExit(1)

for n in nodes:
    nid = n["node"]
    sid, i, tot, role, task = where[nid]
    lr = n.get("layer_range")
    pos = f"{i+1}/{tot}"
    tag = " ← 段首（接收请求）" if n.get("is_head") else \
          " ← 段尾（出 token / 交给后段）" if n.get("is_tail") else ""
    print(f"\n{B}┌─ {nid}  {addr.get(nid,'?')}{R}")
    print(f"{B}│{R}  段 {sid}（{role}{'/'+task if task else ''}），段内第 {pos}{tag}")
    print(f"{B}│{R}  层 {lr[0]}–{lr[1]}（{len(n.get('layers') or {})} 层），"
          f"要拉 {n.get('weight_gb',0):.1f} GB，占显存约 {n.get('total_gb',0):.1f} GB")
    print(f"{B}└─{R}")
    print(f"""
  mkdir -p {workdir} && cd {workdir} \\
    && curl -fsSL http://{ip}:{port}/p2pmoe.tar.gz | tar xz \\
    && curl -fsSL http://{ip}:{port}/plan.json -o /tmp/plan.json \\
    && {py} -m pip install -q -r requirements-node.txt \\
    && {py} -m p2pmoe.deploy.fetch --plan /tmp/plan.json --node {nid} \\
         --repo {repo} --out {weights} \\
    && setsid nohup {py} -m p2pmoe.deploy.agent --id {nid} --bind 0.0.0.0:9101 \\
         > /tmp/agent-{nid}.log 2>&1 &
""")

if not only:
    gb = sum(n.get("weight_gb", 0) for n in m["nodes"])
    print(f"{D}  ── 15 台合计拉 {gb:.0f} GB（全模型 160GB × 15 = 2400GB）。"
          f"各台互不依赖，可以同时跑。{R}")
    print(f"{D}  ── 都起来之后回控制机：./deploy_15.sh check 然后 ./deploy_15.sh measure{R}")
PYEOF
}

# 代码与依赖不会自己长到节点上 —— 先把它们送过去。
cmd_sync() {
  say "S. 推代码 + 装依赖到 15 台"
  command -v rsync >/dev/null || { echo "✗ 控制机没有 rsync"; exit 1; }
  local hosts
  hosts=$(awk '{sub(/#.*/,"")} NF>=2 {split($2,a,":"); print a[1]}' "$HOSTS")
  for h in $hosts; do
    echo "  → $h"
    ssh -o BatchMode=yes ${SSH_OPTS:-} "${SSH_USER:+$SSH_USER@}$h" \
        "mkdir -p '$WORKDIR' '$WEIGHTS'" </dev/null
    rsync -az --delete ${SSH_OPTS:+-e "ssh $SSH_OPTS"} \
      --exclude '__pycache__' --exclude '.git' --exclude 'task' \
      --exclude '*.pyc' --exclude '.pytest_cache' --exclude '.*-logs' \
      ./ "${SSH_USER:+$SSH_USER@}$h:$WORKDIR/"
    ssh -o BatchMode=yes ${SSH_OPTS:-} "${SSH_USER:+$SSH_USER@}$h" \
        "cd '$WORKDIR' && $PY -m pip install -q -r requirements-node.txt" </dev/null
  done
  echo "  ✓ 15 台都有代码与 torch/safetensors"
  echo "  （没 ssh：手工 rsync 到各机 $WORKDIR，再各自 pip install -r requirements-node.txt）"
}

cmd_check() {
  say "0. 前置检查"
  need "$HOSTS"; need "$PLAN"; need "$PROFILE"
  [ -n "$ADVERTISE" ] || { echo "✗ 必须设 ADVERTISE=<控制机对节点可见的IP>"; exit 1; }
  $PY -c "import numpy, tokenizers, jinja2" 2>/dev/null \
    || { echo "✗ 控制机缺依赖：pip3 install -r requirements-control.txt"; exit 1; }
  echo "  控制机依赖 ✓（只要 numpy+tokenizers+jinja2，**不装 torch**）"
  $PY - "$PLAN" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
nodes = {n["node"] for n in m["nodes"]}
print(f"  清单：L₀={m['l0']}，{len(nodes)} 台，{len(m['segments'])} 条段")
for sid, s in sorted(m["segments"].items()):
    print(f"    {sid:<16}{s['role']:<16}{s['nodes']}")
PYEOF
  say "1. ssh 可达（BatchMode：必须免密，密码提示会挂住）"
  # 并行探 —— 串行 15 台 × 10s 超时要等两分半，而这只是个前置检查
  local d; d=$(mktemp -d)
  for h in $(awk '{sub(/#.*/,"")} NF>=2 {split($2,a,":"); print a[1]}' "$HOSTS"); do
    ( ssh -o BatchMode=yes -o ConnectTimeout=10 ${SSH_OPTS:-} \
          "${SSH_USER:+$SSH_USER@}$h" true </dev/null 2>/dev/null \
        && echo ok > "$d/$h" || echo no > "$d/$h" ) &
  done
  wait
  local bad=0
  for h in $(awk '{sub(/#.*/,"")} NF>=2 {split($2,a,":"); print a[1]}' "$HOSTS"); do
    if [ "$(cat "$d/$h" 2>/dev/null)" = ok ]; then printf '  ✓ %s\n' "$h"
    else printf '  ✗ %s  —— 免密没配好，或用户/端口不对\n' "$h"; bad=$((bad+1)); fi
  done
  rm -rf "$d"
  [ "$bad" -eq 0 ] || echo "  （$bad 台不通。改用 ./deploy_15.sh bootstrap 走无 ssh 那条路）"

  say "2. 节点在线与两两可达"
  $PY -m p2pmoe.deploy.launch status --hosts "$HOSTS" "${SSHARG[@]}" || true
  echo "  —— 数据面是节点**直接互发**，控制机不在中间。下面查两两可达："
  $PY -m p2pmoe.deploy.launch probe --hosts "$HOSTS" --k 8 "${SSHARG[@]}" || true
}

cmd_fetch() {
  say "3. 各节点只拉自己那份权重"
  echo "  合计约 141GB（全模型 160GB × 15 台 = 2400GB，省 94%）"
  echo "  先看会下多少："
  $PY -m p2pmoe.deploy.launch fetch --hosts "$HOSTS" --plan "$PLAN" \
      --repo "$REPO" --out "$WEIGHTS" "${SSHARG[@]}" --dry-run
  read -rp "  继续？[y/N] " a; [ "$a" = y ] || return 0
  $PY -m p2pmoe.deploy.launch fetch --hosts "$HOSTS" --plan "$PLAN" \
      --repo "$REPO" --out "$WEIGHTS" "${SSHARG[@]}" ${HF_ENDPOINT:+--endpoint "$HF_ENDPOINT"}
}

cmd_start() {
  say "4. 起 agent"
  $PY -m p2pmoe.deploy.launch start --hosts "$HOSTS" --workdir "$WORKDIR" "${SSHARG[@]}"
  sleep 5
  $PY -m p2pmoe.deploy.launch status --hosts "$HOSTS" "${SSHARG[@]}"
}

# 动态模式：请求随机到达任意前段 → 前段本地识别 task → 派发到该 task 的后段。
# **不加 --static** —— 静态配对下一条前段绑死一个 task，随机来的请求接不住。
cmd_serve() {
  say "5. 建链 + 服务（动态：盲绑 → 识别 → 派发）"
  $PY -m p2pmoe.deploy.control \
      --agents "$(AGENTS)" --advertise "$ADVERTISE" \
      --load-plan "$PLAN" \
      --model-dir "$WEIGHTS" --device "$DEVICE" --ctx 2048 \
      --profile "$PROFILE" --coverage "$COVERAGE" \
      --tasks "$TASKS" --chat \
      --tokens "$TOKENS" --requests "$REQUESTS" --concurrency 3 --once "$@"
}

cmd_measure() {
  say "5'. 服务 + 逐请求时序"
  echo "  --warmup 不是可选的：torch 首次前向含 kernel 选择，不预热量到的是冷启动"
  cmd_serve --verbose
}

cmd_stop() { say "停"; $PY -m p2pmoe.deploy.launch stop --hosts "$HOSTS" "${SSHARG[@]}"; }

case "${1:-all}" in
  sync)      cmd_sync ;;
  bootstrap) cmd_bootstrap ;;
  cmds)      cmd_cmds "${2:-}" ;;
  check)   cmd_check ;;
  fetch)   cmd_fetch ;;
  start)   cmd_start ;;
  serve)   cmd_serve ;;
  measure) cmd_measure ;;
  stop)    cmd_stop ;;
  all)     cmd_sync && cmd_check && cmd_fetch && cmd_start && cmd_serve ;;
  *) echo "用法: $0 {sync|bootstrap|cmds [节点]|check|fetch|start|serve|measure|stop|all}"; exit 1 ;;
esac

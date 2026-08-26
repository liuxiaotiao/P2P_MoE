#!/usr/bin/env bash
# ============================================================================
# 在 15 个真实节点上部署 Qwen3-Next-80B-A3B
#
#   ./deploy_15.sh check      # 只检查连通性与依赖，不动任何东西
#   ./deploy_15.sh fetch      # 各节点只拉自己那份权重（合计 141GB，不是 2400GB）
#   ./deploy_15.sh start      # 起 agent
#   ./deploy_15.sh serve      # 建链 + 打请求（动态模式：随机到达、在线识别）
#   ./deploy_15.sh measure    # 同上，外加逐请求时序与算力使用率
#   ./deploy_15.sh stop
#   ./deploy_15.sh all        # check → fetch → start → serve
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
AGENTS() { $PY -m p2pmoe.deploy.launch agents --hosts "$HOSTS"; }
say() { printf '\n\033[1m══ %s\033[0m\n' "$*"; }
need() { [ -f "$1" ] || { echo "缺文件：$1"; exit 1; }; }

cmd_check() {
  say "0. 前置检查"
  need "$HOSTS"; need "$PLAN"; need "$PROFILE"
  [ -n "$ADVERTISE" ] || { echo "✗ 必须设 ADVERTISE=<控制机对节点可见的IP>"; exit 1; }
  $PY -c "import numpy, tokenizers, jinja2" 2>/dev/null \
    || { echo "✗ 控制机缺依赖：pip3 install -r requirements-control.txt"; exit 1; }
  echo "  控制机依赖 ✓"
  $PY - "$PLAN" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
nodes = {n["node"] for n in m["nodes"]}
print(f"  清单：L₀={m['l0']}，{len(nodes)} 台，{len(m['segments'])} 条段")
for sid, s in sorted(m["segments"].items()):
    print(f"    {sid:<16}{s['role']:<16}{s['nodes']}")
PYEOF
  say "1. 节点在线与两两可达"
  $PY -m p2pmoe.deploy.launch status --hosts "$HOSTS" || true
  echo "  —— 数据面是节点**直接互发**，控制机不在中间。下面查两两可达："
  $PY -m p2pmoe.deploy.launch probe --hosts "$HOSTS" --k 8 || true
}

cmd_fetch() {
  say "2. 各节点只拉自己那份权重"
  echo "  合计约 141GB（全模型 160GB × 15 台 = 2400GB，省 94%）"
  echo "  先看会下多少："
  $PY -m p2pmoe.deploy.launch fetch --hosts "$HOSTS" --plan "$PLAN" \
      --repo "$REPO" --out "$WEIGHTS" --dry-run
  read -rp "  继续？[y/N] " a; [ "$a" = y ] || return 0
  $PY -m p2pmoe.deploy.launch fetch --hosts "$HOSTS" --plan "$PLAN" \
      --repo "$REPO" --out "$WEIGHTS" ${HF_ENDPOINT:+--endpoint "$HF_ENDPOINT"}
}

cmd_start() {
  say "3. 起 agent"
  $PY -m p2pmoe.deploy.launch start --hosts "$HOSTS" --workdir "$WORKDIR"
  sleep 5
  $PY -m p2pmoe.deploy.launch status --hosts "$HOSTS"
}

# 动态模式：请求随机到达任意前段 → 前段本地识别 task → 派发到该 task 的后段。
# **不加 --static** —— 静态配对下一条前段绑死一个 task，随机来的请求接不住。
cmd_serve() {
  say "4. 建链 + 服务（动态：盲绑 → 识别 → 派发）"
  $PY -m p2pmoe.deploy.control \
      --agents "$(AGENTS)" --advertise "$ADVERTISE" \
      --load-plan "$PLAN" \
      --model-dir "$WEIGHTS" --device "$DEVICE" --ctx 2048 \
      --profile "$PROFILE" --coverage "$COVERAGE" \
      --tasks "$TASKS" --chat \
      --tokens "$TOKENS" --requests "$REQUESTS" --concurrency 3 --once "$@"
}

cmd_measure() {
  say "4'. 服务 + 逐请求时序"
  echo "  --warmup 不是可选的：torch 首次前向含 kernel 选择，不预热量到的是冷启动"
  cmd_serve --verbose
}

cmd_stop() { say "停"; $PY -m p2pmoe.deploy.launch stop --hosts "$HOSTS"; }

case "${1:-all}" in
  check)   cmd_check ;;
  fetch)   cmd_fetch ;;
  start)   cmd_start ;;
  serve)   cmd_serve ;;
  measure) cmd_measure ;;
  stop)    cmd_stop ;;
  all)     cmd_check && cmd_fetch && cmd_start && cmd_serve ;;
  *) echo "用法: $0 {check|fetch|start|serve|measure|stop|all}"; exit 1 ;;
esac

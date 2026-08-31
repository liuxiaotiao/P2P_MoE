#!/usr/bin/env bash
# ============================================================================
# 在 15 个真实节点上部署 Qwen3-Next-80B-A3B
#
#   ./deploy_15.sh sync       # 有 ssh：推代码 + 装依赖到 15 台
#   ./deploy_15.sh bootstrap  # 没 ssh：起个 HTTP 服务，打印每台该跑的一条命令
#   ./deploy_15.sh cmds       # 逐台打印它自己该跑的命令（层区间/拉取量各不同）
#   ./deploy_15.sh check      # 只检查连通性与依赖，不动任何东西
#   ./deploy_15.sh fetch      # 各节点只拉自己那份权重（合计 141GB，不是 2400GB）
#   ./deploy_15.sh meta       # 控制机取 config+tokenizer（10MB，不含权重）
#   ./deploy_15.sh diag       # 源站诊断：大文件和小文件是不是两个域名
#   ./deploy_15.sh serve-weights <目录>  # 上游拉不动时：一台发，15 台切片
#   ./deploy_15.sh verify     # 逐台核对权重齐不齐（只读文件头，几秒）
#   ./deploy_15.sh whereis    # 权重在哪、内存被谁占着（目录空了但内存满时用）
#   ./deploy_15.sh start      # 起 agent
#   ./deploy_15.sh serve      # 建链 + 打请求（动态模式：随机到达、在线识别）
#   ./deploy_15.sh measure    # 同上，外加逐请求时序与算力使用率
#   ./deploy_15.sh report     # 把 results/run.json 读成一张人能看的表
#   ./deploy_15.sh logs [节点] # 拉 agent 日志
#   ./deploy_15.sh doctor     # 逐台体检：代码/torch/权重/端口/日志
#   ./deploy_15.sh stop [force]  # force 连占着端口的进程一起清
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
WORKDIR=${WORKDIR:-/home/ubuntu/P2P_MoE}       # 各节点上的代码目录
# 权重放哪儿（**各机相同的路径**）。默认在项目目录下 —— /data 之类的系统目录
# 通常要 root，而这套东西不该需要 root。
#
# 放在 $WORKDIR 里有个陷阱：`sync` 用 `rsync --delete`，源端没有的东西会被删掉，
# 而权重是各节点自己拉的、源端根本没有 —— 一次 sync 就能抹掉 141GB。
# 所以下面 sync 会**自动把它排除**（bootstrap 打包时同理）。
WEIGHTS=${WEIGHTS:-$WORKDIR/weights}
# 节点上跑 torch 的那个解释器。**和控制机的 PY 是两回事** —— 控制机不装 torch。
# torch 在 conda 环境里的话，这里要填**绝对路径**，不能只写 python3：
#     NODE_PY=/home/ubuntu/miniconda3/envs/moe/bin/python
# ssh 非交互 shell 不会执行 conda init，`conda activate` 和裸 `python3` 都找不到它。
# 不知道路径就跑 ./deploy_15.sh doctor —— 它会在各节点上找出来并告诉你填什么。
NODE_PY=${NODE_PY:-python3}
CONDA_ENV=${CONDA_ENV:-moe}                    # doctor 找 conda 环境时按这个名字找
ADVERTISE=${ADVERTISE:-}                       # 控制机对节点可见的 IP —— 必填
DEVICE=${DEVICE:-cuda:0}
COVERAGE=${COVERAGE:-0.70}
TASKS=${TASKS:-mbpp=5,gsm8k=3}
TOKENS=${TOKENS:-64}
REQUESTS=${REQUESTS:-20}
# 权重从哪儿来。三选一，优先级从上到下：
#   SRC_DIR   各节点都能读到的全量 checkpoint（NFS/共享盘）—— 走本地文件读，不用网
#   SRC_URL   局域网里的权重源 —— 一台跑 serve-weights，其余从它切片
#   （都不设） 直连 HF（HF_ENDPOINT 可换镜像）
# 上游拉不动时用前两个：先在一台机器上下一次全量，再局域网分发。
# 注意上游拉不动**通常不是 HF 的问题** —— 见 3a 预检失败时打印的三个常见原因。
SRC_DIR=${SRC_DIR:-}
SRC_URL=${SRC_URL:-}
# 取权重用哪条传输。auto = 先用 Python 的 urllib，失败一次就整轮改用 curl。
# 「curl 能下、Python 下不动」很常见 —— 两者用的不是同一套 TLS
# （conda 环境自带 OpenSSL，系统 curl 用系统的）。
TRANSPORT=${TRANSPORT:-auto}
MODE=${MODE:-slice}                             # slice=逐张量；shard=整分片（源站不支持 Range 时）
FETCH_TIMEOUT=${FETCH_TIMEOUT:-14400}           # 单节点下载超时（秒）
PROGRESS_EVERY=${PROGRESS_EVERY:-30}            # 下载期间多久报一次进度（0=关）
PROMPTS=${PROMPTS:-task/cases.txt}              # 测试集：一行一条 prompt
RESULTS=${RESULTS:-results/run.json}            # 逐请求结果落盘
WARMUP=${WARMUP:-2}                             # measure 前丢掉几条（冷启动）
# ---------------------------------------------------------------------------

PY=${PY:-python3}

# 权重目录在代码目录里面的话，同步与打包都必须绕开它。
# 这不是优化 —— rsync --delete 会真的把 141GB 删掉，tar 会真的把它打进包里。
WEIGHTS_REL=""
case "$WEIGHTS" in
  "$WORKDIR"/*) WEIGHTS_REL=${WEIGHTS#"$WORKDIR"/} ;;
esac

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
      --exclude '.pytest_cache' --exclude '.*-logs' --exclude 'task/*.csv' \
      --exclude 'results' ${WEIGHTS_REL:+--exclude "$WEIGHTS_REL"} .
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
    && $NODE_PY -m pip install -q -r requirements-node.txt \\
    && $NODE_PY -m p2pmoe.deploy.fetch --plan /tmp/plan.json --node \$ID \\
         --repo $REPO --out $WEIGHTS \\
    && setsid nohup $NODE_PY -m p2pmoe.deploy.agent --id \$ID --bind 0.0.0.0:9101 \\
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
  if [ -n "$WEIGHTS_REL" ]; then
    echo "  权重在 $WEIGHTS（代码目录里面）—— rsync 会排除 $WEIGHTS_REL/"
    echo "  否则 --delete 会把各节点自己拉的权重当成「源端没有」删掉"
  fi
  command -v rsync >/dev/null || { echo "✗ 控制机没有 rsync"; exit 1; }
  local hosts
  hosts=$(awk '{sub(/#.*/,"")} NF>=2 {split($2,a,":"); print a[1]}' "$HOSTS")
  # 控制机自己可能就是这 15 台之一，而且就坐在 $WORKDIR 里 ——
  # 那样 `rsync -az --delete ./ host:$WORKDIR/` 是对着自己跑，--delete 会咬人。
  # 放一个一次性标记，从远端看得见就说明是同一个目录，跳过 rsync。
  local stamp=".p2pmoe-syncid-$$"
  : > "$stamp"
  trap 'rm -f "$stamp"' RETURN

  for h in $hosts; do
    printf '  → %-16s' "$h"
    ssh -o BatchMode=yes ${SSH_OPTS:-} "${SSH_USER:+$SSH_USER@}$h" \
        "mkdir -p '$WORKDIR' '$WEIGHTS'" </dev/null
    if ssh -o BatchMode=yes ${SSH_OPTS:-} "${SSH_USER:+$SSH_USER@}$h" \
           "[ -e '$WORKDIR/$stamp' ]" </dev/null 2>/dev/null; then
      echo "同一个目录（控制机就在这台的 $WORKDIR）—— 跳过 rsync"
    else
      rsync -az --delete ${SSH_OPTS:+-e "ssh $SSH_OPTS"} \
        --exclude '__pycache__' --exclude '.git' --exclude 'task' \
        --exclude '*.pyc' --exclude '.pytest_cache' --exclude '.*-logs' \
        --exclude '.p2pmoe-syncid-*' --exclude 'results' \
        ${WEIGHTS_REL:+--exclude "$WEIGHTS_REL"} \
        ./ "${SSH_USER:+$SSH_USER@}$h:$WORKDIR/"
      printf '代码 ✓  '
    fi
    ssh -o BatchMode=yes ${SSH_OPTS:-} "${SSH_USER:+$SSH_USER@}$h" \
        "cd '$WORKDIR' && '$NODE_PY' -m pip install -q -r requirements-node.txt" </dev/null \
      && echo "依赖 ✓" || echo "依赖 ✗（$NODE_PY 装不上，跑 ./deploy_15.sh doctor 看看）"
  done
  # task/ 整个被排除（里面有几百 MB 的 CSV），但清单必须过去 ——
  # 节点上的 `fetch --plan task/plan_deploy.json` 要读它。
  echo "  推清单 $PLAN"
  for h in $hosts; do
    ssh -o BatchMode=yes ${SSH_OPTS:-} "${SSH_USER:+$SSH_USER@}$h" \
        "mkdir -p '$WORKDIR/$(dirname "$PLAN")'" </dev/null
    rsync -az ${SSH_OPTS:+-e "ssh $SSH_OPTS"} \
      "$PLAN" "${SSH_USER:+$SSH_USER@}$h:$WORKDIR/$PLAN" 2>/dev/null || true
  done
  echo "  ✓ 15 台都有代码、清单与 torch/safetensors（用 $NODE_PY 装的）"
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

  if [ ! -f "$WEIGHTS/config.json" ]; then
    echo "  ✗ 控制机缺 $WEIGHTS/config.json —— 跑 ./deploy_15.sh meta（约 10MB）"
    echo "    （控制机只要 config 与 tokenizer，不要权重）"
  else
    echo "  控制机元数据 ✓ $WEIGHTS/config.json"
  fi

  say "2. 节点在线与两两可达"
  $PY -m p2pmoe.deploy.launch status --hosts "$HOSTS" "${SSHARG[@]}" || true
  echo "  —— 数据面是节点**直接互发**，控制机不在中间。下面查两两可达："
  $PY -m p2pmoe.deploy.launch probe --hosts "$HOSTS" --k 8 "${SSHARG[@]}" || true
}

# 开跑之前先确认 NODE_PY 指对了。
#
# 默认值是 `python3`，而 torch 装在 conda 环境里时那是系统 python ——
# 看不见环境里的包。这个错要到「已经 ssh 过去、已经开始跑」才暴露，
# 而那时的报错（No module named 'numpy'）看着像代码没同步，不像解释器选错。
_require_node_py() {
  local first firstip
  first=$(awk '{sub(/#.*/,"")} NF>=2 {print $1; exit}' "$HOSTS")
  firstip=$(awk '{sub(/#.*/,"")} NF>=2 {split($2,a,":"); print a[1]; exit}' "$HOSTS")
  [ -n "$firstip" ] || return 0
  local out rc
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 ${SSH_OPTS:-} \
          "${SSH_USER:+$SSH_USER@}$firstip" \
          "$(printf %q "$NODE_PY") -c 'import numpy,torch,safetensors;\
print(\"ok\", numpy.__version__, torch.__version__)'" \
          </dev/null 2>&1) && rc=0 || rc=$?
  if [ "$rc" = 0 ]; then
    echo "  NODE_PY ✓ $NODE_PY —— ${out}（在 $first 上验的）"
    return 0
  fi
  echo "  ✗ NODE_PY 用不了：$NODE_PY"
  echo "    在 $first 上跑 \`\$NODE_PY -c 'import numpy,torch'\` 得到："
  echo "$out" | head -3 | sed 's/^/      /'
  case "$NODE_PY" in
    python3|python)
      echo "    这是**系统 python**。torch 在 conda 环境里的话它看不见 ——"
      echo "    ssh 起的是非交互 shell，不执行 conda init，"
      echo "    \`conda activate\` 在那里不存在。要填绝对路径：" ;;
    *) echo "    这个解释器里缺依赖。换一个，或在它里面装：" ;;
  esac
  echo
  echo "        export NODE_PY=/home/ubuntu/anaconda3/envs/${CONDA_ENV:-moe}/bin/python"
  echo
  echo "    不确定路径：bash ./deploy_15.sh doctor —— 它会在各节点上找出来。"
  return 1
}

cmd_fetch() {
  say "3. 各节点只拉自己那份权重"
  _require_node_py || return 1
  # 来源三选一
  SRCARG=()
  local ep
  if [ -n "$SRC_DIR" ]; then
    SRCARG=(--src "$SRC_DIR"); ep="本地目录 $SRC_DIR（各节点自己读，不走网）"
  elif [ -n "$SRC_URL" ]; then
    SRCARG=(--base-url "$SRC_URL"); ep="局域网权重源 $SRC_URL"
  else
    SRCARG=(--repo "$REPO" ${HF_ENDPOINT:+--endpoint "$HF_ENDPOINT"})
    ep=${HF_ENDPOINT:-https://huggingface.co}
  fi
  echo "  合计约 141GB（全模型 160GB × 15 台 = 2400GB，省 94%）"
  echo "  源  $ep"
  echo "    ↑ **各节点**从这里拉，不是控制机。控制机的网络好不好在这里不算数。"
  if [ -z "$SRC_DIR$SRC_URL" ] && [ -z "${HF_ENDPOINT:-}" ]; then
    echo "    拉不动时：先看 3a 的预检怎么说；实在不行就下一次全量再局域网分发"
    echo "    （./deploy_15.sh serve-weights <全量目录>）"
  fi

  # 先用**一台**试通。141GB 下到一半才发现源站不通、或者不支持 Range，
  # 代价是几小时；这里花几十秒就能问清楚。
  local first; first=$(awk '{sub(/#.*/,"")} NF>=2 {print $1; exit}' "$HOSTS")
  local firstip; firstip=$(awk '{sub(/#.*/,"")} NF>=2 {split($2,a,":"); print a[1]; exit}' "$HOSTS")
  say "3a. 先用 $first 试通源站（读文件头，不下张量）"
  local probe out rc
  local srcstr=""
  for x in "${SRCARG[@]}"; do srcstr="$srcstr $(printf %q "$x")"; done
  probe="cd $(printf %q "$WORKDIR") && $(printf %q "$NODE_PY") -m p2pmoe.deploy.fetch \
--plan $(printf %q "$PLAN") --node $first --out $(printf %q "$WEIGHTS")$srcstr \
--transport $(printf %q "$TRANSPORT") --dry-run"
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 ${SSH_OPTS:-} \
          "${SSH_USER:+$SSH_USER@}$firstip" "$probe" </dev/null 2>&1) && rc=0 || rc=$?
  echo "$out" | sed 's/^/    /'
  if [ "$rc" != 0 ]; then
    echo
    echo "  ✗ $first 连源站就失败了 —— 15 台一起下只会失败 15 次。"
    case "$out" in
      *UNEXPECTED_EOF*|*"Connection reset"*|*"timed out"*)
        echo "    链路在传输中途被掐断（代码已会续传，仍失败说明每次都断在很早）。"
        echo "    先在那台机器上用 curl 复现，看是不是同一处断："
        echo "        curl -v -o /dev/null https://huggingface.co/$REPO/resolve/main/config.json"
        echo "    常见三个原因，都不是 HF 的问题："
        echo "      · TLS 检查型代理/防火墙掐长连接 —— 查 \$https_proxy，或让 IT 放行"
        echo "      · MTU 黑洞（小包过、大包丢，VPN/隧道上常见）—— 试把 MTU 降到 1400"
        echo "      · 出口 CDN 抖动 —— 换时间，或 --endpoint 换个镜像"
        echo "    绕过去：一台机器下全量，其余从局域网切片"
        echo "        bash ./deploy_15.sh serve-weights <全量目录>"
        echo "        SRC_URL=http://<那台的IP>:9400 bash ./deploy_15.sh fetch" ;;
      *RangeNotSupported*|*"不支持 Range"*)
        echo "    源站不支持 Range 请求 —— 「只拉需要的张量」这件事做不成。"
        echo "    换一个支持 Range 的镜像，或者退到整分片模式（省得少但能跑）：" 
        echo "        MODE=shard bash ./deploy_15.sh fetch" ;;
      *"No module named 'p2pmoe'"*|*'No module named "p2pmoe"'*)
        echo "    代码不在 $WORKDIR —— 先 bash ./deploy_15.sh sync" ;;
      *"No module named"*)
        # 找得到 p2pmoe 却缺别的（numpy/torch/safetensors）——
        # 代码在，是**解释器不对**。这两件事的修法完全相反：
        # 前者要 sync，后者要改 NODE_PY。混成一句会把人送去错的方向。
        echo "    代码在，但这个解释器里缺依赖 —— **NODE_PY 指错了**。"
        echo "    当前 NODE_PY = $NODE_PY"
        case "$NODE_PY" in
          python3|python) echo "    这是系统 python，看不见 conda 环境里的包。" ;;
        esac
        echo "    torch 在 conda 环境里的话要填**绝对路径**："
        echo "        export NODE_PY=/home/ubuntu/anaconda3/envs/${CONDA_ENV:-moe}/bin/python"
        echo "    不确定路径就跑：bash ./deploy_15.sh doctor" ;;
    esac
    return 1
  fi
  echo "  ✓ 源站可达、支持 Range、清单读得通"

  say "3b. 15 台一起下"
  echo "  单台最多 22.4GB，视带宽可能要几十分钟到几小时。"
  echo "  超时上限 ${FETCH_TIMEOUT}s（FETCH_TIMEOUT=），不够就调大。"
  echo "  每 ${PROGRESS_EVERY}s 报一次逐台进度（PROGRESS_EVERY=0 关掉）——"
  echo "  问的是目录大小，所以「进程还在但一个字节没动」也看得见。"
  read -rp "  继续？[y/N] " a; [ "$a" = y ] || return 0
  $PY -m p2pmoe.deploy.launch fetch --hosts "$HOSTS" --plan "$PLAN" \
      --out "$WEIGHTS" --python "$NODE_PY" --workdir "$WORKDIR" \
      --mode "$MODE" --timeout "$FETCH_TIMEOUT" --transport "$TRANSPORT" \
      --progress-every "$PROGRESS_EVERY" \
      "${SSHARG[@]}" "${SRCARG[@]}"
}

cmd_start() {
  say "4. 起 agent"
  _require_node_py || return 1
  # 起之前先把上一轮的残留清掉 —— 否则 Address already in use，
  # 而那个错要翻日志才看得见（start 只报 fork 成功）。
  echo "  先清上一轮的残留…"
  $PY -m p2pmoe.deploy.launch stop --hosts "$HOSTS" "${SSHARG[@]}" --by-port \
    2>&1 | grep -vE "^\\s*$|空闲" | sed 's/^/    /' || true
  $PY -m p2pmoe.deploy.launch start --hosts "$HOSTS" --workdir "$WORKDIR" \
      --python "$NODE_PY" "${SSHARG[@]}"
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
      --tokens "$TOKENS" --requests "$REQUESTS" --concurrency 3 \
      ${PROMPTS:+$([ -f "$PROMPTS" ] && echo "--prompts-file $PROMPTS")} \
      ${RESULTS:+--save-results "$RESULTS"} \
      --once "$@"
}

cmd_measure() {
  say "5'. 服务 + 逐请求时序"
  echo "  预热 $WARMUP 条再计时：torch 首次前向含 kernel 选择、显存池分配、"
  echo "  权重进页缓存 —— 只发生一次，混进 p50 会把结果拖歪"
  cmd_serve --warmup "$WARMUP" --verbose "$@"
}

# 把 --save-results 落下来的 JSON 读成一张表。测完看这个，不用翻滚屏日志。
cmd_report() {
  local f=${1:-$RESULTS}
  [ -f "$f" ] || { echo "没有 $f —— 先跑 ./deploy_15.sh measure"; exit 1; }
  $PY - "$f" <<'PYEOF'
import json, sys, statistics as st
d = json.load(open(sys.argv[1], encoding="utf-8"))
rs = d["requests"]; sm = d["summary"]
B, R = "\033[1m", "\033[0m"
print(f"\n{B}{d['model']}  L₀={d['l0']}  {d['mode']}  "
      f"miss={d['miss_policy']}  cov={d['coverage']}{R}")
print(f"{'req':<7}{'真实':<7}{'识别':<7}{'':<3}{'通道':<16}"
      f"{'总ms':>8}{'排队':>7}{'首tok':>7}{'/tok':>7}{'算力':>7}")
print("─" * 78)
for r in rs:
    mark = "" if r["correct"] is None else ("✓" if r["correct"] else "✗")
    print(f"{r['req']:<7}{str(r['true_task']):<7}{str(r['task']):<7}{mark:<3}"
          f"{r['front']+'×'+r['back']:<16}"
          f"{r['total_ms']:>8.0f}{r['queue_ms']:>7.0f}{r['prefill_ms']:>7.0f}"
          f"{r['per_token_p50_ms']:>7.1f}{r['utilisation']:>6.0%}")
print("─" * 78)
acc = "—" if sm["accuracy"] is None else f"{sm['accuracy']:.0%}"
print(f"  {len(rs)} 条：总时延 p50 {sm['total_ms_p50']:.0f}ms，"
      f"逐 token p50 {sm['per_token_ms_p50']:.1f}ms，"
      f"算力使用率均值 {sm['utilisation_mean']:.1%}，"
      f"识别 {acc}，换绑 {sm['rebinds']} 次")

# 时延去哪了 —— 三段相加恒等于总时延，所以这张表不会骗人
c = st.mean(r["compute_ms"] for r in rs)
q = st.mean(r["queue_ms"] for r in rs)
o = st.mean(r["network_and_overhead_ms"] for r in rs)
tot = c + q + o
print(f"\n{B}  时延构成（均值）{R}")
for name, v in (("计算", c), ("排队", q), ("网络+协议+调度", o)):
    bar = "█" * round(40 * v / tot) if tot else ""
    print(f"    {name:<16}{v:>8.0f}ms  {v/tot:>5.1%}  {bar}")
print("    ⚠ 「网络」是 总时延 − 计算 − 排队 **反推**的上界，不是测量值 ——")
print("      15 台没有时钟同步，跨机的绝对时刻拼不到一条轴上。")

# 谁在干活 —— 算力使用率低时，先看是不是某一跳独吞
print(f"\n{B}  各节点算力占比（占总时延，均值）{R}")
agg = {}
for r in rs:
    for n in r["nodes"]:
        a = agg.setdefault(n["node"], {"share": 0.0, "seg": n["segment"],
                                       "ly": n["layers"], "ne": n["n_experts"],
                                       "cnt": 0})
        a["share"] += n["share"]; a["cnt"] += 1
for nid, a in sorted(agg.items(), key=lambda kv: -kv[1]["share"] / kv[1]["cnt"]):
    sh = a["share"] / a["cnt"]
    print(f"    {nid:<6}{a['seg']:<12}层 {a['ly']:<8}{a['ne']:>4} 专家"
          f"{sh:>7.1%}  {'█' * round(60 * sh)}")
miss = {m for r in rs for m in r["missing_traces"]}
if miss:
    print(f"\n  ⚠ 这些节点没上报埋点：{sorted(miss)} —— 它们的计算被算进了「网络」")
PYEOF
}

# 一次问清楚每台节点到底缺什么。agent 起来就死时先跑这个。
#
# 为什么不是「拉日志」就够：日志**不存在**和 ssh 不通，症状看起来一样，
# 但一个是代码没到、一个是网络问题 —— 修法完全相反。所以这里逐项分别问。
_probe_script() {
  local id=$1
  cat <<EOS
W=$WORKDIR; L=/tmp/p2pmoe/agent-$id.log; D=$WEIGHTS; NP='$NODE_PY'; CE='$CONDA_ENV'
printf 'workdir  %-12s ' "\$W"; [ -d "\$W" ] && echo ok || echo '✗ 目录不存在'
printf 'package  %-12s ' 'p2pmoe/'; [ -f "\$W/p2pmoe/deploy/agent.py" ] && echo ok || echo '✗ 代码不在这儿'
printf 'NODE_PY  %-12s ' ''; if command -v "\$NP" >/dev/null 2>&1 || [ -x "\$NP" ]; then
  echo "\$NP -> \$("\$NP" -V 2>&1 | head -1)"
else echo "✗ \$NP 不存在"; fi
printf 'torch    %-12s ' ''
"\$NP" -c 'import torch;print(torch.__version__, torch.cuda.is_available() and ("cuda "+torch.cuda.get_device_name(0)) or "CPU-only")' 2>/dev/null \
  || echo "✗ \$NP 里没有 torch"
# torch 不在当前解释器里的话，去 conda 环境里找一遍 —— 找到就直接给出该填的值
if ! "\$NP" -c 'import torch' >/dev/null 2>&1; then
  for B in "\$HOME/miniconda3" "\$HOME/anaconda3" "\$HOME/miniforge3" "\$HOME/mambaforge" /opt/conda "\$(conda info --base 2>/dev/null)"; do
    [ -n "\$B" ] || continue
    for E in "\$CE" base; do
      C="\$B/envs/\$E/bin/python"; [ "\$E" = base ] && C="\$B/bin/python"
      if [ -x "\$C" ] && "\$C" -c 'import torch' >/dev/null 2>&1; then
        echo "  ↳ 找到了：NODE_PY=\$C   (torch \$("\$C" -c 'import torch;print(torch.__version__)' 2>/dev/null))"
        break 2
      fi
    done
  done
fi
printf 'weights  %-12s ' "\$D"; [ -d "\$D" ] && echo "ok (\$(du -sh "\$D" 2>/dev/null | cut -f1))" || echo '✗ 没拉过'
printf 'disk     %-12s ' ''; df -h --output=avail "\$(dirname "\$D")" 2>/dev/null | tail -1 | tr -d ' ' | sed 's/$/ 可用/' || echo '?'
printf 'port     %-12s ' '9101'; (ss -lnt 2>/dev/null || netstat -lnt 2>/dev/null) | grep -q ':9101 ' && echo '在听' || echo '空闲'
printf 'log      %-12s ' ''; if [ -s "\$L" ]; then echo "\$L (\$(wc -l < "\$L") 行)"; else echo "\$L ✗ 不存在（进程连启动都没到）"; fi
[ -s "\$L" ] && { echo '--- 日志末尾 ---'; tail -n 8 "\$L"; }
exit 0
EOS
}

# 逐台体检。ssh 失败与「远端某项缺失」分开报 —— 它们的修法相反。
cmd_doctor() {
  need "$HOSTS"
  local only=${1:-}
  say "节点体检"
  local d; d=$(mktemp -d); trap 'rm -rf "$d"' RETURN
  local ids=()
  while read -r id ip; do
    [ -n "$only" ] && [ "$id" != "$only" ] && continue
    ids+=("$id:$ip")
    ( _probe_script "$id" | ssh -o BatchMode=yes -o ConnectTimeout=10 ${SSH_OPTS:-} \
        "${SSH_USER:+$SSH_USER@}$ip" 'sh -s' > "$d/$id.out" 2> "$d/$id.err"
      echo $? > "$d/$id.rc" ) &
  done < <(awk '{sub(/#.*/,"")} NF>=2 {split($2,a,":"); print $1, a[1]}' "$HOSTS")
  wait

  local nossh=0 nocode=0 notorch=0
  for e in "${ids[@]}"; do
    local id=${e%%:*} ip=${e#*:}
    printf '\n\033[1m── %s @ %s\033[0m\n' "$id" "$ip"
    # 这台按清单要拉多少 —— 和上面的 disk 一行放一起看才有意义
    $PY - "$PLAN" "$id" <<'PYEOF' 2>/dev/null || true
import json, sys
m = json.load(open(sys.argv[1]))
n = next((x for x in m["nodes"] if x["node"] == sys.argv[2]), None)
if n:
    lr = n.get("layer_range") or [0, 0]
    print(f"   本机份额     层 {lr[0]}–{lr[1]}，要拉 {n['weight_gb']:.1f} GB"
          f"（全模型 160GB，**不需要全量**）")
PYEOF
    if [ "$(cat "$d/$id.rc" 2>/dev/null)" != 0 ]; then
      echo "   ✗ ssh 连不上：$(head -1 "$d/$id.err" 2>/dev/null)"
      nossh=$((nossh+1)); continue
    fi
    sed 's/^/   /' "$d/$id.out"
    grep -q '代码不在这儿' "$d/$id.out" && nocode=$((nocode+1))
    grep -q '没有 torch' "$d/$id.out" && notorch=$((notorch+1))
    grep -h '↳ 找到了：NODE_PY=' "$d/$id.out" >> "$d/_found" 2>/dev/null || true
  done

  echo
  say "结论"
  [ "$nossh"   -gt 0 ] && echo "  · $nossh 台 ssh 连不上 —— 检查 SSH_USER/SSH_OPTS，或走 ./deploy_15.sh bootstrap"
  [ "$nocode"  -gt 0 ] && echo "  · $nocode 台没有代码 —— ./deploy_15.sh sync"
  if [ "$notorch" -gt 0 ]; then
    echo "  · $notorch 台的 NODE_PY（$NODE_PY）里没有 torch"
    if [ -s "$d/_found" ]; then
      local uniq; uniq=$(sed 's/.*NODE_PY=//;s/ .*//' "$d/_found" | sort -u)
      local n_u; n_u=$(echo "$uniq" | wc -l)
      if [ "$n_u" = 1 ]; then
        echo "    在各节点上找到了带 torch 的解释器，照这个设："
        echo
        echo "        export NODE_PY=$uniq"
        echo
      else
        echo "    各节点路径不一致，逐台确认："
        echo "$uniq" | sed 's/^/        /'
      fi
    else
      echo "    也没找到别的带 torch 的环境 —— ./deploy_15.sh sync 会装（用 $NODE_PY）"
    fi
  fi
  [ "$nossh$nocode$notorch" = "000" ] && echo "  · 每台该有的都有了。agent 还是起不来的话，看上面各自的日志末尾。"
  return 0
}

# 拉各节点的 agent 日志。
#
#   ./deploy_15.sh logs        # 15 台各拉最后 15 行
#   ./deploy_15.sh logs N07    # 只看一台，拉 60 行
cmd_logs() {
  need "$HOSTS"
  local only=${1:-} n=15
  [ -n "$only" ] && n=60
  say "各节点 agent 日志（/tmp/p2pmoe/agent-<id>.log）"
  local miss=0 nossh=0 got=0
  while read -r id ip; do
    [ -n "$only" ] && [ "$id" != "$only" ] && continue
    printf '\n\033[1m── %s @ %s\033[0m\n' "$id" "$ip"
    local out rc
    # **分开判**：ssh 失败 与 日志不存在 是两回事。以前把它们混成一句
    # 「ssh 不通」，于是「代码没同步过去」被误报成网络问题。
    out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 ${SSH_OPTS:-} \
            "${SSH_USER:+$SSH_USER@}$ip" \
            "if [ -f /tmp/p2pmoe/agent-$id.log ]; then tail -n $n /tmp/p2pmoe/agent-$id.log; \
             else echo __NOLOG__; fi" </dev/null 2>&1) && rc=0 || rc=$?
    if [ "$rc" != 0 ]; then
      echo "   ✗ ssh 连不上：$(echo "$out" | head -1)"; nossh=$((nossh+1))
    elif [ "$out" = "__NOLOG__" ]; then
      echo "   ✗ 日志不存在 —— 进程连启动都没到（cd $WORKDIR 就失败了）"; miss=$((miss+1))
    else
      echo "$out" | sed 's/^/   /'; got=$((got+1))
    fi
  done < <(awk '{sub(/#.*/,"")} NF>=2 {split($2,a,":"); print $1, a[1]}' "$HOSTS")
  echo
  [ "$nossh" -gt 0 ] && echo "  $nossh 台 ssh 连不上。"
  [ "$miss"  -gt 0 ] && echo "  $miss 台没有日志 —— 说明 \`cd $WORKDIR\` 就失败了，代码根本不在那儿。先 ./deploy_15.sh sync"
  [ "$got"   -gt 0 ] && echo "  $got 台有日志，死因见上面各自的末尾几行。"
  return 0
}

# 控制机也需要 $WEIGHTS 这个目录 —— 但只要里面的 config.json 与 tokenizer。
#
# 它一个张量都不碰：用 config 算模型规格（层数、专家数、切点的内存账），
# 用 tokenizer 把 prompt 编成 id、把 id 解回文本。权重是各节点自己拉的，
# 不经过控制机。所以这里只要约 10MB，不是 141GB。
cmd_meta() {
  say "M. 控制机取模型元数据（约 10MB，不含权重）"
  if ! mkdir -p "$WEIGHTS" 2>/dev/null; then
    echo "  ✗ 建不了 $WEIGHTS —— 没权限。"
    echo "    把 WEIGHTS 设到你写得进去的地方，比如项目目录下："
    echo "        export WEIGHTS=$WORKDIR/weights"
    exit 1
  fi
  echo "  → $WEIGHTS"
  $PY -m p2pmoe.deploy.fetch --meta-only --repo "$REPO" --out "$WEIGHTS" \
      --transport "$TRANSPORT" ${HF_ENDPOINT:+--endpoint "$HF_ENDPOINT"}
}

# 上游拉不动时：先在**一台**机器上下一份全量，再让 15 台从局域网切片。
#
#     git clone https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct   # 160GB，一次
#     bash ./deploy_15.sh serve-weights /path/to/Qwen3-Next-80B-A3B-Instruct
#     # 另开终端：
#     SRC_URL=http://<这台的IP>:9400 bash ./deploy_15.sh fetch
#
# 合计仍然只拉 141GB，只是源站换成了自己人。
#
# 为什么不能用 `python -m http.server`：它**不支持 Range** —— 无视 Range 头，
# 返回 200 加整个文件。而「只拉自己那份张量」完全靠 Range，用它的话每台会
# 拿到整个分片（fetch 认得出来并报错，不会将就）。所以这里自己起一个。
#
# 有 NFS/共享盘的话连服务都不用起：SRC_DIR=/mnt/shared/ckpt bash ./deploy_15.sh fetch
cmd_serve_weights() {
  local d=${1:-$SRC_DIR}
  [ -n "$d" ] || { echo "用法: $0 serve-weights <全量checkpoint目录>"; exit 1; }
  say "W. 局域网权重源"
  $PY -m p2pmoe.deploy.serve_weights --dir "$d" --bind "0.0.0.0:${WSRC_PORT:-9400}"
}

# 源站诊断：小文件与大文件是不是同一个域名、CDN 那个域名通不通。
#
# 「小文件过得去、大文件过不去」有两种解释，修法完全不同：
#   · 链路把长传输掐断（代理 / MTU / CDN 抖动）→ 调网络
#   · **它们根本是两个域名** → 让 IT 放行第二个
# HF 把 config.json 由 huggingface.co 直接发，LFS 文件（tokenizer.json 与全部
# safetensors）302 到 CDN。白名单只放行前者时，症状看着完全像前一种。
cmd_diag() {
  say "D. 源站诊断"
  echo "  控制机这边："
  $PY -m p2pmoe.deploy.fetch --diagnose --repo "$REPO" \
      ${HF_ENDPOINT:+--endpoint "$HF_ENDPOINT"} 2>&1 | sed 's/^/  /'
  local first firstip
  first=$(awk '{sub(/#.*/,"")} NF>=2 {print $1; exit}' "$HOSTS" 2>/dev/null)
  firstip=$(awk '{sub(/#.*/,"")} NF>=2 {split($2,a,":"); print a[1]; exit}' "$HOSTS" 2>/dev/null)
  [ -n "$firstip" ] || return 0
  echo
  echo "  节点 $first 那边（网络策略可能和控制机不同）："
  ssh -o BatchMode=yes -o ConnectTimeout=10 ${SSH_OPTS:-} \
      "${SSH_USER:+$SSH_USER@}$firstip" \
      "cd $(printf %q "$WORKDIR") && $(printf %q "$NODE_PY") -m p2pmoe.deploy.fetch \
--diagnose --repo $(printf %q "$REPO") ${HF_ENDPOINT:+--endpoint $(printf %q "$HF_ENDPOINT")}" \
      </dev/null 2>&1 | sed 's/^/  /' || echo "  （ssh 不通，上机自己跑）"
}

# 逐台核对：清单要的每个 key，本机的分片里是不是都在。
#
# `check` 里那个预检很便宜（目录在不在、依赖装没装），**不查 key 齐不齐** ——
# 而缺 key 要到 measure 装载模型的那一刻才炸，那时探测与建链已经白跑了几分钟。
# 这里只读 safetensors 的文件头，几秒钟就能回答，代价与收益完全不成比例。
#
# 最常见的触发场景：改了 COVERAGE 之后没重跑 fetch —— 驻留集变了，
# 权重还是旧的那一份。
cmd_verify() {
  need "$HOSTS"; need "$PLAN"
  say "V. 逐台核对权重完整性（只读文件头）"
  local only=${1:-} d; d=$(mktemp -d); trap 'rm -rf "$d"' RETURN
  local ids=()
  while read -r id ip; do
    [ -n "$only" ] && [ "$id" != "$only" ] && continue
    ids+=("$id:$ip")
    ( ssh -o BatchMode=yes -o ConnectTimeout=10 ${SSH_OPTS:-} \
        "${SSH_USER:+$SSH_USER@}$ip" \
        "cd $(printf %q "$WORKDIR") && $(printf %q "$NODE_PY") - \
$(printf %q "$PLAN") $(printf %q "$id") $(printf %q "$WEIGHTS")" \
        <<'PYEOF' > "$d/$id.out" 2>&1
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from p2pmoe.deploy.fetch import keys_for_node
from p2pmoe.planner.manifest import DeploymentManifest
from p2pmoe.runtime.weights import WeightIndex

plan, node, wdir = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.loads((Path(wdir) / "config.json").read_text(encoding="utf-8"))
man = DeploymentManifest.from_json(Path(plan).read_text(encoding="utf-8"))
want = keys_for_node(man, node, config=cfg)
have = set(WeightIndex(wdir).weight_map)
miss = want - have
gb = sum(f.stat().st_size for f in Path(wdir).glob("*.safetensors")) / 1e9
if miss:
    print(f"MISS {len(miss)}/{len(want)} {gb:.1f}GB {sorted(miss)[0]}")
    sys.exit(1)
print(f"OK {len(want)} {gb:.1f}GB")
PYEOF
      echo $? > "$d/$id.rc" ) &
  done < <(awk '{sub(/#.*/,"")} NF>=2 {split($2,a,":"); print $1, a[1]}' "$HOSTS")
  wait

  local ok=0 bad=0
  for e in "${ids[@]}"; do
    local id=${e%%:*} ip=${e#*:} out
    out=$(head -3 "$d/$id.out" 2>/dev/null)
    case "$out" in
      OK\ *)   set -- $out; printf '  ✓ %-5s %s 个张量  %s\n' "$id" "$2" "$3"; ok=$((ok+1)) ;;
      MISS\ *) set -- $out
               printf '  ✗ %-5s 缺 %s  已有 %s  首个: %s\n' "$id" "$2" "$3" "$4"
               bad=$((bad+1)) ;;
      *)       printf '  ? %-5s %s\n' "$id" "$(echo "$out" | head -1)"; bad=$((bad+1)) ;;
    esac
  done
  echo
  if [ "$bad" = 0 ]; then
    echo "  $ok/$ok 台的权重与清单一致 —— 可以 start 了。"
    return 0
  fi
  echo "  $bad 台对不上。最常见的原因：**改了 COVERAGE 之后没重跑 fetch** ——"
  echo "  驻留集变了，权重还是旧的那份。重跑：bash ./deploy_15.sh fetch"
  echo "  （fetch 会跳过已有的分片，只补缺的）"
  return 1
}

# 停 agent。
#
#   ./deploy_15.sh stop        # 按命令行匹配，只杀本框架的 agent
#   ./deploy_15.sh stop force  # 连**占着 9101 的进程**一起清
#
# 为什么要有 force：按命令行匹配只认得出自己起的那些。上一轮如果是手工起的、
# 参数顺序不同、或者进程卡在退出中途，端口就还占着 —— 下一轮 start 报
# Address already in use，而 stop 却说「成功」。
# 权重到底在哪、内存被谁占着。
#
# 典型的一对症状：`ls $WEIGHTS` 空空如也，内存/磁盘却还满着。
# Linux 下 unlink 一个**正在被 mmap 的文件**，空间不会立刻释放 ——
# 目录里看不见了，但持有 fd 的进程还占着它。文件真正消失要等那个进程退出。
#
# 最常见的来路：某次 `sync` 用的是加排除项之前的脚本，`rsync --delete`
# 把 weights/ 当成「源端没有」删掉了，而 agent 正开着那些分片。
cmd_whereis() {
  need "$HOSTS"
  say "W. 权重在哪 / 内存被谁占着"
  local only=${1:-} d; d=$(mktemp -d); trap 'rm -rf "$d"' RETURN
  local ids=()
  while read -r id ip; do
    [ -n "$only" ] && [ "$id" != "$only" ] && continue
    ids+=("$id:$ip")
    ( ssh -o BatchMode=yes -o ConnectTimeout=15 ${SSH_OPTS:-} \
        "${SSH_USER:+$SSH_USER@}$ip" "bash -s" <<EOS > "$d/$id.out" 2>&1
W=$(printf %q "$WEIGHTS")

echo "配置路径 \$W"
if [ -d "\$W" ]; then
  N=\$(ls "\$W"/*.safetensors 2>/dev/null | wc -l)
  echo "  在，\$N 个分片，\$(du -sh "\$W" 2>/dev/null | cut -f1)"
else
  echo "  ✗ 目录不存在"
fi

echo "别处的 safetensors（找 \$HOME 与 /data，深度 4）"
find "\$HOME" /data -maxdepth 4 -name '*.safetensors' -printf '%h\n' 2>/dev/null \
  | sort | uniq -c | sort -rn | head -5 | sed 's/^/  /' || echo "  没找到"

echo "agent 进程"
for pid in \$(pgrep -f 'p2pmoe.deploy.agent' 2>/dev/null); do
  case "\$(ps -o comm= -p \$pid 2>/dev/null)" in python*|Python*) ;; *) continue;; esac
  echo "  pid \$pid  RSS \$(awk '/VmRSS/{printf "%.1f GB", \$2/1048576}' /proc/\$pid/status 2>/dev/null)"
  # **重点**：已被删除但仍被打开的文件 —— 它们还占着空间
  DEL=\$(ls -l /proc/\$pid/fd 2>/dev/null | grep -c '(deleted)')
  MAPDEL=\$(grep -c '(deleted)' /proc/\$pid/maps 2>/dev/null || echo 0)
  echo "    已删除但仍打开: fd \$DEL 个, 内存映射 \$MAPDEL 段"
  if [ "\$MAPDEL" != 0 ]; then
    grep '(deleted)' /proc/\$pid/maps 2>/dev/null \
      | awk '{print \$(NF-1)}' | sort -u | head -3 | sed 's/^/      /'
  fi
done
[ -z "\$(pgrep -f 'p2pmoe.deploy.agent' 2>/dev/null)" ] && echo "  （没有 agent 在跑）"

echo "磁盘"
df -h "\$(dirname "\$W")" 2>/dev/null | tail -1 | awk '{print "  "\$2" 总 / "\$3" 用 / "\$4" 可用"}'
echo "内存"
free -g 2>/dev/null | awk '/Mem:/{print "  "\$2" GB 总 / "\$3" 用 / "\$7" 可用"}'
EOS
      echo $? > "$d/$id.rc" ) &
  done < <(awk '{sub(/#.*/,"")} NF>=2 {split($2,a,":"); print $1, a[1]}' "$HOSTS")
  wait

  local ghost=0
  for e in "${ids[@]}"; do
    local id=${e%%:*} ip=${e#*:}
    printf '\n\033[1m── %s @ %s\033[0m\n' "$id" "$ip"
    if [ "$(cat "$d/$id.rc" 2>/dev/null)" != 0 ]; then
      echo "   ✗ ssh 连不上"; continue
    fi
    sed 's/^/   /' "$d/$id.out"
    grep -q '已删除但仍打开: fd 0 个, 内存映射 0' "$d/$id.out" || \
      grep -q '已删除但仍打开' "$d/$id.out" && \
      grep -q '内存映射 [1-9]' "$d/$id.out" && ghost=$((ghost+1))
  done

  echo
  say "怎么读"
  echo "  · 目录不存在 / 分片为 0，而「已删除但仍打开」不是 0"
  echo "    → 文件被删了，但 agent 还开着它们。空间要等进程退出才回来。"
  echo "    → 处置：bash ./deploy_15.sh stop force   然后重新 fetch"
  echo "  · 「别处的 safetensors」指向另一个目录"
  echo "    → 上一轮 fetch 的 --out 与现在的 WEIGHTS 不一致。改 WEIGHTS 指过去，"
  echo "      或重新 fetch（15 台路径必须一致）。"
  echo "  · 目录在、分片齐、RSS 大 —— 那是正常的：模型装载后就该占着内存。"
  [ "$ghost" -gt 0 ] && echo && echo "  ⚠ $ghost 台有「已删除但仍打开」的映射。"
  return 0
}

cmd_stop() {
  say "停 agent"
  local extra=()
  [ "${1:-}" = force ] && { extra+=(--by-port); echo "  force：连占着 9101 的进程一起清（会先打印是谁）"; }
  $PY -m p2pmoe.deploy.launch stop --hosts "$HOSTS" "${SSHARG[@]}" "${extra[@]}"
}

# serve / measure 后面多写的参数**原样透传给 control.py**，例如
#     ./deploy_15.sh measure --coverage 0.60 --miss-policy drop_noscale
# argparse 取最后一次出现，所以透传的值会盖掉脚本里设的默认值。
action=${1:-all}
shift 2>/dev/null || true

case "$action" in
  sync)      cmd_sync ;;
  bootstrap) cmd_bootstrap ;;
  cmds)      cmd_cmds "${1:-}" ;;
  check)   cmd_check ;;
  fetch)   cmd_fetch ;;
  meta)    cmd_meta ;;
  diag)    cmd_diag ;;
  verify)  cmd_verify "${1:-}" ;;
  whereis) cmd_whereis "${1:-}" ;;
  serve-weights) cmd_serve_weights "${1:-}" ;;
  start)   cmd_start ;;
  serve)   cmd_serve "$@" ;;
  measure) cmd_measure "$@" ;;
  report)  cmd_report "${1:-}" ;;
  logs)    cmd_logs "${1:-}" ;;
  doctor)  cmd_doctor "${1:-}" ;;
  stop)    cmd_stop "${1:-}" ;;
  all)     cmd_sync && cmd_check && cmd_fetch && cmd_start && cmd_serve "$@" ;;
  *) echo "用法: $0 {sync|bootstrap|cmds|check|fetch|meta|diag|serve-weights|verify|whereis|start|serve|measure|report|logs|doctor|stop|all}
       serve/measure 后面的参数原样透传给 control.py"; exit 1 ;;
esac

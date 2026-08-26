"""批量拉起 / 查看 / 停止节点 agent —— 15 台机器不用手敲 15 次。

    python -m p2pmoe.deploy.launch start   --hosts hosts.txt
    python -m p2pmoe.deploy.launch status  --hosts hosts.txt
    python -m p2pmoe.deploy.launch stop    --hosts hosts.txt
    python -m p2pmoe.deploy.launch agents  --hosts hosts.txt   # 打印 --agents 串

hosts 文件每行一台，`#` 注释：

    # 节点id  地址              [附加参数]
    v1        10.0.0.11
    v2        10.0.0.12         --mem-mb 32000
    g1        10.0.0.21:9102    --mem-mb 47000

**agent 不需要同时启动。** 它们是常驻守护进程，先后起来都行；控制器只要求
「跑规划的那一刻它们都在」。没起来的会在采集能力那一步被剔除，不影响其余节点 ——
分散环境下这是常态，不是异常。

生产环境建议用 systemd 而不是这个脚本（`launch systemd` 可以生成 unit 模板）：
它管开机自启、崩溃重启、日志轮转，这些本脚本都不管。
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..runtime.wire import rpc

__all__ = ["Host", "read_hosts", "SYSTEMD_UNIT"]

DEFAULT_PORT = 9101


@dataclass
class Host:
    node_id: str
    host: str
    port: int = DEFAULT_PORT
    extra: list[str] = field(default_factory=list)

    @property
    def addr(self) -> tuple[str, int]:
        return (self.host, self.port)


def read_hosts(path: Path) -> list[Host]:
    out: list[Host] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = shlex.split(line)
        node_id, target, extra = parts[0], parts[1], parts[2:]
        if node_id.endswith("=") or "=" in target:
            raise ValueError(f"hosts 文件格式是「id 地址 [参数]」，收到: {raw!r}")
        host, _, port = target.rpartition(":")
        if not host:
            host, port = target, str(DEFAULT_PORT)
        out.append(Host(node_id, host, int(port), extra))
    if not out:
        raise ValueError(f"{path} 里没有任何节点")
    ids = [h.node_id for h in out]
    if len(set(ids)) != len(ids):
        raise ValueError("节点 id 必须全池唯一")
    return out


# --------------------------------------------------------------------------- #
def _brief(msg: str, width: int = 200) -> str:
    """把多行输出压成一行，**保留头部**。

    以前是 `msg[-160:]` —— 留尾巴。可失败原因几乎总在开头：
    「读不到 https://… 的 config.json：SSL EOF」被切成「ingface.co/…/main」，
    读的人看到的是半截 URL，看不出这是个网络错误。
    """
    lines = [x.strip() for x in msg.strip().splitlines() if x.strip()]
    if not lines:
        return "（没有输出）"
    head = lines[0]
    if len(lines) > 1:
        head += f"  … +{len(lines) - 1} 行（末行：{lines[-1][:60]}）"
    return head if len(head) <= width else head[:width - 1] + "…"


def _ssh(h: Host, cmd: str, *, ssh: str, user: str | None,
         timeout: float = 60.0) -> tuple[bool, str]:
    """在 h 上跑一条命令。

    `timeout` 默认 60 秒 —— 那是给 start/stop/status 这类**控制命令**的。
    下载不能用这个数：单台最多要拉 22GB，60 秒必然超时，而超时抛出的
    `TimeoutExpired` 消息里是一整串 argv，读起来完全不像「超时了」。
    """
    target = f"{user}@{h.host}" if user else h.host
    try:
        r = subprocess.run(shlex.split(ssh) + [target, cmd],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, (f"ssh 超过 {timeout:.0f}s 没返回。"
                       f"下载类命令要给足时间：--timeout <秒>")
    return r.returncode == 0, (r.stdout or r.stderr).strip()


def _local_start(h: Host, workdir: str, python: str, logdir: str) -> tuple[bool, str]:
    """本机模式不经 shell 直接 spawn。

    走 `bash -c "... &"` + capture_output 会挂住：父进程要等管道关闭，而后台
    子进程即使重定向了自己的 fd，shell 的这次捕获仍可能悬着。直接 Popen 加
    start_new_session 干净得多 —— 而且它跟 ssh 那条路是**两条独立实现**，
    本机模式只用于演练与自测，真机永远走 ssh 那条。
    """
    Path(logdir).mkdir(parents=True, exist_ok=True)
    args = [python, "-m", "p2pmoe.deploy.agent",
            "--id", h.node_id, "--bind", f"0.0.0.0:{h.port}"] + h.extra
    lf = open(Path(logdir) / f"agent-{h.node_id}.log", "a", encoding="utf-8")
    proc = subprocess.Popen(args, cwd=workdir, stdout=lf, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    return True, f"pid {proc.pid}"


def _local_stop(h: Host) -> tuple[bool, str]:
    r = subprocess.run(["pkill", "-f", f"p2pmoe.deploy.agent --id {h.node_id} "],
                       capture_output=True, text=True, timeout=30)
    return r.returncode in (0, 1), "已停止" if r.returncode == 0 else "本来就没在跑"


def _start_cmd(h: Host, workdir: str, python: str, logdir: str) -> str:
    args = [python, "-m", "p2pmoe.deploy.agent",
            "--id", h.node_id, "--bind", f"0.0.0.0:{h.port}"] + h.extra
    return (
        f"mkdir -p {shlex.quote(logdir)} && cd {shlex.quote(workdir)} && "
        f"setsid nohup {' '.join(shlex.quote(a) for a in args)} "
        f">> {shlex.quote(logdir)}/agent-{h.node_id}.log 2>&1 < /dev/null & echo $!"
    )


def _expected_bytes(plan_path, per_node_dir: bool) -> dict[str, int]:
    """清单里每台该拉多少字节。进度条的分母。"""
    try:
        m = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {n["node"]: int(float(n.get("weight_gb", 0)) * 1e9) for n in m.get("nodes", [])}


def _du(h: Host, path: str, *, ssh: str, user: str | None) -> int:
    """远端目录当前多大。`du -sb` 很便宜，30 秒问一次不心疼。"""
    ok, out = _ssh(h, f"du -sb {shlex.quote(path)} 2>/dev/null | cut -f1",
                   ssh=ssh, user=user, timeout=20)
    if not ok:
        return -1
    tok = out.strip().split()
    return int(tok[0]) if tok and tok[0].isdigit() else -1


def _progress_loop(hosts, out_of: dict[str, int], out_dir, *, ssh, user,
                   every: float, stop: "threading.Event", per_node_dir: bool,
                   local: bool) -> None:
    """每 `every` 秒打一次进度表。

    为什么要有：`fetch` 是**几小时**的操作，而每个节点的输出被 `capture_output`
    扣着，要等它整个跑完才吐出来。中间什么都没有的话，人分不清「在下」「卡住了」
    和「早就死了」—— 而这三件事的处置完全不同。

    问的是目录大小，不是进程状态：进程活着但一个字节没动，才是最需要看见的那种卡。
    """
    t0 = time.monotonic()
    last: dict[str, int] = {}
    while not stop.wait(every):
        rows = []
        for h in hosts:
            d = f"{out_dir}/{h.node_id}" if per_node_dir else out_dir
            got = -1 if local else _du(h, d, ssh=ssh, user=user)
            want = out_of.get(h.node_id, 0)
            # 第一次采样没有「上一次」可比 —— delta 记 None，而不是 0。
            # 记 0 的话首屏会把每台都标成「无增长」，而那正是最刺眼的告警。
            delta = None if h.node_id not in last else got - last[h.node_id]
            rows.append((h.node_id, got, want, delta))
            if got >= 0:
                last[h.node_id] = got
        done = sum(1 for _, g, w, _ in rows if w and g >= w * 0.98)
        tot_got = sum(max(0, g) for _, g, _, _ in rows)
        tot_want = sum(w for _, _, w, _ in rows) or 1
        el = time.monotonic() - t0
        rate = tot_got / el if el > 0 else 0
        eta = (tot_want - tot_got) / rate if rate > 1e3 else 0
        print(f"\n  ── 进度 +{el/60:.0f}min  "
              f"{tot_got/1e9:.1f}/{tot_want/1e9:.0f}GB  "
              f"({tot_got/tot_want:.0%})  {rate/1e6:.0f}MB/s  "
              f"{done}/{len(rows)} 台完成"
              + (f"  剩约 {eta/60:.0f}min" if eta else ""))
        for nid, got, want, delta in sorted(rows):
            if got < 0:
                print(f"     {nid:<6} ?（问不到目录大小）")
                continue
            frac = got / want if want else 0
            bar = "█" * round(24 * min(1.0, frac)) + "·" * (24 - round(24 * min(1.0, frac)))
            # 两次之间没长 = 卡住了。进程还在不代表在下东西。
            mark = ("  ⏸ 无增长" if delta == 0 and got < want * 0.98 else "")
            print(f"     {nid:<6} {bar} {got/1e9:>5.1f}/{want/1e9:<5.1f}GB{mark}")


def _warn_default_workdir(args) -> None:
    """`--workdir` 默认是 `"."`，而远端的 `cd .` 落在 **ssh 登录目录**（通常是
    $HOME），不是代码目录。症状是每台都报 `No module named 'p2pmoe'` ——
    看起来像没装，其实是跑在了错的目录里。

    本机模式（`--local`）没这个问题：`.` 就是当前进程的工作目录。
    """
    if args.local or args.workdir not in (".", "./"):
        return
    print("  ⚠ 没给 --workdir，远端会 `cd .` —— 那是 ssh 登录目录（通常 $HOME），")
    print("    不是代码目录。如果各节点的代码不在那儿，会报 No module named 'p2pmoe'。")
    print("    加上 --workdir /path/to/p2p-framework（deploy_15.sh 用 $WORKDIR）")


def _stop_cmd(h: Host) -> str:
    # 精确匹配 --id，避免误杀同机上别的 agent
    return f"pkill -f {shlex.quote(f'p2pmoe.deploy.agent --id {h.node_id} ')} || true"


def probe_status(h: Host) -> tuple[Host, str]:
    try:
        r = rpc(h.addr, {"type": "capabilities"}, timeout=8.0)
    except Exception as e:
        return h, f"✗ 不可达（{type(e).__name__}）"
    tag = "已配置" if r.get("configured") else "空闲"
    return h, (f"✓ {tag}  内存 {r['mem_mb']:.0f}MB  基准 {r['ms_per_layer']:.3f}ms"
               + (f"  接入模拟 {r['access_ms']:.0f}ms" if r.get("access_ms") else ""))


def _probe_matrix(hosts: list[Host], *, k: int, parallel: int) -> int:
    """全网逐对探测，只报告、不规划。

    **上真机的第一件事就该跑它。** 这套方法的全部价值建立在一个前提上：
    网络延迟是主项（文档 I.2.1：分散环境下网络占单 token 延迟约九成）。
    如果你的 15 台在同一个机房同一台交换机后面，逐对延迟是零点几毫秒，
    那么「拉平组合延迟」这个目标本身就没有对象 —— 均匀性机制没有用武之地，
    该用的是别的架构。先量清楚，再决定要不要往下走。
    """
    from statistics import median

    from .probe import RemoteNetworkOracle

    addrs = {h.node_id: h.addr for h in hosts}
    ids = sorted(addrs)
    print(f"逐对探测 {len(ids)} 台（由节点自己发起，k={k}）…")
    o = RemoteNetworkOracle(addrs, k_default=k, symmetric=True)
    n = o.warm_all(ids, k=k, workers=parallel)

    p50s, jits = [], []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            pr = o.probe(a, b, k)
            if pr.p50 != float("inf"):
                p50s.append(pr.p50)
                jits.append(pr.p95 - pr.p50)

    if not p50s:
        print("没有任何一对可达 —— 检查防火墙与端口")
        return 1

    reach = o.reachability(ids)
    print(f"\n{n} 对，可达 {len(p50s)} 对")
    print(f"  单向 p50   {min(p50s):7.2f} – {max(p50s):7.2f} ms（中位 {median(p50s):.2f}）")
    print(f"  抖动 p95−p50 {min(jits):5.2f} – {max(jits):7.2f} ms（中位 {median(jits):.2f}）")

    bad = {n_: c for n_, c in reach.items() if c < len(ids) - 1}
    if bad:
        print(f"  可达性不完整: {bad} —— 规划会自然绕开不可达链路")

    # 按接入质量排序：δ̂_v = 该节点对全网的 p50 中位
    off = []
    for a in ids:
        v = [o.probe(a, b, k).p50 for b in ids if b != a]
        v = [x for x in v if x != float("inf")]
        if v:
            off.append((median(v), a))
    off.sort()
    print("\n  接入质量（对全网的 p50 中位，越小越好）：")
    for m, a in off:
        print(f"    {a:<10} {m:7.2f} ms")

    med = median(p50s)
    print()
    if med < 1.0:
        print("⚠ 中位延迟 < 1ms —— 这是 LAN/同机量级。")
        print("  这套方法针对的是**分散环境**：逐对数十毫秒、抖动同量级、网络占")
        print("  单 token 延迟九成（文档 I.2.1）。你的池子里网络几乎不花时间，")
        print("  「拉平组合延迟」没有对象 —— 公共带宽度 W = η·median T 会小到")
        print("  凑不出人口，规划会（正确地）失败。")
        print("  想在这样的池子上验证部署路径，给每个 agent 加 --access-ms 模拟接入段。")
        return 2
    if med < 5.0:
        print("△ 中位延迟 1–5ms —— 介于 LAN 与 WAN 之间。方法能跑，但收益有限：")
        print("  网络不占主项时，跳数优先与均匀性收紧的杠杆都变小了。")
        return 0
    print(f"✓ 中位 {med:.0f}ms，符合分散环境的设定 —— 网络是主项，这套方法有对象。")
    return 0


# --------------------------------------------------------------------------- #
SYSTEMD_UNIT = """\
# /etc/systemd/system/p2pmoe-agent.service
# 装好后：systemctl daemon-reload && systemctl enable --now p2pmoe-agent
[Unit]
Description=P2P MoE node agent ({node_id})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={user}
WorkingDirectory={workdir}
ExecStart={python} -m p2pmoe.deploy.agent --id {node_id} --bind 0.0.0.0:{port}{extra}
Restart=always
RestartSec=3
# agent 是无状态的：崩溃重启后回到未配置态，等控制器重新下发即可
StandardOutput=append:/var/log/p2pmoe-agent-{node_id}.log
StandardError=inherit

[Install]
WantedBy=multi-user.target
"""


# --------------------------------------------------------------------------- #
def _fetch_all(hosts, args) -> int:
    """让每台节点**自己**去拉它那一份权重。

    为什么是各节点自己拉而不是控制机拉了再分发：控制机的上行是单点，15 台机器
    从它这里拿等于把并行的下载串行化了。各节点直连上游（或镜像），带宽是加起来的。

    清单**不**推过去 —— 各节点只需要知道自己那一份，而 `--node` 加上清单里的
    条目就够了。所以清单文件得在各节点上能读到（跟代码一起 rsync 过去即可）。
    """
    if not args.plan:
        print("fetch 需要 --plan（部署清单）")
        return 2
    if not (args.repo or args.endpoint or args.base_url or args.src):
        print("fetch 需要一个权重源：--repo / --endpoint / --base-url / --src")
        return 2
    out = args.out or "/data/p2pmoe-weights"
    _warn_default_workdir(args)

    def cmd_for(h) -> str:
        parts = [args.python, "-m", "p2pmoe.deploy.fetch",
                 "--plan", str(args.plan), "--node", h.node_id,
                 "--out", f"{out}/{h.node_id}" if args.per_node_dir else out,
                 "--mode", args.mode]
        # 权重源三选一，与 fetch 的口径一致
        if args.src:
            parts += ["--src", args.src]
        elif args.base_url:
            parts += ["--base-url", args.base_url]
        else:
            if args.repo:
                parts += ["--repo", args.repo, "--revision", args.revision]
            if args.endpoint:
                parts += ["--endpoint", args.endpoint]
        if args.transport and args.transport != "auto":
            parts += ["--transport", args.transport]
        return " ".join(shlex.quote(x) for x in parts)

    if args.dry_run:
        # 打**真正会执行的那条**，含 `cd`。少打 cd 的代价：workdir 配错时
        # dry-run 看起来完全正常，直到真跑才报 No module named 'p2pmoe'。
        for h in hosts:
            shown = (cmd_for(h) if args.local
                     else f"cd {shlex.quote(args.workdir)} && {cmd_for(h)}")
            print(f"  {h.node_id:<8} {shown}")
        return 0

    def run_one(h):
        try:
            if args.local:
                import subprocess
                r = subprocess.run(cmd_for(h), shell=True, cwd=args.workdir,
                                   capture_output=True, text=True)
                return h, r.returncode == 0, (r.stdout or r.stderr).strip()[-200:]
            full = f"cd {shlex.quote(args.workdir)} && {cmd_for(h)}"
            return (h, *_ssh(h, full, ssh=args.ssh, user=args.user,
                             timeout=args.timeout))
        except Exception as e:
            return h, False, f"{type(e).__name__}: {e}"[:200]

    # 各节点并行拉 —— 这正是不经控制机中转的意义
    stop = threading.Event()
    mon: threading.Thread | None = None
    if args.progress_every > 0 and not args.local:
        exp = _expected_bytes(args.plan, args.per_node_dir)
        if exp:
            print(f"  每 {args.progress_every:.0f}s 报一次进度"
                  f"（问的是目录大小，不是进程状态）")
            mon = threading.Thread(
                target=_progress_loop, args=(hosts, exp, out),
                kwargs=dict(ssh=args.ssh, user=args.user,
                            every=args.progress_every, stop=stop,
                            per_node_dir=args.per_node_dir, local=args.local),
                daemon=True)
            mon.start()
    try:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            rows = list(ex.map(run_one, hosts))
    finally:
        stop.set()
        if mon:
            mon.join(timeout=3)
        print()
    n_ok = 0
    for h, ok, msg in rows:
        print(f"  {h.node_id:<8} {'✓' if ok else '✗'} {_brief(msg)}")
        n_ok += ok
    print(f"\nfetch: {n_ok}/{len(hosts)} 成功")
    if n_ok == len(hosts):
        where = f"{out}/<节点id>" if args.per_node_dir else out
        print(f"\n各节点的权重在 {where}（每台只有自己那一份张量）")
        if not args.per_node_dir:
            print(f"下一步：控制器加 --model-dir {out} —— 对 15 台是同一个值，"
                  f"因为每台机器上那个目录里装的正好是它自己要的")
        print("提醒：这份权重是**按当时的驻留集**拉的。改了画像、覆盖率或分层之后"
              "要重拉，否则节点加载时会报「缺 N 个 key」")
    return 0 if n_ok == len(hosts) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="p2pmoe-launch",
                                 description="批量管理节点 agent")
    ap.add_argument("action",
                    choices=["start", "stop", "status", "probe", "agents", "systemd",
                             "fetch"])
    ap.add_argument("--hosts", type=Path, help="hosts 文件")
    ap.add_argument("--workdir", default=".", help="各节点上代码所在目录")
    ap.add_argument("--python", default="python3")
    ap.add_argument("--logdir", default="/tmp/p2pmoe", help="各节点上的日志目录")
    ap.add_argument("--ssh", default="ssh -o BatchMode=yes -o ConnectTimeout=10")
    ap.add_argument("--user", default=None)
    ap.add_argument("--local", action="store_true",
                    help="不走 ssh，在本机执行（用于同机演练与自测）")
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--progress-every", type=float, default=30.0, metavar="秒",
                    help="fetch 期间多久报一次进度（0=关）。问的是各节点输出目录"
                         "的大小 —— 几小时的操作没有中间输出的话，"
                         "分不清「在下」「卡住了」和「早就死了」")
    ap.add_argument("--timeout", type=float, default=4 * 3600, metavar="秒",
                    help="fetch 的单节点超时。默认 4 小时 —— 单台最多拉 22GB，"
                         "控制命令那套 60 秒的口径在这里必然超时")
    ap.add_argument("--k", type=int, default=8, help="probe 的采样次数")
    # fetch 子命令用
    ap.add_argument("--plan", type=Path, help="部署清单 JSON（fetch 用）")
    ap.add_argument("--repo", default=None, help="HF 仓库，如 Qwen/Qwen3-30B-A3B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--endpoint", default=None, help="HF 镜像地址")
    ap.add_argument("--base-url", default=None,
                    help="任意 base URL —— 局域网权重源（serve_weights）用这个")
    ap.add_argument("--src", default=None,
                    help="各节点都能读到的全量 checkpoint 目录（NFS/共享盘）")
    ap.add_argument("--transport", choices=("auto", "urllib", "curl"),
                    default="auto",
                    help="透传给各节点的 fetch。curl 能下而 Python 下不动时用")
    ap.add_argument("--out", default=None,
                    help="各节点上放权重的目录。**各机同一个路径** —— "
                         "节点 id 决定拉什么，不决定放哪，这样 --model-dir 对 "
                         "15 台是同一个值")
    ap.add_argument("--per-node-dir", action="store_true",
                    help="改成 <out>/<节点id>。只在同机演练（多个 agent 在一台机器上）"
                         "时需要，真机上别开")
    ap.add_argument("--mode", choices=("slice", "shard"), default="slice")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印各节点上该跑的命令，不执行。"
                         "**没配 ssh 就用它**：把命令拷到各机自己跑，效果一样 —— "
                         "ssh 只是这个脚本的便利，框架本身不需要它")
    # systemd 子命令用
    ap.add_argument("--id", default="v1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args(argv)

    if args.action == "systemd":
        print(SYSTEMD_UNIT.format(
            node_id=args.id, port=args.port, user=args.user or "p2pmoe",
            workdir=str(Path(args.workdir).resolve()), python=args.python, extra="",
        ))
        return 0

    if not args.hosts:
        ap.error("除 systemd 外都需要 --hosts")
    hosts = read_hosts(args.hosts)

    if args.action == "agents":
        print(",".join(f"{h.node_id}={h.host}:{h.port}" for h in hosts))
        return 0

    if args.action == "fetch":
        return _fetch_all(hosts, args)

    if args.action == "probe":
        return _probe_matrix(hosts, k=args.k, parallel=args.parallel)

    if args.action == "status":
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            rows = list(ex.map(probe_status, hosts))
        up = 0
        for h, s in rows:
            print(f"  {h.node_id:<8} {h.host}:{h.port:<6} {s}")
            up += s.startswith("✓")
        print(f"\n{up}/{len(hosts)} 台在线")
        if up < len(hosts):
            print("离线的节点会在控制器采集能力那步被剔除，不影响其余节点 —— "
                  "分散环境下这是常态")
        return 0 if up else 1

    if args.dry_run:
        # ssh 是 launch 的便利，不是框架的依赖。打印出来自己跑，结果一模一样。
        for h in hosts:
            cmd = (_start_cmd(h, args.workdir, args.python, args.logdir)
                   if args.action == "start" else _stop_cmd(h))
            print(f"# {h.node_id} @ {h.host}\n{cmd}\n")
        return 0

    if args.action == "start":
        _warn_default_workdir(args)

    def run_one(h: Host):
        try:
            if args.local:
                ok, msg = (_local_start(h, args.workdir, args.python, args.logdir)
                           if args.action == "start" else _local_stop(h))
            else:
                cmd = (_start_cmd(h, args.workdir, args.python, args.logdir)
                       if args.action == "start" else _stop_cmd(h))
                ok, msg = _ssh(h, cmd, ssh=args.ssh, user=args.user)
            return h, ok, msg[:120]
        except Exception as e:
            return h, False, f"{type(e).__name__}: {e}"[:120]

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        rows = list(ex.map(run_one, hosts))
    n_ok = 0
    for h, ok, msg in rows:
        print(f"  {h.node_id:<8} {h.host:<16} {'✓' if ok else '✗'} {_brief(msg)}")
        n_ok += ok
    print(f"\n{args.action}: {n_ok}/{len(hosts)} 成功")

    if args.action == "start" and n_ok:
        # PID **不是**存活证明 —— `setsid nohup ... &` 立刻返回 PID，进程随后
        # 因为缺依赖、端口占用、代码不在 workdir 而死掉，start 却报了 ✓。
        # 这里补一次真正的存活检查：等几秒，然后按 hosts 逐台探端口。
        import socket as _sock
        import time as _time

        print("\n验活（PID 只说明 fork 成功了，不说明进程还在）…")
        _time.sleep(5)
        dead = []
        for h in hosts:
            try:
                with _sock.create_connection((h.host, h.port), timeout=3):
                    pass
            except OSError:
                dead.append(h)
        if not dead:
            print(f"  ✓ {len(hosts)}/{len(hosts)} 台在听 {hosts[0].port}")
        else:
            print(f"  ✗ {len(dead)}/{len(hosts)} 台起来就死了："
                  f"{', '.join(h.node_id for h in dead)}")
            print("  —— 下面是它们各自日志的最后几行（死因通常就在这里）：")
            logf = f"{args.logdir}/agent-{{}}.log"
            for h in dead[:5]:
                print(f"\n  ── {h.node_id} @ {h.host} : {logf.format(h.node_id)}")
                if args.local:
                    try:
                        tail = Path(logf.format(h.node_id)).read_text(
                            encoding="utf-8", errors="replace").splitlines()[-12:]
                    except OSError as e:
                        tail = [f"（读不到日志：{e}）"]
                else:
                    # ssh 失败 与 日志不存在 是两回事 —— 前者是网络，后者是
                    # 代码没到。tail 对缺文件返回非零，混在一起会把「没同步」
                    # 误报成「连不上」，修法正好相反。
                    lp = shlex.quote(logf.format(h.node_id))
                    ok2, out = _ssh(
                        h, f"if [ -s {lp} ]; then tail -n 12 {lp}; "
                           f"else echo __NOLOG__; fi",
                        ssh=args.ssh, user=args.user)
                    if not ok2:
                        tail = [f"（ssh 连不上：{out.splitlines()[0] if out else ''}）"]
                    elif out.strip() == "__NOLOG__":
                        tail = [f"（日志不存在 —— 进程连启动都没到，"
                                f"`cd {args.workdir}` 就失败了，代码不在那儿）"]
                    else:
                        tail = out.splitlines()
                for ln in tail:
                    print(f"     {ln[:160]}")
            if len(dead) > 5:
                print(f"\n  （还有 {len(dead)-5} 台，同样看 {logf.format('<id>')}）")
            print("\n  最常见的三个原因：")
            print("    1. 节点上没装依赖 —— `pip3 install -r requirements-node.txt`")
            print(f"    2. 代码不在 {args.workdir} —— 先 ./deploy_15.sh sync")
            print(f"    3. 端口 {hosts[0].port} 被占 —— `ss -lntp | grep {hosts[0].port}`")
            return 1

    if args.action == "start":
        print("\n下一步（等几秒让 agent 起完）：")
        print(f"  python -m p2pmoe.deploy.launch status --hosts {args.hosts}"
              + (" --local" if args.local else ""))
        print(f"  python -m p2pmoe.deploy.control --agents "
              f"{','.join(f'{h.node_id}={h.host}:{h.port}' for h in hosts)} "
              f"--advertise <控制机IP>")
    return 0 if n_ok == len(hosts) else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""在**一台机器上**演练多机部署的完整流程。

    python examples/multinode_local.py [--nodes 8] [--requests 3]

它做的事和真机部署一字不差：用 `subprocess` 拉起 N 个**独立的 agent 进程**
（`python -m p2pmoe.deploy.agent`），然后跑真正的控制器
（`python -m p2pmoe.deploy.control`）去发现、探测、规划、下发、服务。

唯一的差别是地址都指向 127.0.0.1，于是探测出来的延迟是几十**微秒**而不是几十
毫秒。换到真机只需要把 `--agents` 里的地址换成各机 IP、加上 `--advertise`。

**注意这里没有任何延迟注入**：`e2e.py` 是单进程 fork + 人造延迟，用来验证协议
在分散环境的时序下成立；这个脚本走的是真实 socket、真实探测、真实 RPC，用来
验证部署路径本身。两者验证的是不同的东西。
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def free_port() -> int:
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_up(addr: tuple[str, int], timeout: float = 20.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            socket.create_connection(addr, timeout=0.5).close()
            return True
        except OSError:
            time.sleep(0.1)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--requests", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--mem-mb", type=float, default=26.0,
                    help="每台 agent 声明的可用内存（MB）。toy 模型下 26MB 够放一条前段")
    ap.add_argument("--emulate-wan", action="store_true", default=True,
                    help="给每个 agent 配一个模拟的出口接入段延迟（默认开）。"
                         "关掉就是纯本地速度 —— 那时「网络主导」的前提不成立，"
                         "公共带会窄到凑不出人口，规划会（正确地）失败")
    ap.add_argument("--no-emulate-wan", dest="emulate_wan", action="store_false")
    ap.add_argument("--bad-frac", type=float, default=0.2,
                    help="劣质接入节点的比例 —— 异类入口诊断要靠它才有戏演")
    ap.add_argument("--keep-logs", action="store_true")
    args = ap.parse_args()

    ids = [f"n{i+1}" for i in range(args.nodes)]
    ports = {i: free_port() for i in ids}

    # 每台 agent 一个自己的出口接入段（II.3.1 的加法结构）：
    # 逐对 RTT ≈ access(a) + access(b)，而**探测量到的就是它**，不是另开一套账。
    import numpy as np
    rng = np.random.default_rng(7)
    n_bad = int(round(args.bad_frac * len(ids)))
    bad = set(rng.choice(ids, size=n_bad, replace=False).tolist()) if n_bad else set()
    access = {
        i: (float(rng.uniform(28, 33)) if i in bad else float(rng.uniform(12, 16)),
            float(rng.uniform(4, 9)) * (1.7 if i in bad else 1.0))
        for i in ids
    }
    logdir = ROOT / ".multinode-logs"
    logdir.mkdir(exist_ok=True)

    procs: list[subprocess.Popen] = []
    print(f"启动 {len(ids)} 个 agent 进程…")
    try:
        for i in ids:
            lf = open(logdir / f"agent-{i}.log", "w", encoding="utf-8")
            p = subprocess.Popen(
                [sys.executable, "-m", "p2pmoe.deploy.agent",
                 "--id", i, "--bind", f"127.0.0.1:{ports[i]}",
                 "--mem-mb", str(args.mem_mb)]
                + (["--access-ms", f"{access[i][0]:.2f}",
                    "--access-jitter-ms", f"{access[i][1]:.2f}"]
                   if args.emulate_wan else []),
                cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
            )
            procs.append(p)

        for i in ids:
            if not wait_up(("127.0.0.1", ports[i])):
                print(f"agent {i} 没起来，看 {logdir/f'agent-{i}.log'}")
                return 1
        print(f"全部就绪：{', '.join(f'{i}:{ports[i]}' for i in ids)}")
        if args.emulate_wan:
            print("模拟接入段（ms）：" + ", ".join(
                f"{i}={access[i][0]:.0f}±{access[i][1]:.0f}"
                + ("[劣]" if i in bad else "") for i in ids) + "\n")

        agents = ",".join(f"{i}=127.0.0.1:{ports[i]}" for i in ids)
        rc = subprocess.call(
            [sys.executable, "-m", "p2pmoe.deploy.control",
             "--agents", agents, "--advertise", "127.0.0.1",
             "--requests", str(args.requests), "--tokens", str(args.tokens),
             "--once", "--save-plan", str(logdir / "plan.json")],
            cwd=ROOT,
        )
        print(f"\n控制器退出码 {rc}；清单存于 {logdir/'plan.json'}")
        return rc
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        if not args.keep_logs:
            for f in logdir.glob("agent-*.log"):
                f.unlink(missing_ok=True)
        print("agent 进程已全部停止")


if __name__ == "__main__":
    raise SystemExit(main())

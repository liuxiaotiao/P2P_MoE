#!/usr/bin/env python3
"""**给定布局就跑** —— 在一台机器上演练 `deploy/run.py` 的完整流程。

    python examples/manual_deploy.py                     # toy 模型，2 条通道
    python examples/manual_deploy.py --channels 5 --front 1 --back 2
    python examples/manual_deploy.py --real              # 造个微型真 checkpoint 来跑

它做的事和真机一字不差：起 N 个独立的 agent 进程，写一个布局文件，然后跑
`python -m p2pmoe.deploy.run`。唯一差别是地址都指向 127.0.0.1。

没有探测、没有规划 —— 布局是这个脚本按 `--channels/--front/--back` 直接排出来的，
你也可以 `--spec 自己的.json` 换成手写的。
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
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
    ap.add_argument("--channels", type=int, default=2, help="几条通道")
    ap.add_argument("--front", type=int, default=1, help="每条前段几台机器")
    ap.add_argument("--back", type=int, default=2, help="每条后段几台机器")
    ap.add_argument("--l0", type=int, default=None, help="前后段切点。默认取层数的 1/3")
    ap.add_argument("--spec", type=Path, default=None,
                    help="用你自己写的布局文件（这时 --channels 等被忽略）")
    ap.add_argument("--real", action="store_true",
                    help="造一个微型的真 Qwen3-MoE checkpoint 来跑（需要 torch）")
    ap.add_argument("--model-dir", default=None, help="真实 checkpoint 目录")
    ap.add_argument("--mem-mb", type=float, default=4000.0)
    ap.add_argument("--relay", action="store_true",
                    help="演练「节点之间没有直连」：起一个中继，agent 不监听端口，"
                         "全部挂到中继上")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--chat", action="store_true")
    args, passthrough = ap.parse_known_args()

    tmp = tempfile.TemporaryDirectory(prefix="p2pmoe-manual-")
    tmpd = Path(tmp.name)
    model_dir = args.model_dir
    n_layers = 8                       # toy 模型的层数

    if args.real and not model_dir:
        from p2pmoe.sim.fake_checkpoint import TINY_QWEN3_MOE, write_fake_checkpoint
        cfg = dict(TINY_QWEN3_MOE, num_hidden_layers=n_layers)
        model_dir = str(write_fake_checkpoint(tmpd / "ckpt", cfg, seed=0))
        print(f"造了个微型 checkpoint：{model_dir}（权重随机，输出无语义）")
    elif model_dir:
        # `{node}` 占位：各节点各一份权重。config 各节点一样，随便代一个读层数
        probe = model_dir
        if "{node}" in probe:
            import glob
            hits = sorted(glob.glob(probe.replace("{node}", "*") + "/config.json"))
            if not hits:
                raise SystemExit(f"{probe} 展开后找不到 config.json —— 权重拉好了吗？")
            probe = str(Path(hits[0]).parent)
        n_layers = int(json.loads(
            (Path(probe) / "config.json").read_text(encoding="utf-8")
        )["num_hidden_layers"])

    # ---- 布局 ---- #
    if args.spec:
        raw = json.loads(args.spec.read_text(encoding="utf-8"))
        ids = sorted({v for ch in raw["channels"]
                      for side in ("front", "back")
                      for v in ([ch[side]] if isinstance(ch[side], str) else
                                [x if isinstance(x, str) else x["node"]
                                 for x in ch[side]])})
        spec_path = args.spec
    else:
        l0 = args.l0 or max(1, n_layers // 3)
        per = args.front + args.back
        ids = [f"n{i+1}" for i in range(args.channels * per)]
        raw = {
            "channels": [
                {"front": ids[i * per: i * per + args.front],
                 "back": ids[i * per + args.front: (i + 1) * per]}
                for i in range(args.channels)
            ],
            "l0": l0,
        }
        if model_dir:
            raw["model_dir"] = model_dir
        spec_path = tmpd / "deploy.json"
        spec_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        print(f"布局（{spec_path}）：\n{json.dumps(raw, indent=2, ensure_ascii=False)}\n")

    ports = {i: free_port() for i in ids}
    logdir = ROOT / ".manual-logs"
    logdir.mkdir(exist_ok=True)
    procs: list[subprocess.Popen] = []

    try:
        relay_proc = None
        relay_addr = None
        if args.relay:
            rp = free_port()
            relay_addr = f"127.0.0.1:{rp}"
            lf = open(logdir / "relay.log", "w", encoding="utf-8")
            relay_proc = subprocess.Popen(
                [sys.executable, "-m", "p2pmoe.deploy.relay", "--bind", relay_addr],
                cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
            procs.append(relay_proc)
            if not wait_up(("127.0.0.1", rp)):
                print("中继没起来")
                return 1
            print(f"中继在 {relay_addr}；agent 将不监听任何端口\n")

        print(f"启动 {len(ids)} 个 agent 进程…")
        for i in ids:
            lf = open(logdir / f"agent-{i}.log", "w", encoding="utf-8")
            procs.append(subprocess.Popen(
                [sys.executable, "-m", "p2pmoe.deploy.agent", "--id", i,
                 "--bind", f"127.0.0.1:{ports[i]}", "--mem-mb", str(args.mem_mb)]
                + (["--relay", relay_addr] if relay_addr else []),
                cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT))
        if relay_addr:
            time.sleep(2.0)      # 等它们把连接挂上去（没有端口可探）
            print(f"{len(ids)} 个 agent 已挂到中继\n")
        else:
            for i in ids:
                if not wait_up(("127.0.0.1", ports[i])):
                    print(f"agent {i} 没起来，看 {logdir/f'agent-{i}.log'}")
                    return 1
            print(f"全部就绪：{', '.join(f'{i}:{ports[i]}' for i in ids)}\n")

        cmd = [sys.executable, "-m", "p2pmoe.deploy.run",
               "--spec", str(spec_path),
               # --spec 用的是你自己的文件，里面未必写了 model_dir；
               # 这里显式转过去，否则 --real 造出来的 checkpoint 用不上
               *(["--model-dir", model_dir] if model_dir else []),
               "--agents", ",".join(f"{i}=127.0.0.1:{ports[i]}" for i in ids),
               "--advertise", "127.0.0.1", "--tokens", str(args.tokens),
               "--ctx", "256", "--dtype-bytes", "4", "--once"]
        if relay_addr:
            cmd += ["--relay", relay_addr]
        for p in (args.prompt or []):
            cmd += ["--prompt", p]
        if args.chat:
            cmd.append("--chat")
        rc = subprocess.call(cmd + passthrough, cwd=ROOT)
        if relay_addr:
            # 问一下中继：接通了多少次、搬了多少字节。
            # 这是「数据面确实经过它」的直接证据 —— agent 压根没监听端口。
            try:
                from p2pmoe.runtime.wire import recv_msg, send_msg
                import socket as _s
                c = _s.create_connection(("127.0.0.1", rp), timeout=5)
                send_msg(c, {"type": "status"})
                st, _ = recv_msg(c)
                c.close()
                print(f"\n中继：接通 {st['spliced']} 次，搬运 "
                      f"{st['bytes']/1e6:.2f}MB，仍挂起 {st['parked']}")
            except OSError as e:
                print(f"\n（问不到中继统计：{e}）")
        print(f"\n退出码 {rc}")
        return rc
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
        print("agent 进程已全部停止")
        tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

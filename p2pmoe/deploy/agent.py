"""节点 agent —— 在每台真实机器上跑的守护进程。

    python -m p2pmoe.deploy.agent --id v1 --bind 0.0.0.0:9101
    python -m p2pmoe.deploy.agent --id g1 --bind 0.0.0.0:9101 --mem-mb 47000

agent 启动时**什么都不知道**：不知道自己要装哪几层、不知道池子里还有谁。
它先把服务起起来，然后等控制器来问、来量、来下发清单。

这个顺序不是设计选择，是被逼的：规划要先量到全网的逐对延迟才能决定层怎么切，
而量延迟又要求 agent 已经在跑并且能互相打包。先起服务、后收清单是唯一能解开
这个循环的顺序。

未配置态下 agent 只响应三类消息：
  capabilities  报告自己有多少内存、算力多快
  echo          给别人量延迟当靶子
  probe         按指令去量某个对端（**由本节点发起**，见 NodeServer.probe_peer）

收到 configure 后才加载模型 —— 而且只加载清单里点名的那几层的那些专家。
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from ..runtime.node import NodeServer

log = logging.getLogger("p2pmoe.agent")


def parse_bind(s: str) -> tuple[str, int]:
    if ":" not in s:
        return ("0.0.0.0", int(s))
    host, port = s.rsplit(":", 1)
    return (host or "0.0.0.0", int(port))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="p2pmoe-agent", description="P2P MoE 节点 agent"
    )
    ap.add_argument("--id", required=True, help="节点 id，全池唯一（如 g1 / v3）")
    ap.add_argument("--bind", default="0.0.0.0:9101", help="监听地址，默认 0.0.0.0:9101")
    ap.add_argument("--relay", default=None, metavar="HOST:PORT",
                    help="**节点之间没有直连时用这个**（家宽 NAT 后面等）："
                         "不监听任何端口，改成挂到中继上，只需要一条出站连接。"
                         "代价是每跳绕一圈，逐 token 延迟大致翻倍 —— "
                         "见 deploy/relay.py")
    ap.add_argument("--mem-mb", type=float, default=None,
                    help="声明可用内存（MB）。不给则读 /proc/meminfo；"
                         "真实 GPU 节点上应填显存")
    ap.add_argument("--access-ms", type=float, default=0.0,
                    help="模拟本节点的出口接入段延迟（ms）。**真机部署不要用** —— "
                         "网络自己会给。它只用于在一台机器上演练多机部署：本地 "
                         "socket 是几十微秒，跑不出分散环境的行为")
    ap.add_argument("--access-jitter-ms", type=float, default=0.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    host, port = parse_bind(args.bind)
    relay = None
    if args.relay:
        rh, _, rp = args.relay.rpartition(":")
        relay = (rh or "127.0.0.1", int(rp))
    srv = NodeServer(args.id, host=host, port=port, mem_mb=args.mem_mb,
                     access_ms=args.access_ms, access_jitter_ms=args.access_jitter_ms,
                     relay=relay)
    caps = srv.capabilities()
    log.info(
        "agent %s 已启动，%s；可用内存 %.0fMB，基准 %.3fms/单位%s",
        args.id,
        f"经中继 {relay[0]}:{relay[1]}（不监听端口）" if relay
        else f"监听 {host}:{srv.port}",
        caps["mem_mb"], caps["ms_per_layer"],
        f"；模拟接入段 {args.access_ms:.0f}±{args.access_jitter_ms:.0f}ms"
        if args.access_ms > 0 else "",
    )
    log.info("等待控制器下发清单…（未配置态只响应 capabilities / echo / probe）")

    def _bye(*_a):
        log.info("收到停止信号，退出")
        srv._stop.set()

    signal.signal(signal.SIGINT, _bye)
    signal.signal(signal.SIGTERM, _bye)

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    log.info("agent %s 已停止（累计处理 %d 条消息）", args.id, srv.n_msgs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""把一份全量 checkpoint 当作权重源发出去 —— **支持 Range 请求**。

    python -m p2pmoe.deploy.serve_weights --dir /data/Qwen3-Next-80B --bind 0.0.0.0:9400

然后各节点：

    python -m p2pmoe.deploy.fetch --plan plan.json --node N07 \\
        --base-url http://<这台的IP>:9400 --out ./weights

为什么需要这个
--------------
「只拉自己那份张量」靠的是 HTTP Range：先读几百 KB 的 safetensors 文件头，算出
要的张量在哪些字节区间，再按区间取。**源站不支持 Range，这件事就不成立。**

而 `python -m http.server` 恰恰不支持 —— 它无视 Range 头，返回 200 加整个文件。
`fetch` 会认出这一点并报 `RangeNotSupported`（不能将就：那意味着「省下载」根本
没发生，而按区间长度切出来的文件还会是错的）。所以这里自己实现一个。

什么时候用它
------------
上游拉不动的时候：链路掐大文件、没有外网、或者镜像不支持 Range。
流程变成「下一次全量 → 局域网切片」：

    一台机器  git clone / hf download  一份完整 checkpoint（160GB）
       ↓      serve_weights
    15 台     各自 fetch --base-url，合计再拉 141GB —— 但走的是局域网

比每台各拉一份全量省 94%，和直连上游的账一样；区别只是源站换成了自己人。

共享存储（NFS/Lustre）的话更简单：不用起服务，各节点直接
`fetch --src /mnt/shared/checkpoint`，走本地文件读，连 HTTP 都省了。

实现上的取舍
------------
只读、不列目录、路径限制在 `--dir` 之下。这不是通用文件服务器，是一次性的
权重源 —— 少一个功能就少一个能被绕出去的洞。
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import socketserver
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

log = logging.getLogger("p2pmoe.serve_weights")

__all__ = ["WeightServer", "make_handler"]

_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")
_CHUNK = 1 << 20


class _ThreadingServer(socketserver.ThreadingTCPServer):
    """15 台会同时来拉 —— 串行的话它们排成一队，局域网带宽白白空着。"""

    allow_reuse_address = True
    daemon_threads = True


def make_handler(root: Path, stats: dict):
    root = root.resolve()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "p2pmoe-weights"

        def log_message(self, fmt, *args):
            log.debug("%s - %s", self.address_string(), fmt % args)

        # -- 路径 ------------------------------------------------------- #
        def _resolve(self) -> Path | None:
            """把 URL 路径解到 root 之下。跳出 root 的一律拒绝。

            `..` 与符号链接都要挡：前者靠 resolve() 之后的归属检查，
            后者也是 —— resolve() 会跟随链接，所以链接指向 root 外面时
            同一个检查就能拦下。
            """
            rel = self.path.split("?", 1)[0].lstrip("/")
            try:
                p = (root / rel).resolve()
            except OSError:
                return None
            if p != root and root not in p.parents:
                return None
            return p if p.is_file() else None

        # -- 请求 ------------------------------------------------------- #
        def do_HEAD(self) -> None:
            self._serve(body=False)

        def do_GET(self) -> None:
            self._serve(body=True)

        def _serve(self, *, body: bool) -> None:
            p = self._resolve()
            if p is None:
                self.send_error(404, "not found")
                return
            size = p.stat().st_size
            rng = self.headers.get("Range")
            start, end = 0, size            # [start, end)

            if rng:
                m = _RANGE.match(rng.strip())
                if not m:
                    # 语法不认识就必须报 416，**不能退回 200** ——
                    # 退回去的话调用方会把整个文件当成它要的那一小段。
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                lo, hi = m.group(1), m.group(2)
                if lo == "":                       # bytes=-N：最后 N 字节
                    n = int(hi or 0)
                    start, end = max(0, size - n), size
                else:
                    start = int(lo)
                    end = size if hi == "" else min(size, int(hi) + 1)
                if start >= size or start >= end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

            n = end - start
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(n))
            self.send_header("Accept-Ranges", "bytes")
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end-1}/{size}")
            self.end_headers()
            if not body:
                return

            with open(p, "rb") as fh:
                fh.seek(start)
                left = n
                while left > 0:
                    buf = fh.read(min(_CHUNK, left))
                    if not buf:
                        break
                    try:
                        self.wfile.write(buf)
                    except (BrokenPipeError, ConnectionResetError):
                        return          # 对端走了，不是错误
                    left -= len(buf)
            stats["bytes"] = stats.get("bytes", 0) + (n - left)
            stats["reqs"] = stats.get("reqs", 0) + 1

    return Handler


class WeightServer:
    """起一个只读的、支持 Range 的权重源。"""

    def __init__(self, directory: str | Path, host: str = "0.0.0.0", port: int = 9400):
        self.root = Path(directory).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"{self.root} 不是目录")
        self.stats: dict = {"bytes": 0, "reqs": 0}
        self._srv = _ThreadingServer((host, port), make_handler(self.root, self.stats))
        self.host, self.port = self._srv.server_address[:2]
        self._t: threading.Thread | None = None

    def start(self) -> "WeightServer":
        self._t = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._t.start()
        return self

    def stop(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()
        if self._t:
            self._t.join(timeout=5)

    def __enter__(self) -> "WeightServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="p2pmoe-serve-weights",
        description="把一份全量 checkpoint 当作支持 Range 的权重源发出去")
    ap.add_argument("--dir", required=True, help="全量 checkpoint 目录")
    ap.add_argument("--bind", default="0.0.0.0:9400", help="监听地址")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")
    host, _, port = args.bind.rpartition(":")
    d = Path(args.dir)
    shards = sorted(d.glob("*.safetensors"))
    if not shards:
        log.error("%s 里没有 .safetensors —— 逐张量拉取要它的文件头。", d)
        log.error("  （.bin 是 pickle，做不到只读部分张量）")
        return 2

    # 起服务之前先看一眼分片是不是真的权重。
    # 发一堆 LFS 指针出去的话，15 台会各自拿到 130 字节的文本，然后在解析
    # 文件头时报「头长不合理」—— 那个错离真正的原因隔了三层。
    from p2pmoe.deploy.fetch import looks_like_lfs_pointer

    ptr = [f for f in shards
           if f.stat().st_size < 4096
           and looks_like_lfs_pointer(f.read_bytes()[:64])]
    if ptr:
        log.error("%s 里有 %d/%d 个分片是 **git-lfs 指针**，不是权重。",
                  d, len(ptr), len(shards))
        log.error("  `GIT_LFS_SKIP_SMUDGE=1 git clone` 只拉仓库结构 —— "
                  "目录看着对，每个分片却只有 130 字节文本。")
        log.error("  补上：")
        log.error("      cd %s && git lfs pull", d)
        log.error("  或者换个工具（能断点续传，比 git 稳）：")
        log.error("      hf download <仓库> --local-dir %s", d)
        return 2
    total = sum(f.stat().st_size for f in shards)

    srv = WeightServer(d, host or "0.0.0.0", int(port))
    log.info("权重源 http://%s:%d  ←  %s", host or "0.0.0.0", srv.port, srv.root)
    log.info("  %d 个分片，共 %.1f GB，**支持 Range**", len(shards), total / 1e9)
    log.info("")
    log.info("  各节点这样拉（只拉自己那份）：")
    log.info("      python -m p2pmoe.deploy.fetch --plan <清单> --node <ID> \\")
    log.info("          --base-url http://<这台的IP>:%d --out <本地目录>", srv.port)
    log.info("")
    log.info("  Ctrl-C 结束")
    srv.start()
    try:
        while True:
            threading.Event().wait(30)
            if srv.stats["reqs"]:
                log.info("  已发出 %d 次请求，%.2f GB",
                         srv.stats["reqs"], srv.stats["bytes"] / 1e9)
    except KeyboardInterrupt:
        pass
    srv.stop()
    log.info("停。共 %d 次请求，%.2f GB", srv.stats["reqs"], srv.stats["bytes"] / 1e9)
    return 0


if __name__ == "__main__":
    sys.exit(main())

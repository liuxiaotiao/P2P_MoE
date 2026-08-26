#!/usr/bin/env python3
"""只拉这台机器要的那部分权重 —— 不下整个 checkpoint。

    # 每台节点上（n3 只是它自己的节点名）
    python3 -m p2pmoe.deploy.fetch --repo Qwen/Qwen3-30B-A3B \\
        --plan plan.json --node n3 --out /data/qwen3-part

61GB 的模型，一台只承载几层的几十个专家 —— 让它下 61GB 是荒谬的。而选择性加载
已经知道自己要哪些 key 了（`runtime/weights.py`），这里把同一份 key 集合往上游推
一层，变成「只下这些字节」。

两种粒度
--------
**`--mode slice`（默认，逐张量）** —— safetensors 的文件头里有每个张量的
`data_offsets`，而 HuggingFace 的 CDN 支持 HTTP Range。于是可以只读文件头（几百 KB），
算出需要的字节区间，再把这些区间拉下来，本地重新拼成一个**更小但完全合法**的
safetensors。下载量 ≈ 驻留量。

**`--mode shard`（整分片）** —— 只下含有目标 key 的那几个分片文件。省得少，但不依赖
Range 请求，代理/镜像不支持 Range 时用它兜底。

产出的是一个**可以直接用的 checkpoint 目录**（config.json + 单文件权重 + 索引），
`--model-dir` 指向它即可，节点侧代码一行都不用改。

限制，说清楚
------------
* 只认 safetensors。pickle 的 `.bin` 没有逐张量偏移，做不了这件事 —— 这也是
  整套方案一开始就要求 safetensors 的原因；
* 逐张量模式要求上游支持 Range 请求。HF 官方 CDN 支持；自建镜像不一定，
  不支持时脚本会说出来并建议 `--mode shard`；
* **拉下来的是「按当前驻留集」的那一份**。改了画像、改了覆盖率、重新分层之后，
  驻留集变了，得重拉（增量补拉见 TODO.md）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..planner.manifest import DeploymentManifest
from ..runtime.weights import KeyPlan, qwen_moe_keys

log = logging.getLogger("p2pmoe.fetch")

__all__ = ["Source", "TensorSpec", "FetchPlan", "plan_fetch", "fetch",
           "RangeNotSupported"]


class RangeNotSupported(RuntimeError):
    """上游无视了 Range 请求。

    单独一个类型是因为它**不能和「文件不存在」混为一谈**：两者都会让
    `exists()` 返回假，于是脚本会以为这是个单文件布局的 checkpoint，
    接着去找一个不存在的 model.safetensors，最后报一个 404 —— 而真正的问题
    是镜像不支持 Range。诊断信息跑偏比失败本身更费时间。
    """

_ALIGN = 8            # safetensors 要求数据段按 8 字节对齐
_HDR_MAX = 200 << 20  # 文件头上限，防止畸形文件把内存吃光


# --------------------------------------------------------------------------- #
def _total_from_headers(text: str) -> int:
    """从响应头里读文件总长。Content-Range 优先 —— 只有它在分段响应里也是全长。"""
    total = 0
    for line in text.splitlines():
        low = line.lower()
        if low.startswith("content-range:"):
            tail = line.rpartition("/")[2].strip()
            if tail.isdigit():
                total = int(tail)
        elif low.startswith("content-length:") and not total:
            v = line.partition(":")[2].strip()
            if v.isdigit():
                total = int(v)
    return total


class Source:
    """权重从哪来 —— HF 仓库、任意 base URL，或本地目录。"""

    def __init__(self, *, repo: str | None = None, revision: str = "main",
                 base_url: str | None = None, local: str | Path | None = None,
                 token: str | None = None, endpoint: str | None = None,
                 timeout: float = 60.0, retries: int = 5,
                 transport: str = "auto"):
        self.local = Path(local) if local else None
        self.timeout = timeout
        self.retries = max(1, retries)
        self._len: dict[str, int] = {}
        # "urllib" | "curl" | "auto"（urllib 失败一次就永久切到 curl）
        #
        # 为什么要有第二条传输：curl 能下、Python 下不动，是很常见的一对症状。
        # 两者用的**不是同一套 TLS** —— conda 环境自带 OpenSSL，系统 curl 用系统的；
        # 版本、cipher、ALPN、代理处理都可能不同。与其查清是哪一处不同，
        # 不如换用那个已经证明能跑的。
        self.transport = transport
        self._curl_ok: bool | None = None
        self.token = token or os.environ.get("HF_TOKEN")
        if self.local is None:
            if base_url:
                self.base = base_url.rstrip("/")
            elif repo:
                ep = (endpoint or os.environ.get("HF_ENDPOINT")
                      or "https://huggingface.co").rstrip("/")
                self.base = f"{ep}/{repo}/resolve/{revision}"
            else:
                raise ValueError("要么给 --repo/--base-url，要么给 --src")
        else:
            self.base = str(self.local)

    def __str__(self) -> str:
        return self.base

    # -- 取字节 ------------------------------------------------------------ #
    def read(self, name: str, start: int | None = None, end: int | None = None) -> bytes:
        """读 name 的 [start, end) 字节；不给区间就整读。

        **断了会从断点续。** 一次性 `urlopen().read()` 的问题是：连接在中途被
        掐断时，已经收到的几 MB 全部作废，重来一次大概率断在同一个地方。

        「小文件过得去、大文件过不去」是个很常见的形状 —— TLS 检查型代理掐长
        连接、MTU 黑洞丢大包、CDN 重置，都会这样。它们的共同点是**总能传一段**，
        所以按已收字节数续传就能爬完，不必弄清到底是哪一种。

        续传本身要求上游支持 Range。不支持的话第一次就会被 `RangeNotSupported`
        挡下来，走不到这里。
        """
        if self.local is not None:
            with open(self.local / name, "rb") as fh:
                if start is None:
                    return fh.read()
                fh.seek(start)
                return fh.read(end - start)

        buf = bytearray()
        want = None if start is None else end - start
        stalls = 0          # **连续**没进展的次数。有进展就清零。
        while True:
            lo = (start or 0) + len(buf)
            before = len(buf)
            try:
                chunk, status = self._get(name, lo, end,
                                          ranged=start is not None or bool(buf))
                buf += chunk
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                stalls += 1
                if stalls >= self.retries:
                    raise
                log.debug("%s 断于 %d 字节（%s），%.1fs 后续传…",
                          name, len(buf), type(e).__name__, 0.5 * stalls)
                time.sleep(0.5 * stalls)
                continue

            if want is not None:
                if len(buf) >= want:
                    return bytes(buf[:want])
            else:
                # 整读：拿 Content-Length / Content-Range 判完没完。
                # **不能因为「返回了 200」就认为读完了** —— 200 说的是请求成功，
                # 不是响应体完整；连接在中途断掉时状态码早就发出去了。
                total = self._len.get(name) or 0
                if total:
                    if len(buf) >= total:
                        return bytes(buf[:total])
                elif chunk:
                    # 两个长度头都没有，只能信「一次读完」——
                    # 否则会一直向后要，拿回空响应，转到重试上限为止。
                    return bytes(buf)

            # 重试计数只数**没进展**的轮次：链路每次只肯传几百 KB 时，
            # 那不是失败，是慢 —— 数成失败的话大文件永远爬不完。
            if len(buf) == before:
                stalls += 1
                if stalls >= self.retries:
                    if want is None:
                        return bytes(buf)
                    raise OSError(
                        f"{name}: 只拿到 {len(buf)}/{want} 字节，"
                        f"连续 {stalls} 次没有进展")
                time.sleep(0.5 * stalls)
            else:
                stalls = 0

    def _have_curl(self) -> bool:
        """PATH 里有还不够 —— 真跑一次 `curl --version`。

        装了但跑不起来（缺库、权限、被壳子包住）的情况是有的，
        而在那种情况下切过去只是把一个错换成另一个错。
        """
        if self._curl_ok is None:
            if shutil.which("curl") is None:
                self._curl_ok = False
            else:
                try:
                    r = subprocess.run(["curl", "--version"],
                                       capture_output=True, timeout=10)
                    self._curl_ok = r.returncode == 0
                    if self._curl_ok:
                        first = r.stdout.decode(errors="replace").splitlines()[:1]
                        log.debug("curl 可用：%s", first[0] if first else "?")
                except Exception:
                    self._curl_ok = False
        return self._curl_ok

    def _get(self, name: str, lo: int, hi: int | None, *, ranged: bool):
        if self.transport == "curl":
            return self._get_curl(name, lo, hi, ranged=ranged)
        try:
            return self._get_urllib(name, lo, hi, ranged=ranged)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            # RangeNotSupported 是对端行为，换传输也不会变 —— 不该触发切换
            if isinstance(e, RangeNotSupported) or self.transport != "auto":
                raise
            if not self._have_curl():
                raise
            # **决定切的那一刻就粘住**，不是等 curl 成功之后。
            # 放在成功之后的话，curl 本身有问题时每一次请求都会重走一遍
            # urllib（同样的超时、同样的等待），然后打同一句话 —— 而真正的
            # curl 报错被淹没在重复里。
            self.transport = "curl"
            log.warning("urllib 取 %s 失败（%s）—— 本次运行余下的请求改用 curl。",
                        name, e)
            log.warning("  这两者用的不是同一套 TLS："
                        "conda 环境自带 OpenSSL，系统 curl 用系统的。")
            return self._get_curl(name, lo, hi, ranged=ranged)

    def _get_curl(self, name: str, lo: int, hi: int | None, *, ranged: bool):
        """用 curl 取一段。`-r lo-hi` 就是 Range 头。

        body 与 header 分开落盘 —— body 是二进制，混在一起没法解析；
        而状态码是必须拿到的：206 还是 200 决定了对端有没有理会 Range。
        """
        if not self._have_curl():
            raise OSError("需要 curl，但 PATH 里没有")
        with tempfile.TemporaryDirectory() as td:
            body, hdr = Path(td) / "b", Path(td) / "h"
            # 只用**很老的 curl 也有**的选项：-sS -L --max-time -D -o -r -H。
            # 别用 --fail / --fail-with-body：前者会把错误响应体丢掉，
            # 后者要 curl ≥ 7.76 —— 而集群里的 curl 常常比这老得多。
            # 状态码从 -D 落下来的头文件里读，判断权留在自己手上。
            cmd = ["curl", "-sS", "-L",
                   "--max-time", str(int(self.timeout)),
                   "-D", str(hdr), "-o", str(body)]
            if self.token:
                cmd += ["-H", f"Authorization: Bearer {self.token}"]
            if ranged:
                cmd += ["-r", f"{lo}-{hi - 1}" if hi is not None else f"{lo}-"]
            cmd.append(f"{self.base}/{name}")
            r = subprocess.run(cmd, capture_output=True, timeout=self.timeout + 30)
            text = hdr.read_text(errors="replace") if hdr.exists() else ""
            # -L 之后 header 文件里有多段，最后一段才是最终响应
            codes = [int(l.split()[1]) for l in text.splitlines()
                     if l.startswith("HTTP/") and len(l.split()) > 1]
            status = codes[-1] if codes else 0
            if r.returncode != 0:
                raise OSError(f"curl 退出码 {r.returncode}："
                              f"{r.stderr.decode(errors='replace').strip()[:200]}")
            # **两条传输必须抛同一种异常。** 调用方靠 HTTPError.code 区分
            # 「仓库里没有这个文件」（404，可选文件正常缺席）和「取失败了」；
            # 这边抛 OSError 的话，一个可有可无的 special_tokens_map.json
            # 就能让整轮下载失败。
            if status >= 400:
                raise urllib.error.HTTPError(
                    f"{self.base}/{name}", status, f"HTTP {status}", None, None)
            if ranged and hi is not None and lo == 0 and status != 206:
                raise RangeNotSupported(
                    f"{self.base} 不支持 Range 请求（请求 {name} 的一小段，"
                    f"对端返回 {status} 把整个文件发了回来）—— "
                    f"改用 --mode shard，或换一个支持 Range 的源")
            self._len[name] = self._len.get(name) or _total_from_headers(text)
            return body.read_bytes(), status

    def _get_urllib(self, name: str, lo: int, hi: int | None, *, ranged: bool):
        req = urllib.request.Request(f"{self.base}/{name}")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        if ranged:
            req.add_header("Range",
                           f"bytes={lo}-{hi - 1}" if hi is not None else f"bytes={lo}-")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            if ranged and hi is not None and lo == 0 and r.status != 206:
                # 200 表示对端无视了 Range 把整个文件发了过来。**必须报错而不是
                # 将就**：那意味着「省下载」这件事根本没发生，而拼出来的文件还
                # 会是错的（我们按区间长度去切）。
                raise RangeNotSupported(
                    f"{self.base} 不支持 Range 请求（请求 {name} 的一小段，"
                    f"对端返回 {r.status} 把整个文件发了回来）—— "
                    f"改用 --mode shard，或换一个支持 Range 的源"
                )
            self._len[name] = self._len.get(name) or (
                int(r.headers.get("Content-Range", "").rpartition("/")[2] or 0)
                or (int(r.headers.get("Content-Length") or 0) if r.status == 200 else 0))
            return r.read(), r.status

    def _done(self, name: str, got: int) -> bool:
        total = self._len.get(name) or 0
        return bool(total) and got >= total

    def read_json(self, name: str) -> dict:
        return json.loads(self.read(name).decode("utf-8"))

    def exists(self, name: str) -> bool:
        """文件在不在。**不吞 RangeNotSupported** —— 那是另一回事，见该类的说明。"""
        if self.local is not None:
            return (self.local / name).exists()
        try:
            self.read(name, 0, 1)
            return True
        except (urllib.error.HTTPError, urllib.error.URLError):
            return False


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TensorSpec:
    """一个张量在某个分片里的位置。dtype 与 shape 原样搬运，不做任何解释。"""

    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    """相对**文件开头**的绝对偏移（已加上 8 + 头长）。"""
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass
class FetchPlan:
    tensors: list[TensorSpec] = field(default_factory=list)
    shards_needed: list[str] = field(default_factory=list)
    shards_total: int = 0
    bytes_total: int = 0
    """整个 checkpoint 的权重字节数。"""
    bytes_shards: int = 0
    """整分片模式要下多少。"""
    shards_read: int = 0
    """为了算区间读了几个分片的头 —— 只读需要的那几个。"""
    missing: list[str] = field(default_factory=list)

    @property
    def bytes_slice(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    def summary(self) -> str:
        def gb(x: int) -> str:
            return f"{x / 1e9:.2f}GB" if x >= 1e8 else f"{x / 1e6:.1f}MB"

        return (
            f"{len(self.tensors)} 个张量，分布在 {len(self.shards_needed)}/"
            f"{self.shards_total} 个分片里\n"
            f"    逐张量  {gb(self.bytes_slice)}"
            f"（{self.bytes_slice / max(self.bytes_total, 1):.1%}）\n"
            f"    整分片  {gb(self.bytes_shards)}"
            f"（{self.bytes_shards / max(self.bytes_total, 1):.1%}）\n"
            f"    全量    {gb(self.bytes_total)}"
        )


# --------------------------------------------------------------------------- #
_LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def looks_like_lfs_pointer(head: bytes) -> bool:
    """这几个字节是 LFS 指针文本，不是 safetensors。

    `GIT_LFS_SKIP_SMUDGE=1 git clone` 只把仓库结构拉下来，每个大文件留一个
    130 字节左右的指针::

        version https://git-lfs.github.com/spec/v1
        oid sha256:9f86d0…
        size 4294967296

    目录看起来是对的 —— 41 个 `.safetensors` 都在，只是每个都是文本。
    不认出来的话，报错会变成「头长 1936026161 不合理」之类，指向完全错误的方向。
    """
    return head[:len(_LFS_MAGIC)] == _LFS_MAGIC


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """把重定向拦下来，好看清它想把我们送到哪个域名去。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise _Redirected(newurl)


class _Redirected(Exception):
    def __init__(self, url: str):
        super().__init__(url)
        self.url = url


def lack_would_be_empty(got: list[str], have: list[str]) -> bool:
    """两个必需文件都到位了 —— 有几个可选文件取失败无所谓，别做多余的诊断。"""
    ok = set(got) | set(have)
    return {"config.json", "tokenizer.json"} <= ok


def _meta_file_ok(path: "Path") -> bool:
    """本地已有的元数据文件是不是真能用。

    手工下载最常见的坑不是没下到，是**下到了一个 HTML 错误页**（登录墙、
    403、镜像的提示页），文件存在、大小也不为零，直到 tokenizer 加载时才炸。
    这里当场验一次，比留到几步之后便宜得多。

    只做**结构**检查，不 import `tokenizers` —— 这个模块跑在 15 台推理节点上，
    而节点自始至终只见 token id，不该为了校验一个文件多装一套分词器
    （`test_requirements.py::test_node_never_needs_a_tokenizer` 守着这条线）。
    结构检查足够挡住 HTML 错误页，那才是要挡的东西。
    """
    try:
        if not path.name.endswith(".json"):
            return path.stat().st_size > 0
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False
        if path.name == "tokenizer.json":
            # HF 的 tokenizer.json 一定有 model；多数还有 added_tokens。
            # 错误页连合法 JSON 都不是，走不到这儿。
            return isinstance(data.get("model"), dict)
        return True
    except Exception:
        return False


def diagnose(src: Source, probe: str = "tokenizer.json") -> None:
    """「小文件过得去、大文件过不去」的第二种解释：**它们不在同一个域名上**。

    HF 把小文件（config.json、tokenizer_config.json）由 huggingface.co 直接发，
    而 LFS 文件（tokenizer.json、所有 safetensors）**302 重定向到 CDN** ——
    `cdn-lfs*.huggingface.co` 或 `*.xethub.hf.co`。

    企业网络里常见的情形是白名单只放行了 huggingface.co，CDN 那个域名没放。
    症状就长得像「大文件传不动」，但真正要做的不是调 MTU、不是换镜像，
    而是让 IT 把 CDN 域名也加进去。这个函数就是把该报给 IT 的域名找出来。
    """
    from urllib.parse import urlsplit

    if src.local is not None:
        log.info("  本地源，无需诊断网络")
        return

    log.info("  ── 源站诊断 ──")
    base_host = urlsplit(src.base).netloc

    # 1. 小文件：走不走得通 base 这个域名
    try:
        src.read("config.json")
        log.info("  ✓ %s 可达（config.json 取到了）", base_host)
    except Exception as e:
        log.error("  ✗ %s 不可达：%s", base_host, e)
        log.error("    连主域名都不通 —— 先查代理与 DNS，别往下看了。")
        return

    # 2. 大文件被重定向到哪儿
    op = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(f"{src.base}/{probe}", method="HEAD")
    if src.token:
        req.add_header("Authorization", f"Bearer {src.token}")
    target = None
    try:
        op.open(req, timeout=src.timeout)
        log.info("  · %s 没有重定向 —— 与小文件同一个域名", probe)
    except _Redirected as r:
        target = r.url
        log.info("  · %s → 重定向到 %s", probe, urlsplit(target).netloc)
    except Exception as e:
        log.info("  · 问不出 %s 的重定向目标（%s）", probe, type(e).__name__)

    if not target:
        log.error("  大文件和小文件在同一个域名上，那不是域名的问题 —— "
                  "回去查代理 / MTU / CDN 抖动。")
        return

    # 3. 那个域名连不连得上
    host = urlsplit(target).netloc
    try:
        r2 = urllib.request.Request(target)
        with urllib.request.urlopen(r2, timeout=src.timeout) as resp:
            resp.read(65536)
        log.info("  ✓ %s 也可达 —— 域名不是原因", host)
    except Exception as e:
        log.error("  ✗ **%s 连不上**：%s", host, e)
        log.error("")
        log.error("  这就是原因：小文件由 %s 直接发，", base_host)
        log.error("  而 LFS 文件（tokenizer.json 与**全部 safetensors**）"
                  "重定向到 %s。", host)
        log.error("  企业网络的白名单常常只放行了前者。")
        log.error("")
        log.error("  要 IT 放行的域名：")
        log.error("      %s", base_host)
        log.error("      %s", host)
        log.error("      （HF 的 CDN 还会用 cdn-lfs*.huggingface.co、"
                  "*.xethub.hf.co，一并放行）")
        log.error("")
        log.error("  等不及的话：在一台能连上 %s 的机器上下全量，", host)
        log.error("  再用 ./deploy_15.sh serve-weights 从局域网发给 15 台。")


def read_header(src: Source, shard: str) -> tuple[dict, int]:
    """读 safetensors 的文件头。返回 (头 JSON, 数据段起始偏移)。

    布局是 `[8 字节小端 u64 = 头长 N][N 字节 JSON][数据]`，头里每个张量的
    `data_offsets` 是**相对数据段**的，所以绝对偏移要加 8 + N。
    """
    head = src.read(shard, 0, 64)
    if looks_like_lfs_pointer(head):
        raise ValueError(
            f"{shard} 是 **git-lfs 指针**，不是权重。\n"
            f"  `GIT_LFS_SKIP_SMUDGE=1 git clone` 只拉仓库结构，大文件留一个"
            f"130 字节的指针 —— 目录看着是对的，41 个分片都在，但每个都是文本。\n"
            f"  补上真正的权重：\n"
            f"      cd <克隆出来的目录> && git lfs pull\n"
            f"  或者换个工具（能断点续传，比 git 稳）：\n"
            f"      hf download Qwen/Qwen3-Next-80B-A3B-Instruct --local-dir ./ckpt")
    n = int.from_bytes(head[:8], "little")
    if not 0 < n <= _HDR_MAX:
        raise ValueError(f"{shard} 的头长 {n} 不合理 —— 这是 safetensors 文件吗？")
    return json.loads(src.read(shard, 8, 8 + n).decode("utf-8")), 8 + n


def plan_fetch(src: Source, keys: Iterable[str]) -> FetchPlan:
    """算清楚要下哪些字节。**只读文件头**，一个张量都不碰。"""
    want = set(keys)
    index = (src.read_json("model.safetensors.index.json")
             if src.exists("model.safetensors.index.json") else None)
    weight_map: dict[str, str] = dict(index["weight_map"]) if index else {}

    if weight_map:
        all_shards = sorted(set(weight_map.values()))
        # **只读需要的那几个分片的头。** 61GB 的模型有十几个分片，为了算个
        # 「全量多大」去挨个读头是白费往返 —— 索引的 metadata 里就有总量。
        touch = sorted({weight_map[k] for k in want if k in weight_map})
        missing_from_index = sorted(want - set(weight_map))
    else:
        all_shards = touch = ["model.safetensors"]     # 单文件布局
        missing_from_index = []

    plan = FetchPlan(shards_total=len(all_shards))
    per_shard_bytes: dict[str, int] = {}

    for shard in touch:
        header, data_at = read_header(src, shard)
        end_of_data = 0
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            st, en = meta["data_offsets"]
            end_of_data = max(end_of_data, en)
            if name in want:
                plan.tensors.append(TensorSpec(
                    name=name, shard=shard, dtype=meta["dtype"],
                    shape=tuple(int(x) for x in meta["shape"]),
                    start=data_at + st, end=data_at + en,
                ))
        per_shard_bytes[shard] = end_of_data

    got = {t.name for t in plan.tensors}
    plan.missing = sorted(want - got) or missing_from_index
    plan.shards_needed = sorted({t.shard for t in plan.tensors})
    plan.bytes_shards = sum(per_shard_bytes.get(s, 0) for s in plan.shards_needed)
    meta_total = int((index or {}).get("metadata", {}).get("total_size", 0))
    plan.bytes_total = meta_total or sum(per_shard_bytes.values())
    plan.shards_read = len(touch)
    # 顺序稳定：按 (分片, 偏移)，这样下载是顺序读，也便于合并相邻区间
    plan.tensors.sort(key=lambda t: (t.shard, t.start))
    return plan


# --------------------------------------------------------------------------- #
def _coalesce(specs: Sequence[TensorSpec], gap: int) -> list[tuple[int, int]]:
    """把相邻的字节区间并成一次请求 —— 隔一小段就多发一次请求不划算。"""
    out: list[tuple[int, int]] = []
    for t in specs:
        if out and t.start - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], t.end))
        else:
            out.append((t.start, t.end))
    return out


def _write_safetensors(path: Path, specs: Sequence[TensorSpec],
                       blobs: Mapping[str, bytes]) -> int:
    """按 safetensors 的格式把拿到的张量拼成一个新文件。

    dtype 与 shape 原样搬运 —— 我们从头到尾没有解释过一个字节，只是搬运。
    这也是为什么不需要 torch：拼文件是纯字节操作。
    """
    header: dict = {"__metadata__": {"format": "pt"}}
    off = 0
    for t in specs:
        header[t.name] = {"dtype": t.dtype, "shape": list(t.shape),
                          "data_offsets": [off, off + t.nbytes]}
        off += t.nbytes
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(raw)) % _ALIGN
    raw += b" " * pad

    with open(path, "wb") as fh:
        fh.write(len(raw).to_bytes(8, "little"))
        fh.write(raw)
        for t in specs:
            b = blobs[t.name]
            if len(b) != t.nbytes:
                raise RuntimeError(
                    f"{t.name} 拿到 {len(b)} 字节，应该是 {t.nbytes} —— "
                    f"上游可能没有正确处理 Range 请求")
            fh.write(b)
    return path.stat().st_size


def fetch(src: Source, keys: Iterable[str], out_dir: str | Path, *,
          mode: str = "slice", gap: int = 1 << 20,
          extra_files: Sequence[str] = (), dry_run: bool = False) -> FetchPlan:
    """拉权重并在 out_dir 拼出一个可用的 checkpoint 目录。"""
    out = Path(out_dir)
    plan = plan_fetch(src, keys)
    if plan.missing:
        raise KeyError(
            f"上游缺 {len(plan.missing)} 个 key，首个是 {plan.missing[0]} —— "
            f"模型对不上？（层号是 1-based 转 0-based 的，见 weights.qwen_moe_keys）")
    if dry_run:
        return plan

    out.mkdir(parents=True, exist_ok=True)
    for name in extra_files:
        if src.exists(name):
            (out / name).write_bytes(src.read(name))
            log.info("  取 %s", name)

    if mode == "shard":
        for shard in plan.shards_needed:
            log.info("  下整分片 %s", shard)
            (out / shard).write_bytes(src.read(shard))
        idx = {"metadata": {"total_size": plan.bytes_shards},
               "weight_map": {t.name: t.shard for t in plan.tensors}}
        (out / "model.safetensors.index.json").write_text(
            json.dumps(idx, indent=2), encoding="utf-8")
        return plan

    # ---- 逐张量 ---- #
    blobs: dict[str, bytes] = {}
    done = 0
    for shard in plan.shards_needed:
        specs = [t for t in plan.tensors if t.shard == shard]
        ranges = _coalesce(specs, gap)
        chunks = {}
        for lo, hi in ranges:
            chunks[(lo, hi)] = src.read(shard, lo, hi)
        for t in specs:
            for (lo, hi), buf in chunks.items():
                if lo <= t.start and t.end <= hi:
                    blobs[t.name] = buf[t.start - lo: t.end - lo]
                    break
        done += sum(hi - lo for lo, hi in ranges)
        log.info("  %s：%d 个张量 / %d 次请求 / %.1fMB（累计 %.1fMB）",
                 shard, len(specs), len(ranges),
                 sum(t.nbytes for t in specs) / 1e6, done / 1e6)

    size = _write_safetensors(out / "model.safetensors", plan.tensors, blobs)
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": size},
                    "weight_map": {t.name: "model.safetensors"
                                   for t in plan.tensors}}, indent=2),
        encoding="utf-8")
    return plan


# --------------------------------------------------------------------------- #
def keys_for_node(manifest: DeploymentManifest, node: str, *,
                  config: dict | None = None,
                  tie_word_embeddings: bool | None = None) -> set[str]:
    """清单里这台机器要哪些 key —— 与节点加载时**按同一套规则**分派。

    下载侧和加载侧对「要哪些 key」的理解必须一致，否则症状是
    「下完了，加载时报缺 N 个 key」—— 而那时 141GB 已经下完了。

    分派按 checkpoint 自报的架构走，与 `NodeServer._build_torch` 里那段
    逐字同源。**不按「有没有某个字段」去猜** —— 猜错的代价是下错一整套权重。

    Qwen3-Next 与 Qwen3-MoE 的 key 集合差得很远，不是能糊弄过去的小差异：

    * 逐层不同 —— 36 层 Gated DeltaNet 用 `linear_attn.*`，12 层全注意力才有
      `self_attn.*`。拿 MoE 的方案去要，前者的 key 在 checkpoint 里根本不存在。
    * **共享专家** `mlp.shared_expert.*` —— 每个 token 都激活，不参与驻留集裁剪。
      MoE 的方案里没有它，漏了不会报错，只会让后段每一层都少加一路输出。

    `config` 就是 checkpoint 的 `config.json`。不给的话退回 Qwen3-MoE 方案 ——
    那是这个函数原来的行为，保留给只有清单、拿不到 config 的调用方。
    """
    p = manifest.plan_for(node)
    if p is None:
        raise SystemExit(
            f"清单里没有节点 {node}；有的是 {sorted(x.node for x in manifest.nodes)}")
    plan = KeyPlan(
        layer_experts={l.layer: list(l.experts) for l in p.layers},
        with_embed=(p.role == "front" and p.is_head),
        with_lm_head=(p.role.startswith("back:") and p.is_tail),
    )
    cfg = config or {}
    if tie_word_embeddings is None:
        tie_word_embeddings = bool(cfg.get("tie_word_embeddings"))

    arch = str(cfg.get("model_type", "")).lower()
    archs = [str(a).lower() for a in cfg.get("architectures", [])]
    if arch == "qwen3_next" or any("qwen3next" in a for a in archs):
        from p2pmoe.runtime.qwen3_next import NextModelConfig
        from p2pmoe.runtime.weights import qwen3_next_keys

        mcfg = NextModelConfig.from_hf(cfg)
        return qwen3_next_keys(plan, layer_types=list(mcfg.layer_types),
                               shared_expert=mcfg.shared_intermediate > 0,
                               tie_word_embeddings=mcfg.tie_word_embeddings)
    return qwen_moe_keys(plan, tie_word_embeddings=tie_word_embeddings)


TOKENIZER_FILES = ("config.json", "generation_config.json", "tokenizer.json",
                   "tokenizer_config.json", "special_tokens_map.json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="p2pmoe-fetch",
                                 description="只拉本机需要的那部分权重")
    ap.add_argument("--repo", default=None, help="HF 仓库，如 Qwen/Qwen3-30B-A3B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--endpoint", default=None,
                    help="HF 镜像地址（也可用 HF_ENDPOINT 环境变量）")
    ap.add_argument("--base-url", default=None, help="任意 base URL（自建权重服务）")
    ap.add_argument("--src", default=None, help="本地目录（共享存储/已下好的全量）")
    ap.add_argument("--plan", type=Path, default=None,
                    help="部署清单 JSON（--meta-only 时不需要）")
    ap.add_argument("--node", default=None,
                    help="本机在清单里的节点 id（--meta-only 时不需要）")
    ap.add_argument("--out", type=Path, default=None,
                    help="拼好的 checkpoint 放哪（--diagnose 时不需要）")
    ap.add_argument("--mode", choices=("slice", "shard"), default="slice",
                    help="slice=逐张量（省得多，要 Range 支持）；shard=整分片（兜底）")
    ap.add_argument("--gap-mb", type=float, default=1.0,
                    help="相邻区间间隔小于它就并成一次请求")
    ap.add_argument("--meta-only", action="store_true",
                    help="只取 config.json 与 tokenizer（约 10MB），**不碰任何张量**。"
                         "控制机要用 —— 它不跑模型，但要读 config 算模型规格、"
                         "读 tokenizer 编解码文本。给它 141GB 权重是浪费")
    ap.add_argument("--no-tokenizer", action="store_true",
                    help="不取 tokenizer（节点其实用不到，默认取是为了目录自洽）")
    ap.add_argument("--transport", choices=("auto", "urllib", "curl"),
                    default="auto",
                    help="auto=先用 Python 的 urllib，失败一次就整轮改用 curl。"
                         "curl 能下而 Python 下不动很常见 —— 两者用的不是同一套 "
                         "TLS（conda 自带 OpenSSL，系统 curl 用系统的）")
    ap.add_argument("--force", action="store_true",
                    help="--meta-only 时重取已存在的文件（默认跳过）")
    ap.add_argument("--diagnose", action="store_true",
                    help="只做源站诊断：小文件与大文件是不是同一个域名、"
                         "CDN 那个域名通不通。企业白名单漏放 CDN 时用它定位")
    ap.add_argument("--dry-run", action="store_true", help="只算要下多少，不下")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")
    src = Source(repo=args.repo, revision=args.revision, base_url=args.base_url,
                 local=args.src, endpoint=args.endpoint,
                 transport=args.transport)
    if not args.diagnose and args.out is None:
        ap.error("要给 --out")
    if not (args.meta_only or args.diagnose) and not (args.plan and args.node):
        ap.error("要么给 --plan 与 --node（拉本机那份权重），"
                 "要么给 --meta-only（只拉 config 与 tokenizer）"
                 "或 --diagnose（只诊断源站）")
    man = (DeploymentManifest.from_json(args.plan.read_text(encoding="utf-8"))
           if args.plan else None)

    # --diagnose 要在读 config 之前 —— 读不到 config 恰恰是它要诊断的情形之一
    if args.diagnose:
        diagnose(src)
        return 0

    try:
        cfg = src.read_json("config.json")
    except (urllib.error.URLError, OSError, ValueError) as e:
        log.error("读不到 %s 的 config.json：%s", src, e)
        log.error("  · 私有/受限仓库要 HF_TOKEN（`hf auth login` 之后在 "
                  "~/.cache/huggingface/token）")
        log.error("  · 连接被中途掐断：查代理（$https_proxy）、MTU、"
                  "或换 --endpoint")
        log.error("  · 已经有全量 checkpoint 的话用 --src <目录> 从本地取")
        return 2

    if args.meta_only:
        # 控制机这条路：不看清单、不算 key、一个张量都不下。
        try:
            args.out.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            # /data 之类的系统目录要 root，而这套东西不该需要 root。
            # 裸 traceback 会让人去查 pathlib，而真正要改的是一个路径。
            log.error("建不了 %s：%s", args.out, e)
            log.error("  换一个写得进去的地方，比如项目目录下：")
            log.error("      export WEIGHTS=$PWD/weights")
            log.error("      ./deploy_15.sh meta")
            log.error("  注意 15 台节点上这个路径要**一致** —— 清单按同一个 "
                      "--model-dir 下发。")
            return 2
        got: list[str] = []
        have: list[str] = []                 # 已经在本地了，跳过
        absent: list[str] = []               # 仓库里确实没有（404）
        failed: list[tuple[str, str]] = []   # 有，但没取下来 —— 完全是另一回事
        for f in TOKENIZER_FILES:
            # 已经有了就不重取。手工 curl 补上的那一个尤其不能被覆盖 ——
            # 在会掐断的链路上，那可能是唯一一次成功。
            cur = args.out / f
            if not args.force and cur.exists() and cur.stat().st_size > 0:
                if _meta_file_ok(cur):
                    have.append(f)
                    continue
                log.warning("  %s 已存在但读不通（%d 字节）—— 重取。"
                            "手工下载常见的坑：拿到的是错误页而不是文件。",
                            f, cur.stat().st_size)
            try:
                # tokenizer.json 是这批里唯一的大文件（真模型上约 11MB），
                # 也是唯一会因为超时而失败的。重试两次 —— 比让人重跑整条命令便宜。
                last: Exception | None = None
                for attempt in range(3):
                    try:
                        (args.out / f).write_bytes(src.read(f))
                        last = None
                        break
                    except (urllib.error.HTTPError, FileNotFoundError):
                        raise      # 404 / 本地真的没这个文件 —— 重试只是白等
                    except (urllib.error.URLError, TimeoutError, OSError) as e:
                        last = e
                        if attempt < 2:
                            log.info("  %s 第 %d 次失败（%s），重试…",
                                     f, attempt + 1, type(e).__name__)
                            time.sleep(1.5 * (attempt + 1))
                if last is not None:
                    raise last
                got.append(f)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    absent.append(f)
                else:
                    failed.append((f, f"HTTP {e.code}"))
            except FileNotFoundError:
                absent.append(f)
            except (urllib.error.URLError, OSError, ValueError) as e:
                failed.append((f, f"{type(e).__name__}: {e}"))
        n = sum((args.out / f).stat().st_size for f in got + have)
        log.info("元数据 → %s（%d 个文件，%.1f MB）",
                 args.out, len(got) + len(have), n / 1e6)
        if got:
            log.info("  取到：%s", ", ".join(got))
        if have:
            log.info("  已有（跳过，--force 可强制重取）：%s", ", ".join(have))
        if absent:
            log.info("  仓库里没有：%s", ", ".join(absent))

        # 「仓库里没有」与「下载失败」必须分开报。混成一句的话，一次网络抖动
        # 看起来会像「这个仓库就是没有 tokenizer」—— 而那个错要到几步之后
        # （--chat 渲染模板、prompt 编码）才发作，到时候没人会想到是这里丢的。
        if failed:
            log.error("这些文件**取失败了**（不是仓库里没有）：")
            for f, why in failed:
                log.error("    %s —— %s", f, why)
            # SSL EOF / 连接被重置：小文件过得去、大文件过不去，是链路在中途
            # 掐断，不是 HF 的问题。重试同一个地址没用 —— 换出口才有用。
            cut = any(k in w for _, w in failed for k in
                      ("UNEXPECTED_EOF", "ConnectionReset", "Connection reset",
                       "EOF occurred", "timed out", "TimeoutError"))
            ep = os.environ.get("HF_ENDPOINT")
            if cut:
                log.error("  这个错的形状是**链路在传输中途被掐断**："
                          "几 KB 的文件都过了，只有 11MB 的 tokenizer.json 没过。")
                log.error("  代码已经会按已收字节续传，还是不行的话，"
                          "先用 curl 复现一次、看是不是同一处断：")
                log.error("      curl -v -o /dev/null %s/tokenizer.json",
                          src.base)
                log.error("  常见的三个原因（都不是 HF 的问题）：")
                log.error("    · TLS 检查型代理/防火墙掐长连接 —— 查 "
                          "$https_proxy / $HTTPS_PROXY，或让 IT 放行 huggingface.co")
                log.error("    · MTU 黑洞（小包过、大包丢，VPN/隧道上常见）—— "
                          "试 `ip link set dev <网卡> mtu 1400` 再跑")
                log.error("    · 出口 CDN 抖动 —— 换个时间，或换 --endpoint 到"
                          "别的镜像")
                if ep:
                    log.error("  当前 --endpoint 是 %s；试试去掉它直连。", ep)
            else:
                log.error("  重跑一次试试。")
            log.error("  绕过去也行 —— 就一个文件，用什么下都可以：")
            log.error("      curl -L -o %s/tokenizer.json \\", args.out)
            log.error("          %s/tokenizer.json", src.base)
            log.error("      # 或者已有全量 checkpoint：--src <目录>")

        if failed and not lack_would_be_empty(got, have):
            try:
                diagnose(src)
            except Exception as e:                  # 诊断本身不能盖住原始错误
                log.debug("诊断失败：%s", e)

        need = {"config.json": "控制机没它算不了模型规格",
                "tokenizer.json": "控制机没它没法把 prompt 编成 id、把 id 解回文本"}
        ok_set = set(got) | set(have)
        lack = [(f, why) for f, why in need.items() if f not in ok_set]
        if lack:
            for f, why in lack:
                log.error("✗ 缺 %s —— %s", f, why)
            return 2
        if "tokenizer_config.json" not in ok_set:
            log.warning("没有 tokenizer_config.json —— --chat 套不了对话模板，"
                        "指令模型的输出会像坏了")
        log.info("控制机的 --model-dir 指到这里即可。**这里没有权重**，"
                 "节点上的同名目录才有。")
        return 0

    keys = keys_for_node(man, args.node, config=cfg)
    p = man.plan_for(args.node)
    arch = str(cfg.get("model_type", "?"))
    log.info("节点 %s：%s，层 %d–%d，%d 个专家（架构 %s，%d 个张量）",
             args.node, p.role, p.layer_range[0], p.layer_range[1],
             sum(len(l.experts) for l in p.layers), arch, len(keys))
    log.info("来源 %s", src)

    try:
        args.out.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        log.error("建不了 %s：%s", args.out, e)
        log.error("  换一个写得进去的地方（15 台要一致），比如 $WORKDIR/weights")
        return 2

    t0 = time.perf_counter()
    try:
        plan = fetch(src, keys, args.out, mode=args.mode,
                     gap=int(args.gap_mb * 1e6),
                     extra_files=() if args.no_tokenizer else TOKENIZER_FILES,
                     dry_run=args.dry_run)
    except (KeyError, RuntimeError, urllib.error.URLError, OSError) as e:
        log.error("%s", e)
        return 2
    log.info("要下的：\n    %s", plan.summary())
    if args.dry_run:
        log.info("（--dry-run，没有真下）")
        return 0
    log.info("完成，用时 %.1fs → %s", time.perf_counter() - t0, args.out)
    log.info("节点上把 --model-dir 指到 %s 即可", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

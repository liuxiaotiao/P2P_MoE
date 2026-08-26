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
class Source:
    """权重从哪来 —— HF 仓库、任意 base URL，或本地目录。"""

    def __init__(self, *, repo: str | None = None, revision: str = "main",
                 base_url: str | None = None, local: str | Path | None = None,
                 token: str | None = None, endpoint: str | None = None,
                 timeout: float = 60.0):
        self.local = Path(local) if local else None
        self.timeout = timeout
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
        """读 name 的 [start, end) 字节；不给区间就整读。"""
        if self.local is not None:
            with open(self.local / name, "rb") as fh:
                if start is None:
                    return fh.read()
                fh.seek(start)
                return fh.read(end - start)

        req = urllib.request.Request(f"{self.base}/{name}")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        if start is not None:
            req.add_header("Range", f"bytes={start}-{end - 1}")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            if start is not None and r.status != 206:
                # 200 表示对端无视了 Range 把整个文件发了过来。**必须报错而不是
                # 将就**：那意味着「省下载」这件事根本没发生，而拼出来的文件还
                # 会是错的（我们按区间长度去切）。
                raise RangeNotSupported(
                    f"{self.base} 不支持 Range 请求（请求 {name} 的一小段，"
                    f"对端返回 {r.status} 把整个文件发了回来）—— "
                    f"改用 --mode shard，或换一个支持 Range 的镜像"
                )
            return r.read()

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
def read_header(src: Source, shard: str) -> tuple[dict, int]:
    """读 safetensors 的文件头。返回 (头 JSON, 数据段起始偏移)。

    布局是 `[8 字节小端 u64 = 头长 N][N 字节 JSON][数据]`，头里每个张量的
    `data_offsets` 是**相对数据段**的，所以绝对偏移要加 8 + N。
    """
    n = int.from_bytes(src.read(shard, 0, 8), "little")
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
                  tie_word_embeddings: bool = False) -> set[str]:
    """清单里这台机器要哪些 key —— 与节点加载时用的是**同一个函数**。

    同一个函数很重要：下载和加载对「要哪些 key」的理解必须一致，
    否则会出现「下了却加载不到」或「加载时才发现少文件」。
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
    ap.add_argument("--plan", type=Path, required=True, help="部署清单 JSON")
    ap.add_argument("--node", required=True, help="本机在清单里的节点 id")
    ap.add_argument("--out", type=Path, required=True, help="拼好的 checkpoint 放哪")
    ap.add_argument("--mode", choices=("slice", "shard"), default="slice",
                    help="slice=逐张量（省得多，要 Range 支持）；shard=整分片（兜底）")
    ap.add_argument("--gap-mb", type=float, default=1.0,
                    help="相邻区间间隔小于它就并成一次请求")
    ap.add_argument("--no-tokenizer", action="store_true",
                    help="不取 tokenizer（节点其实用不到，默认取是为了目录自洽）")
    ap.add_argument("--dry-run", action="store_true", help="只算要下多少，不下")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")
    src = Source(repo=args.repo, revision=args.revision, base_url=args.base_url,
                 local=args.src, endpoint=args.endpoint)
    man = DeploymentManifest.from_json(args.plan.read_text(encoding="utf-8"))

    cfg = src.read_json("config.json")
    keys = keys_for_node(man, args.node,
                         tie_word_embeddings=bool(cfg.get("tie_word_embeddings")))
    p = man.plan_for(args.node)
    log.info("节点 %s：%s，层 %d–%d，%d 个专家",
             args.node, p.role, p.layer_range[0], p.layer_range[1],
             sum(len(l.experts) for l in p.layers))
    log.info("来源 %s", src)

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

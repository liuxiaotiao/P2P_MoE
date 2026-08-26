#!/usr/bin/env python3
"""按**你给定的布局**部署并推理 —— 不探测、不规划、不优化。

    python3 -m p2pmoe.deploy.run --spec deploy.json \\
        --agents "$(python3 -m p2pmoe.deploy.launch agents --hosts hosts.txt)" \\
        --advertise 10.0.0.1 --chat --prompt "你好"

与 `deploy/control.py` 的分工：

| | control.py | run.py（本文件） |
|---|---|---|
| 放置 | 规划器算（分档 → L₀ → 建段 → 收紧） | **你写在布局文件里** |
| 连接 | 自动配对或 `--wiring` | **你写在布局文件里** |
| 探测 | 逐对实测，几分钟 | 不测 |
| 前提 | 组合极差压到抖动量级以下 | 不提供 |

所以本文件跑出来的东西**不享有**方案文档的那些结论（零后悔、任意组合、
延迟均匀）—— 它只保证「按你说的装上、连上、能出 token」。

流程只有四步：连上 → 预检 → 下发 → 服务。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Mapping

from ..planner.hf_config import model_spec_from_hf
from ..planner.types import ModelSpec
from ..runtime.coordinator import Coordinator
from ..runtime.model import ToyMoEConfig
from ..runtime.text import TextIO
from ..runtime.wire import Addr, LinkTable, PeerPool, rpc
from . import control
from .control import (
    ModelSetup,
    check_model_dirs,
    distribute,
    parse_agents,
    toy_model_spec,
    _guess_local_ip,
)
from ..runtime.profile import (
    ActivationRecord,
    load_profile,
    save_profile,
    summarise,
)
from .manual import ManualSpec, build_manual_manifest, memory_report

log = logging.getLogger("p2pmoe.run")


# --------------------------------------------------------------------------- #
def collect_memory(addrs: dict[str, Addr], *, real_units: bool) -> dict[str, float]:
    """只问内存，不建 Node 表 —— 手动放置不需要规划器的那套输入。"""
    out: dict[str, float] = {}
    for name, addr in addrs.items():
        try:
            r = rpc(addr, {"type": "capabilities"}, timeout=15.0,
                    relay=control.RELAY, to=name)
        except Exception as e:
            log.warning("  %s 问不到内存（%s）—— 内存核对会跳过它", name, e)
            continue
        mb = float(r.get("mem_mb", 0.0))
        out[name] = mb / 1024.0 if real_units else mb
    return out


def _any_node_dir(model_dir: str, spec_raw: Mapping) -> str:
    """把 `{node}` 代成布局里任意一个节点 —— 控制机只读 config/tokenizer。

    这两个文件各节点完全一样（`fetch` 会把它们复制到每一份里），所以代谁都行。
    真正逐节点不同的是权重，那是节点自己加载时才用的。
    """
    if "{node}" not in model_dir:
        return model_dir
    for ch in spec_raw.get("channels", []):
        for side in ("front", "back"):
            v = ch.get(side)
            if isinstance(v, str):
                return model_dir.replace("{node}", v)
            if isinstance(v, (list, tuple)) and v:
                first = v[0]
                return model_dir.replace(
                    "{node}", first if isinstance(first, str) else str(first["node"]))
    raise SystemExit("--model-dir 里有 {node} 占位，但布局里一个节点都没有")


def collect_profiles(addrs: dict[str, Addr], man, *, n_experts: int
                     ) -> dict[str, ActivationRecord]:
    """向各后段节点收逐层激活画像，按 task 合并。

    只收后段：前段是 task 无关的（I.1.1），它装全集或并集，不按 task 裁。
    同一个 task 有多条通道时，各通道的同一层加在一起 —— 样本更多，分布更稳。
    """
    out: dict[str, ActivationRecord] = {}
    for p in man.nodes:
        if not p.role.startswith("back:") or p.node not in addrs:
            continue
        task = p.role.split(":", 1)[1]
        try:
            r = rpc(addrs[p.node], {"type": "get_profile"}, timeout=60.0,
                    relay=control.RELAY, to=p.node)
        except Exception as e:
            log.warning("      %s 的画像取不到（%s）—— 它那几层会缺", p.node, e)
            continue
        if not r.get("layers"):
            log.warning("      %s 没有画像数据 —— 这一轮没开 --profile-out？", p.node)
            continue
        rec = out.setdefault(task, ActivationRecord(task=task, n_experts=n_experts))
        rec.add_wire(r)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="p2pmoe-run",
                                 description="按给定布局部署并推理（不规划）")
    ap.add_argument("--spec", type=Path, required=True, help="布局文件，见 manual.py")
    ap.add_argument("--agents", default=None, help="v1=host:port,… 节点 id 与地址")
    ap.add_argument("--plan-only", action="store_true",
                    help="只把布局翻译成部署清单并存下来，**不连节点**。"
                         "先有清单才能用 deploy.fetch 只拉本机那部分权重 —— "
                         "而清单只是 JSON，不需要权重也不需要机器在线")
    ap.add_argument("--advertise", default=None,
                    help="控制机对节点可见的 IP。跨机部署必填")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--model-dir", default=None,
                    help="覆盖布局文件里的 model_dir。不给就是 toy 模型")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--dtype-bytes", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--chat", action="store_true", help="套对话模板（指令模型必须）")
    ap.add_argument("--system", default=None)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--requests", type=int, default=None,
                    help="打几条。默认等于 prompt 条数")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--profile", default=None,
                    help="激活画像 JSON —— 后段按它只装子集。"
                         "不给就全装（无近似）")
    ap.add_argument("--coverage", type=float, default=0.95,
                    help="--profile 的覆盖率阈值：逐层按激活质量降序累加到这个比例。"
                         "调它不用重新采样")
    ap.add_argument("--profile-out", type=Path, default=None,
                    help="**采画像**：这一轮结束后把各后段的逐层路由统计写出来。"
                         "必须在全装（不给 --profile）的前提下采，否则采到的是"
                         "被 drop-expert 带偏的路由")
    ap.add_argument("--warmup", type=int, default=0,
                    help="正式计时前先打几条丢弃的请求。**测时延时务必给** —— "
                         "torch 的第一次前向包含 kernel 选择与惰性初始化，"
                         "不预热的话首请求量到的大半是冷启动开销，不是稳态时延")
    ap.add_argument("--timing", action="store_true",
                    help="每条请求打一份时序报告：总时延怎么分掉的、"
                         "各节点算了多久、算力使用率")
    ap.add_argument("--miss-policy", default="drop",
                    choices=("drop", "drop_noscale", "local_topk"),
                    help="专家不在本地时怎么补救。见 runtime/node.py 的 NodeConfig")
    ap.add_argument("--relay", default=None, metavar="HOST:PORT",
                    help="节点之间没有直连时的中继（deploy/relay.py）。"
                         "各节点的 agent 也要加同一个 --relay")
    ap.add_argument("--skip-model-check", action="store_true")
    ap.add_argument("--save-plan", type=Path, default=None)
    ap.add_argument("--once", action="store_true", help="跑完就退出（不常驻）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if not args.agents and not args.plan_only:
        ap.error("需要 --agents（或用 --plan-only 只出清单）")
    if args.plan_only and not args.save_plan:
        ap.error("--plan-only 得配 --save-plan 说明存到哪")
    if args.relay:
        rh, _, rp = args.relay.rpartition(":")
        control.RELAY = (rh or "127.0.0.1", int(rp))
        log.info("中继模式：%s:%d —— 节点之间不直连，每跳绕一圈，"
                 "逐 token 延迟大致翻倍", *control.RELAY)
    addrs = parse_agents(args.agents) if args.agents else {}

    # ---------------- 0. 模型 -------------------------------------------- #
    raw = json.loads(args.spec.read_text(encoding="utf-8"))
    model_dir = args.model_dir or raw.get("model_dir")
    if model_dir:
        # `{node}` 占位：各节点上那份权重的目录。真机上各机同路径（不写占位），
        # 同机演练时每个 agent 一个目录。控制机这边只读 config/tokenizer ——
        # 它们各节点是一样的，所以拿任意一个节点的目录代入即可。
        d = Path(_any_node_dir(model_dir, raw))
        if not (d / "config.json").exists():
            log.error("%s/config.json 不存在 —— model_dir 要指向 HF checkpoint 目录", d)
            return 2
        hf = json.loads((d / "config.json").read_text(encoding="utf-8"))
        spec_m, info = model_spec_from_hf(hf, name=d.name, ctx_max=args.ctx,
                                          dtype_bytes=args.dtype_bytes)
        backend, node_model, name = "torch", hf, d.name
        log.info("[0/4] %s", info.summary())
    else:
        mcfg = ToyMoEConfig()
        spec_m, backend, node_model, name = (toy_model_spec(mcfg), "numpy",
                                             dict(mcfg.__dict__), "toy")
        log.info("[0/4] toy 模型（%d 层 / %d 专家，numpy）—— 只验链路，不是真模型",
                 mcfg.n_layers, mcfg.n_experts)

    # ---------------- 1. 布局 -------------------------------------------- #
    try:
        layout = ManualSpec.from_dict(raw, n_layers=spec_m.n_layers)
    except ValueError as e:
        log.error("%s", e)
        return 2
    if args.profile_out and args.profile:
        log.error("--profile-out 和 --profile 不能同时给：采画像必须在全装的前提下做，"
                  "只驻留子集时输出被 drop-expert 带偏，后面几层的路由就不是真实路由了")
        return 2
    if args.profile:
        try:
            layout.apply_profile(args.profile, coverage=args.coverage,
                                 n_experts=spec_m.n_experts, top_k=spec_m.top_k)
        except (ValueError, FileNotFoundError) as e:
            log.error("画像用不了：%s", e)
            return 2

    want = set(layout.all_nodes())
    if args.plan_only:
        man, wired = build_manual_manifest(layout, spec_m, model_name=name)
        args.save_plan.write_text(man.to_json(), encoding="utf-8")
        log.info("[1/4] 布局 → 清单：%d 条通道，%d 台机器，L₀=%d",
                 len(layout.channels), len(want), layout.l0)
        for f, (b, u) in sorted(wired.items()):
            log.info("      %s%s → %s%s  [%s]", f, tuple(man.segments[f]["nodes"]),
                     b, tuple(man.segments[b]["nodes"]), u)
        log.info("清单已存 %s", args.save_plan)
        log.info("下一步：各节点按它只拉自己那部分权重\n"
                 "  python3 -m p2pmoe.deploy.launch fetch --hosts hosts.txt "
                 "--plan %s --repo <模型> --out /data/part", args.save_plan)
        return 0

    missing = want - set(addrs)
    if missing:
        log.error("布局里的节点 %s 不在 --agents 里", sorted(missing))
        return 2
    idle = set(addrs) - want
    log.info("[1/4] 布局：%d 条通道，用 %d 台机器%s，L₀=%d（前段 1..%d，后段 %d..%d）",
             len(layout.channels), len(want),
             f"（另有 %d 台没用上：%s）" % (len(idle), sorted(idle)) if idle else "",
             layout.l0, layout.l0, layout.l0 + 1, spec_m.n_layers)
    for i, ch in enumerate(layout.channels):
        log.info("      ch%d [%s]  前段 %s → 后段 %s", i, ch.task,
                 " ".join(f"{v}:{lo}-{hi}"
                          for v, (lo, hi) in zip(ch.front.nodes, ch.front.splits)),
                 " ".join(f"{v}:{lo}-{hi}"
                          for v, (lo, hi) in zip(ch.back.nodes, ch.back.splits)))
    if args.profile:
        for line in summarise(load_profile(args.profile), coverage=args.coverage,
                              min_experts=spec_m.top_k):
            log.info("      专家（画像 @ 覆盖率 %.0f%%）：%s", args.coverage * 100, line)
    else:
        n_sub = sum(1 for l in range(1, spec_m.n_layers + 1) if l in layout.experts)
        log.info("      专家：%s", f"{n_sub} 层由 experts 指定，其余全装" if n_sub else
                 "全装（无 drop-expert 近似，输出与单机逐位一致）")
        if args.profile_out:
            log.info("      本轮采画像 → %s", args.profile_out)

    # ---------------- 2. 预检 -------------------------------------------- #
    log.info("[2/4] 预检")
    avail = collect_memory({k: v for k, v in addrs.items() if k in want},
                           real_units=backend == "torch")
    unit = "GB" if backend == "torch" else "MB(当GB用)"
    rows = memory_report(layout, spec_m, avail)
    tight = [r for r in rows if r[2] and r[1] > r[2]]
    for v, need, have in rows[:3] if tight else []:
        log.warning("      ⚠ %s 需要 %.1f%s，只报了 %.1f%s —— 加载时可能 OOM",
                    v, need, unit, have, unit)
    if not tight:
        v, need, have = rows[0]
        fmt = "%.1f" if need >= 0.1 else "%.3f"      # 微型模型别显示成 0.0
        log.info(f"      内存：最紧的是 %s（要 {fmt} / 有 %.1f %s）", v, need, have, unit)
    if model_dir and not args.skip_model_check:
        bad = check_model_dirs({k: v for k, v in addrs.items() if k in want}, model_dir)
        # check_model 是逐节点问的，占位由各节点自己代入 —— 见 control.distribute
        if bad:
            log.error("以下节点读不到 checkpoint：")
            for b in bad:
                log.error("    %s", b)
            log.error("权重分发还没做 —— --model-dir 是**各节点上的本地路径**。"
                      "先同步到每台机器，或加 --skip-model-check 自担风险")
            return 3
        log.info("      checkpoint：%d 台都读得到", len(want))

    # ---------------- 3. 下发 -------------------------------------------- #
    man, wired = build_manual_manifest(layout, spec_m, model_name=name)
    if args.save_plan:
        args.save_plan.write_text(man.to_json(), encoding="utf-8")
        log.info("      清单已存 %s", args.save_plan)

    textio = None
    if model_dir:
        try:
            textio = TextIO.from_model_dir(_any_node_dir(model_dir, raw),
                                           chat=args.chat, system=args.system)
            log.info("      tokenizer：vocab %d，停止 token %s，%s",
                     textio.tok.vocab_size, sorted(textio.stop.ids),
                     "套对话模板" if args.chat else "completion 模式")
        except (FileNotFoundError, ImportError) as e:
            log.warning("      没有文本层（%s）→ 请求走 token id", e)

    host = args.advertise or (
        "127.0.0.1" if all(a[0] in ("127.0.0.1", "localhost") for a in addrs.values())
        else _guess_local_ip(next(iter(addrs.values()))))
    coord = Coordinator(man, baselines={}, priors={}, static_wiring=wired,
                        host=args.bind, textio=textio, relay=control.RELAY)
    coord.max_tokens = args.tokens
    log.info("[3/4] 下发（协调器 %s:%d）", host, coord.port)
    setup = ModelSetup(label=name, spec=spec_m, n_layers=spec_m.n_layers,
                       n_experts=spec_m.n_experts, backend=backend,
                       node_model=node_model, front_plc=None, back_plcs={}, tasks=[])
    t0 = time.perf_counter()
    node_addrs = {k: v for k, v in addrs.items() if k in want}
    distribute(man, node_addrs, setup, None, (host, coord.port), layout.l0, wired,
               stop_ids=sorted(textio.stop.ids) if textio else None,
               model_dir=model_dir, device=args.device,
               profile=bool(args.profile_out), miss_policy=args.miss_policy)
    log.info("      %d 台加载完毕，用时 %.1fs", len(want), time.perf_counter() - t0)

    pool = PeerPool("__coord__", LinkTable(), seed=0)
    pool.use_relay(control.RELAY)
    for n, a in node_addrs.items():
        pool.register(n, a)
    coord.start(pool)
    pool.warm(node_addrs)

    # ---------------- 4. 服务 -------------------------------------------- #
    prompts = args.prompt or (["你好"] if textio else None)
    n_req = args.requests if args.requests is not None else (
        len(prompts) if prompts else len(layout.channels))
    tasks = sorted({ch.task for ch in layout.channels})
    log.info("[4/4] 服务：%d 条请求 × %d token，通道 %s",
             n_req, args.tokens, {u: len(q) for u, q in coord.front_pools.items()})

    def drain(batch):
        for rec in batch:
            if not rec.done.wait(timeout=600):
                log.error("%s 超时。已发生的事件：", rec.req)
                for e in rec.events:
                    log.error("    %s", e)
                continue
            if args.timing:
                # 埋点跟在 release 后面回来，比完成事件晚一点 —— 显式等一下。
                # 不能让请求的完成去等它，那会把控制面往返算进用户看到的时延。
                coord.wait_trace(rec, timeout=5.0)
                from ..runtime.timing import summarise_request
                log.info("\n%s\n", summarise_request(rec, coord).render())
            per = sorted(rec.token_ms)
            log.info("  %-7s %s×%s  停于 %-11s %3d token  首token %5.0fms  "
                     "逐token p50 %4.0fms%s",
                     rec.req, rec.front, rec.back, rec.stop_reason or "-",
                     len(rec.tokens), (rec.t_first - rec.t0) * 1000,
                     per[len(per) // 2] if per else 0.0,
                     f"  «{rec.text[:100]}»" if rec.text else "")

    if args.warmup:
        log.info("      预热 %d 条（结果丢弃）—— torch 首次前向含 kernel 选择",
                 args.warmup)
        for w in range(args.warmup):
            u = tasks[w % len(tasks)]
            r = (coord.submit(f"warm{w}", text=prompts[0], task=u)
                 if (textio and prompts) else
                 coord.submit(f"warm{w}", list(range(8)), task=u))
            r.done.wait(timeout=600)
        log.info("      预热完成")

    batch = []
    for i in range(n_req):
        u = tasks[i % len(tasks)]
        if textio and prompts:
            batch.append(coord.submit(f"req{i}", text=prompts[i % len(prompts)], task=u))
        else:
            batch.append(coord.submit(f"req{i}", list(range(8)), task=u))
        if len(batch) >= args.concurrency:
            drain(batch)
            batch = []
    if batch:
        drain(batch)

    # ---------------- 采画像 --------------------------------------------- #
    if args.profile_out:
        recs = collect_profiles(node_addrs, man, n_experts=spec_m.n_experts)
        if not recs:
            log.error("一条画像都没收到")
            return 4
        save_profile(args.profile_out, recs, model=name, n_layers=spec_m.n_layers)
        log.info("画像已存 %s", args.profile_out)
        for line in summarise(load_profile(args.profile_out), coverage=args.coverage,
                              min_experts=spec_m.top_k):
            log.info("  @ 覆盖率 %.0f%%  %s", args.coverage * 100, line)
        log.info("  下一轮加 --profile %s 就只装这些专家", args.profile_out)

    if coord.errors:
        log.error("节点上报的错误：")
        for e in coord.errors[:5]:
            log.error("  %s", e.replace("\n", " ")[:300])

    if not args.once:
        log.info("保持运行中，Ctrl-C 停止（节点 agent 不受影响）")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    coord.stop()
    pool.close()
    return 1 if coord.errors else 0


if __name__ == "__main__":
    sys.exit(main())

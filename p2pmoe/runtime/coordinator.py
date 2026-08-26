"""在线协议与协调器（II.5）+ 单机多进程集群启动器。

协调器只做 O(1) 的事（文档 II.5「在线零计算」）：

    到达:  f = pop(前段空闲队列)                # 盲绑，零后悔（推论 III.3.2）
    识别:  tail(f) 本地分类 → 上报 û 与置信
    派发:  b = pop(û 池空闲队列)                # O(1)，零后悔（定理 III.3.1）
    检出:  通道二 —— 滑窗 miss 率 > 基线×factor → 换绑
    换绑:  b′ = pop(u 池)；L₀ 缓存重发 b′；旧 b 释放 KV；**前段 KV 不动**

「盲绑」这件事在这里是字面意义的：`pop()` 就是队列头，不比较任何延迟。它之所以
成立，是因为离线已经把组合极差压到抖动量级以下 —— 在线再挑也只是在测噪声
（命题 III.3.3）。这是整套方案在运行时最直观的体现。

分散环境无兜底池（命题 III.7.5）：低置信请求绑最大先验池并全程监控，靠换绑兜底。
"""

from __future__ import annotations

import multiprocessing as mp
import socket
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..planner.experts import ExpertPlacement
from ..planner.manifest import DeploymentManifest
from .identify import HistogramClassifier
from .text import TextIO
from .model import ToyMoEConfig
from .node import NodeConfig, run_agent
from .wire import Addr, LinkTable, PeerPool, RelayListener, recv_msg, rpc

__all__ = ["RequestRecord", "Coordinator", "LocalCluster"]


# --------------------------------------------------------------------------- #
@dataclass
class RequestRecord:
    req: str
    true_task: str | None
    ids: list[int] = field(default_factory=list)
    """prompt token。排队时要留着，出队才发得出去。"""
    front: str = ""
    back: str = ""
    task: str | None = None
    zone: str = ""
    conf: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    tokens: list[int] = field(default_factory=list)
    prompt: str = ""
    """原始文本（走文本入口时才有；直接喂 id 的话是空串）。"""
    text: str = ""
    """已生成的文本 —— 由增量解码逐 token 拼出来，不是最后 decode 一次。"""
    stop_reason: str = ""
    """"eos" | "stop_string" | "max_tokens"。空串表示还没结束。"""
    _detok: object = None
    traces: dict = field(default_factory=dict)
    """节点 id → 该节点在这条请求上的埋点（计算时长、前向次数、发出字节）。

    只有时长，没有时刻 —— 15 台机器没有时钟同步，跨机的绝对时刻拼不到一条轴上。
    所以「网络时间」是 `总时延 − 各节点计算 − 排队` 反推出来的，不是测出来的。
    """
    token_ms: list[float] = field(default_factory=list)
    miss_window: deque = field(default_factory=lambda: deque(maxlen=16))
    rebinds: int = 0
    force_task: str | None = None
    """故障注入：强制绑到这个池（无视识别结果）。"""
    tried: set = field(default_factory=set)
    """已经绑过的 task —— 换绑不回头，否则 miss 报警会让它在两个池之间来回跳。"""
    events: list[str] = field(default_factory=list)
    t0: float = 0.0
    t_first: float = 0.0
    t_front: float = 0.0
    """拿到前段的时刻。与 t0 的差就是前段排队时长。"""
    t_back: float = 0.0
    """拿到后段的时刻。"""
    _last: float = 0.0
    done: threading.Event = field(default_factory=threading.Event)

    def log(self, msg: str) -> None:
        self.events.append(f"t+{(time.perf_counter() - self.t0)*1000:7.1f}ms  {msg}")

    @property
    def window_miss(self) -> float:
        return sum(self.miss_window) / len(self.miss_window) if self.miss_window else 0.0

    @property
    def correct(self) -> bool | None:
        return None if self.true_task is None else self.task == self.true_task

    @property
    def wait_front_ms(self) -> float:
        return max(0.0, (self.t_front - self.t0) * 1000) if self.t_front else 0.0

    @property
    def wait_back_ms(self) -> float:
        """识别完成到拿到后段的等待。池子够用时接近 0。"""
        return max(0.0, (self.t_back - self._t_classified) * 1000) if self.t_back else 0.0

    _t_classified: float = 0.0
    finished: bool = False
    """完成判定已经做过。**不能用 `done` 代替** —— 见 `Coordinator._finish`。"""


# --------------------------------------------------------------------------- #
class Coordinator:
    def __init__(
        self,
        manifest: DeploymentManifest,
        *,
        baselines: Mapping[str, float],
        priors: Mapping[str, float],
        alarm_factor: float = 5.0,
        min_window: int = 6,
        host: str = "127.0.0.1",
        port: int = 0,
        static_wiring: Mapping[str, tuple[str, str]] | None = None,
        textio: "TextIO | None" = None,
        on_text=None,
        relay: Addr | None = None,
    ):
        self.man = manifest
        # 文本进出是**可选**的：不给就是 id 进 id 出（toy 模型、协议测试都这么用）。
        # 给了才有 prompt 编码、增量解码、EOS 停止。tokenizer 只活在控制机上，
        # 节点那边什么都不用变 —— 见 runtime/text.py 开头。
        self.text = textio
        self.on_text = on_text
        """流式回调 on_text(rec, delta)。每吐出一段新文本调一次。"""
        # 静态模式：{前段 id: (后段 id, task)}，配对在部署时定死。
        # 给了它就不走识别→派发那条路，请求直接说明自己是哪个 task。
        self.static_wiring = dict(static_wiring or {})
        self.static = bool(self.static_wiring)
        self.baselines = dict(baselines)
        self.priors = dict(priors)
        self.alarm_factor = alarm_factor
        self.min_window = min_window

        # 空闲队列：盲绑就是从这里 pop
        self.free_fronts: deque[str] = deque(
            sorted({p.segment for p in manifest.nodes if p.role == "front"})
        )
        self.free_backs: dict[str, deque[str]] = {}
        for p in manifest.nodes:
            if p.role.startswith("back:"):
                u = p.role.split(":", 1)[1]
                self.free_backs.setdefault(u, deque())
        for u in self.free_backs:
            self.free_backs[u] = deque(
                sorted({p.segment for p in manifest.nodes
                        if p.role == f"back:{u}"})
            )
        # 池的**容量**（建了几条），与「此刻空闲几条」是两回事。
        #
        # 分清楚它们，是因为「队列空」有两种截然不同的含义：
        #   容量 > 0，空闲 = 0  → 都忙着，等就是了（II.5 的有界等待）
        #   容量 = 0            → 这个 task 压根没有通道，**等到超时也不会有**
        # 混为一谈的症状是：请求排进队列，300 秒后报「超时」，而事件日志只说
        # 「池无空闲后段」—— 看起来像拥塞，其实是规划阶段就没给它建通道。
        self.back_capacity: dict[str, int] = {
            u: len(q) for u, q in self.free_backs.items()}

        if self.static:
            # 前段按 task 分池 —— 静态模式下一条前段只服务一个 task
            self.free_fronts = deque()
            self.front_pools: dict[str, deque[str]] = {}
            for f, (_, task) in sorted(self.static_wiring.items()):
                self.front_pools.setdefault(task, deque()).append(f)
        else:
            self.front_pools = {}

        self.seg_head = {sid: info["head"] for sid, info in manifest.segments.items()}
        self.seg_nodes = {sid: list(info["nodes"]) for sid, info in manifest.segments.items()}

        self.records: dict[str, RequestRecord] = {}
        # 有界等待（II.5「池满 → 有界等待」）：池子空了不是错误，是排队。
        # 15 台节点建出 5 条前段就只能同时服务 5 条请求 —— 第 6 条在这里等。
        # 并发度 = 段数，是排他独占（I.2.4）的直接后果。批处理见 TODO.md P1。
        self._arrivals: deque[RequestRecord] = deque()
        self._await_back: dict[str, deque[RequestRecord]] = {}
        self.pairings: list[tuple[str, str, str, str]] = []
        """(req, front, back, task) 的配对历史 —— 用来核对「每次都是新组合」。"""
        self._lock = threading.Lock()
        self.pool: PeerPool | None = None
        self.errors: list[str] = []

        # 中继模式：协调器也挂到中继上（节点连不到它，只能反过来）
        self.relay = tuple(relay) if relay else None
        if self.relay:
            self.sock = RelayListener(self.relay, "__coord__")
            self.host, self.port = self.relay
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((host, port))
            self.sock.listen(128)
            self.host, self.port = self.sock.getsockname()[:2]
        self._stop = threading.Event()
        self.max_tokens = 12

    # -- 服务 -------------------------------------------------------------- #
    def start(self, pool: PeerPool) -> None:
        self.pool = pool
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()

    def _serve(self) -> None:
        self.sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._conn, args=(conn,), daemon=True).start()

    def _conn(self, conn: socket.socket) -> None:
        try:
            while not self._stop.is_set():
                h, _ = recv_msg(conn)
                try:
                    self._on(h)
                except Exception:
                    self.errors.append(traceback.format_exc()[-500:])
        except (ConnectionError, OSError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def queue_depths(self) -> dict:
        with self._lock:
            if self.static:
                return {
                    "mode": "static",
                    "waiting": len(self._arrivals),
                    "free_channels": {u: len(q) for u, q in self.front_pools.items()},
                }
            return {
                "waiting_front": len(self._arrivals),
                "waiting_back": {u: len(q) for u, q in self._await_back.items() if q},
                "free_fronts": len(self.free_fronts),
                "free_backs": {u: len(q) for u, q in self.free_backs.items()},
            }

    def stop(self) -> None:
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass

    # -- 在线协议 ---------------------------------------------------------- #
    def submit(
        self,
        req: str,
        ids: Sequence[int] | None = None,
        *,
        text: str | None = None,
        true_task: str | None = None,
        force_task: str | None = None,
        task: str | None = None,
    ) -> RequestRecord:
        """到达：盲绑一条空闲前段。不比不挑（推论 III.3.2）。

        静态模式下 `task` 必填（或由 true_task 兜底）：配对是部署时定死的，
        协调器只需按 task 取一条空闲通道，不做识别也不做派发。

        force_task 是**故障注入**：无视识别结果，强行绑到指定的池。用来验证
        通道二（miss 率检出）与换绑路径确实工作 —— 就像给一个正确的系统注入
        一个已知错误，看它能不能自己发现并纠正。
        """
        if text is not None:
            if self.text is None:
                raise ValueError(
                    "协调器没配 tokenizer，收不了文本 —— "
                    "要么传 ids，要么 Coordinator(textio=TextIO.from_model_dir(...))"
                )
            if ids is not None:
                raise ValueError("text 与 ids 二选一")
            ids = self.text.encode_prompt(text)
        elif ids is None:
            raise ValueError("submit 需要 ids 或 text")

        rec = RequestRecord(req=req, true_task=true_task, ids=list(ids))
        rec.prompt = text or ""
        if self.text is not None:
            rec._detok = self.text.stream()
        rec.force_task = force_task
        rec.t0 = time.perf_counter()

        if self.static:
            return self._submit_static(rec, task or true_task)

        with self._lock:
            self.records[req] = rec
            if self.free_fronts:
                rec.front = self.free_fronts.popleft()
            else:
                self._arrivals.append(rec)
                depth = len(self._arrivals)
        if not rec.front:
            rec.log(f"到达 → 前段池空，排队（队深 {depth}）—— 有界等待（II.5）")
            return rec
        self._start_front(rec)
        return rec

    def _submit_static(self, rec: RequestRecord, task: str | None) -> RequestRecord:
        """静态模式的到达：按 task 取一条通道，前后段是绑好的。"""
        if not task:
            raise ValueError("静态模式下 submit 必须给 task —— 配对在部署时就定死了")
        with self._lock:
            self.records[rec.req] = rec
            q = self.front_pools.get(task)
            if q is None:
                raise ValueError(f"没有服务 task {task} 的静态通道；有的是 "
                                 f"{sorted(self.front_pools)}")
            if q:
                rec.front = q.popleft()
            else:
                self._arrivals.append(rec)
                depth = len(self._arrivals)
        if not rec.front:
            rec.task = task
            rec.log(f"到达（task={task}）→ {task} 通道池空，排队（队深 {depth}）")
            return rec
        rec.task = task
        rec.back = self.static_wiring[rec.front][0]
        with self._lock:
            self.pairings.append((rec.req, rec.front, rec.back, task))
        rec.t_front = rec.t_back = time.perf_counter()
        rec.log(f"到达（task={task}）→ 静态通道 {rec.front} × {rec.back}"
                f"（配对在部署时定死，无识别、无派发 RTT）")
        self.pool.send(self.seg_head[rec.front],
                       {"type": "prefill", "req": rec.req, "ids": rec.ids}, delay=False)
        return rec

    def _start_front(self, rec: RequestRecord) -> None:
        rec.t_front = time.perf_counter()
        w = rec.wait_front_ms
        rec.log(f"盲绑 {rec.front}（队列头，未做任何比较）"
                + (f"，排队等了 {w:.0f}ms" if w > 1 else ""))
        self.pool.send(self.seg_head[rec.front],
                       {"type": "prefill", "req": rec.req, "ids": rec.ids}, delay=False)

    def _release_front(self, f: str) -> None:
        """前段用完归还。若有请求在排队，直接交给队首而不是先入池再出池。"""
        if self.static:
            task = self.static_wiring[f][1]
            with self._lock:
                waiting = [r for r in self._arrivals if r.task == task]
                if waiting:
                    nxt = waiting[0]
                    self._arrivals.remove(nxt)
                    nxt.front = f
                    nxt.back = self.static_wiring[f][0]
                    self.pairings.append((nxt.req, f, nxt.back, task))
                else:
                    self.front_pools[task].append(f)
                    return
            nxt.t_front = nxt.t_back = time.perf_counter()
            nxt.log(f"出队 → 静态通道 {f} × {nxt.back}（等了 {nxt.wait_front_ms:.0f}ms）")
            self.pool.send(self.seg_head[f],
                           {"type": "prefill", "req": nxt.req, "ids": nxt.ids},
                           delay=False)
            return
        with self._lock:
            if self._arrivals:
                nxt = self._arrivals.popleft()
                nxt.front = f
            else:
                self.free_fronts.append(f)
                return
        self._start_front(nxt)

    def _release_back(self, task: str, b: str) -> None:
        """后段用完归还。同理，等这个池的请求优先。"""
        with self._lock:
            q = self._await_back.get(task)
            if q:
                nxt = q.popleft()
            else:
                self.free_backs.setdefault(task, deque()).append(b)
                return
        self._attach_back(nxt, task, b, rebind=False)

    def _on(self, h: dict) -> None:
        t = h.get("type")
        if t == "classified":
            self._on_classified(h)
        elif t == "token":
            self._on_token(h)
        elif t == "error":
            self.errors.append(f"{h.get('node')}: {h.get('trace')}")
        elif t == "static_forward":
            rec = self.records.get(h.get("req", ""))
            if rec:
                rec.log(f"{h['node']} 静态转发 → {h['peer']}（task={h.get('task')}）")
        elif t == "req_trace":
            rec = self.records.get(h.get("req", ""))
            if rec is not None:
                rec.traces[h["node"]] = h
        elif t == "kv_dropped":
            rec = self.records.get(h.get("req", ""))
            if rec:
                rec.log(f"{h['node']} 释放后段 KV（{'成功' if h['dropped'] else '本就没有'}）")

    def _on_classified(self, h: dict) -> None:
        rec = self.records[h["req"]]
        rec._t_classified = time.perf_counter()
        rec.task, rec.conf, rec.zone, rec.scores = h["task"], h["conf"], h["zone"], h["scores"]
        rec.log(
            f"{h['node']} 本地识别: {rec.task} (c={rec.conf:.2f}, {rec.zone} 区)"
            f"  scores={ {k: round(v,2) for k,v in rec.scores.items()} }"
        )
        if rec.zone == "prior":
            rec.log("低置信 → 绑最大先验池并全程监控（无兜底池，命题 III.7.5）")
        target = rec.task
        if rec.force_task and rec.force_task != rec.task:
            rec.log(f"⚠ 故障注入：无视识别结果，强行绑到 {rec.force_task} 池")
            target = rec.task = rec.force_task
        self._bind(rec, target, rebind=False)

    def _bind(self, rec: RequestRecord, task: str, *, rebind: bool) -> None:
        # 容量为 0 = 规划阶段就没给这个 task 建后段。排队没有意义 ——
        # 队列里没有任何东西会被归还，等的是一个永远不会到的事件。
        # 立刻失败，并把原因说清楚，比让它在 300 秒后报「超时」诚实得多。
        if self.back_capacity.get(task, 0) == 0:
            have = {u: n for u, n in self.back_capacity.items() if n}
            rec.log(f"✗ {task} 池**一条后段都没有**（容量 0）—— 不是拥塞，是规划"
                    f"阶段没给它建通道。现有 {have or '无'}")
            self._fail(rec, f"task {task} 没有后段通道（容量 0）。"
                            f"现有通道 {have or '无'}。"
                            f"降低 --coverage 让每条段更小、或增加节点，"
                            f"让规划器能给 {task} 分到配额")
            return
        with self._lock:
            q = self.free_backs.get(task)
            if q:
                b = q.popleft()
            else:
                self._await_back.setdefault(task, deque()).append(rec)
                depth = len(self._await_back[task])
                b = None
        if b is None:
            rec.log(f"{task} 池 {self.back_capacity[task]} 条全忙，排队"
                    f"（队深 {depth}）—— 有界等待（II.5）")
            return
        self._attach_back(rec, task, b, rebind=rebind)

    def _fail(self, rec: RequestRecord, why: str) -> None:
        """把请求判死并唤醒等它的人。**不走 `_finish`** —— 那条路要归还通道，
        而这条请求从来没拿到过通道，归还会把池子搞乱。"""
        if rec.finished:
            return
        rec.finished = True
        rec.stop_reason = "no_channel"
        self.errors.append(f"{rec.req}: {why}")
        rec.done.set()

    def _attach_back(self, rec: RequestRecord, task: str, b: str, *, rebind: bool) -> None:
        old, rec.back = rec.back, b
        rec.task = task
        rec.t_back = time.perf_counter()
        w = rec.wait_back_ms
        rec.log(f"{'换绑' if rebind else '派发'}: pop({task} 池) → {b}"
                + (f"（原 {old}）" if rebind and old else "")
                + (f"，排队等了 {w:.0f}ms" if w > 1 else ""))
        with self._lock:
            self.pairings.append((rec.req, rec.front, b, task))
        # 告诉 tail(f) 绑到谁；它会把**缓存的** L₀ 输出发过去，前段一层不重算
        tail = self.seg_nodes[rec.front][-1]
        self.pool.send(tail, {"type": "bind", "req": rec.req, "target": self.seg_head[b],
                              "task": task, "rebind": rebind}, delay=False)

    def _on_token(self, h: dict) -> None:
        rec = self.records.get(h["req"])
        if rec is None or rec.finished:
            # 完成判定与「release 送达各节点」之间有一个窗口，环里可能还有一个
            # 在途的 token 正绕回来。它是**已经算出来**的、合法的 token，只是
            # 到晚了 —— 收下它会让 tokens 超出 max_tokens 并二次触发 _finish，
            # 于是同一条通道被归还两次，池深凭空长大。丢弃是对的：环是异步的，
            # 「停」这个决定必然要在某个已经发出的计算之后生效。
            return
        st = h["back_stats"]
        rate = st["miss"] / st["ntl"] if st["ntl"] else 0.0
        # 只把 decode 步计入滑窗：prefill 的 miss 率明显低于 decode（输入还没漂），
        # 混进去会让窗口值忽高忽低。基线那边也是只算 decode，两边口径必须一致。
        if h.get("phase") == "decode":
            rec.miss_window.append(rate)
        now = time.perf_counter()
        if not rec.tokens:
            rec.t_first = now
            rec.log(f"首 token（后段 prefill 完成）token={h['token']}")
        else:
            rec.token_ms.append((now - rec._last) * 1000)
        rec._last = now
        token = h["token"]
        rec.tokens.append(token)

        # ---- 文本层：增量解码与停止判定（都在控制面） ----
        if self.text is not None:
            if self.text.stop.hit_id(token):
                # EOS 不进文本 —— 它是控制符，不是内容。
                rec.tokens.pop()
                rec.stop_reason = "eos"
                rec.log(f"模型自己收尾（EOS token={token}）")
                self._finish(rec)
                return
            delta = rec._detok.push(token)
            if delta:
                rec.text += delta
                if self.on_text:
                    self.on_text(rec, delta)
            hit = self.text.stop.hit_text(rec.text)
            if hit:
                rec.stop_reason = "stop_string"
                rec.log(f"命中停止串 {hit!r}")
                self._finish(rec)
                return

        # ---- 通道二：滑窗 miss 率 vs 基线（II.5） ----
        # 静态模式下 task 是给定的，不存在误绑 —— 仍然统计 miss 率（它反映
        # 驻留集覆盖得够不够），但不触发换绑。
        base = self.baselines.get(rec.task, 0.03)
        if (not self.static
                and len(rec.miss_window) >= self.min_window
                and rec.window_miss > base * self.alarm_factor
                and rec.rebinds < 2):
            rec.log(
                f"通道二报警: 滑窗 miss 率 {rec.window_miss:.1%} > 基线 {base:.1%}"
                f" × {self.alarm_factor:g} —— 误绑的直接症状"
            )
            self._rebind(rec)
            return

        if len(rec.tokens) >= self.max_tokens:
            rec.stop_reason = "max_tokens"
            self._finish(rec)

    def _rebind(self, rec: RequestRecord) -> None:
        """换绑：按前段分类器的次优 task 重绑（通道一与通道二合流）。"""
        rec.tried.add(rec.task)
        order = sorted(rec.scores, key=lambda u: -rec.scores[u])
        nxt = next((u for u in order if u not in rec.tried and self.free_backs.get(u)), None)
        if nxt is None:
            rec.log("已试过的池之外没有候选 —— 继续用 drop-expert 近似跑完"
                    "（运维近似，非无损；II.5）")
            rec.miss_window.clear()
            return
        rec.rebinds += 1
        old, old_task = rec.back, rec.task
        rec.miss_window.clear()
        rec.log(f"换绑决策: {old_task} → {nxt}（前段 KV 保留，只重算后段 prefill）")
        # 旧后段 KV 作废并归还队列；**前段 KV 一层都不动**（命题 III.7.1）
        for n in self.seg_nodes[old]:
            self.pool.send(n, {"type": "drop_kv", "req": rec.req}, delay=False)
        rec.task = nxt
        self._bind(rec, nxt, rebind=True)
        self._release_back(old_task, old)   # 归还旧后段，可能立刻被排队者接走

    def expected_trace_nodes(self, rec: RequestRecord) -> list[str]:
        return sorted(set(self.seg_nodes.get(rec.front, []))
                      | set(self.seg_nodes.get(rec.back, [])))

    def wait_trace(self, rec: RequestRecord, timeout: float = 3.0) -> bool:
        """等各节点把埋点报回来。

        `release` 是 fire-and-forget 的，埋点跟在它后面回来，所以完成事件
        （`rec.done`）比埋点早到。要看时序报告就得显式等一下 —— 但**不能**让
        请求的完成去等它，那会把控制面的往返算进用户看到的时延里。
        """
        want = set(self.expected_trace_nodes(rec))
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            if want <= set(rec.traces):
                return True
            time.sleep(0.02)
        return want <= set(rec.traces)

    def _finish(self, rec: RequestRecord) -> None:
        if rec.finished:
            return
        rec.finished = True
        if rec._detok is not None:
            # 收尾把攒着的半个字符吐出来 —— 截断在多字节字符中间时会有
            tail = rec._detok.flush()
            if tail:
                rec.text += tail
                if self.on_text:
                    self.on_text(rec, tail)
        rec.log(f"完成 {len(rec.tokens)} 个 token，释放前后段")
        for sid in (rec.front, rec.back):
            for n in self.seg_nodes.get(sid, []):
                self.pool.send(n, {"type": "release", "req": rec.req}, delay=False)
        rec.done.set()
        # 归还：若有请求在排队，直接交给队首（「重新进入可用池，等待下一个 request」）
        # 静态模式下前后段是一体的，归还前段即等于归还整条通道。
        if rec.back and not self.static:
            self._release_back(rec.task, rec.back)
        self._release_front(rec.front)


# --------------------------------------------------------------------------- #
class LocalCluster:
    """单机多进程集群：一个 manifest 节点一个进程，本地 socket + 延迟注入。"""

    def __init__(
        self,
        manifest: DeploymentManifest,
        model_cfg: ToyMoEConfig | None,
        links: LinkTable,
        classifier: HistogramClassifier | None = None,
        *,
        baselines: Mapping[str, float] | None = None,
        priors: Mapping[str, float] | None = None,
        alarm_factor: float = 5.0,
        static_wiring: Mapping[str, tuple[str, str]] | None = None,
        backend: str = "numpy",
        model_dir: str | None = None,
        model_hf: Mapping | None = None,
        device: str = "cpu",
        miss_policy: str = "drop",
        textio: TextIO | None = None,
        on_text=None,
    ):
        self.man = manifest
        self.mcfg = model_cfg
        self.links = links
        self.clf = classifier
        self.backend = backend
        self.model_dir = model_dir
        self.model_hf = dict(model_hf or {})
        self.device = device
        self.miss_policy = miss_policy
        self.static_wiring = dict(static_wiring or {})
        self.textio = textio
        self.coord = Coordinator(manifest, baselines=baselines or {}, priors=priors or {},
                                 alarm_factor=alarm_factor,
                                 static_wiring=static_wiring,
                                 textio=textio, on_text=on_text)
        self.procs: list[mp.Process] = []
        self.ports: dict[str, int] = {}
        self.pool: PeerPool | None = None

    def __enter__(self) -> "LocalCluster":
        """两阶段装配 —— 与真机部署（deploy/control.py）同一条路径。

        先把所有 agent 起起来（各自 bind 0、把真实端口报回来），再按拿到的
        地址表下发 configure。这样既没有端口预分配的竞态，也保证本地跑通的
        装配流程就是上真机时跑的那一套。
        """
        nodes = self.man.nodes
        ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
        ready: mp.Queue = ctx.Queue()

        for p in nodes:
            proc = ctx.Process(target=run_agent, args=(p.node, ready), daemon=True)
            proc.start()
            self.procs.append(proc)

        for _ in nodes:
            try:
                nid, port = ready.get(timeout=60)
            except Exception as e:
                alive = sum(1 for x in self.procs if x.is_alive())
                raise RuntimeError(
                    f"只有 {len(self.ports)}/{len(nodes)} 个 agent 报到"
                    f"（{alive} 个进程仍存活）：{e}"
                ) from e
            self.ports[nid] = port

        peers: dict[str, Addr] = {n: ("127.0.0.1", p) for n, p in self.ports.items()}
        coord_addr: Addr = ("127.0.0.1", self.coord.port)

        by_seg: dict[str, list] = {}
        for p in nodes:
            by_seg.setdefault(p.segment, []).append(p)
        for sid in by_seg:
            by_seg[sid].sort(key=lambda x: x.position)

        for p in nodes:
            chain = by_seg[p.segment]
            i = p.position
            role = "front" if p.role.startswith("front") else p.role
            is_back = role.startswith("back")
            task = (role.split(":", 1)[1] if is_back
                    else (self.static_wiring.get(p.segment, (None, None))[1]))
            # 静态模式：前段的 tail 在配置期就知道自己该发给哪个后段 head
            peer = None
            if self.static_wiring and role == "front" and p.is_tail:
                wired = self.static_wiring.get(p.segment)
                if wired:
                    peer = self.man.segments[wired[0]]["head"]
            cfg = NodeConfig(
                node_id=p.node,
                role=role,
                segment=p.segment,
                layer_experts={l.layer: list(l.experts) for l in p.layers},
                next_hop=chain[i + 1].node if i + 1 < len(chain) else None,
                seg_head=chain[0].node,
                is_head=p.is_head,
                is_tail=p.is_tail,
                peers=peers,
                links=self.links.to_dict(),
                coordinator=coord_addr,
                model=(dict(self.mcfg.__dict__) if self.backend == "numpy"
                       else dict(self.model_hf)),
                classifier=(self.clf.to_wire()
                            if (self.clf is not None and role == "front"
                                and p.is_tail and not self.static_wiring)
                            else None),
                static_peer=peer,
                static_task=task,
                backend=self.backend,
                model_dir=self.model_dir,
                device=self.device,
                with_embed=(role == "front" and p.is_head),
                with_lm_head=(is_back and p.is_tail),
                miss_policy=self.miss_policy,
                stop_ids=(sorted(self.textio.stop.ids)
                          if (self.textio and is_back and p.is_tail) else []),
            )
            rpc(peers[p.node], {"type": "configure", "config": cfg.to_dict()}, timeout=120)

        self.pool = PeerPool("__coord__", LinkTable(), seed=0)
        for n, a in peers.items():
            self.pool.register(n, a)
        self.coord.start(self.pool)
        # 保温网格：先把所有连接建起来（II.4 Step 6「保温」）
        self.pool.warm(peers)
        return self

    def __exit__(self, *exc) -> None:
        try:
            for n in self.ports:
                try:
                    self.pool.send(n, {"type": "shutdown"}, delay=False)
                except Exception:
                    pass
            self.coord.stop()
            for p in self.procs:
                p.join(timeout=3)
                if p.is_alive():
                    p.terminate()
        finally:
            if self.pool:
                self.pool.close()

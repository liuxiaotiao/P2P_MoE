"""节点 agent 进程 —— 一台分散 GPU 节点在本 demo 里的化身。

一个节点只做三件事：
  1. 按下发的清单加载**属于自己那一段的那几层的那些专家**（`NodePlan.layers`）；
  2. 收到 hidden state 就跑自己这几层，然后往链上下一跳转发；
  3. 如果自己是段的出口，按协调器给的绑定把结果送到对面那一段的入口。

节点**不知道**组合矩阵、不知道配额、不知道公共带 —— 那些是离线规划的产物。
在线时它只认三样：自己的层、自己的下一跳、当前请求绑到了哪条对面段。这正是
文档 II.5「在线零计算」的含义：所有智力离线预支，在线只做 O(1) 的转发。

排他独占（I.2.4）：一个进程只服务一条段、一条请求。批处理被显式放弃，换取确定性
延迟 —— 代价是并发度等于段数。这是方案的核心取舍之一，不是实现限制；要改的话
见 TODO.md 的 P1 条目（它会同时影响均匀性目标、零后悔定理与命题 III.3.3）。
"""

from __future__ import annotations

import queue
import socket
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .identify import HistogramClassifier
from .model import MoEStats, SegmentModel, ToyMoEConfig, embed_tokens, lm_head
from .wire import Addr, LinkTable, PeerPool, recv_msg, send_msg

__all__ = ["NodeConfig", "NodeServer", "run_agent", "run_node"]


# --------------------------------------------------------------------------- #
@dataclass
class ReqState:
    """一个请求在本节点上的在线状态。"""

    cached_l0: np.ndarray | None = None
    """tail(f) 缓存的 layer L₀ 输出（II.5：识别未提交时保留，供换绑重发）。"""
    bound_to: str | None = None
    """tail(f) 记的当前绑定：对面后段的 head 节点 id。"""
    task: str | None = None
    loop_to: str | None = None
    """后段侧记的回环目标：当前配的那条前段的 head。"""
    pending: np.ndarray | None = None
    """已算完但还没等到 bind 的 hidden。"""
    n_decode: int = 0
    n_sampled: int = 0


@dataclass
class NodeConfig:
    node_id: str
    role: str
    segment: str
    layer_experts: dict[int, list[int]]
    next_hop: str | None
    """段内下一跳；None 表示自己是 tail。"""
    seg_head: str
    is_head: bool
    is_tail: bool
    peers: dict[str, Addr]
    links: dict
    coordinator: Addr
    model: dict
    classifier: dict | None = None
    """只有 front 的 tail 需要 —— 识别在那里做（II.5）。"""

    def to_dict(self) -> dict:
        return {
            **self.__dict__,
            "peers": {k: list(v) for k, v in self.peers.items()},
            "coordinator": list(self.coordinator),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NodeConfig":
        d = dict(d)
        d["peers"] = {k: tuple(v) for k, v in d["peers"].items()}
        d["coordinator"] = tuple(d["coordinator"])
        d["layer_experts"] = {int(k): list(v) for k, v in d["layer_experts"].items()}
        return cls(**d)


# --------------------------------------------------------------------------- #
class NodeServer:
    """两阶段：先起服务，再等控制器下发 configure。

    真机部署必须是两阶段的 —— agent 启动时还不知道自己该装哪几层，因为规划要
    先量到全网的逐对延迟，而量延迟又要求 agent 已经在跑。先起服务、后收清单
    是唯一能解开这个循环的顺序。

    未配置状态下 agent 只响应三类消息：capabilities（报告自己有多少内存/算力）、
    echo（给别人量延迟当靶子）、probe（去量指定对端）。
    """

    def __init__(
        self,
        node_id: str,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        mem_mb: float | None = None,
        access_ms: float = 0.0,
        access_jitter_ms: float = 0.0,
    ):
        self.me = node_id
        self.declared_mem_mb = mem_mb
        self.access_ms = access_ms
        self.access_jitter_ms = access_jitter_ms
        self.cfg: NodeConfig | None = None
        self.mcfg: ToyMoEConfig | None = None
        self.model: SegmentModel | None = None
        self.clf: HistogramClassifier | None = None
        self.links = LinkTable()
        self.pool = PeerPool(self.me, self.links, seed=abs(hash(self.me)) % (2**31),
                             egress_ms=access_ms, egress_jitter_ms=access_jitter_ms)
        self.load_ms = 0.0

        self._reqs: dict[str, ReqState] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.compute_ms = 0.0
        self.n_msgs = 0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(128)
        self.host, self.port = self.sock.getsockname()[:2]

    # -- 第二阶段：装载 ---------------------------------------------------- #
    def apply_config(self, cfg: NodeConfig) -> dict:
        """收到清单：加载**属于自己那一段的那几层的那些专家**。"""
        self.cfg = cfg
        self.mcfg = ToyMoEConfig(**cfg.model)
        t0 = time.perf_counter()
        self.model = SegmentModel(self.mcfg, cfg.layer_experts)
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.clf = (
            HistogramClassifier.from_wire(cfg.classifier) if cfg.classifier else None
        )
        self.links = LinkTable.from_dict(cfg.links)
        self.pool = PeerPool(self.me, self.links, seed=abs(hash(self.me)) % (2**31),
                             egress_ms=self.access_ms,
                             egress_jitter_ms=self.access_jitter_ms)
        for n, a in cfg.peers.items():
            self.pool.register(n, a)
        self.pool.register("__coord__", cfg.coordinator)
        return {
            "node": self.me, "segment": cfg.segment, "role": cfg.role,
            "layers": sorted(cfg.layer_experts),
            "n_experts": sum(len(v) for v in cfg.layer_experts.values()),
            "resident_mb": round(self.model.resident_bytes / 1e6, 3),
            "full_mb": round(self.model.full_bytes / 1e6, 3),
            "load_ms": round(self.load_ms, 1),
        }

    # -- 服务循环 ---------------------------------------------------------- #
    def serve_forever(self) -> None:
        self.sock.settimeout(0.2)
        threads: list[threading.Thread] = []
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            t = threading.Thread(target=self._serve_conn, args=(conn,), daemon=True)
            t.start()
            threads.append(t)
        self.pool.close()
        try:
            self.sock.close()
        except OSError:
            pass

    def _serve_conn(self, conn: socket.socket) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while not self._stop.is_set():
                header, arr = recv_msg(conn)
                self.n_msgs += 1
                try:
                    self._dispatch(header, arr, conn)
                except Exception:
                    self._report({"type": "error", "node": self.me,
                                  "trace": traceback.format_exc()[-800:]})
        except (ConnectionError, OSError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    # -- 分发 -------------------------------------------------------------- #
    def _dispatch(self, h: dict, arr: np.ndarray | None, conn=None) -> None:
        t = h.get("type")
        if t == "ping":
            return
        if t == "shutdown":
            self._stop.set()
            return

        # ---- 未配置态也要能响应的三类：报能力、当靶子、去量别人 ----
        if t == "echo":
            # 回包也要过本节点的出口接入段 —— 否则探测量到的是单向的一半
            if conn is not None:
                self.pool.egress_delay()
                send_msg(conn, {"type": "echo_ack", "seq": h.get("seq"), "node": self.me})
            return
        if t == "capabilities":
            if conn is not None:
                send_msg(conn, {"type": "capabilities_ack", **self.capabilities()})
            return
        if t == "probe":
            res = self.probe_peer(tuple(h["addr"]), int(h.get("k", 8)))
            if conn is not None:
                send_msg(conn, {"type": "probe_ack", "peer": h["peer"],
                                "src": self.me, **res})
            return
        if t == "configure":
            info = self.apply_config(NodeConfig.from_dict(h["config"]))
            if conn is not None:
                send_msg(conn, {"type": "configure_ack", **info})
            return

        if self.model is None:
            if conn is not None:
                send_msg(conn, {"type": "error", "node": self.me,
                                "trace": f"收到 {t} 但还没配置"})
            return

        if t == "prefill":
            self._on_prefill(h)
        elif t == "hop":
            self._on_hop(h, arr)
        elif t == "seg_in":
            self._on_seg_in(h, arr)
        elif t == "bind":
            self._on_bind(h)
        elif t == "loop":
            self._on_loop(h)
        elif t == "drop_kv":
            dropped = self.model.drop_kv(h["req"])
            self._report({"type": "kv_dropped", "req": h["req"], "node": self.me,
                          "dropped": dropped})
        elif t == "release":
            self.model.drop_kv(h["req"])
            with self._lock:
                self._reqs.pop(h["req"], None)
        elif t == "stats":
            self._report({"type": "node_stats", "node": self.me, **self.stats()})

    # -- 数据面 ------------------------------------------------------------ #
    def _state(self, req: str) -> ReqState:
        with self._lock:
            return self._reqs.setdefault(req, ReqState())

    def _forward(self, x: np.ndarray, req: str) -> tuple[np.ndarray, MoEStats]:
        t0 = time.perf_counter()
        h, st = self.model.forward(req, x)
        self.compute_ms += (time.perf_counter() - t0) * 1000
        return h, st

    def _on_prefill(self, h: dict) -> None:
        """协调器 → head(f)：一条新请求到达（盲绑已经在协调器侧完成）。"""
        req = h["req"]
        x = embed_tokens(self.mcfg, h["ids"])
        y, st = self._forward(x, req)
        self._advance(req, "prefill", y, st, self.cfg.seg_head)

    def _on_hop(self, h: dict, arr: np.ndarray) -> None:
        """段内下一跳收到 hidden state（+ 捎带的直方图）。"""
        req = h["req"]
        y, st = self._forward(arr, req)
        st = st.merge(MoEStats.from_wire(h["stats"]))
        self._advance(req, h["phase"], y, st, h.get("loop_to"))

    def _on_seg_in(self, h: dict, arr: np.ndarray) -> None:
        """对面那一段的入口收到 hidden state（跨接口，正向）。"""
        req = h["req"]
        # 回环目标是**逐请求**的：同一条后段这一刻可能配着 F0，下一刻配着 F3。
        # 所以它随消息走，而不是配置期定死 —— 这正是「任意组合」在运行时的样子。
        loop_to = h.get("loop_to")
        if loop_to:
            self._state(req).loop_to = loop_to
        else:
            loop_to = self._state(req).loop_to
        y, st = self._forward(arr, req)
        self._advance(req, h["phase"], y, st, loop_to)

    def _advance(
        self, req: str, phase: str, y: np.ndarray, st: MoEStats, loop_to: str | None
    ) -> None:
        """算完本节点的层之后往下走：段内还有下一跳就转发，否则按角色出段。"""
        if self.cfg.next_hop:
            self.pool.send(
                self.cfg.next_hop,
                {"type": "hop", "req": req, "phase": phase,
                 "stats": st.to_wire(), "loop_to": loop_to},
                y,
            )
            return

        if self.cfg.role == "front":
            self._front_tail(req, phase, y, st)
        else:
            self._back_tail(req, phase, y, st, loop_to)

    # -- 前段出口：识别 + 绑定 --------------------------------------------- #
    def _front_tail(self, req: str, phase: str, y: np.ndarray, st: MoEStats) -> None:
        s = self._state(req)
        if phase == "prefill":
            # 缓存 layer L₀ 输出 —— 换绑时靠它避免重放前段（命题 III.7.1/III.7.2）
            s.cached_l0 = y
            v = self.clf.predict(st.hist) if self.clf else None
            self._report({
                "type": "classified", "req": req, "node": self.me,
                "task": v.task if v else None,
                "conf": round(v.confidence, 4) if v else 0.0,
                "zone": v.zone if v else "none",
                "scores": {k: round(x, 4) for k, x in (v.scores if v else {}).items()},
                "front_stats": st.to_wire(),
            })
            s.pending = y  # 等 bind
            return

        # decode：已经绑好了，直接送对面
        if s.bound_to:
            self.pool.send(
                s.bound_to,
                {"type": "seg_in", "req": req, "phase": "decode",
                 "loop_to": self.cfg.seg_head},
                y,
            )

    def _on_bind(self, h: dict) -> None:
        """协调器 → tail(f)：本请求绑到哪条后段。也用于换绑。"""
        req = h["req"]
        s = self._state(req)
        s.bound_to = h["target"]
        s.task = h.get("task")
        payload = s.pending if s.pending is not None else s.cached_l0
        if payload is None:
            self._report({"type": "error", "req": req, "node": self.me,
                          "trace": "bind 时没有可发的 L₀ 输出"})
            return
        s.pending = None
        # 换绑走的也是这条路：把**缓存的** L₀ 输出重发给新后段，前段一层都不重算
        self.pool.send(
            s.bound_to,
            {"type": "seg_in", "req": req, "phase": "prefill",
             "rebind": h.get("rebind", False), "loop_to": self.cfg.seg_head},
            payload,
        )

    def _on_loop(self, h: dict) -> None:
        """回环：tail(b) 把采样出的 token id 传回 head(f)，开始下一个 decode step。"""
        req = h["req"]
        s = self._state(req)
        s.n_decode += 1
        x = embed_tokens(self.mcfg, [h["token"]])
        y, st = self._forward(x, req)
        self._advance(req, "decode", y, st, self.cfg.seg_head)

    # -- 后段出口：采样 + 回环 --------------------------------------------- #
    def _back_tail(
        self, req: str, phase: str, y: np.ndarray, st: MoEStats, loop_to: str | None
    ) -> None:
        # 温度采样而非 argmax：权重绑定 + argmax 会让生成塌成一个不动点，
        # decode 阶段就永远在同一个 token 上打转，验证不到路由的动态。
        st_ = self._state(req)
        logits = lm_head(self.mcfg, y[-1]) / max(self.mcfg.sample_temp, 1e-6)
        pr = np.exp(logits - logits.max())
        pr /= pr.sum()
        rng = np.random.default_rng(abs(hash((req, st_.n_sampled))) % (2**31))
        token = int(rng.choice(len(pr), p=pr))
        st_.n_sampled += 1
        self._report({
            "type": "token", "req": req, "node": self.me, "segment": self.cfg.segment,
            "phase": phase, "token": token, "back_stats": st.to_wire(),
        })
        if not loop_to:
            self._report({"type": "error", "req": req, "node": self.me,
                          "trace": "后段出口不知道回环目标"})
            return
        # 回环接口：字节级 payload，纯延迟（第〇部分）
        self.pool.send(loop_to, {"type": "loop", "req": req, "token": token})

    # -- 未配置态：能力与探测 ---------------------------------------------- #
    def capabilities(self) -> dict:
        """上报本机能力。规划器的 Node 表就是由这些回包拼出来的。"""
        return {
            "node": self.me,
            "host": self.host,
            "port": self.port,
            "mem_mb": self.declared_mem_mb if self.declared_mem_mb is not None
            else _detect_mem_mb(),
            "access_ms": self.access_ms,
            "ms_per_layer": _bench_ms_per_layer(),
            "configured": self.model is not None,
        }

    def probe_peer(self, addr: tuple[str, int], k: int) -> dict:
        """**由本节点发起**去量到 addr 的延迟。

        为什么必须是节点自己去量：逐对代价 h(v,v′) 里最大的一项是两端的接入段
        （II.3.1），从控制器去 ping 两台机器，量到的是控制器自己的接入质量，
        跟这两台之间的链路没什么关系。所以探测指令由控制器下发、探测动作由
        节点执行、结果回传 —— 这是把 SimNetwork 换成真实网络时唯一的结构性改动。

        [口径说明] 这里测的是应用层 RTT，取 RTT/2 作为单向估计。文档把正向接口
        与回环当作两个独立的有向量，但**没有时钟同步就观测不到方向不对称**：
        RTT 只给出两个方向之和。所以真机上 ŵ(a,b) 与 ŵ(b,a) 只能取同一个估计。
        若链路确有明显不对称（常见于非对称带宽的家宽接入），这会低估其中一向、
        高估另一向，需要靠 II.6 的在线仪表（decode 每 token 实付的 w 与 d_loop）
        事后修正 —— 那才是真正逐向的观测。
        """
        rtts: list[float] = []
        s = None
        try:
            s = socket.create_connection(addr, timeout=5.0)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            for i in range(k):
                t0 = time.perf_counter()
                self.pool.egress_delay()   # 本节点发出去也要过自己的接入段
                send_msg(s, {"type": "echo", "seq": i})
                recv_msg(s)
                rtts.append((time.perf_counter() - t0) * 1000.0)
        except (OSError, ConnectionError) as e:
            return {"ok": False, "error": str(e), "p50": 0.0, "p95": 0.0, "k": k}
        finally:
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        one_way = np.array(rtts) / 2.0
        return {
            "ok": True,
            "p50": float(np.quantile(one_way, 0.5)),
            "p95": float(np.quantile(one_way, 0.95)),
            "k": k,
        }

    # -- 控制面 ------------------------------------------------------------ #
    def _report(self, msg: dict) -> None:
        """发给协调器。控制面不注入延迟 —— 它不占 decode 的每 token 预算。"""
        try:
            self.pool.send("__coord__", msg, delay=False)
        except Exception:
            pass

    def stats(self) -> dict:
        if self.cfg is None or self.model is None:
            return {"node": self.me, "configured": False, "msgs": self.n_msgs}
        return {
            "role": self.cfg.role,
            "segment": self.cfg.segment,
            "layers": sorted(self.cfg.layer_experts),
            "resident_mb": round(self.model.resident_bytes / 1e6, 2),
            "full_mb": round(self.model.full_bytes / 1e6, 2),
            "load_ms": round(self.load_ms, 1),
            "compute_ms": round(self.compute_ms, 1),
            "msgs": self.n_msgs,
            **self.pool.stats(),
        }


# --------------------------------------------------------------------------- #
def _detect_mem_mb() -> float:
    """粗略探测可用内存（MB）。真实 GPU 节点上应改成查显存。"""
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        pass
    return 1024.0


_BENCH_CACHE: float | None = None


def _bench_ms_per_layer() -> float:
    """跑一个小 matmul 基准估算「每层每 token」的耗时。

    异构池里算力近同质但不完全相同（文档设定：同档差 ≤ 1.5×），而规划器的
    计算项需要这个数。与其让运维填，不如让节点自己量 —— 量到的还包含了当时的
    实际负载与降频，比铭牌值更接近真相。
    """
    global _BENCH_CACHE
    if _BENCH_CACHE is not None:
        return _BENCH_CACHE
    d = 256
    a = np.random.default_rng(0).standard_normal((d, d))
    b = np.random.default_rng(1).standard_normal((d, d))
    t0 = time.perf_counter()
    for _ in range(20):
        a @ b
    per = (time.perf_counter() - t0) / 20 * 1000.0
    _BENCH_CACHE = round(per, 4)
    return _BENCH_CACHE


def run_agent(
    node_id: str, ready, host: str = "127.0.0.1",
    access_ms: float = 0.0, access_jitter_ms: float = 0.0,
) -> None:
    """进程入口：起一个**未配置**的 agent，把真实端口报回父进程。

    与 `p2pmoe.deploy.agent` 走的是同一条路径 —— 先起服务、再等 configure。
    这样单机多进程与真机部署共用一套装配流程，不会出现「本地能跑、上真机不行」。

    （早先的版本让父进程预分配端口再传给子进程，那有竞态：bind→close→子进程
    rebind 之间端口可能被别的进程抢走。让子进程自己 bind 0 并把端口报回来，
    这个窗口就不存在了。）
    """
    srv = NodeServer(node_id, host=host, port=0,
                     access_ms=access_ms, access_jitter_ms=access_jitter_ms)
    ready.put((node_id, srv.port))
    srv.serve_forever()


def run_node(cfg_dict: dict, port: int, ready, host: str = "127.0.0.1") -> None:
    """旧的一步式进程入口（配置随构造传入）。保留给不需要两阶段的场景。"""
    cfg = NodeConfig.from_dict(cfg_dict)
    srv = NodeServer(cfg.node_id, host=host, port=port)
    srv.apply_config(cfg)
    ready.put((cfg.node_id, srv.port))
    srv.serve_forever()

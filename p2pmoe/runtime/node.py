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

import importlib.util
import queue
import socket
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .identify import HistogramClassifier
from .model import MoEStats, SegmentModel, ToyMoEConfig, embed_tokens, lm_head
from .wire import (
    Addr,
    LinkTable,
    PeerPool,
    RelayListener,
    dial_via_relay,
    recv_msg,
    send_msg,
)

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

    # ---- 逐请求的时序埋点（用来算总时延与算力使用率）----
    compute_ms: float = 0.0
    """本节点在这条请求上**实际算**了多久。

    只用单调时钟量**时长**，不记绝对时刻 —— 15 台机器的墙钟不同步，跨机的
    绝对时刻拼不到一条时间轴上。时长是本地量、本地可信的，这也是为什么
    「网络时间」只能由 `总时延 − 各节点计算 − 排队` 反推，而不能直接测。
    """
    n_forward: int = 0
    """前向次数（prefill 1 次 + 每个 decode 步 1 次）。"""
    bytes_out: int = 0
    """本节点往下游发出的 hidden state 字节数。"""
    first_seen: float = 0.0
    """本节点第一次碰这条请求的本地时刻（单调）—— 只用来算本地存续时长。"""
    last_seen: float = 0.0


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
    """只有 front 的 tail 需要 —— 识别在那里做（II.5）。静态模式下为 None。"""

    miss_policy: str = "drop"
    """路由到的专家不在本地时怎么补救。三选一：

    * `"drop"`（默认，文档 II.5）—— 跳过缺失的，剩下的**重归一**；
    * `"drop_noscale"` —— 跳过缺失的，但**不重归一**：活下来的保持原权重，
      于是 FFN 贡献按丢掉的门控质量成比例缩小，缺得越多越接近「只走残差」；
    * `"local_topk"` —— 把路由概率限制到驻留集再取 top-k，永远用满 k 个。

    默认保持 `drop` 是为了忠于文档。但实测（`examples/drop_expert_impact.py`）
    在合成权重上 `drop_noscale` 明显更好 —— 重归一等于宣称「路由本来就只想要
    剩下这几个」，而那不是事实。真权重上该测一遍再定。"""

    profile: bool = False
    """开启逐层激活画像累计（runtime/profile.py）。

    只有想采画像的那一轮才开。开着的代价是每层一次向量加法 —— 相对前向可以忽略，
    但**采画像必须在全装（无 drop-expert 近似）的前提下做**：只驻留子集时输出被
    近似带偏，后面几层的路由就不是真实路由了。"""

    stop_ids: list[int] = field(default_factory=list)
    """EOS token id。**只是一个抄近路的优化，不是权威。**

    停止是控制面的决定（协调器判 EOS / stop 串 / max_tokens）。但采样出 EOS 的
    那一刻，tail(b) 就已经知道这一步不必再绕环了 —— 不告诉它的话，环会白转
    一整圈（两跳网络 + 一次前后段前向），只为算出一个马上要被丢掉的 token。

    所以两边都判：节点判了就不发 `loop`，协调器仍按自己收到的 token 收尾。
    两条判定互不依赖，节点这份配错了也只是少省一圈，不会漏停。

    节点因此**不需要 tokenizer** —— 它只拿到几个整数。""" 

    # ---- 静态配对模式 ----------------------------------------------------- #
    static_peer: str | None = None
    """前段 tail 的固定下游：对面后段的 head 节点 id。

    **给了它就进入静态模式**：前段 prefill 完直接发过去，不做识别、不上报
    `classified`、不等协调器的 `bind`。省掉一个控制面 RTT，也省掉分类器、
    置信三区、通道二检出、换绑这一整套。

    代价是放弃了「到达时 task 未知」这个前提（文档 I.1.1）—— 调用方必须知道
    自己是哪个 task。以及放弃了任意组合：配对写死在配置里，想改要重新下发
    （但**不用重新加载权重**，只要前段装的是全集或并集）。
    """
    static_task: str | None = None
    """本段服务的 task。静态模式下前后段都带，用于协调器按 task 选通道。"""

    # ---- 真实模型后端 ------------------------------------------------------ #
    backend: str = "numpy"
    """"numpy"（toy 模型，验证协议用）或 "torch"（真实 MoE 权重）。"""
    model_dir: str | None = None
    """backend="torch" 时的 checkpoint 目录（safetensors 格式）。"""
    device: str = "cpu"
    with_embed: bool = False
    """本节点是否需要词嵌入 —— 只有前段的 head 需要。"""
    with_lm_head: bool = False
    """本节点是否需要最终 norm + 输出头 —— 只有后段的 tail 需要。"""

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
        relay: Addr | None = None,
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

        # 中继模式：不监听任何端口，改成在中继上挂几条连接等人来接。
        # `RelayListener` 的接口与 socket 对齐，所以 serve_forever 一行不用改 ——
        # 节点不需要知道自己是在监听端口还是挂在中继上。
        self.relay = tuple(relay) if relay else None
        if self.relay:
            self.pool.use_relay(self.relay)
            self.sock = RelayListener(self.relay, self.me)
            self.host, self.port = self.relay
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((host, port))
            self.sock.listen(128)
            self.host, self.port = self.sock.getsockname()[:2]

    # -- 第二阶段：装载 ---------------------------------------------------- #
    def apply_config(self, cfg: NodeConfig) -> dict:
        """收到清单：加载**属于自己那一段的那几层的那些专家**。

        两个后端共用这一条路径，因为它们的契约相同
        （`forward(req, x) -> (y, MoEStats)`、`drop_kv` 等）。
        """
        self.cfg = cfg
        t0 = time.perf_counter()
        extra: dict = {}
        if cfg.backend == "torch":
            self.mcfg, extra = self._load_torch(cfg)
        else:
            self.mcfg = ToyMoEConfig(**cfg.model)
            self.model = SegmentModel(self.mcfg, cfg.layer_experts)
        if cfg.profile:
            self.model.enable_profiling()
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.clf = (
            HistogramClassifier.from_wire(cfg.classifier) if cfg.classifier else None
        )
        self.links = LinkTable.from_dict(cfg.links)
        self.pool = PeerPool(self.me, self.links, seed=abs(hash(self.me)) % (2**31),
                             egress_ms=self.access_ms,
                             egress_jitter_ms=self.access_jitter_ms)
        self.pool.use_relay(self.relay)   # 重建连接池时别把中继设置丢了
        for n, a in cfg.peers.items():
            self.pool.register(n, a)
        self.pool.register("__coord__", cfg.coordinator)
        return {
            "node": self.me, "segment": cfg.segment, "role": cfg.role,
            "backend": cfg.backend,
            "layers": sorted(cfg.layer_experts),
            "n_experts": sum(len(v) for v in cfg.layer_experts.values()),
            "resident_mb": round(self.model.resident_bytes / 1e6, 3),
            "full_mb": round(self.model.full_bytes / 1e6, 3),
            "load_ms": round(self.load_ms, 1),
            "static_peer": cfg.static_peer,
            "task": cfg.static_task,
            **extra,
        }

    def _load_torch(self, cfg: NodeConfig):
        """真实模型后端：按清单从 safetensors 只读需要的张量。

        torch / safetensors 是**可选依赖**（requirements-node.txt）——
        只有配成 backend="torch" 的节点才会走到这里，控制机永远不会。
        """
        from .weights import (
            KeyPlan, SelectiveLoader, WeightIndex, qwen3_next_keys, qwen_moe_keys,
        )

        if not cfg.model_dir:
            raise ValueError("backend='torch' 需要 model_dir")

        # 按 checkpoint 自报的架构分派。**不要按「有没有某个字段」去猜** ——
        # 猜错的代价是加载一个结构不对的模型，而它不会报错，只会算出垃圾。
        arch = str(cfg.model.get("model_type", "")).lower()
        archs = [str(a).lower() for a in cfg.model.get("architectures", [])]
        is_next = arch == "qwen3_next" or any("qwen3next" in a for a in archs)

        idx = WeightIndex(self.resolve_dir(cfg.model_dir))
        plan = KeyPlan(layer_experts=cfg.layer_experts,
                       with_embed=cfg.with_embed, with_lm_head=cfg.with_lm_head)
        if is_next:
            from .qwen3_next import NextModelConfig, TorchNextSegmentModel

            mcfg = NextModelConfig.from_hf(cfg.model)
            keys = qwen3_next_keys(
                plan, layer_types=list(mcfg.layer_types),
                shared_expert=mcfg.shared_intermediate > 0,
                tie_word_embeddings=mcfg.tie_word_embeddings)
            SegModel = TorchNextSegmentModel
        else:
            from .torch_model import TorchModelConfig, TorchSegmentModel

            mcfg = TorchModelConfig.from_hf(cfg.model)
            keys = qwen_moe_keys(plan, tie_word_embeddings=mcfg.tie_word_embeddings)
            SegModel = TorchSegmentModel
        tensors, rep = SelectiveLoader(idx).load(
            keys, device=cfg.device, dtype=mcfg.torch_dtype
        )
        if rep.missing:
            raise KeyError(f"checkpoint 缺 {len(rep.missing)} 个 key，首个: {rep.missing[0]}")
        self.model = SegModel(mcfg, cfg.layer_experts, tensors, device=cfg.device,
                              miss_policy=cfg.miss_policy)
        return mcfg, {
            "loaded_gb": round(rep.bytes_loaded / 1e9, 3),
            "ckpt_gb": round(rep.bytes_total / 1e9, 3),
            "load_fraction": round(rep.fraction, 4),
            "shards": f"{rep.shards_opened}/{rep.shards_total}",
            "model_dir": self.resolve_dir(cfg.model_dir),
            "arch": "qwen3_next" if is_next else "qwen3_moe",
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
            res = self.probe_peer(tuple(h["addr"]), int(h.get("k", 8)),
                                  peer=h.get("peer"))
            if conn is not None:
                send_msg(conn, {"type": "probe_ack", "peer": h["peer"],
                                "src": self.me, **res})
            return
        if t == "check_model":
            # 预检必须在 configure **之前**能答 —— 它问的就是「你能不能加载」
            if conn is not None:
                send_msg(conn, {"type": "check_model_ack",
                                **self.check_model(str(h.get("dir", "")))})
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
            # 释放前把这条请求的时序埋点报回去 —— 之后状态就没了。
            # 报的是**时长**不是时刻：跨机的绝对时刻拼不到一条轴上（无时钟同步）。
            st_ = self._reqs.get(h["req"])
            if st_ is not None:
                self._report({
                    "type": "req_trace", "req": h["req"], "node": self.me,
                    "segment": self.cfg.segment, "role": self.cfg.role,
                    "layers": sorted(self.cfg.layer_experts),
                    "n_experts": sum(len(v) for v in self.cfg.layer_experts.values()),
                    "compute_ms": round(st_.compute_ms, 3),
                    "n_forward": st_.n_forward,
                    "bytes_out": st_.bytes_out,
                    "local_span_ms": round((st_.last_seen - st_.first_seen) * 1000, 3)
                    if st_.first_seen else 0.0,
                })
            self.model.drop_kv(h["req"])
            with self._lock:
                self._reqs.pop(h["req"], None)
        elif t == "get_profile":
            if conn is not None:
                send_msg(conn, {"type": "profile_ack", "node": self.me,
                                "segment": self.cfg.segment, "role": self.cfg.role,
                                "task": self.cfg.static_task,
                                **(self.model.profiler.to_wire()
                                   if self.model.profiler else {"layers": {}})})
        elif t == "stats":
            self._report({"type": "node_stats", "node": self.me, **self.stats()})

    # -- 数据面 ------------------------------------------------------------ #
    def _state(self, req: str) -> ReqState:
        with self._lock:
            return self._reqs.setdefault(req, ReqState())

    def _forward(self, x: np.ndarray, req: str) -> tuple[np.ndarray, MoEStats]:
        st_ = self._state(req)
        t0 = time.perf_counter()
        if not st_.first_seen:
            st_.first_seen = t0
        h, st = self.model.forward(req, x)
        dt = (time.perf_counter() - t0) * 1000
        self.compute_ms += dt
        st_.compute_ms += dt
        st_.n_forward += 1
        st_.last_seen = time.perf_counter()
        st_.bytes_out += int(getattr(h, "nbytes", 0)
                             or (h.numel() * 4 if hasattr(h, "numel") else 0))
        return h, st

    def _on_prefill(self, h: dict) -> None:
        """协调器 → head(f)：一条新请求到达（盲绑已经在协调器侧完成）。"""
        req = h["req"]
        x = (self.model.embed_tokens(h["ids"]) if self.cfg.backend == "torch"
             else embed_tokens(self.mcfg, h["ids"]))
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

        # ---- 静态模式：配对在部署时就定死了，不识别、不问协调器 ----
        if self.cfg.static_peer:
            s.bound_to = s.bound_to or self.cfg.static_peer
            s.cached_l0 = y if phase == "prefill" else s.cached_l0
            self.pool.send(
                s.bound_to,
                {"type": "seg_in", "req": req, "phase": phase,
                 "loop_to": self.cfg.seg_head, "task": self.cfg.static_task},
                y,
            )
            if phase == "prefill":
                self._report({
                    "type": "static_forward", "req": req, "node": self.me,
                    "task": self.cfg.static_task, "peer": s.bound_to,
                    "front_stats": st.to_wire(),
                })
            return

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
        x = (self.model.embed_tokens([h["token"]]) if self.cfg.backend == "torch"
             else embed_tokens(self.mcfg, [h["token"]]))
        y, st = self._forward(x, req)
        self._advance(req, "decode", y, st, self.cfg.seg_head)

    # -- 后段出口：采样 + 回环 --------------------------------------------- #
    def _back_tail(
        self, req: str, phase: str, y: np.ndarray, st: MoEStats, loop_to: str | None
    ) -> None:
        # 温度采样而非 argmax：权重绑定 + argmax 会让生成塌成一个不动点，
        # decode 阶段就永远在同一个 token 上打转，验证不到路由的动态。
        st_ = self._state(req)
        if self.cfg.backend == "torch":
            # 采样交给模型自己做 —— node.py 不 import torch，否则 toy 模型路径
            # 也会被迫装 torch（见 test_requirements 的依赖边界）
            token = self.model.sample(y, temperature=0.0)
        else:
            logits = lm_head(self.mcfg, y[-1]) / max(self.mcfg.sample_temp, 1e-6)
            pr = np.exp(logits - logits.max())
            pr /= pr.sum()
            rng = np.random.default_rng(abs(hash((req, st_.n_sampled))) % (2**31))
            token = int(rng.choice(len(pr), p=pr))
        st_.n_sampled += 1
        stop = token in self.cfg.stop_ids
        self._report({
            "type": "token", "req": req, "node": self.me, "segment": self.cfg.segment,
            "phase": phase, "token": token, "back_stats": st.to_wire(), "stop": stop,
        })
        if stop:
            return          # 抄近路：不再绕环。收尾由协调器做（见 NodeConfig.stop_ids）
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
            "relay": list(self.relay) if self.relay else None,
        }

    def resolve_dir(self, model_dir: str | None) -> str | None:
        """把 `{node}` 代成本机的节点名。

        为什么由节点代而不是控制机代：**只在一处代，就只有一处会代错。**
        预检问的是「你能不能加载」，加载做的是「按这个路径加载」—— 两者必须
        是同一个路径，而唯一保证这一点的办法是让同一段代码算它。

        真机上各机权重放在同一个路径，压根不用占位；占位是给「多个 agent 挤在
        一台机器上」的演练用的。
        """
        if not model_dir:
            return model_dir
        return model_dir.replace("{node}", self.me)

    def check_model(self, model_dir: str) -> dict:
        """本机能不能加载这个 checkpoint —— 在下发清单**之前**回答。

        权重分发还没做（TODO.md P0），`model_dir` 是各节点上的本地路径。
        控制机读得到不代表这台读得到，而不预检的话故障会推迟到下发那一刻才爆，
        那时前面几分钟的探测已经白跑。所以这里只做便宜的检查：目录在不在、
        config.json 在不在、权重索引在不在、torch/safetensors 装没装。
        """
        if not model_dir:
            return {"ok": False, "why": "没给目录"}
        model_dir = self.resolve_dir(model_dir)
        d = Path(model_dir)
        if not d.is_dir():
            return {"ok": False, "why": f"{d} 不存在或不是目录"}
        if not (d / "config.json").exists():
            return {"ok": False, "why": f"{d}/config.json 不存在"}
        idx = d / "model.safetensors.index.json"
        shards = sorted(d.glob("*.safetensors"))
        if not idx.exists() and not shards:
            return {"ok": False, "why": f"{d} 里没有 safetensors 权重"
                                        f"（.bin 格式不支持：pickle 做不到只读部分张量）"}
        for mod in ("torch", "safetensors"):
            if importlib.util.find_spec(mod) is None:
                return {"ok": False, "why": f"没装 {mod}"
                                            f"（pip install -r requirements-node.txt）"}
        return {
            "ok": True,
            "node": self.me,
            "shards": len(shards),
            "bytes": sum(f.stat().st_size for f in shards),
        }

    def probe_peer(self, addr: tuple[str, int], k: int, peer: str | None = None) -> dict:
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
            if self.relay:
                # 中继模式下量到的是**经中继的**往返，不是两台之间的真实延迟。
                # 规划器据此做的放置仍然自洽（它优化的就是实际付出的延迟），
                # 但别拿这些数字去推断链路质量。
                if not peer:
                    return {"ok": False, "error": "中继模式下探测要给对端节点 id",
                            "p50": 0.0, "p95": 0.0, "k": k}
                s = dial_via_relay(self.relay, self.me, peer, timeout=15.0)
                s.settimeout(15.0)
            else:
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

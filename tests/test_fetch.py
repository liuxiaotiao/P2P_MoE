"""只拉本机要的那部分权重。

这一层的正确性判据很硬：**拼出来的张量必须与上游逐位一致**。
搬运字节的代码错一个偏移不会抛异常，只会让权重变成噪声 —— 而随机权重跑起来
一样出 token，肉眼看不出来。所以这里逐个张量比对。
"""

from __future__ import annotations

import functools
import http.server
import io
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.deploy.fetch import (
    RangeNotSupported,
    Source,
    fetch,
    keys_for_node,
    plan_fetch,
    read_header,
)
from p2pmoe.deploy.manual import ManualSpec, build_manual_manifest
from p2pmoe.planner.hf_config import model_spec_from_hf

pytest.importorskip("torch")
pytest.importorskip("safetensors")

CFG = None


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def upstream(tmp_path_factory) -> tuple[Path, dict]:
    """一个 8 层 / 6 分片的 checkpoint 当上游。"""
    from p2pmoe.sim.fake_checkpoint import TINY_QWEN3_MOE, write_fake_checkpoint

    cfg = dict(TINY_QWEN3_MOE, num_hidden_layers=8)
    d = tmp_path_factory.mktemp("upstream")
    write_fake_checkpoint(d, cfg, seed=0, n_shards=6)
    return d, cfg


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler 不支持 Range，这里补上。"""

    def log_message(self, *a) -> None:
        pass

    def send_head(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().send_head()
        path = self.translate_path(self.path)
        size = Path(path).stat().st_size
        lo, _, hi = rng.split("=")[1].partition("-")
        lo, hi = int(lo), (int(hi) if hi else size - 1)
        with open(path, "rb") as f:
            f.seek(lo)
            buf = f.read(hi - lo + 1)
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Range", f"bytes {lo}-{hi}/{size}")
        self.send_header("Content-Length", str(len(buf)))
        self.end_headers()
        return io.BytesIO(buf)


class _NoRangeHandler(_RangeHandler):
    """无视 Range，整个文件发回去 —— 模拟不支持 Range 的镜像。"""

    def send_head(self):
        return http.server.SimpleHTTPRequestHandler.send_head(self)


@pytest.fixture(scope="module")
def server(upstream):
    d, _ = upstream
    srv = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_RangeHandler, directory=str(d)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def make_manifest(cfg: dict):
    spec, _ = model_spec_from_hf(cfg, name="t", ctx_max=128, dtype_bytes=4)
    layout = ManualSpec.from_dict({"channels": [{
        "front": [{"node": "n1", "layers": [1, 2]}],
        "back": [{"node": "n2", "layers": [3, 5]}, {"node": "n3", "layers": [6, 8]}],
    }], "experts": {str(l): [0, 1, 2] for l in range(3, 9)}}, n_layers=8)
    return build_manual_manifest(layout, spec)[0]


def load_all(d: Path) -> dict:
    from safetensors.torch import load_file

    out: dict = {}
    for f in sorted(d.glob("*.safetensors")):
        out.update(load_file(str(f)))
    return out


# --------------------------------------------------------------------------- #
# 1. 算：只读文件头
# --------------------------------------------------------------------------- #
def test_header_gives_absolute_offsets(upstream) -> None:
    d, _ = upstream
    src = Source(local=d)
    header, data_at = read_header(src, "model-00001-of-00006.safetensors")
    assert data_at > 8                       # 8 字节长度 + 头本身
    assert "__metadata__" in header or header


def test_only_the_needed_shards_are_opened(upstream) -> None:
    """61GB 的模型十几个分片，为了算个总量挨个读头是白费往返。"""
    d, cfg = upstream
    man = make_manifest(cfg)
    p = plan_fetch(Source(local=d), keys_for_node(man, "n1"))     # 只要 1–2 层
    assert p.shards_read < p.shards_total
    assert p.shards_read == len(p.shards_needed)


def test_slice_beats_shard_which_beats_full(upstream) -> None:
    d, cfg = upstream
    man = make_manifest(cfg)
    p = plan_fetch(Source(local=d), keys_for_node(man, "n1"))
    assert p.bytes_slice < p.bytes_shards <= p.bytes_total


def test_a_key_the_upstream_lacks_is_reported(upstream) -> None:
    d, _ = upstream
    p = plan_fetch(Source(local=d), {"model.layers.999.mlp.gate.weight"})
    assert p.missing == ["model.layers.999.mlp.gate.weight"]


# --------------------------------------------------------------------------- #
# 2. 拉：逐位一致
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("node", ["n1", "n2", "n3"])
def test_slices_are_bit_identical_to_upstream(upstream, server, tmp_path, node) -> None:
    """**本文件最重要的一条。** 搬错一个偏移不会抛异常，只会让权重变成噪声。"""
    import torch

    d, cfg = upstream
    man = make_manifest(cfg)
    out = tmp_path / node
    fetch(Source(base_url=server), keys_for_node(man, node), out, mode="slice")

    full, got = load_all(d), load_all(out)
    assert got
    for k, v in got.items():
        assert torch.equal(v, full[k]), f"{k} 数值对不上"


def test_only_the_requested_keys_come_down(upstream, server, tmp_path) -> None:
    d, cfg = upstream
    man = make_manifest(cfg)
    keys = keys_for_node(man, "n2")
    fetch(Source(base_url=server), keys, tmp_path / "n2", mode="slice")
    assert set(load_all(tmp_path / "n2")) == set(keys)


def test_shard_mode_also_works(upstream, server, tmp_path) -> None:
    """不支持 Range 的镜像上的兜底 —— 省得少，但拿到的张量一样对。"""
    import torch

    d, cfg = upstream
    man = make_manifest(cfg)
    fetch(Source(base_url=server), keys_for_node(man, "n1"), tmp_path / "s",
          mode="shard")
    full, got = load_all(d), load_all(tmp_path / "s")
    assert set(keys_for_node(man, "n1")) <= set(got)     # 整分片会多拿一些
    for k, v in got.items():
        assert torch.equal(v, full[k])


def test_local_source_needs_no_server(upstream, tmp_path) -> None:
    """共享存储/已下好的全量也能当上游 —— 同一套逻辑，不走网络。"""
    d, cfg = upstream
    man = make_manifest(cfg)
    fetch(Source(local=d), keys_for_node(man, "n3"), tmp_path / "n3", mode="slice")
    assert load_all(tmp_path / "n3")


# --------------------------------------------------------------------------- #
# 3. 产出的是一个能用的 checkpoint
# --------------------------------------------------------------------------- #
def test_output_is_a_valid_checkpoint_directory(upstream, server, tmp_path) -> None:
    d, cfg = upstream
    man = make_manifest(cfg)
    out = tmp_path / "n2"
    fetch(Source(base_url=server), keys_for_node(man, "n2"), out, mode="slice",
          extra_files=("config.json", "tokenizer.json"))
    assert (out / "config.json").exists()
    assert (out / "model.safetensors.index.json").exists()
    idx = json.loads((out / "model.safetensors.index.json").read_text())
    assert set(idx["weight_map"].values()) == {"model.safetensors"}


def test_the_node_loader_reads_it_unchanged(upstream, server, tmp_path) -> None:
    """节点侧代码一行不用改 —— 这才是「产出一个更小的合法 checkpoint」的意义。"""
    from p2pmoe.runtime.weights import KeyPlan, SelectiveLoader, WeightIndex, qwen_moe_keys

    d, cfg = upstream
    man = make_manifest(cfg)
    out = tmp_path / "n2"
    fetch(Source(base_url=server), keys_for_node(man, "n2"), out, mode="slice",
          extra_files=("config.json",))

    p = man.plan_for("n2")
    keys = qwen_moe_keys(KeyPlan(
        layer_experts={l.layer: list(l.experts) for l in p.layers}))
    tensors, rep = SelectiveLoader(WeightIndex(out)).load(keys)
    assert not rep.missing
    assert set(tensors) == set(keys)


def test_a_fetched_part_actually_serves_inference(upstream, server, tmp_path) -> None:
    """端到端：拉下来的那一份权重能真的跑出 token。"""
    import torch

    from p2pmoe.runtime.torch_model import TorchModelConfig, TorchSegmentModel
    from p2pmoe.runtime.weights import (
        KeyPlan, SelectiveLoader, WeightIndex, qwen_moe_keys,
    )

    d, cfg = upstream
    man = make_manifest(cfg)
    out = tmp_path / "n1"
    fetch(Source(base_url=server), keys_for_node(man, "n1"), out, mode="slice",
          extra_files=("config.json",))

    mcfg = TorchModelConfig.from_hf(cfg)
    p = man.plan_for("n1")
    le = {l.layer: list(l.experts) for l in p.layers}
    keys = qwen_moe_keys(KeyPlan(layer_experts=le, with_embed=True))
    tensors, _ = SelectiveLoader(WeightIndex(out)).load(keys, dtype=torch.float32)
    m = TorchSegmentModel(mcfg, le, tensors)
    y, st = m.forward("r0", m.embed_tokens([1, 2, 3]))
    assert y.shape == (3, mcfg.d_model)
    assert torch.isfinite(y).all()
    assert st.n_token_layer > 0


# --------------------------------------------------------------------------- #
# 4. 上游不支持 Range 时必须报错，不能将就
# --------------------------------------------------------------------------- #
def test_a_server_ignoring_range_is_an_error(upstream, tmp_path) -> None:
    """返回 200 意味着「省下载」根本没发生，而按区间长度切出来的还会是错的。"""
    d, cfg = upstream
    srv = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_NoRangeHandler, directory=str(d)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        src = Source(base_url=f"http://127.0.0.1:{srv.server_address[1]}")
        # 关键是**报对了原因**：不能退化成「找不到 model.safetensors」的 404
        with pytest.raises(RangeNotSupported, match="--mode shard"):
            plan_fetch(src, keys_for_node(make_manifest(cfg), "n1"))
    finally:
        srv.shutdown()


# --------------------------------------------------------------------------- #
# 5. key 集合由同一个函数算
# --------------------------------------------------------------------------- #
def test_download_and_load_agree_on_what_is_needed(upstream) -> None:
    """下载和加载对「要哪些 key」必须同源，否则会出现「下了却加载不到」。"""
    from p2pmoe.runtime.weights import KeyPlan, qwen_moe_keys

    d, cfg = upstream
    man = make_manifest(cfg)
    p = man.plan_for("n3")
    same = qwen_moe_keys(KeyPlan(
        layer_experts={l.layer: list(l.experts) for l in p.layers},
        with_embed=False, with_lm_head=True))
    assert keys_for_node(man, "n3") == same


def test_head_gets_the_embedding_tail_gets_the_lm_head(upstream) -> None:
    d, cfg = upstream
    man = make_manifest(cfg)
    assert "model.embed_tokens.weight" in keys_for_node(man, "n1")
    assert "model.embed_tokens.weight" not in keys_for_node(man, "n2")
    assert "lm_head.weight" in keys_for_node(man, "n3")     # 后段的 tail
    assert "lm_head.weight" not in keys_for_node(man, "n2")


def test_an_unknown_node_lists_what_exists(upstream) -> None:
    d, cfg = upstream
    with pytest.raises(SystemExit, match="n1"):
        keys_for_node(make_manifest(cfg), "nope")


# --------------------------------------------------------------------------- #
# 6. `{node}` 占位由**节点自己**代入
# --------------------------------------------------------------------------- #
def _bare_node(name: str):
    from p2pmoe.runtime.node import NodeServer

    n = object.__new__(NodeServer)
    n.me = name
    return n


def test_the_node_substitutes_its_own_name() -> None:
    """只在一处代，就只有一处会代错。

    预检问的是「你能不能加载」，加载做的是「按这个路径加载」—— 两者必须是同一个
    路径，而唯一保证这一点的办法是让同一段代码算它。
    """
    assert _bare_node("n7").resolve_dir("/data/parts/{node}") == "/data/parts/n7"


def test_a_path_without_a_placeholder_is_untouched() -> None:
    """真机上各机同路径，压根不用占位 —— 那条路必须原样通过。"""
    assert _bare_node("n7").resolve_dir("/data/qwen3-part") == "/data/qwen3-part"


def test_preflight_and_load_resolve_to_the_same_place(tmp_path) -> None:
    """预检说「读得到」，加载就必须读得到同一个目录 —— 否则预检等于没做。"""
    d = tmp_path / "n3"
    d.mkdir()
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"x" * 16)

    node = _bare_node("n3")
    tmpl = str(tmp_path / "{node}")
    assert node.check_model(tmpl)["ok"]
    assert node.resolve_dir(tmpl) == str(d)


def test_a_wrong_node_name_fails_the_preflight(tmp_path) -> None:
    (tmp_path / "n3").mkdir()
    assert not _bare_node("n9").check_model(str(tmp_path / "{node}"))["ok"]

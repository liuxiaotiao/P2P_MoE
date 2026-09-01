"""核对本机分片里是不是有清单要的每个 key。

`check` 里的预检很便宜（目录在不在、依赖装没装），但**不查 key 齐不齐**。
而缺 key 要到 `measure` 装载模型的那一刻才炸 —— 那时探测与建链已经白跑了几分钟。
只读 safetensors 的文件头就能回答，代价与收益完全不成比例。

最常见的触发场景：改了 `COVERAGE` 之后没重跑 `fetch` —— 驻留集变了，
权重还是旧的那一份。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.deploy.fetch import keys_for_node
from p2pmoe.planner.manifest import DeploymentManifest
from p2pmoe.runtime.weights import WeightIndex
from p2pmoe.sim.fake_checkpoint import TINY_QWEN3_NEXT, write_fake_next_checkpoint

ROOT = Path(__file__).resolve().parent.parent
CFG = dict(TINY_QWEN3_NEXT)
L, E = CFG["num_hidden_layers"], CFG["num_experts"]
L0 = L // 3


def _manifest(step: int) -> dict:
    """`step` 越大驻留集越小 —— 用来模拟不同覆盖率下的清单。"""
    return {"l0": L0, "model": {}, "segments": {"B0": {
        "role": "back:u", "task": "u", "nodes": ["nb"],
        "splits": [[L0 + 1, L]], "head": "nb", "tail": "nb", "hops": 0,
        "compute_ms": 1., "hop_ms": 0., "delay_ms": 1.}},
        "nodes": [{"node": "nb", "role": "back:u", "segment": "B0", "position": 0,
                   "is_head": True, "is_tail": True, "layer_range": [L0 + 1, L],
                   "weight_gb": .1, "kv_gb": 0., "total_gb": .1,
                   "layers": [{"layer": l, "experts": list(range(0, E, step)),
                               "weight_gb": .01, "kv_gb": 0.}
                              for l in range(L0 + 1, L + 1)]}]}


@pytest.fixture(scope="module")
def full(tmp_path_factory) -> Path:
    return Path(write_fake_next_checkpoint(
        str(tmp_path_factory.mktemp("full")), CFG, seed=0))


def _fetch_with(plan: dict, full: Path, out: Path) -> None:
    pf = out.parent / "plan.json"
    pf.write_text(json.dumps(plan), encoding="utf-8")
    r = subprocess.run(
        [sys.executable, "-m", "p2pmoe.deploy.fetch", "--plan", str(pf),
         "--node", "nb", "--src", str(full), "--out", str(out)],
        capture_output=True, text=True, timeout=180, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


def _missing(plan: dict, weights: Path) -> set[str]:
    """核对本身 —— 与 `deploy_15.sh verify` 在节点上跑的是同一段逻辑。"""
    want = keys_for_node(DeploymentManifest.from_dict(plan), "nb", config=CFG)
    return want - set(WeightIndex(str(weights)).weight_map)


def test_a_matching_fetch_leaves_nothing_missing(full, tmp_path) -> None:
    plan = _manifest(3)
    out = tmp_path / "w"
    _fetch_with(plan, full, out)
    assert _missing(plan, out) == set()


def test_a_stale_fetch_is_caught(full, tmp_path) -> None:
    """**这条是重点。** 按旧覆盖率拉的权重，配上新清单，缺的 key 要被找出来。"""
    _fetch_with(_manifest(6), full, tmp_path / "w")      # 旧：驻留集更小
    miss = _missing(_manifest(3), tmp_path / "w")        # 新：要得更多
    assert miss, "旧权重配新清单竟然没缺 key —— 核对失效了"
    assert any(".experts." in k for k in miss)


def test_the_check_reads_only_headers(full, tmp_path, monkeypatch) -> None:
    """只读索引与文件头 —— 真机上单台 22GB，读一遍数据就不是「几秒钟」了。"""
    plan = _manifest(3)
    out = tmp_path / "w"
    _fetch_with(plan, full, out)

    opened: list[str] = []
    real = Path.open

    def spy(self, *a, **k):
        opened.append(self.name)
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "open", spy)
    _missing(plan, out)
    assert not any(n.endswith(".safetensors") for n in opened), \
        "核对时打开了权重文件本体"


def test_the_script_has_the_verify_step() -> None:
    """脚本里要真的有这一步，且排在 start 之前 —— 放在后面就失去意义了。"""
    src = (ROOT / "deploy_15.sh").read_text(encoding="utf-8")
    assert "cmd_verify" in src
    assert "verify)" in src
    i_v = src.index("#   ./deploy_15.sh verify")
    i_s = src.index("#   ./deploy_15.sh start")
    assert i_v < i_s, "verify 该排在 start 前面"


def test_the_failure_points_at_the_actual_cause() -> None:
    """「缺 key」最常见的原因是改了覆盖率没重拉，而不是 checkpoint 坏了。"""
    src = (ROOT / "deploy_15.sh").read_text(encoding="utf-8")
    i = src.index("cmd_verify")
    tail = src[i:i + 4000]
    assert "COVERAGE" in tail
    assert "fetch" in tail


# --------------------------------------------------------------------------- #
# 「全缺」时要能分清是哪一种全缺
# --------------------------------------------------------------------------- #
def test_the_report_shows_which_layers_are_actually_present(full, tmp_path) -> None:
    """**真踩过。**

    一台报「缺 3321/3321，本地 22.5GB」—— 缺全部，却又装着东西。
    只看这两个数字判断不了；对上「本地实际有层 21–48 / 要的是层 12–20」
    才知道它装的是**别的节点**那一份。

    两种全缺的修法完全不同：装错人要单台重拉，驻留集变了要整轮重跑。
    """
    import re

    plan_a = _manifest(3)
    plan_b = json.loads(json.dumps(plan_a))
    # B 要的是另一段层 —— 模拟「装了别人那份」
    for n in plan_b["nodes"]:
        n["layer_range"] = [L0 + 1, L0 + 2]
        n["layers"] = [l for l in n["layers"] if l["layer"] <= L0 + 2]
    plan_b["segments"]["B0"]["splits"] = [[L0 + 1, L0 + 2]]

    out = tmp_path / "w"
    _fetch_with(plan_a, full, out)          # 装的是 A（层 L0+1..L）

    def span(keys):
        ls = {int(m.group(1)) + 1 for k in keys
              for m in [re.search(r"model\.layers\.(\d+)\.", k)] if m}
        return (min(ls), max(ls)) if ls else None

    want_b = keys_for_node(DeploymentManifest.from_dict(plan_b), "nb", config=CFG)
    have = set(WeightIndex(str(out)).weight_map)
    assert span(have) != span(want_b), "构造失败：两边层区间该不同"
    # 本地有的层区间必须能算出来 —— 这就是报告里 have= 的来源
    assert span(have) is not None


def test_the_script_reports_have_and_want_spans() -> None:
    """脚本里要真的把这两个区间打出来，否则上面那条推理在现场用不上。"""
    src = (ROOT / "deploy_15.sh").read_text(encoding="utf-8")
    i = src.index("cmd_verify")
    blk = src[i:i + 4000]
    assert "have=" in blk and "want=" in blk
    assert "def span" in blk, "没有算层区间的函数"


def test_refetch_clears_before_pulling() -> None:
    """**重来一次必须真的从零开始。**

    `fetch` 正常会覆盖 model.safetensors，但只要有一步先失败，
    目录里那份不属于这台的旧产物就一直留着 —— 于是「重跑了还是不对」。
    """
    src = (ROOT / "deploy_15.sh").read_text(encoding="utf-8")
    i = src.index("cmd_refetch")
    blk = src[i:i + 2000]
    assert "rm -f" in blk, "refetch 没有先清旧产物"
    i_rm, i_fetch = blk.index("rm -f"), blk.index("p2pmoe.deploy.fetch")
    assert i_rm < i_fetch, "先拉后删等于没删"
    assert "index.json" in blk, "索引文件也要清 —— 留着会指向不存在的张量"

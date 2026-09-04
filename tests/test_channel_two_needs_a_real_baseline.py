"""一个偏差已知的估计值，不该驱动一个不可逆的动作。

真机上量到的事实（results/cov70.json，20 条请求）：

    换绑 0 次的 5 条 —— 5/5 全对
    换绑 1 次的 15 条 —— 15/15 全错

完美相关。`task` 字段记的是换绑**之后**的 task，所以真相是：分类器 20/20 全
判对了（置信 0.79–0.91，全在 commit 区，两个参考的余弦 0.82 分得很开），
然后通道二的误绑告警把其中 15 条改成了错的。

告警线是怎么来的：

    baselines = 1 − 覆盖率 = 0.30        # control.py 里注释写明「偏低 3–6 倍」
    告警线 = 0.30 × alarm_factor 3.0 = 0.90
    而真实 miss 率 ≈ 0.30 × (3~6) = 0.9 ~ 1.8

**告警线正好落在正常工作区间的中间**，于是它成了一枚硬币。而只有两个 task
时，换绑必然换到另一个 —— 硬币一朝上，一条对的绑定就变成错的。

修法不是把系数调大（那只是把硬币换个位置），而是：估计出来的基线**不报警**，
改成拿运行中没换过绑的请求在线自校准，攒够样本再启用。
"""

from __future__ import annotations

import pytest

from p2pmoe.runtime.coordinator import Coordinator
from p2pmoe.planner.manifest import DeploymentManifest


def _man() -> DeploymentManifest:
    """两条前段、两个 task 各一条后段的最小清单。"""
    nodes, segs = [], {}

    def add(nid: str, sid: str, role: str, task, layers):
        segs[sid] = {"role": role, "task": task, "nodes": [nid],
                     "splits": [list(layers)], "head": nid, "tail": nid,
                     "hops": 0, "compute_ms": 1.0, "hop_ms": 0.0, "delay_ms": 1.0}
        nodes.append({"node": nid, "role": role, "segment": sid,
                      "position": 0, "is_head": True, "is_tail": True,
                      "layer_range": [layers[0], layers[-1]],
                      "weight_gb": 1.0, "kv_gb": 0.0, "total_gb": 1.0,
                      "layers": [{"layer": l, "experts": [0], "weight_gb": .5,
                                  "kv_gb": 0.0} for l in layers]})

    add("f0", "F0", "front", None, (1, 2))
    add("f1", "F1", "front", None, (1, 2))
    add("bm", "Bm0", "back:mbpp", "mbpp", (3, 4))
    add("bg", "Bg0", "back:gsm8k", "gsm8k", (3, 4))
    return DeploymentManifest.from_dict(
        {"l0": 2, "model": {}, "segments": segs, "nodes": nodes})


def _coord(**kw) -> Coordinator:
    c = Coordinator(_man(), baselines={"mbpp": 0.30, "gsm8k": 0.30},
                    priors={"mbpp": 0.62, "gsm8k": 0.38}, **kw)
    return c


def test_an_estimated_baseline_never_arms_the_alarm() -> None:
    """核心断言：估计值进来，告警线出去必须是 None。

    None 的意思是「还不能报警」，而不是「基线是 0」—— 后者会让**任何**
    miss 率都超线，正好把 bug 放大到极致。
    """
    c = _coord(baselines_measured=False)
    assert c.alarm_baseline("mbpp") is None
    assert c.alarm_baseline("gsm8k") is None


def test_a_measured_baseline_is_used_as_is() -> None:
    """toy 模型能拿同一份语料回放，基线是实测的，照用不误。"""
    c = _coord(baselines_measured=True)
    assert c.alarm_baseline("mbpp") == pytest.approx(0.30)


def test_measured_is_the_default() -> None:
    """既有调用方（toy 那条路、各处测试）传的都是实测值，默认不能变。"""
    c = _coord()
    assert c.baselines_measured is True
    assert c.alarm_baseline("mbpp") == pytest.approx(0.30)


def test_calibration_arms_the_alarm_after_enough_samples() -> None:
    """自校准：拿没换过绑的请求实测，攒够 calibrate_n 条就启用。"""
    c = _coord(baselines_measured=False, calibrate_n=3, min_window=2)
    c._miss_seen["mbpp"] = [0.80, 0.90]
    assert c.alarm_baseline("mbpp") is None, "样本不够就不该报警"
    c._miss_seen["mbpp"].append(1.00)
    assert c.alarm_baseline("mbpp") == pytest.approx(0.90), "取中位数"


def test_calibration_uses_the_median_not_the_mean() -> None:
    """一条离群请求不该把告警线拉走。

    均值会被 5.0 那条拖到 1.4，从此再也不报警；中位数只动一格。
    """
    c = _coord(baselines_measured=False, calibrate_n=5)
    c._miss_seen["mbpp"] = [0.8, 0.85, 0.9, 0.95, 5.0]
    assert c.alarm_baseline("mbpp") == pytest.approx(0.9)


def test_only_unrebound_requests_feed_the_calibration() -> None:
    """换过绑的请求多半绑错了，它的 miss 率不能当「绑对时的样子」。

    「没换过绑」是运行期能拿到的、最接近「绑对了」的信号 —— 真实 task 只有
    评分时才知道，运行期看它就是作弊。
    """
    c = _coord(baselines_measured=False, min_window=2)

    class Rec:
        task = "mbpp"
        rebinds = 0
        miss_window = [0.9, 0.9]
        window_miss = 0.9

        def log(self, _m):
            pass

    good = Rec()
    c._note_baseline_sample(good)
    assert c._miss_seen["mbpp"] == [0.9]

    bad = Rec()
    bad.rebinds = 1
    c._note_baseline_sample(bad)
    assert c._miss_seen["mbpp"] == [0.9], "换过绑的请求不该进校准样本"


def test_measured_mode_collects_nothing() -> None:
    """实测基线已经可信，不需要也不应该被运行期样本改写。"""
    c = _coord(baselines_measured=True, min_window=2)

    class Rec:
        task = "mbpp"
        rebinds = 0
        miss_window = [0.9, 0.9]
        window_miss = 0.9

        def log(self, _m):
            pass

    c._note_baseline_sample(Rec())
    assert c._miss_seen == {}


def test_the_real_numbers_no_longer_fire() -> None:
    """把真机上那组数原样复现一遍。

        基线 0.30（1−覆盖率）× alarm_factor 3.0 = 告警线 0.90
        实测滑窗 miss 率 ≈ 0.95（真实值是估计值的 3–6 倍）
        旧代码：0.95 > 0.90 → 报警 → 换绑 → 一条对的绑定变成错的
    """
    observed = 0.95

    old = _coord(baselines_measured=True, alarm_factor=3.0)   # 旧行为
    assert observed > old.alarm_baseline("mbpp") * old.alarm_factor, \
        "这组数在旧代码下确实会报警 —— 前提没变，说明复现是对的"

    new = _coord(baselines_measured=False, alarm_factor=3.0)  # 新行为
    assert new.alarm_baseline("mbpp") is None, \
        "估计值仍在当告警线用 —— 15/20 会再错一次"


def test_a_calibrated_alarm_still_fires_on_a_real_anomaly() -> None:
    """修的是基线，不是把功能关掉。

    自校准出真实基线之后，一条**确实异常**的请求照样报得出来 —— 否则这就
    不是修复，是把通道二删了。
    """
    c = _coord(baselines_measured=False, calibrate_n=3, alarm_factor=3.0)
    c._miss_seen["mbpp"] = [0.30, 0.32, 0.31]

    base = c.alarm_baseline("mbpp")
    assert base == pytest.approx(0.31)
    assert 0.95 > base * c.alarm_factor, "绑错的请求仍然要能被抓出来"
    assert not 0.33 > base * c.alarm_factor, "绑对的请求不该被误报"


def test_the_control_path_flags_the_estimate_as_unmeasured() -> None:
    """真模型那条路必须把 `baselines_measured=False` 传下去。

    control.py 里那段注释早就写明估计值「偏低 3–6 倍」「会对绑对的池持续
    误报」—— 但没有人把这个认识接到代码上。这条测试就是那根接线。
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "p2pmoe/deploy/control.py").read_text(encoding="utf-8")
    assert "baselines_measured=measured" in src
    assert 'measured = setup.backend == "numpy"' in src

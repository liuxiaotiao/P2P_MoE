"""权重目录放在代码目录里时，`sync` 不能把它删掉。

`rsync --delete` 的语义是「目的端有、源端没有 → 删」。而权重恰恰是
**各节点自己拉的、源端根本没有** —— 不排除的话，一次 sync 抹掉 141GB，
而且没有任何提示。这条测试盯的就是那个排除项还在不在。

不依赖系统装没装 rsync：这里模拟它的 --delete 语义，测的是「排除项算得对不对」，
而排除项是脚本里那个 case 语句算出来的。
"""

from __future__ import annotations

import fnmatch
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "deploy_15.sh"

pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="没有 deploy_15.sh")


def weights_rel(workdir: str, weights: str) -> str:
    """跑脚本里那段 case，拿它真正算出来的排除项 —— 而不是在这儿重写一遍。"""
    src = SCRIPT.read_text(encoding="utf-8")
    m = re.search(r'WEIGHTS_REL=""\n(case .*?esac)', src, re.S)
    assert m, "脚本里找不到 WEIGHTS_REL 的推导 —— 排除逻辑被删了？"
    r = subprocess.run(
        ["bash", "-c", f'WORKDIR={workdir}; WEIGHTS={weights}\n'
                       f'WEIGHTS_REL=""\n{m.group(1)}\nprintf %s "$WEIGHTS_REL"'],
        capture_output=True, text=True, timeout=30)
    return r.stdout


def rsync_delete(src: Path, dst: Path, excludes: list[str]) -> list[str]:
    """模拟 `rsync -a --delete --exclude ...` 的删除那一半。"""
    def skip(rel: str) -> bool:
        return any(fnmatch.fnmatch(rel, e) or rel.startswith(e.rstrip("/") + "/")
                   for e in excludes)
    killed = []
    for p in sorted(dst.rglob("*"), key=lambda x: -len(x.parts)):
        rel = str(p.relative_to(dst))
        if skip(rel) or (src / rel).exists():
            continue
        killed.append(rel)
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    return killed


def _scene(tmp: Path) -> tuple[Path, Path]:
    src, dst = tmp / "src", tmp / "dst"
    (src / "p2pmoe").mkdir(parents=True)
    (src / "p2pmoe" / "a.py").write_text("new")
    (dst / "p2pmoe").mkdir(parents=True)
    (dst / "p2pmoe" / "gone.py").write_text("旧代码，该被删")
    (dst / "weights").mkdir()
    (dst / "weights" / "m.safetensors").write_bytes(b"x" * 4096)
    return src, dst


def test_weights_under_workdir_are_excluded() -> None:
    assert weights_rel("/home/u/P2P_MoE", "/home/u/P2P_MoE/weights") == "weights"


def test_weights_outside_workdir_need_no_exclusion() -> None:
    """不在代码目录里就不该凭空加排除项 —— 那会掩盖真正该同步的东西。"""
    assert weights_rel("/home/u/P2P_MoE", "/data/w") == ""


def test_a_nested_weights_dir_keeps_its_full_relative_path() -> None:
    assert weights_rel("/home/u/P", "/home/u/P/a/b/w") == "a/b/w"


def test_the_exclusion_actually_saves_the_weights(tmp_path) -> None:
    """**这条是这个文件存在的理由。**"""
    src, dst = _scene(tmp_path)
    rsync_delete(src, dst, [weights_rel("/w", "/w/weights")])
    assert (dst / "weights" / "m.safetensors").exists(), "权重被 --delete 干掉了"


def test_without_it_the_weights_are_gone(tmp_path) -> None:
    """反证：排除项一旦丢失，损失是静默的 —— 没有报错，只是 141GB 没了。"""
    src, dst = _scene(tmp_path)
    killed = rsync_delete(src, dst, [])
    assert not (dst / "weights").exists()
    assert "weights/m.safetensors" in killed


def test_stale_code_is_still_deleted(tmp_path) -> None:
    """别为了保权重把 --delete 也废掉 —— 旧代码留在节点上会跑出错误结果。"""
    src, dst = _scene(tmp_path)
    killed = rsync_delete(src, dst, [weights_rel("/w", "/w/weights")])
    assert "p2pmoe/gone.py" in killed


def test_the_script_passes_the_exclusion_to_both_rsync_and_tar() -> None:
    """sync 会删，bootstrap 会把 141GB 打进 tar 包 —— 两处都得挡。"""
    src = SCRIPT.read_text(encoding="utf-8")
    assert src.count('${WEIGHTS_REL:+--exclude "$WEIGHTS_REL"}') >= 2, (
        "rsync 与 tar 至少各要有一处排除")


def test_the_default_weights_dir_does_not_need_root() -> None:
    """默认值踩在 /data 上的话，第一条命令就会因为权限失败。"""
    src = SCRIPT.read_text(encoding="utf-8")
    m = re.search(r'^WEIGHTS=\$\{WEIGHTS:-(.*?)\}', src, re.M)
    assert m, "找不到 WEIGHTS 的默认值"
    assert not m.group(1).startswith("/data"), f"默认值 {m.group(1)} 需要 root"

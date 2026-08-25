"""依赖声明与实际 import 必须对得上，且重依赖不能越界。

分三层，每层的边界都值得守：

    requirements.txt        numpy          —— 控制机与 toy 模型路径
    requirements-node.txt   + torch/safetensors —— **只有跑真实模型的节点**
    requirements-dev.txt    + pytest       —— 只有开发机

最要紧的一条是**控制机不能碰 torch**：它跑规划、探测、下发清单，一个张量都不碰。
让它装 torch 意味着多几 GB 镜像和一堆 CUDA 依赖，白付。
`test_control_plane_never_imports_torch` 守着这条线。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

# import 的模块名 → PyPI 包名。多数一致，不一致的在这里登记。
ALIASES = {
    "yaml": "PyYAML",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
}

# 本仓库自己的模块 / 示例之间的相对 import
LOCAL = {"p2pmoe", "e2e", "examples", "tests"}


def _third_party_imports(files) -> dict[str, set[str]]:
    """扫出所有非标准库、非本地的顶层 import。"""
    std = set(sys.stdlib_module_names)
    out: dict[str, set[str]] = {}
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module.split(".")[0]]
            for m in mods:
                if m in std or m in LOCAL:
                    continue
                out.setdefault(m, set()).add(str(f.relative_to(ROOT)))
    return out


def _declared(path: Path) -> set[str]:
    """读 requirements 文件里声明的包名（小写，去掉版本约束）。"""
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            names |= _declared(path.parent / line[3:].strip())
            continue
        for sep in ("==", ">=", "<=", "~=", ">", "<", "["):
            if sep in line:
                line = line.split(sep)[0]
        names.add(line.strip().lower())
    return names


def _py_files(*dirs: str):
    for d in dirs:
        for f in (ROOT / d).rglob("*.py"):
            if "__pycache__" not in str(f):
                yield f


# --------------------------------------------------------------------------- #
def test_runtime_imports_are_declared() -> None:
    """p2pmoe/ 与 examples/ 用到的第三方包，requirements-node.txt 里必须有。

    用 node 那份（它 `-r` 了 requirements.txt），因为仓库里现在既有只要 numpy 的
    控制面，也有要 torch 的执行层。
    """
    used = _third_party_imports(_py_files("p2pmoe", "examples"))
    declared = _declared(ROOT / "requirements-node.txt")
    missing = {
        m: sorted(files) for m, files in used.items()
        if ALIASES.get(m, m).lower() not in declared
    }
    assert not missing, (
        "以下第三方包被 import 了但没写进 requirements.txt：\n"
        + "\n".join(f"  {m}  ←  {', '.join(f_)}" for m, f_ in missing.items())
    )


def test_test_imports_are_declared() -> None:
    used = _third_party_imports(_py_files("tests"))
    declared = _declared(ROOT / "requirements-dev.txt")
    missing = {
        m: sorted(files) for m, files in used.items()
        if ALIASES.get(m, m).lower() not in declared
    }
    assert not missing, (
        "测试用到但 requirements-dev.txt 没声明：\n"
        + "\n".join(f"  {m}  ←  {', '.join(f_)}" for m, f_ in missing.items())
    )


# 允许使用 torch/safetensors 的模块 —— 只有真实模型执行层
HEAVY_ALLOWED = {
    "p2pmoe/runtime/torch_model.py",
    "p2pmoe/runtime/weights.py",
    "p2pmoe/sim/fake_checkpoint.py",   # 生成测试用 checkpoint
}
HEAVY = {"torch", "transformers", "safetensors", "accelerate"}


def test_control_plane_never_imports_torch() -> None:
    """**最要紧的一条边界。**

    规划器（分档估算、公共带、回环裁剪）与部署层（agent、探测、控制器）
    一个张量都不碰 —— 它们不该 import torch。控制机因此只要 numpy，
    镜像小、装得快、没有 CUDA 依赖。

    真要打破它，先想清楚 15 台机器 + 控制机都得装 torch 值不值。
    """
    used = _third_party_imports(_py_files("p2pmoe/planner", "p2pmoe/deploy"))
    heavy = set(used) & HEAVY
    assert not heavy, (
        f"控制面 import 了 {sorted(heavy)}：\n"
        + "\n".join(f"  {m} ← {', '.join(sorted(used[m]))}" for m in heavy)
    )


def test_heavy_deps_stay_in_the_execution_layer() -> None:
    """torch / safetensors 只允许出现在真实模型执行层，别处不行。

    toy 模型路径（runtime/model.py、corpus.py、node.py、coordinator.py）必须
    保持纯 numpy —— 「只验证部署路径」的场景不该被迫装 torch。
    """
    used = _third_party_imports(_py_files("p2pmoe", "examples"))
    for m in HEAVY:
        for f in used.get(m, ()):
            assert f.replace("\\", "/") in HEAVY_ALLOWED, (
                f"{f} import 了 {m}，但它不在执行层白名单里。"
                f"要么把重依赖挪进执行层，要么显式扩白名单并更新 DEPLOY.md"
            )


def test_numpy_is_still_the_only_hard_dependency() -> None:
    """requirements.txt（控制机那份）必须只有 numpy。"""
    assert _declared(ROOT / "requirements.txt") == {"numpy"}


def test_declared_packages_are_actually_used() -> None:
    """反向检查：声明了却没人用的包应该删掉。"""
    used = {ALIASES.get(m, m).lower()
            for m in _third_party_imports(_py_files("p2pmoe", "examples", "tests"))}
    declared = _declared(ROOT / "requirements-dev.txt")
    unused = declared - used
    assert not unused, f"requirements 里声明了但没用到: {sorted(unused)}"

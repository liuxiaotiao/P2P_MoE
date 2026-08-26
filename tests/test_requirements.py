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
LOCAL = {"p2pmoe", "examples", "tests"}
"""包名意义上的本地模块。同目录的兄弟模块另有规则 —— 见 `_third_party_imports`。"""


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
                # 同目录的兄弟模块是本地的，不是依赖 —— examples 里 serving.py
                # 从 e2e.py 借池子构造，tests 里也互相借 fake。按「文件是否
                # 就在旁边」判定，比维护一张名字白名单可靠。
                if m in std or m in LOCAL or (f.parent / f"{m}.py").exists():
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

    依赖按机器角色分三份（见 requirements.txt 的注释），这里查**并集** ——
    「这个包有没有人声明过」。谁该装谁不该装是另外两条测试的事：
    `test_control_plane_never_imports_torch`（控制机不碰张量）与
    `test_node_never_needs_a_tokenizer`（节点不碰文本）。
    """
    used = _third_party_imports(_py_files("p2pmoe", "examples"))
    declared = (_declared(ROOT / "requirements-node.txt")
                | _declared(ROOT / "requirements-control.txt"))
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
    "p2pmoe/runtime/torch_model.py",       # Qwen3-MoE 执行层
    "p2pmoe/runtime/qwen3_next.py",        # Qwen3-Next 执行层（混合注意力）
    "p2pmoe/runtime/weights.py",
    "p2pmoe/sim/fake_checkpoint.py",       # 生成测试用 checkpoint
    "examples/drop_expert_impact.py",      # 离线评估脚本，不在部署路径上
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
    """requirements.txt（两边共用那份）必须只有 numpy。"""
    assert _declared(ROOT / "requirements.txt") == {"numpy"}


TEXT_DEPS = {"tokenizers", "jinja2", "transformers"}
# 允许 import 文本层依赖的模块 —— 控制机侧，外加造测试数据的
TEXT_ALLOWED = {
    "p2pmoe/runtime/text.py",
    "p2pmoe/sim/fake_checkpoint.py",     # 造一份微型 tokenizer.json
}


def test_node_never_needs_a_tokenizer() -> None:
    """**与 torch 那条对称的另一半边界。**

    节点自始至终只见 token id：head(f) 拿 id 查嵌入表、tail(b) 采样出 id，
    环里传的是 hidden state，`stop_ids` 也只是几个整数。文本在控制机的入口
    编码、出口解码。

    所以推理节点不装 tokenizers，控制机不装 torch —— 两份依赖互不包含。
    真要打破它，先想清楚 15 台节点上多装一套分词器换来了什么。
    """
    used = _third_party_imports(_py_files("p2pmoe", "examples"))
    for m in TEXT_DEPS:
        for f in used.get(m, ()):
            assert f.replace("\\", "/") in TEXT_ALLOWED, (
                f"{f} import 了 {m}。文本层只该出现在控制机侧（runtime/text.py）；"
                f"数据面拿到的应该只有整数"
            )


def test_the_two_side_dependencies_do_not_overlap() -> None:
    """控制机与节点的额外依赖没有交集 —— 分层不是文档口号，是可查的事实。"""
    base = _declared(ROOT / "requirements.txt")
    control = _declared(ROOT / "requirements-control.txt") - base
    node = _declared(ROOT / "requirements-node.txt") - base
    assert control and node
    assert not (control & node), f"两侧都要装 {sorted(control & node)} —— 分层没分干净"


def test_declared_packages_are_actually_used() -> None:
    """反向检查：声明了却没人用的包应该删掉。"""
    used = {ALIASES.get(m, m).lower()
            for m in _third_party_imports(_py_files("p2pmoe", "examples", "tests"))}
    declared = _declared(ROOT / "requirements-dev.txt")
    unused = declared - used
    assert not unused, f"requirements 里声明了但没用到: {sorted(unused)}"


# --------------------------------------------------------------------------- #
# 只允许 shell 出去的模块 —— 批量运维脚本，且它是可选的
SHELL_ALLOWED = {"p2pmoe/deploy/launch.py"}
SHELL_MARKS = ("subprocess", "paramiko", "pexpect", "fabric")


def test_the_data_plane_never_shells_out() -> None:
    """**节点之间只用 TCP，不用 ssh，也不起任何子进程。**

    这条边界值得钉死，因为它决定了部署门槛：节点之间只要开一个端口就能跑，
    不需要互相配免密、不需要装 ssh 客户端、不需要共享账号。分散环境里那些机器
    往往属于不同的人，"两两配 ssh" 是不现实的。

    `ssh` 只出现在 `deploy/launch.py`（控制机批量起停/拉权重的便利脚本），
    而且它是可选的 —— `--dry-run` 把命令打出来自己跑，或者走 systemd。
    """
    used = _third_party_imports(_py_files("p2pmoe"))
    std = set(sys.stdlib_module_names)
    for f in _py_files("p2pmoe"):
        rel = str(f.relative_to(ROOT)).replace("\\", "/")
        if rel in SHELL_ALLOWED:
            continue
        text = f.read_text(encoding="utf-8")
        for mark in SHELL_MARKS:
            assert f"import {mark}" not in text, (
                f"{rel} import 了 {mark}。数据面与节点侧不该起子进程 —— "
                f"节点之间只用 TCP。批量运维放到 deploy/launch.py"
            )
        assert '"ssh' not in text and "'ssh" not in text and " ssh " not in text, (
            f"{rel} 里出现了 ssh。节点之间不该需要 ssh —— "
            f"那会让部署门槛从「开一个端口」变成「两两配免密」"
        )
    assert not (set(used) & {"paramiko", "fabric"}), "别引 ssh 库"

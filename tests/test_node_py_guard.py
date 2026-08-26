"""开跑之前先确认 NODE_PY 指对了。

`NODE_PY` 默认是 `python3`，而 torch 装在 conda 环境里时那是**系统 python** ——
看不见环境里的包。这个错要到「已经 ssh 过去、已经开始跑」才暴露，而那时的报错
（`No module named 'numpy'`）看着像代码没同步，不像解释器选错。

两件事的修法完全相反：一个要 `sync`，一个要改 `NODE_PY`。所以脚本必须分得清。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "deploy_15.sh"

pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="没有 deploy_15.sh")


def _run(fake: str, node_py: str = "python3", action: str = "fetch") -> str:
    """把 ssh 换成一个可控的桩，跑一遍脚本。"""
    stub = ROOT / ".test-ssh-stub.sh"
    stub.write_text(
        "#!/bin/bash\n"
        "while [ $# -gt 1 ]; do shift; done\n"
        'case "$FAKE" in\n'
        '  sysver) echo "ModuleNotFoundError: No module named \'numpy\'"; exit 1 ;;\n'
        '  nocode) echo "ModuleNotFoundError: No module named \'p2pmoe\'"; exit 1 ;;\n'
        '  ok)     echo "ok 1.26.4 2.5.1"; exit 0 ;;\n'
        "esac\n", encoding="utf-8")
    stub.chmod(0o755)
    patched = ROOT / ".test-deploy.sh"
    src = SCRIPT.read_text(encoding="utf-8")
    patched.write_text(
        src.replace("out=$(ssh -o BatchMode=yes -o ConnectTimeout=10",
                    f"out=$({stub} -o BatchMode=yes -o ConnectTimeout=10"),
        encoding="utf-8")
    hosts = ROOT / ".test-hosts.txt"
    hosts.write_text("N01  10.0.0.1:9101\n", encoding="utf-8")
    try:
        r = subprocess.run(
            ["bash", "-c", f"echo n | bash {patched} {action}"],
            capture_output=True, text=True, timeout=120, cwd=ROOT,
            env={"PATH": "/usr/bin:/bin", "FAKE": fake, "NODE_PY": node_py,
                 "HOSTS": str(hosts), "PLAN": str(ROOT / "task/plan_deploy.json"),
                 "WORKDIR": "/home/ubuntu/P2P_MoE", "WEIGHTS": "/w",
                 "CONDA_ENV": "moe", "ADVERTISE": "1.2.3.4",
                 "PROFILE": str(ROOT / "task/plan_deploy.json")})
        return re.sub(r"\x1b\[[0-9;]*m", "", r.stdout + r.stderr)
    finally:
        for f in (stub, patched, hosts):
            f.unlink(missing_ok=True)


def test_a_system_python_is_caught_before_downloading(  ) -> None:
    """**这条是重点。** 141GB 开始之前就要拦住。"""
    out = _run("sysver")
    assert "NODE_PY 用不了" in out
    assert "系统 python" in out
    assert "conda" in out
    assert "3b" not in out, "不该走到真正下载那一步"


def test_the_hint_names_the_env_var_and_a_concrete_path() -> None:
    """光说「配错了」没用 —— 要给出可以直接粘的那一行。"""
    out = _run("sysver")
    assert "export NODE_PY=" in out
    assert "/envs/moe/bin/python" in out


def test_a_working_interpreter_passes_and_reports_versions() -> None:
    """通过时也要说清楚验的是什么、在哪台验的 —— 否则下次还会怀疑它。"""
    out = _run("ok")
    assert "NODE_PY ✓" in out
    assert "在 N01 上验的" in out
    assert "3a" in out, "该继续走下去"


def test_start_is_guarded_too() -> None:
    """起 agent 同样会踩 —— 而且那里的症状是「起来就死」，更难查。"""
    out = _run("sysver", action="start")
    assert "NODE_PY 用不了" in out


def test_missing_p2pmoe_and_missing_numpy_are_different_diagnoses() -> None:
    """`No module named 'p2pmoe'` = 代码不在，要 sync；
    `No module named 'numpy'` = 解释器不对，要改 NODE_PY。
    混成一句会把人送去错的方向。"""
    src = SCRIPT.read_text(encoding="utf-8")
    assert "No module named 'p2pmoe'" in src, "没有单独识别 p2pmoe 缺失"
    i_specific = src.index("No module named 'p2pmoe'")
    i_generic = src.index('*"No module named"*)', i_specific)
    assert i_specific < i_generic, "通配的分支必须排在具体分支后面，否则永远轮不到"

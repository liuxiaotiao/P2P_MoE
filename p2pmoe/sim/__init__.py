from .network import SimNetwork, AccessProfile
from .replay import make_activation_profiles
from .scenario import appendix_c, Stage, APPENDIX_C_MODEL, APPENDIX_C_TASKS, UNION_EXPERTS, P_CURVE

try:
    from .fake_checkpoint import TINY_QWEN3_MOE, write_fake_checkpoint
    _HAS_CKPT = True
except ImportError:  # pragma: no cover — 需要 safetensors
    _HAS_CKPT = False

__all__ = [
    "SimNetwork", "AccessProfile",
    "appendix_c", "Stage", "make_activation_profiles",
    "APPENDIX_C_MODEL", "APPENDIX_C_TASKS", "UNION_EXPERTS", "P_CURVE",
]
if _HAS_CKPT:
    __all__ += ["TINY_QWEN3_MOE", "write_fake_checkpoint"]

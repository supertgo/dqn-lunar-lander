from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC_DQN_DIR = Path(__file__).resolve().parents[1] / "src" / "dqn"
SRC_DQN_FILES = sorted(SRC_DQN_DIR.rglob("*.py"))

FORBIDDEN_ROOT_MODULES: frozenset[str] = frozenset(
    {
        "stable_baselines3",
        "sb3_contrib",
        "stable_baselines",
        "rl_zoo3",
        "keras_rl",
        "rl",  # pacote raiz do keras-rl / keras-rl2
        "ray",  # RLlib (ray.rllib)
        "rllib",
        "tianshou",
        "cleanrl",
        "torchrl",
        "dopamine",
        "tf_agents",
        "acme",
        "run_sb3",  
    }
)

def _is_forbidden_module(module_name: str) -> bool:
    parts = module_name.split(".")
    return parts[0] in FORBIDDEN_ROOT_MODULES or "run_sb3" in parts


def _find_forbidden_imports(file_path: Path) -> list[tuple[int, str]]:
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))
    violations: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden_module(alias.name):
                    violations.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module and _is_forbidden_module(node.module):
                violations.append((node.lineno, node.module))
            else:
                for alias in node.names:
                    full_name = f"{node.module}.{alias.name}" if node.module else alias.name
                    if _is_forbidden_module(full_name):
                        violations.append((node.lineno, full_name))

        elif isinstance(node, ast.Call) and node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                func = node.func
                is_dynamic_import = (
                    (isinstance(func, ast.Name) and func.id in {"__import__", "import_module"})
                    or (isinstance(func, ast.Attribute) and func.attr == "import_module")
                )
                if is_dynamic_import and _is_forbidden_module(first_arg.value):
                    violations.append((node.lineno, first_arg.value))

    return violations


def test_src_dqn_directory_contains_python_files() -> None:
    assert SRC_DQN_DIR.is_dir(), f"Diretório não encontrado: {SRC_DQN_DIR}"
    assert len(SRC_DQN_FILES) > 0, f"Nenhum arquivo .py encontrado em {SRC_DQN_DIR}"

@pytest.mark.parametrize(
    "py_file",
    SRC_DQN_FILES,
    ids=[str(p.relative_to(SRC_DQN_DIR)) for p in SRC_DQN_FILES],
)
def test_src_dqn_file_has_no_forbidden_rl_imports(py_file: Path) -> None:
    violations = _find_forbidden_imports(py_file)

    assert not violations, (
        f"Imports proibidos encontrados em {py_file.name}: "
        + ", ".join(f"linha {lineno} ('{mod}')" for lineno, mod in violations)
    )


@pytest.mark.parametrize(
    ("code", "expected_module"),
    [
        ("import stable_baselines3\n", "stable_baselines3"),
        ("from stable_baselines3 import DQN\n", "stable_baselines3"),
        ("from ray.rllib.algorithms.dqn import DQN\n", "ray.rllib.algorithms.dqn"),
        ("from rl.agents.dqn import DQNAgent\n", "rl.agents.dqn"),
        ("from scripts import run_sb3\n", "scripts"),  # ignorado se não houver run_sb3 no module
        ("from scripts import run_sb3\n", "scripts.run_sb3"),
        ('import importlib\nimportlib.import_module("stable_baselines3")\n', "stable_baselines3"),
        ('__import__("stable_baselines3")\n', "stable_baselines3"),
    ],
)
def test_forbidden_import_detector_catches_violations(tmp_path: Path, code: str, expected_module: str) -> None:
    sample_file = tmp_path / "sample.py"
    sample_file.write_text(code, encoding="utf-8")

    violations = _find_forbidden_imports(sample_file)

    assert len(violations) == 1
    assert violations[0][1] == expected_module

def test_forbidden_import_detector_allows_permitted_libraries(tmp_path: Path) -> None:
    sample_file = tmp_path / "clean.py"
    sample_file.write_text(
        "import gymnasium as gym\n"
        "import numpy as np\n"
        "import torch\n"
        "from torch import nn\n"
        "from dqn.config import EvalConfig\n",
        encoding="utf-8",
    )

    violations = _find_forbidden_imports(sample_file)

    assert violations == []

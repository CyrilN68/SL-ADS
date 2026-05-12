"""Packaging contract for the ``sl-ads`` console entry point.

The CLI exposed in ``pyproject.toml`` must point to an importable module in an
installed wheel/editable install.  This catches the common mismatch where a
console script targets a top-level module that setuptools never packages.
"""
from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _pyproject() -> dict:
    return tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_sl_ads_console_script_target_is_packaged():
    data = _pyproject()
    target = data["project"]["scripts"]["sl-ads"]
    module_name, _, attr_name = target.partition(":")
    assert module_name and attr_name

    setuptools_cfg = data.get("tool", {}).get("setuptools", {})
    py_modules = set(setuptools_cfg.get("py-modules", []))
    assert module_name.startswith("sl_ads.") or module_name in py_modules, (
        f"Console script target {target!r} is not covered by package discovery "
        "or [tool.setuptools].py-modules."
    )


def test_sl_ads_console_script_target_imports():
    target = _pyproject()["project"]["scripts"]["sl-ads"]
    module_name, _, attr_name = target.partition(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attr_name))

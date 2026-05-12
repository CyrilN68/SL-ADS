"""Pytest bootstrap — Phase H reorganisation.

Pytest auto-loads any ``conftest.py`` it finds in the test rootdir or
its ancestors.  Placing this at the project root ensures the new
``src/sl_ads/`` package layout is importable by tests *without*
requiring the user to ``pip install -e .`` first.

Once Phase 7 closes (shims removed, callers migrated), the project can
optionally be installed in editable mode and this file can be deleted.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Ensure the project root is on sys.path for legacy flat imports
# (``import config``, ``import paths``, …) used by tests that pre-date
# the Phase H reorganisation.  Pytest adds it automatically in most
# launch modes, but we make it explicit for safety.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

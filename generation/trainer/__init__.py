"""Compatibility alias for older imports.

New code should import from :mod:`omnivae_generation.trainer`.
"""

from __future__ import annotations

import importlib
import sys

_pkg = importlib.import_module("omnivae_generation.trainer")
sys.modules[__name__] = _pkg

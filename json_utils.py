"""Helpers for writing reports that may contain NumPy scalar values."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def json_safe_default(value: Any):
    """Convert common non-stdlib values into JSON-serializable Python values.

    NumPy scalars (for example numpy.int64 returned by seqeval) are converted
    to native Python scalars. Arrays become nested lists. Unsupported objects
    still raise TypeError rather than being silently stringified.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )

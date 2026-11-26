from __future__ import annotations

import time
from contextlib import contextmanager

from seg3d.utils.memory import peak_memory_mb, reset_peak_memory


@contextmanager
def stage(name: str, record: dict):
    """Time the code inside the `with` block and store the result under `record[name]`."""
    reset_peak_memory()
    start = time.perf_counter()
    try:
        yield
    finally:
        # `finally` so a failed stage still records how long it ran before failing.
        record[name] = {
            "wall_s": round(time.perf_counter() - start, 4),
            "peak_gpu_mb": peak_memory_mb(),
        }

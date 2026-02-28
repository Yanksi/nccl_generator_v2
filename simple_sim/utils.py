from __future__ import annotations
from typing import Iterable, Tuple

DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
}

def numel(shape: Tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n

def bytes_of(shape: Tuple[int, ...], dtype: str) -> int:
    return numel(shape) * DTYPE_BYTES.get(dtype, 2)
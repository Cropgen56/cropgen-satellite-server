"""
In-process TTL + LRU cache with single-flight.

Single-flight matters here: a frontend that loads seven layer endpoints in
parallel for the same field would otherwise run the DEM fetch and hydrology
seven times at once. With single-flight the first request computes, the other
six wait for it and reuse the result.

This cache lives in one process. Under several uvicorn workers each worker has
its own copy; move it to Redis or disk if that duplication starts to cost.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable, Generic, Hashable, TypeVar

T = TypeVar("T")


class SingleFlightCache(Generic[T]):
    def __init__(self, max_items: int, ttl_s: float):
        self._max = max(1, int(max_items))
        self._ttl = float(ttl_s)
        self._data: "OrderedDict[Hashable, tuple[float, T]]" = OrderedDict()
        self._inflight: dict[Hashable, threading.Event] = {}
        self._lock = threading.Lock()

    def _fresh(self, key: Hashable) -> tuple[bool, T | None]:
        hit = self._data.get(key)
        if hit is None:
            return False, None
        stamp, value = hit
        if time.monotonic() - stamp > self._ttl:
            del self._data[key]
            return False, None
        self._data.move_to_end(key)
        return True, value

    def get_or_compute(self, key: Hashable, fn: Callable[[], T]) -> T:
        while True:
            with self._lock:
                ok, value = self._fresh(key)
                if ok:
                    return value  # type: ignore[return-value]
                event = self._inflight.get(key)
                owner = event is None
                if owner:
                    event = threading.Event()
                    self._inflight[key] = event

            if not owner:
                # Someone else is computing this key. Wait, then loop: either the
                # value is there now, or the owner failed and we take over.
                event.wait()
                continue

            try:
                value = fn()
                with self._lock:
                    self._data[key] = (time.monotonic(), value)
                    self._data.move_to_end(key)
                    while len(self._data) > self._max:
                        self._data.popitem(last=False)
                return value
            finally:
                with self._lock:
                    self._inflight.pop(key, None)
                event.set()

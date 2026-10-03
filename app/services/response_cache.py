"""Small thread-safe in-memory cache with a TTL and a size bound.

Used for per-process API response caching. Entries expire after `ttl_seconds`,
and once `max_entries` is reached the least recently used entry is evicted, so
memory stays bounded no matter how many distinct requests a worker serves.
"""
import threading
import time
from collections import OrderedDict
from typing import Any, Hashable, Optional


class TTLCache:
    def __init__(self, ttl_seconds: float, max_entries: int):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._data: "OrderedDict[Hashable, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Optional[Any]:
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return None
            timestamp, value = hit
            if time.time() - timestamp > self.ttl_seconds:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: Hashable, value: Any) -> Any:
        """Store `value` and return it, so callers can `return cache.set(k, v)`."""
        now = time.time()
        with self._lock:
            self._data[key] = (now, value)
            self._data.move_to_end(key)
            if len(self._data) > self.max_entries:
                expired = [k for k, (ts, _) in self._data.items() if now - ts > self.ttl_seconds]
                for k in expired:
                    del self._data[k]
                while len(self._data) > self.max_entries:
                    self._data.popitem(last=False)
        return value

    def __len__(self) -> int:
        return len(self._data)

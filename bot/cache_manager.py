import time
import logging
from typing import Any, Optional, Dict

logger = logging.getLogger("cache_mgr")


class CacheEntry:
    """A cached value with TTL and refresh logic."""
    def __init__(self, value: Any, ttl_seconds: float):
        self.value = value
        self.ttl_seconds = ttl_seconds
        self.created_at = time.time()

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds

    def refresh(self, new_value: Any):
        self.value = new_value
        self.created_at = time.time()


class CacheManager:
    """Simple in-memory cache with TTL for frequently accessed data.
    
    Reduces redundant API calls for:
    - ATR values (expensive pandas + kline fetch)
    - Positions (N+1 query pattern)
    - Settings (string parsing overhead)
    """
    
    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Return cached value if not expired, else None."""
        if key not in self._cache:
            return None
        entry = self._cache[key]
        if entry.is_expired():
            del self._cache[key]
            return None
        return entry.value
    
    def set(self, key: str, value: Any, ttl_seconds: float = 60.0):
        """Cache a value with a TTL in seconds."""
        self._cache[key] = CacheEntry(value, ttl_seconds)
    
    def invalidate(self, key: str):
        """Force expiration of a cache entry."""
        if key in self._cache:
            del self._cache[key]
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching a pattern (e.g., 'atr_*')."""
        to_delete = [k for k in self._cache.keys() if pattern in k]
        for k in to_delete:
            del self._cache[k]
    
    def clear(self):
        """Clear entire cache."""
        self._cache.clear()


# Singleton instance
_cache = CacheManager()


def get_cache() -> CacheManager:
    """Return the global cache manager."""
    return _cache

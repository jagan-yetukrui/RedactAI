"""
Advanced caching system for RedactAI.

This module provides intelligent caching capabilities for processed results,
model outputs, and frequently accessed data to improve performance.
"""

import hashlib
import json
import pickle
import time
import threading
from typing import Any, Optional, Dict, List, Callable, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def touch(self) -> None:
        """Update access information."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def get_age_seconds(self) -> float:
        """Get age of the entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    def get_ttl_seconds(self) -> Optional[float]:
        """Get time to live in seconds."""
        if self.expires_at is None:
            return None
        return (self.expires_at - datetime.now()).total_seconds()


class MemoryCache:
    """In-memory cache with LRU eviction and TTL support."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100, 
                 default_ttl: Optional[int] = None):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default time to live in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
        self.current_memory = 0
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                time.sleep(60)  # Run every minute
                self._cleanup_expired()
                self._cleanup_lru()
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items() 
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                logger.debug(f"Removed {len(expired_keys)} expired cache entries")
    
    def _cleanup_lru(self) -> None:
        """Remove least recently used entries if cache is full."""
        with self.lock:
            # Check size limits
            if len(self.cache) <= self.max_size and self.current_memory <= self.max_memory_bytes:
                return
            
            # Sort by last accessed time (oldest first)
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Remove oldest entries until we're under limits
            removed_count = 0
            for key, _ in sorted_entries:
                if len(self.cache) <= self.max_size and self.current_memory <= self.max_memory_bytes:
                    break
                
                self._remove_entry(key)
                removed_count += 1
            
            if removed_count > 0:
                logger.debug(f"Removed {removed_count} LRU cache entries")
    
    def _remove_entry(self, key: str) -> None:
        """Remove an entry from cache."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_memory -= entry.size_bytes
            del self.cache[key]
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of a value in bytes."""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float, bool)):
                return 8
            else:
                # Use pickle to estimate size
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default estimate
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            if entry.is_expired():
                self._remove_entry(key)
                return None
            
            entry.touch()
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache."""
        with self.lock:
            # Remove existing entry if it exists
            if key in self.cache:
                self._remove_entry(key)
            
            # Calculate TTL
            expires_at = None
            if ttl is not None:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            elif self.default_ttl is not None:
                expires_at = datetime.now() + timedelta(seconds=self.default_ttl)
            
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at,
                size_bytes=size_bytes
            )
            
            # Add to cache
            self.cache[key] = entry
            self.current_memory += size_bytes
            
            # Trigger cleanup if needed
            if len(self.cache) > self.max_size or self.current_memory > self.max_memory_bytes:
                self._cleanup_lru()
    
    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        with self.lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.current_memory = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_entries = len(self.cache)
            total_memory = self.current_memory
            
            # Calculate hit rate (simplified)
            total_accesses = sum(entry.access_count for entry in self.cache.values())
            hit_rate = total_accesses / max(total_entries, 1)
            
            # Age statistics
            ages = [entry.get_age_seconds() for entry in self.cache.values()]
            avg_age = sum(ages) / max(len(ages), 1)
            
            return {
                'total_entries': total_entries,
                'max_entries': self.max_size,
                'total_memory_bytes': total_memory,
                'max_memory_bytes': self.max_memory_bytes,
                'memory_usage_percent': (total_memory / self.max_memory_bytes) * 100,
                'hit_rate': hit_rate,
                'average_age_seconds': avg_age
            }


class FileCache:
    """File-based cache for persistent storage."""
    
    def __init__(self, cache_dir: Path, max_size_mb: int = 1000):
        """
        Initialize file cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.lock = threading.RLock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        # Use hash to avoid filesystem issues with special characters
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.cache"
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                self._cleanup_old_files()
            except Exception as e:
                logger.error(f"Error in file cache cleanup: {e}")
    
    def _cleanup_old_files(self) -> None:
        """Remove old cache files."""
        with self.lock:
            cache_files = list(self.cache_dir.glob("*.cache"))
            
            if not cache_files:
                return
            
            # Sort by modification time (oldest first)
            cache_files.sort(key=lambda x: x.stat().st_mtime)
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in cache_files)
            
            # Remove oldest files if over size limit
            removed_count = 0
            for cache_file in cache_files:
                if total_size <= self.max_size_bytes:
                    break
                
                total_size -= cache_file.stat().st_size
                cache_file.unlink()
                removed_count += 1
            
            if removed_count > 0:
                logger.debug(f"Removed {removed_count} old cache files")
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from file cache."""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            # Check if expired
            if 'expires_at' in data and datetime.now() > data['expires_at']:
                cache_path.unlink()
                return None
            
            return data['value']
        
        except Exception as e:
            logger.error(f"Error reading cache file {cache_path}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in file cache."""
        cache_path = self._get_cache_path(key)
        
        try:
            data = {
                'value': value,
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(seconds=ttl) if ttl else None
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        
        except Exception as e:
            logger.error(f"Error writing cache file {cache_path}: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete a value from file cache."""
        cache_path = self._get_cache_path(key)
        
        if cache_path.exists():
            cache_path.unlink()
            return True
        
        return False
    
    def clear(self) -> None:
        """Clear all cache files."""
        with self.lock:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get file cache statistics."""
        with self.lock:
            cache_files = list(self.cache_dir.glob("*.cache"))
            total_files = len(cache_files)
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                'total_files': total_files,
                'total_size_bytes': total_size,
                'max_size_bytes': self.max_size_bytes,
                'size_usage_percent': (total_size / self.max_size_bytes) * 100,
                'cache_dir': str(self.cache_dir)
            }


class CacheManager:
    """Unified cache manager with multiple cache backends."""
    
    def __init__(self, memory_cache_size: int = 1000, memory_cache_mb: int = 100,
                 file_cache_dir: Optional[Path] = None, file_cache_mb: int = 1000):
        """
        Initialize cache manager.
        
        Args:
            memory_cache_size: Maximum number of entries in memory cache
            memory_cache_mb: Maximum memory usage for memory cache
            file_cache_dir: Directory for file cache
            file_cache_mb: Maximum size for file cache
        """
        self.memory_cache = MemoryCache(memory_cache_size, memory_cache_mb)
        self.file_cache = FileCache(file_cache_dir or Path("data/cache"), file_cache_mb)
        self.lock = threading.RLock()
    
    def get(self, key: str, use_file_cache: bool = True) -> Optional[Any]:
        """Get a value from cache."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try file cache if enabled
        if use_file_cache:
            value = self.file_cache.get(key)
            if value is not None:
                # Store in memory cache for faster access
                self.memory_cache.set(key, value)
                return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            use_file_cache: bool = True) -> None:
        """Set a value in cache."""
        # Store in memory cache
        self.memory_cache.set(key, value, ttl)
        
        # Store in file cache if enabled
        if use_file_cache:
            self.file_cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        memory_deleted = self.memory_cache.delete(key)
        file_deleted = self.file_cache.delete(key)
        return memory_deleted or file_deleted
    
    def clear(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        self.file_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'memory_cache': self.memory_cache.get_stats(),
            'file_cache': self.file_cache.get_stats()
        }


def cached(ttl: Optional[int] = None, use_file_cache: bool = True, 
           key_func: Optional[Callable] = None):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        use_file_cache: Whether to use file cache
        key_func: Function to generate cache key from arguments
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = {
                    'func': func.__name__,
                    'args': args,
                    'kwargs': sorted(kwargs.items())
                }
                cache_key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
            
            # Try to get from cache
            cache_manager = get_cache_manager()
            result = cache_manager.get(cache_key, use_file_cache)
            
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl, use_file_cache)
            
            return result
        
        return wrapper
    return decorator


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def clear_cache() -> None:
    """Clear all caches."""
    get_cache_manager().clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return get_cache_manager().get_stats()

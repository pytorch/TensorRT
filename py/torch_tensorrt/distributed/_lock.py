"""
File-based distributed lock for coordinating work across ranks.

Used to ensure only one rank performs an expensive operation (e.g., TRT engine
build) while others wait and consume the result from a shared cache.

The lock file is created atomically via os.O_CREAT | os.O_EXCL — only one
process can succeed.  Other processes see FileExistsError and know to wait.

If the lock holder crashes, the lock file becomes stale.  A configurable
timeout detects this: if the lock file's modification time is older than
the timeout, it is treated as stale and forcibly removed so another rank
can proceed.
"""

import logging
import os
import time

import torch.distributed as dist

logger = logging.getLogger(__name__)

# Default timeout for stale lock detection (seconds).
# TRT engine builds can take several minutes for large models.
_DEFAULT_STALE_TIMEOUT_S = 600  # 10 minutes


class DistributedFileLock:
    """Atomic file lock for coordinating distributed engine builds.

    Args:
        lock_dir: Directory to create lock files in (typically the cache dir).
        name: Unique name for this lock (typically the engine hash).
        suffix: File suffix for the lock file.
        stale_timeout_s: Seconds after which a lock file is considered stale
            (holder likely crashed). Set to 0 to disable stale detection.
    """

    def __init__(
        self,
        lock_dir: str,
        name: str,
        suffix: str = ".building",
        stale_timeout_s: float = _DEFAULT_STALE_TIMEOUT_S,
    ) -> None:
        self._lock_path = os.path.join(lock_dir, f".{name}{suffix}")
        self._acquired = False
        self._stale_timeout_s = stale_timeout_s

    @property
    def acquired(self) -> bool:
        """Whether this instance holds the lock."""
        return self._acquired

    @property
    def lock_path(self) -> str:
        return self._lock_path

    def _is_stale(self) -> bool:
        """Check if an existing lock file is stale (holder likely crashed).

        A lock is stale if its modification time is older than stale_timeout_s.
        """
        if self._stale_timeout_s <= 0:
            return False
        try:
            mtime = os.path.getmtime(self._lock_path)
            age = time.time() - mtime
            if age > self._stale_timeout_s:
                logger.warning(
                    f"Stale build lock detected: {self._lock_path} "
                    f"(age={age:.0f}s > timeout={self._stale_timeout_s:.0f}s). "
                    f"Lock holder likely crashed. Removing stale lock."
                )
                return True
        except FileNotFoundError:
            pass
        return False

    def _remove_stale(self) -> None:
        """Remove a stale lock file so acquire() can succeed."""
        try:
            os.remove(self._lock_path)
            logger.debug(f"Removed stale lock: {self._lock_path}")
        except FileNotFoundError:
            pass

    def acquire(self) -> bool:
        """Try to acquire the lock atomically.

        If the lock file exists but is stale (older than stale_timeout_s),
        it is removed and acquisition is retried once.

        Returns:
            True if this process acquired the lock (should do the work).
            False if another process already holds it (should wait).
        """
        try:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            self._acquired = True
            logger.debug(f"Acquired build lock: {self._lock_path}")
            return True
        except FileExistsError:
            pass

        if self._is_stale():
            self._remove_stale()
            try:
                fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                self._acquired = True
                logger.debug(
                    f"Acquired build lock after stale removal: {self._lock_path}"
                )
                return True
            except FileExistsError:
                pass

        self._acquired = False
        logger.debug(f"Build lock already held: {self._lock_path}")
        return False

    def release(self) -> None:
        """Release the lock by removing the lock file."""
        if self._acquired:
            try:
                os.remove(self._lock_path)
                logger.debug(f"Released build lock: {self._lock_path}")
            except FileNotFoundError:
                pass
            self._acquired = False

    @staticmethod
    def cleanup_stale_locks(
        lock_dir: str,
        suffix: str = ".building",
        stale_timeout_s: float = _DEFAULT_STALE_TIMEOUT_S,
    ) -> int:
        """Remove all stale lock files in a directory.

        Useful for cleaning up after crashes before starting a new run.

        Args:
            lock_dir: Directory to scan for lock files.
            suffix: Lock file suffix to match.
            stale_timeout_s: Age threshold in seconds.

        Returns:
            Number of stale lock files removed.
        """
        removed = 0
        if not os.path.isdir(lock_dir):
            return 0
        now = time.time()
        for entry in os.scandir(lock_dir):
            if (
                entry.name.startswith(".")
                and entry.name.endswith(suffix)
                and entry.is_file()
            ):
                try:
                    age = now - entry.stat().st_mtime
                    if age > stale_timeout_s:
                        os.remove(entry.path)
                        logger.debug(
                            f"Cleaned up stale lock: {entry.path} (age={age:.0f}s)"
                        )
                        removed += 1
                except (FileNotFoundError, OSError):
                    pass
        if removed > 0:
            logger.info(f"Cleaned up {removed} stale lock file(s) in {lock_dir}")
        return removed

    @staticmethod
    def barrier() -> None:
        """Distributed barrier — all ranks must call this."""
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

    def __enter__(self) -> "DistributedFileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        self.release()
        return None

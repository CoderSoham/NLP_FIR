"""Fixed-window rate limiting and optional API-key auth.

No dependency: Flask-Limiter would pull in a storage abstraction this does not
need, and requirements.txt is a pinned set worth keeping small. The limiter is
per-process, which is correct here because the app runs one worker -- each
worker holds 4.5 GB of model weights, so horizontal scaling is not the plan.

A fixed window rather than a token bucket: the thing being protected is tens of
seconds of CPU per request, so precision at the boundary is irrelevant next to
"how many of these can one caller start per minute".
"""
import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, limit, window_seconds):
        self.limit = limit
        self.window = window_seconds
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key, now=None):
        """Record a hit. Returns (allowed, retry_after_seconds).

        Locked because the sweeper thread and request threads both touch this,
        and a deque is not safe under concurrent mutation.
        """
        now = time.time() if now is None else now
        if self.limit <= 0:
            return True, 0
        cutoff = now - self.window
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, max(1, int(hits[0] + self.window - now) + 1)
            hits.append(now)
            return True, 0

    def forget(self, older_than_seconds, now=None):
        """Drop keys with no recent hits, so the map cannot grow unbounded."""
        now = time.time() if now is None else now
        cutoff = now - older_than_seconds
        with self._lock:
            for key in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
                del self._hits[key]


def client_key(request, trust_proxy=False):
    """Identify the caller.

    X-Forwarded-For is only consulted when trust_proxy is set. Honouring it
    unconditionally would let any caller spoof the header and reset their own
    bucket, which is worse than no limiter at all because it looks like one.
    """
    if trust_proxy:
        forwarded = request.headers.get('X-Forwarded-For', '')
        if forwarded:
            return forwarded.split(',')[0].strip()
    return request.remote_addr or 'unknown'


def api_key_ok(request, expected):
    """Constant-time comparison of the supplied key. Unset means open."""
    import hmac
    if not expected:
        return True
    supplied = request.headers.get('X-API-Key', '')
    return hmac.compare_digest(supplied, expected)

"""Age-based cleanup of stored uploads and generated artefacts.

The input to this application is emergency-call audio. Keeping it indefinitely
is not a storage problem, it is a disclosure one, so everything a request
produces has a lifetime.

Deliberately filesystem-based: there is no database, mtime is what there is to
sort on, and a sweep is idempotent and safe to run from more than one place.
"""
import os
import time


def sweep(*folders, max_age_seconds, now=None):
    """Delete regular files older than max_age_seconds. Returns the count.

    Missing folders are skipped rather than raising -- the sweeper runs at
    startup, potentially before anything has created them.

    Files that vanish between listing and unlinking are ignored: two workers
    sweeping at once is a supported situation, not a race to guard against.
    """
    now = time.time() if now is None else now
    cutoff = now - max_age_seconds
    removed = 0
    for folder in folders:
        if not folder or not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            try:
                if not os.path.isfile(path):
                    continue
                if os.path.getmtime(path) >= cutoff:
                    continue
                os.remove(path)
                removed += 1
            except OSError:
                continue
    return removed


def start_background_sweeper(folders, max_age_seconds, interval_seconds):
    """Sweep now, then on a timer. Returns the thread.

    A daemon thread so it never holds up interpreter shutdown. Swept once
    up-front too, because a process that crashed and restarted should not
    resurrect an old sweep interval's worth of recordings.
    """
    import threading

    def loop():
        while True:
            try:
                sweep(*folders, max_age_seconds=max_age_seconds)
            except Exception:
                pass
            time.sleep(interval_seconds)

    sweep(*folders, max_age_seconds=max_age_seconds)
    thread = threading.Thread(target=loop, daemon=True, name="retention-sweeper")
    thread.start()
    return thread

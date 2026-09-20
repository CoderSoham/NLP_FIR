"""In-process job registry, so a request can report progress while it works.

The pipeline takes tens of seconds and the browser previously saw nothing until
the PDF was written. That is not only a poor wait, it hides which stage is slow
and which one failed.

Deliberately a dict and a worker thread rather than a broker: the bottleneck is
one CPU/GPU-bound pipeline per machine, and Redis or Celery would add a service
to deploy without buying anything until there is more than one machine. The
registry is small enough to swap later; `submit` and `get` are the whole
interface.

Jobs are dropped once they age out, so a long-running process does not
accumulate completed results -- and the artefacts they point at are deleted by
the retention sweeper anyway.
"""
import threading
import time
import traceback
import uuid

_JOBS = {}
_LOCK = threading.Lock()

MAX_AGE_SECONDS = 3600


def _prune(now=None):
    now = now or time.time()
    for job_id, job in list(_JOBS.items()):
        if now - job["created"] > MAX_AGE_SECONDS:
            _JOBS.pop(job_id, None)


def submit(fn, *args, **kwargs):
    """Run fn(*args, progress=..., **kwargs) on a worker thread.

    `fn` receives a `progress` callable it should invoke with a short,
    human-readable phrase. The phrase is what the page displays, so it is
    written for a reader rather than as a stage identifier.
    """
    job_id = uuid.uuid4().hex
    with _LOCK:
        _prune()
        _JOBS[job_id] = {"id": job_id, "status": "queued", "stage": "Queued",
                         "created": time.time(), "started": None,
                         "finished": None, "result": None, "error": None,
                         "history": []}

    def progress(stage):
        with _LOCK:
            job = _JOBS.get(job_id)
            if job is None:
                return
            now = time.time()
            if job["history"]:
                job["history"][-1]["seconds"] = round(now - job["history"][-1]["at"], 1)
            job["history"].append({"stage": stage, "at": now})
            job["stage"] = stage

    def run():
        with _LOCK:
            _JOBS[job_id].update(status="running", started=time.time())
        try:
            result = fn(*args, progress=progress, **kwargs)
            with _LOCK:
                job = _JOBS.get(job_id)
                if job is not None:
                    if job["history"]:
                        job["history"][-1]["seconds"] = round(
                            time.time() - job["history"][-1]["at"], 1)
                    job.update(status="done", result=result,
                               stage="Done", finished=time.time())
        except Exception as exc:
            traceback.print_exc()
            with _LOCK:
                job = _JOBS.get(job_id)
                if job is not None:
                    job.update(status="failed", stage="Failed",
                               error=f"{type(exc).__name__}: {exc}",
                               finished=time.time())

    threading.Thread(target=run, daemon=True, name=f"job-{job_id[:8]}").start()
    return job_id


def get(job_id):
    with _LOCK:
        return _JOBS.get(job_id)


def status(job_id):
    """A JSON-safe view: never includes the result payload."""
    job = get(job_id)
    if job is None:
        return None
    started = job["started"] or job["created"]
    end = job["finished"] or time.time()
    return {
        "id": job["id"],
        "status": job["status"],
        "stage": job["stage"],
        "elapsed": round(end - started, 1),
        "error": job["error"],
        "history": [{"stage": h["stage"], "seconds": h.get("seconds")}
                    for h in job["history"]],
    }

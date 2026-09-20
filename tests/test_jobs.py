"""The job registry: stages, timings, and the partial results a waiter sees.

No pipeline here -- `submit` takes any callable, which is the point of the
`progress` contract. These tests pin that contract, because the progress page
is written against it and nothing else checks the two agree.
"""
import time

import pytest

from utils import jobs


def wait_for(job_id, status=("done", "failed"), timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = jobs.status(job_id)
        if state and state["status"] in status:
            return state
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} never reached {status}")


def test_a_job_runs_and_reports_done():
    job_id = jobs.submit(lambda progress: 42)
    assert wait_for(job_id)["status"] == "done"
    assert jobs.get(job_id)["result"] == 42


def test_arguments_are_passed_through():
    job_id = jobs.submit(lambda a, b, progress: a + b, 1, b=2)
    wait_for(job_id)
    assert jobs.get(job_id)["result"] == 3


def test_a_raising_job_is_recorded_not_propagated():
    def boom(progress):
        raise ValueError("no audio")

    state = wait_for(jobs.submit(boom))
    assert state["status"] == "failed"
    assert state["error"] == "ValueError: no audio"


def test_stages_are_recorded_in_order_with_timings():
    def work(progress):
        progress("Loading audio")
        progress("Transcribing the call")

    state = wait_for(jobs.submit(work))
    assert [h["stage"] for h in state["history"]] == [
        "Loading audio", "Transcribing the call"]
    # Every stage is closed out, including the last one when the job ends.
    assert all(h["seconds"] is not None for h in state["history"])


def test_partial_results_are_published_and_merged():
    def work(progress):
        progress("Transcribing the call")
        progress(None, transcript="there is a fire", language="en")
        progress("Classifying the emergency")
        progress(None, emergency_type="fire")

    state = wait_for(jobs.submit(work))
    assert state["partial"] == {
        "transcript": "there is a fire", "language": "en",
        "emergency_type": "fire"}


def test_a_partial_publish_does_not_create_a_step():
    """Otherwise the step list becomes a list of events, not of stages."""
    def work(progress):
        progress("Summarising")
        progress(None, summary="A fire was reported.")

    state = wait_for(jobs.submit(work))
    assert [h["stage"] for h in state["history"]] == ["Summarising"]


def test_a_partial_is_visible_before_the_job_finishes():
    release = __import__("threading").Event()

    def work(progress):
        progress(None, transcript="there is a fire")
        release.wait(5)
        return "done"

    job_id = jobs.submit(work)
    deadline = time.time() + 5
    while time.time() < deadline:
        if jobs.status(job_id)["partial"].get("transcript"):
            break
        time.sleep(0.01)
    else:
        release.set()
        raise AssertionError("partial never became visible mid-run")

    assert jobs.status(job_id)["status"] == "running"
    release.set()
    wait_for(job_id)


def test_status_never_leaks_the_result_payload():
    job_id = jobs.submit(lambda progress: {"transcription": "secret"})
    state = wait_for(job_id)
    assert "result" not in state


def test_an_unknown_job_has_no_status():
    assert jobs.status("0" * 32) is None


def test_old_jobs_are_pruned(monkeypatch):
    job_id = jobs.submit(lambda progress: None)
    wait_for(job_id)
    jobs._JOBS[job_id]["created"] -= jobs.MAX_AGE_SECONDS + 1
    jobs.submit(lambda progress: None)          # any submit prunes
    assert jobs.status(job_id) is None


def test_progress_after_a_job_is_pruned_is_a_no_op():
    """The worker outlives the registry entry when a process is busy."""
    captured = {}

    def work(progress):
        captured["progress"] = progress
        return None

    job_id = jobs.submit(work)
    wait_for(job_id)
    jobs._JOBS.pop(job_id, None)
    captured["progress"]("Writing the report")   # must not raise


# ---- spawn: overlapping one stage with another ------------------------------

def test_spawn_returns_the_value_on_join():
    join = jobs.spawn(lambda a, b: a * b, 6, b=7)
    assert join() == 42


def test_spawn_re_raises_on_join_not_in_the_thread():
    def boom():
        raise RuntimeError("provider is down")

    join = jobs.spawn(boom)
    with pytest.raises(RuntimeError, match="provider is down"):
        join()


def test_spawn_actually_overlaps():
    """Two half-second calls must finish in well under a second together."""
    started = __import__("threading").Event()
    join = jobs.spawn(lambda: (started.set(), time.sleep(0.3), "llm")[-1])
    assert started.wait(1), "spawn did not start until join"
    time.sleep(0.3)                     # stands in for the GPU stages
    begin = time.time()
    assert join() == "llm"
    assert time.time() - begin < 0.2    # the wait was already paid for


def test_joining_twice_is_safe():
    join = jobs.spawn(lambda: "once")
    assert join() == "once" and join() == "once"

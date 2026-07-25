"""In-process async job store for the optimise pipeline.

Single-user local tool: jobs live in a dict and run on daemon threads. A failed
job is recorded as state="error" with the exception message — never swallowed,
per shared context §8. `runner` is injectable so tests drive the store without
an LLM.
"""

import threading
import uuid

from rho.api.entry import run_optimize
from rho.models.api import JobStatus, OptimizeJobRequest, OptimizeResult


def _default_runner(req: OptimizeJobRequest, on_stage) -> OptimizeResult:
    return run_optimize(req.resume, req.jd_text, on_stage=on_stage)


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}
        self._lock = threading.Lock()

    def create(self, req: OptimizeJobRequest, runner=_default_runner) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = JobStatus(id=job_id, state="queued")
        threading.Thread(target=self._run, args=(job_id, req, runner), daemon=True).start()
        return job_id

    def get(self, job_id: str) -> JobStatus | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _set(self, job_id: str, **fields) -> None:
        with self._lock:
            current = self._jobs[job_id]
            self._jobs[job_id] = current.model_copy(update=fields)

    def _run(self, job_id: str, req: OptimizeJobRequest, runner) -> None:
        self._set(job_id, state="running")
        try:
            result = runner(req, lambda name: self._set(job_id, stage=name))
            self._set(job_id, state="done", stage="done", result=result)
        except Exception as exc:  # a dead model must not look like a clean run
            self._set(job_id, state="error", error=f"{type(exc).__name__}: {exc}")

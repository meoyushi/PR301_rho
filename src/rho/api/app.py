import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from rho.api.docx_export import build_docx
from rho.api.jobs import JobStore
from rho.extraction import extract
from rho.ingestion import ingest
from rho.models.api import ExportDocxRequest, JobStatus, OptimizeJobRequest, ParseResponse

logger = logging.getLogger(__name__)


def _warm_models() -> None:
    """Load the heavy model weights once, so no request pays that cost.

    Docling's converter and the sentence-transformers embedder both loaded their
    weights lazily on first use — and Docling rebuilt the converter per call, so
    every PDF ingest reloaded them. Warming here loads each once at boot.
    """
    try:
        from rho.ingestion.docling_adapter import warm_up

        warm_up()
    except Exception as exc:  # a warm-up failure must not down the server
        logger.warning("Docling warm-up skipped: %s", exc)
    try:
        from rho.matching.embed import Embedder

        Embedder().encode(["warm up"])
    except Exception as exc:
        logger.warning("embedder warm-up skipped: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm in a background thread so the server accepts connections immediately;
    # the first heavy request waits on the same cached models rather than
    # triggering a second load.
    threading.Thread(target=_warm_models, daemon=True).start()
    yield


app = FastAPI(title="rho", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs = JobStore()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/parse", response_model=ParseResponse)
async def parse(file: UploadFile):
    data = await file.read()
    try:
        md, prov = ingest(data, file.filename or "resume.txt")
        resume = extract(md, prov)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"parse failed: {exc}")
    return ParseResponse(structured_resume=resume, provenance_map=prov)


@app.post("/optimize", response_model=JobStatus)
def optimize(req: OptimizeJobRequest):
    job_id = _jobs.create(req)
    return _jobs.get(job_id)


@app.get("/optimize/{job_id}", response_model=JobStatus)
def optimize_status(job_id: str):
    js = _jobs.get(job_id)
    if js is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return js


_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@app.post("/export/docx")
def export_docx(req: ExportDocxRequest):
    try:
        data = build_docx(req.resume, req.section_order, req.accent, req.hidden_sections)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"docx export failed: {exc}")
    return Response(
        content=data,
        media_type=_DOCX_MIME,
        headers={"Content-Disposition": 'attachment; filename="resume.docx"'},
    )

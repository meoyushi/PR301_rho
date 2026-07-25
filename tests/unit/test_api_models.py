from rho.models.api import JobStatus, OptimizeJobRequest, OptimizeResult, ParseResponse
from rho.models.resume import StructuredResume


def test_parse_response_holds_resume_and_prov():
    r = ParseResponse(structured_resume=StructuredResume(name="X"), provenance_map={"doc_id": "d", "spans": {}})
    assert r.structured_resume.name == "X"


def test_optimize_request_requires_resume_and_jd():
    req = OptimizeJobRequest(resume=StructuredResume(name="X"), jd_text="jd")
    assert req.jd_text == "jd"


def test_job_status_defaults_to_queued_with_no_result():
    js = JobStatus(id="abc")
    assert js.state == "queued" and js.result is None and js.error is None

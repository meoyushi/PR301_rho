from rho.jd.schema import to_requirement_set
from rho.models.jd import RequirementSet


def analyze_jd(jd_text: str, _schema_fn=None) -> RequirementSet:
    """jd text -> RequirementSet"""
    if _schema_fn is None:
        from rho.jd.llm import analyze_jd_schema as _schema_fn_default

        _schema_fn = _schema_fn_default
    js = _schema_fn(jd_text)
    return to_requirement_set(js)

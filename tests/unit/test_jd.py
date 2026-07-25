from rho.jd import analyze_jd
from rho.jd.schema import JDSchema


def test_analyze_jd_maps_requirements():
    fake = lambda t: JDSchema(
        reasoning="x",
        title="Backend Engineer",
        requirements=[
            {"text": "Python", "kind": "skill", "priority": "must", "years": None},
            {"text": "AWS", "kind": "skill", "priority": "nice", "years": None},
            {
                "text": "5 years backend",
                "kind": "experience",
                "priority": "must",
                "years": 5.0,
            },
        ],
    )
    rs = analyze_jd("...", _schema_fn=fake)
    assert rs.title == "Backend Engineer"
    musts = [r for r in rs.requirements if r.priority == "must"]
    assert len(musts) == 2
    assert any(r.years == 5.0 for r in rs.requirements)

"""Guards the frozen component surface from shared-context Section 6.

Deliberately free of per-phase hand-editing: rather than naming whichever
function happens to still be a stub, this enumerates every Section-6 callable,
checks it is importable at its frozen path, and asserts that the ones whose
bodies are still `raise NotImplementedError` actually do so when called.
As each phase lands, its function stops being detected as a stub with no
change to this file.
"""

import importlib
import inspect

import pytest

# (module path, attribute) for every callable in shared-context Section 6.
SECTION_6 = [
    ("rho.ingestion", "ingest"),
    ("rho.extraction", "extract"),
    ("rho.jd", "analyze_jd"),
    ("rho.matching", "match"),
    ("rho.ats", "harvest_ats"),
    ("rho.ats", "Calibrator"),
    ("rho.rewrite", "rewrite"),
    ("rho.rewrite", "verify"),
    ("rho.graph", "run_pipeline"),
]


def _resolve(module_path: str, attr: str):
    return getattr(importlib.import_module(module_path), attr)


def _is_stub(fn) -> bool:
    """True when the body is nothing but `raise NotImplementedError`."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    body = [
        ln.strip()
        for ln in src.splitlines()
        if ln.strip() and not ln.strip().startswith(("def ", "class ", "@", '"""', "#"))
    ]
    return any(ln.startswith("raise NotImplementedError") for ln in body)


@pytest.mark.parametrize("module_path,attr", SECTION_6, ids=lambda v: v)
def test_section_6_symbol_is_importable(module_path, attr):
    """Frozen signatures must stay importable at their documented paths."""
    assert _resolve(module_path, attr) is not None


def test_remaining_stubs_raise_not_implemented():
    """Whatever is still unimplemented must fail loudly, never return None.

    Phase 6 landed `run_pipeline`, the last Section-6 stub, so this now walks an
    empty set and the `checked > 0` assertion is gone (as this file always said
    it should be). It is kept rather than deleted: it still guards against a
    regression that reintroduces a silently-returning stub.
    """
    checked = 0
    for module_path, attr in SECTION_6:
        obj = _resolve(module_path, attr)
        if inspect.isclass(obj):
            # bind an instance so `self` is supplied and not counted as an arg
            instance = obj()
            targets = [
                getattr(instance, m) for m in ("fit", "predict") if hasattr(instance, m)
            ]
        else:
            targets = [obj]
        for fn in targets:
            if not _is_stub(fn):
                continue
            checked += 1
            sig = inspect.signature(fn)
            args = [
                None
                for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.name not in ("self", "cls")
            ]
            with pytest.raises(NotImplementedError):
                fn(*args)
    assert checked == 0, f"{checked} Section-6 stub(s) left after Phase 6"

"""HTTP layer for the KAVACH frontend in `kavach/`.

    schemas.py     Pydantic models mirroring `kavach/src/api/types.ts`
    converters.py  domain objects -> those models, in one place
    store.py       SQLite records, audio on disk
    pipeline.py    the models, enrolment, and the verification path
    attacks.py     the Attack Lab
    evaluation.py  the Evaluation page
    app.py         the routes

Run it with::

    uvicorn kavach.api.app:app --reload --port 8000

and point the frontend at it by setting `VITE_USE_MOCK=false` in
`kavach/.env`.

Nothing here is imported eagerly. `app.py` pulls in FastAPI, which the CSBG
research core does not depend on, so importing `kavach.api` to get at
`schemas` in a test must not drag a web framework in with it.
"""

from __future__ import annotations

__all__ = ["schemas"]


def __getattr__(name: str):  # pragma: no cover - trivial lazy re-export
    if name in {"app", "create_app"}:
        from . import app as _app

        return getattr(_app, name)
    if name in {"schemas", "converters", "store", "pipeline", "evaluation", "attacks"}:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

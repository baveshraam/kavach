"""HTTP layer for the KAVACH frontend in `kavach/`.

    schemas.py     Pydantic models mirroring `kavach/src/api/types.ts`
    converters.py  domain objects -> those models, in one place
    store.py       SQLite records, audio on disk
    pipeline.py    the models, enrolment, and the verification path
    attacks.py     the Attack Lab
    evaluation.py  the Evaluation page
    app.py         the routes

Run it with::

    uvicorn kavach.api.app:app --reload --port 8000 --app-dir backend

and point the frontend at it by setting `VITE_USE_MOCK=false` in
`kavach/.env`.

Note the target is `kavach.api.app:app`, not `kavach.api:app`. The ASGI
application cannot be re-exported here under the name `app`, because `app.py`
already owns that name on this package: importing the submodule -- by any
route, including this module's own lazy loader -- makes Python bind the module
to `kavach.api.app` afterwards, so a re-export would mean the module or the
ASGI app depending on what a given process happened to import first. A server
that starts or doesn't depending on import order is worse than a longer
command, so `kavach.api.app` is always the module and `create_app` is the one
callable re-exported here.

Nothing here is imported eagerly. `app.py` pulls in FastAPI, which the CSBG
research core does not depend on, so importing `kavach.api` to get at
`schemas` in a test must not drag a web framework in with it.
"""

from __future__ import annotations

import importlib

__all__ = ["schemas"]

#: Submodules reachable as `kavach.api.<name>` without importing them eagerly.
_SUBMODULES = frozenset(
    {"app", "schemas", "converters", "store", "pipeline", "evaluation", "attacks"}
)

#: Names re-exported out of `app.py`. Deliberately not `app` -- see the module
#: docstring; that name belongs to the submodule and cannot be held against it.
_FROM_APP = frozenset({"create_app"})


def __getattr__(name: str):
    # `from . import app` is the natural spelling here and it is a trap: before
    # the import machinery will load the submodule it asks
    # `hasattr(kavach.api, "app")`, that question re-enters this very function,
    # and the answer never arrives -- RecursionError rather than an import, on
    # the one path a server start-up depends on. importlib addresses the
    # submodule by its full name and never consults this hook.
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    if name in _FROM_APP:
        return getattr(importlib.import_module(f"{__name__}.app"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

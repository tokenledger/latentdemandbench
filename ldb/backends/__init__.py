"""Orchestrators for the LLM baseline.

A backend answers one constrained question: given a system prompt, a task, and a
JSON schema, return a JSON string matching that schema. Nothing about the
benchmark task lives here — that is `ldb.llm`.

    class Backend(Protocol):
        def complete(self, *, system: str, task: str,
                     schema: dict, model: str | None) -> str: ...
"""

import functools
import hashlib
import json
import os
from importlib import import_module

CACHE_DIR = os.environ.get("LDB_CACHE", ".cache/responses")


TRANSIENT = ("429", "rate", "503", "502", "500", "overload",
             "unavailable", "timeout", "deadline")
# A depleted balance also arrives as 429, but waiting will never fix it. Retrying
# it five times per call turns an instant, obvious failure into a slow one.
FATAL = ("credit", "billing", "prepayment", "payment", "insufficient",
         "api key", "unauthenticated", "permission denied")


def with_backoff(fn, attempts: int = 5, base: float = 2.0):  # noqa: D401
    """Retry transient provider errors with exponential backoff and jitter.

    Without this, `retries` fired instantly and a rate-limited worker simply
    burned its attempts in milliseconds, which is also why worker counts had to
    stay low. With it, concurrency is limited by the provider rather than by us.
    """
    import functools
    import random
    import time

    @functools.wraps(fn)  # without this the wrapper's __module__ is this file,
    def wrapper(*a, **kw):  # and describe() is looked up in the wrong module
        last = None
        for i in range(attempts):
            try:
                return fn(*a, **kw)
            except Exception as e:  # noqa: BLE001 - provider SDKs differ
                last = e
                low = str(e).lower()
                if any(t in low for t in FATAL):
                    raise  # account problem: fail now, do not sleep on it
                if not any(t in low for t in TRANSIENT):
                    raise
                time.sleep(min(base ** i, 60) * (0.5 + random.random()))
        raise last

    return wrapper


_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool,
          "number": (int, float), "integer": int}


def _validate(obj, schema):
    """Structural check: types, required keys, numeric bounds, finiteness, and
    unexpected properties. Previously it ignored strings and booleans, treated
    booleans as numbers (bool is a subclass of int), and accepted NaN."""
    import math

    expected = schema.get("type")
    if expected in _TYPES:
        if expected in ("number", "integer") and isinstance(obj, bool):
            raise ValueError("boolean where a number was expected")
        if not isinstance(obj, _TYPES[expected]):
            raise ValueError(f"expected {expected}, got {type(obj).__name__}")
    if expected == "object":
        for key in schema.get("required", []):
            if not isinstance(obj, dict) or key not in obj:
                raise ValueError(f"response missing required key {key!r}")
        props = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            extra = set(obj) - set(props)
            if extra:
                raise ValueError(f"unexpected properties: {sorted(extra)}")
        for key, sub in props.items():
            if key in obj:
                _validate(obj[key], sub)
    elif expected == "array":
        if "maxItems" in schema and len(obj) > schema["maxItems"]:
            raise ValueError(f"array of {len(obj)} exceeds maxItems")
        for item in obj:
            _validate(item, schema.get("items") or {})
    elif expected in ("number", "integer"):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite value {obj}")
        lo, hi = schema.get("minimum"), schema.get("maximum")
        if (lo is not None and obj < lo) or (hi is not None and obj > hi):
            raise ValueError(f"{obj} outside [{lo}, {hi}]")


class InfrastructureError(RuntimeError):
    """A bug in this code, not a provider failure. Must never be scored as a
    model result: doing so turns a crash into a plausible-looking 0.500 row."""


def cache_key(module: str, system: str, task: str, schema: dict,
              resolved: dict, replicate: int = 0) -> str:
    """The single definition of a cache key.

    Both the writer (`cached`) and the pruner call this. They used to derive it
    separately, and when a `replicate` field was added to one the other declared
    every live response dead - a `--delete` away from erasing paid work.
    """
    return hashlib.sha256(json.dumps(
        [module, system, task, schema, resolved, replicate], sort_keys=True
    ).encode()).hexdigest()[:32]


def cached(fn):
    """Disk-cache raw model responses, keyed by the exact request.

    Model calls here cost minutes each, so a metric change must never force a
    re-query: with the cache warm, recomputing every metric over every arm is
    free. Also makes an arm reproducible byte-for-byte.
    """
    @functools.wraps(fn)
    def wrapper(*, system, task, schema, model=None, effort=None,
                replicate: int = 0, **kw):
        # Key on the RESOLVED configuration, not the arguments as passed: with
        # `model=None` in the key, changing a backend's default would silently
        # serve responses generated by the previous default.
        import sys
        mod = sys.modules[fn.__module__]
        if not hasattr(mod, "describe"):
            raise InfrastructureError(
                f"backend module {fn.__module__} has no describe(); "
                "a decorator has probably lost functools.wraps")
        resolved = mod.describe(model, effort)
        # replicate index is part of the key: without it, "replicates" were
        # cache hits on the first response and averaged nothing
        key = cache_key(fn.__module__, system, task, schema, resolved, replicate)
        path = os.path.join(CACHE_DIR, f"{key}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)["response"]
        out = fn(system=system, task=task, schema=schema,
                 model=model, effort=effort, **kw)
        # Validate before caching. A truncated response cached as-is would fail
        # identically on every retry and permanently poison that universe; and a
        # schema-invalid `{}` parses as JSON but is not a usable answer.
        _validate(json.loads(out), schema)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"resolved": resolved, "key": key,
                       "prompt_sha": hashlib.sha256(
                           (system + task).encode()).hexdigest()[:16],
                       "response": out}, f)
        return out
    return wrapper

_REGISTRY = {
    "codex": "ldb.backends.codex_cli",
    "anthropic": "ldb.backends.anthropic_api",
    "gemini": "ldb.backends.gemini_api",
    "openai": "ldb.backends.openai_api",
}


def get_backend(name: str):
    if name not in _REGISTRY:
        raise ValueError(f"unknown backend {name!r}; have {sorted(_REGISTRY)}")
    return import_module(_REGISTRY[name])


def backend_names() -> list[str]:
    return sorted(_REGISTRY)

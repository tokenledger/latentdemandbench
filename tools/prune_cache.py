"""Identify unlisted cached responses and optionally quarantine them.

Changing the generator or the prompt silently orphans every cached response:
the key includes both, so the entries stay on disk forever, consuming space and
misleading anyone who reads the cache as a record of what was run. This
recomputes the live key set through the real code path rather than reimplementing
the key derivation, so it cannot drift from `ldb.backends.cached`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import glob
import json
import os

from ldb.backends import cache_key, get_backend
from ldb.llm import SCHEMA, TOP_K, build_system, build_task, response_schema
from ldb.render import SURFACES
from ldb.task import build_instance


def live_keys(backend: str, seeds: range, model=None, effort=None,
              assists=("none", "composition", "gate", "both"),
              replicates: int = 1, configurations=()) -> set[str]:
    mod = f"ldb.backends.{ {'codex': 'codex_cli', 'anthropic': 'anthropic_api', 'gemini': 'gemini_api', 'openai': 'openai_api'}[backend] }"
    resolved = get_backend(backend).describe(model, effort)
    keys = set()
    seeds = list(seeds)

    def keep(inst, top_k=TOP_K):
        # Preserve pre-fix requests too: sensitivity arms originally shared
        # the fixed maxItems=40 schema even when the task requested another K.
        for schema in (SCHEMA, response_schema(top_k)):
            for r in range(replicates):
                keys.add(cache_key(mod, build_system(inst.surface, inst.horizon),
                                   build_task(inst, top_k), schema, resolved, r))

    # twin worlds are separate instances with their own corpora
    from ldb.twins import build_twin
    for s in seeds:
        for surface in SURFACES:
            tp = build_twin(s, surface=surface)
            if tp is not None:
                keep(tp.blocked)
                keep(tp.enabled)
    configs = []
    for surface in SURFACES:
        for assist in assists:
            configs.append({"surface": surface, "assist": assist})
    configs += [{"horizon": h} for h in (4, 12)]
    configs += [{"slope": slope} for slope in (6, 14)]
    configs += [{"n_products": 8, "n_needs": 10, "top_k": 25},
                {"n_products": 18, "n_needs": 22, "top_k": 90}]
    configs += list(configurations)
    seen = set()
    for config in configs:
        tag = json.dumps(config, sort_keys=True)
        if tag in seen:
            continue
        seen.add(tag)
        kw = dict(config)
        top_k = kw.pop("top_k", TOP_K)
        for s in seeds:
            keep(build_instance(s, **kw), top_k)
    return keys


def prune(cache_dir: str, keys: set[str], dry_run: bool = True) -> tuple[int, int]:
    """Unlisted does not prove obsolete: move files into a recoverable archive."""
    files = glob.glob(os.path.join(cache_dir, "*.json"))
    dead = [f for f in files
            if os.path.basename(f)[:-5] not in keys]
    if dead and not dry_run:
        import uuid
        quarantine = Path(cache_dir) / "quarantine" / uuid.uuid4().hex
        quarantine.mkdir(parents=True)
        for f in dead:
            Path(f).rename(quarantine / Path(f).name)
    return len(files), len(dead)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", action="append", default=None,
                    help="repeatable; every backend whose responses to keep")
    ap.add_argument("--effort", action="append", default=[None],
                    help="repeatable; every effort level to keep")
    ap.add_argument("--model", action="append", default=[None])
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--replicates", type=int, default=3,
                    help="highest replicate index in use; keys are per-replicate")
    ap.add_argument("--cache", default=".cache/responses")
    ap.add_argument("--delete", action="store_true", help="quarantine unlisted entries (recoverable)")
    ap.add_argument("--yes-delete-unlisted", action="store_true",
                    help="required with --delete: confirms that every backend, "
                         "model, effort and seed range still in use is listed")
    args = ap.parse_args()

    if args.delete and not args.yes_delete_unlisted:
        raise SystemExit(
            "refusing to delete: responses cost money and the live set covers "
            "only what you list. Re-run with --yes-delete-unlisted once every "
            "backend/model/effort/seed range in use is passed explicitly.")

    backends = args.backend or ["gemini"]
    keys = set()
    for b in backends:
        for m in args.model:
            for e in args.effort:
                keys |= live_keys(b, range(args.seeds), model=m, effort=e,
                                  replicates=args.replicates)
    total, dead = prune(args.cache, keys, dry_run=not args.delete)
    verb = "quarantined" if args.delete else "unlisted (dry run; not proof of obsolescence)"
    print(f"{total} cached responses, {dead} {verb}, {total - dead} live")
    if dead and not args.delete:
        print("--delete moves unlisted entries to cache/quarantine; it does not erase them")
    if dead and args.delete:
        print(f"recoverable responses are under {Path(args.cache) / 'quarantine'}")


if __name__ == "__main__":
    main()

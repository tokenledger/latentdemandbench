"""Run every model arm at once, sized to the account's rate limit.

The arms are independent, so running them sequentially at low concurrency wastes
almost all of a high RPM allowance: seven arms x 30 calls at 5 workers is ~40
sequential batches. This launches them together with a per-arm worker count
chosen from the limit, and reuses the real runners as subprocesses so every
guard (hash-before-run, failure classification, isolation, all-fail refusal)
still applies exactly as it does for a single arm.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ldb.backends import get_backend  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

SURFACES = ("finance", "opaque", "space_opera", "anime_tech", "cyberpunk")


def arms(universes: int, llm_universes: int, backend: str, model, effort,
         twins_seeds: int, llm_twins: int, replicates: int, outdir: str):
    common = ["--backend", backend, "--llm"]
    if model:
        common += ["--model", model]
    if effort:
        common += ["--effort", effort]

    for surface in SURFACES:
        yield (f"surface:{surface}",
               ["run_v0.py", "--universes", str(universes),
                "--llm-universes", str(llm_universes), "--surface", surface,
                "--out", f"{outdir}/surface_{surface}.json", *common])

    # The driver used to yield one assist arm and no difficulty sweep, so it did
    # not reproduce the sixteen arms the paper reports. Filenames match what
    # tools/paper_tables.py expects to find in an arm directory.
    for assist in ("composition", "gate", "both"):
        yield (f"assist:{assist}",
               ["run_v0.py", "--universes", str(universes),
                "--llm-universes", str(llm_universes), "--assist", assist,
                "--out", f"{outdir}/assist_{assist}.json", *common])

    for horizon in (4, 12):
        yield (f"horizon:{horizon}",
               ["run_v0.py", "--universes", str(universes),
                "--llm-universes", str(llm_universes), "--horizon", str(horizon),
                "--out", f"{outdir}/horizon_{horizon}.json", *common])

    for slope in (6, 14):
        yield (f"slope:{slope}",
               ["run_v0.py", "--universes", str(universes),
                "--llm-universes", str(llm_universes), "--slope", str(slope),
                "--out", f"{outdir}/slope_{slope}.json", *common])

    # K scales with the candidate count, so a larger economy does not silently
    # face a tighter filter
    for tag, n_products, n_needs, top_k in (("small", 8, 10, 25),
                                            ("large", 18, 22, 90)):
        yield (f"size:{tag}",
               ["run_v0.py", "--universes", str(universes),
                "--llm-universes", str(llm_universes),
                "--n-products", str(n_products), "--n-needs", str(n_needs),
                "--top-k", str(top_k),
                "--out", f"{outdir}/size_{tag}.json", *common])

    yield ("replicates:3",
           ["run_v0.py", "--universes", str(universes),
            "--llm-universes", str(llm_universes), "--replicates", "3",
            "--out", f"{outdir}/replicates_3.json", *common])

    yield ("twins",
           ["run_twins.py", "--seeds", str(twins_seeds),
            "--llm-twins", str(llm_twins), "--replicates", str(replicates),
            "--out", f"{outdir}/twins.json", *common])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpm", type=int, default=1000,
                    help="account requests-per-minute allowance")
    ap.add_argument("--seconds-per-call", type=float, default=90.0,
                    help="measured latency; concurrency is latency-bound, not "
                         "RPM-bound, once calls take longer than a minute")
    ap.add_argument("--universes", type=int, default=100)
    ap.add_argument("--llm-universes", type=int, default=30)
    ap.add_argument("--twins-seeds", type=int, default=60)
    ap.add_argument("--llm-twins", type=int, default=15)
    ap.add_argument("--replicates", type=int, default=1)
    ap.add_argument("--backend", default="gemini")
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--outdir", default=None,
                    help="default: results/<model>, keeping rungs separate")
    ap.add_argument("--max-workers", type=int, default=None,
                    help="per-arm workers; default derives from --rpm")
    args = ap.parse_args()

    # one directory per model, so ladder rungs cannot overwrite each other
    tag = (args.model or get_backend(args.backend).describe()["model"])
    outdir = args.outdir or f"results/{tag}"
    Path(ROOT / outdir).mkdir(parents=True, exist_ok=True)

    plan = list(arms(args.universes, args.llm_universes, args.backend,
                     args.model, args.effort, args.twins_seeds,
                     args.llm_twins, args.replicates, outdir))

    # in-flight requests the allowance supports, split across arms
    in_flight = max(1, int(args.rpm * args.seconds_per_call / 60))
    per_arm = args.max_workers or max(1, min(40, in_flight // len(plan)))

    print(f"writing to {outdir}/")
    print(f"{len(plan)} arms, {per_arm} workers each "
          f"(~{per_arm * len(plan)} concurrent, {args.rpm} rpm allowance)")

    procs, started = [], time.time()
    logs = ROOT / "results"
    logs.mkdir(exist_ok=True)
    for name, cmd in plan:
        log = logs / f"log_{name.replace(':', '_')}.txt"
        p = subprocess.Popen(
            [sys.executable, *cmd, "--workers", str(per_arm)],
            cwd=ROOT, stdout=open(log, "w"), stderr=subprocess.STDOUT)
        procs.append((name, p, log))
        print(f"  started {name}")

    failed = []
    for name, p, log in procs:
        if p.wait() != 0:
            failed.append(name)
        tail = log.read_text().strip().splitlines()[-1:] or ["(no output)"]
        print(f"  {'FAILED' if p.returncode else 'ok    '} {name}: {tail[0][:90]}")

    print(f"\n{len(procs) - len(failed)}/{len(procs)} arms wrote artifacts "
          f"in {(time.time() - started) / 60:.1f} min")
    if failed:
        print(f"failed: {failed}")
        print("an arm that refused to write is working as intended - check the "
              "log for whether the cause was the account or the model")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

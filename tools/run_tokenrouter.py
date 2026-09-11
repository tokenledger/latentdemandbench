"""TokenRouter (OpenAI-compatible) arm, scored like any other arm.

Same rationale as `tools/run_open_weight.py`: `run_v0._code_hash` hashes every
.py under `ldb/**` and every .py at the repo root, so a new backend cannot live
in `ldb/` without invalidating every published artifact. This file reimplements
only the transport; prompts, schema and scoring are imported from `ldb.llm`
unchanged, and the baseline sweep / artifact layout are imported from `run_v0`.

The endpoint is a *thinking* model behind an OpenAI-compatible API: calls are
slow (minutes), reasoning tokens are billed but never appear in the message
content, and the content may carry prose before the JSON object. Concurrency is
a bounded thread pool; the API key is read from $TR_KEY and never persisted.

    TR_KEY=... python3 tools/run_tokenrouter.py --llm-universes 2 \
        --out results/tokenrouter/smoke.json
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ldb.baselines import BASELINES, SCORE_KIND               # noqa: E402
from ldb.llm import (TOP_K, apply_top_k_protocol,             # noqa: E402
                     build_system, build_task, failed_response, score_response)
from ldb.metrics import aggregate, evaluate, paired_diff      # noqa: E402
from ldb.task import build_instance                           # noqa: E402
from run_v0 import (COLS, CONTRASTS, ELICITED_COLS, PATH_COLS,  # noqa: E402
                    _code_hash, _finite, _value_orders, fmt)

# run_v0 scores calibration only for the rankers that emit probabilities
PROB_SCORERS = {"structured_evidence", "fair_reference", "hidden_oracle"}


def extract_json(text: str) -> str:
    """First balanced top-level JSON object in `text`.

    A thinking model may narrate before answering, or wrap the object in a
    ```json fence. Braces inside string literals are skipped so an
    `evidence_path` containing "{" cannot close the object early.
    """
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    raise ValueError("no balanced JSON object in response")


def rank_from_raw(inst, raw: str, top_k: int = TOP_K):
    """Use the shared scorer after the TokenRouter transport has completed."""
    return score_response(inst, raw, top_k)


def _failure_row(inst):
    """Scored as a failure, not dropped — see run_v0.main."""
    return failed_response(inst)


_USAGE_LOCK = threading.Lock()
USAGE = {"prompt_tokens": 0, "completion_tokens": 0,
         "reasoning_tokens": 0, "total_tokens": 0}
_JSON_MODE = {"ok": True}  # flipped off once the endpoint rejects response_format


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, finish_reason):
        self.message = _Msg(content)
        self.finish_reason = finish_reason


class _Resp:
    """Reassembled stream, shaped like a non-streamed completion."""

    def __init__(self, content, finish_reason, model, usage):
        self.choices = [_Choice(content, finish_reason)]
        self.model = model
        self.usage = usage


def _consume_stream(stream):
    """Collect a streamed completion into a response-shaped object.

    Reasoning deltas are counted but discarded: they are billed and arrive on
    `delta.reasoning_content`, never on `delta.content`, and are not part of the
    answer. A response that is all reasoning yields empty content, which the
    caller treats as a failed call rather than an empty ranking.
    """
    parts, finish, model, usage, n_reason = [], None, None, None, 0
    for ev in stream:
        if getattr(ev, "usage", None):
            usage = ev.usage
        if getattr(ev, "model", None):
            model = ev.model
        if not ev.choices:
            continue
        ch = ev.choices[0]
        if ch.finish_reason:
            finish = ch.finish_reason
        d = ch.delta
        if getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None):
            n_reason += 1
        if getattr(d, "content", None):
            parts.append(d.content)
    return _Resp("".join(parts), finish, model, usage)


def _retryable(e) -> bool:
    status = getattr(e, "status_code", None) or getattr(e, "status", None)
    if status is None:
        status = getattr(getattr(e, "response", None), "status_code", None)
    if status in (408, 409, 429) or (status is not None and status >= 500):
        return True
    if status is not None:
        return False
    # transport-level failures (timeout, connection reset) carry no status
    return isinstance(e, Exception) and "APIStatusError" not in type(e).__name__


def call_one(client, model, system, task, args):
    """One completion, with backoff. Returns (text, resolved_model, latency)."""
    last = None
    for attempt in range(args.retries + 1):
        try:
            kw = dict(model=model, temperature=0.0,
                      messages=[{"role": "system", "content": system},
                                {"role": "user", "content": task}],
                      timeout=args.timeout)
            if args.max_tokens:
                kw["max_tokens"] = args.max_tokens
            if _JSON_MODE["ok"]:
                kw["response_format"] = {"type": "json_object"}
            t0 = time.time()
            if args.stream:
                # Measured on this endpoint: a non-streamed request to this
                # model returns nothing for 30+ min, while a streamed one emits
                # its first event in ~2s. Streaming is not an optimisation here,
                # it is the only mode that makes progress observable.
                kw["stream"] = True
                kw["stream_options"] = {"include_usage": True}
                r = _consume_stream(client.chat.completions.create(**kw))
            else:
                r = client.chat.completions.create(**kw)
            latency = time.time() - t0
            u = getattr(r, "usage", None)
            if u is not None:
                d = u.model_dump() if hasattr(u, "model_dump") else dict(u)
                det = d.get("completion_tokens_details") or {}
                with _USAGE_LOCK:
                    USAGE["prompt_tokens"] += d.get("prompt_tokens") or 0
                    USAGE["completion_tokens"] += d.get("completion_tokens") or 0
                    USAGE["total_tokens"] += d.get("total_tokens") or 0
                    USAGE["reasoning_tokens"] += (
                        (det.get("reasoning_tokens") if isinstance(det, dict) else 0)
                        or d.get("reasoning_tokens") or 0)
            choice = r.choices[0]
            text = choice.message.content or ""
            if not text.strip():
                raise ValueError(
                    f"empty content (finish_reason={choice.finish_reason}); "
                    "thinking model may have spent the budget on reasoning")
            return extract_json(text), (r.model or model), latency, \
                choice.finish_reason
        except Exception as e:  # noqa: BLE001
            last = e
            msg = str(e)
            if _JSON_MODE["ok"] and ("response_format" in msg
                                     or "json_object" in msg):
                print("  endpoint rejected response_format; "
                      "falling back to free-form + JSON extraction")
                _JSON_MODE["ok"] = False
                continue
            if attempt < args.retries and _retryable(e):
                back = min(60.0, 4.0 * (2 ** attempt))
                print(f"  retry {attempt + 1}/{args.retries} after "
                      f"{type(e).__name__}: {msg[:140]} (sleep {back:.0f}s)")
                time.sleep(back)
                continue
            raise
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universes", type=int, default=100)
    ap.add_argument("--llm-universes", type=int, default=30)
    ap.add_argument("--model", default="moonshotai/kimi-k3-free")
    ap.add_argument("--base-url", default="https://api.tokenrouter.com/v1")
    ap.add_argument("--surface", default="finance")
    ap.add_argument("--assist", default="none")
    ap.add_argument("--n-products", type=int, default=12)
    ap.add_argument("--n-needs", type=int, default=15)
    ap.add_argument("--n-complements", type=int, default=8)
    ap.add_argument("--cutoff", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--slope", type=float, default=9.0)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--stream", dest="stream", action="store_true", default=True)
    ap.add_argument("--no-stream", dest="stream", action="store_false")
    ap.add_argument("--json-mode", dest="json_mode", action="store_true",
                    default=False,
                    help="request response_format=json_object. OFF by default: "
                         "on this endpoint the JSON grammar stalls generation "
                         "outright (no stream event for 3.5+ min), while the "
                         "same prompt without it starts emitting in ~2s.")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="0 leaves it to the endpoint; a thinking model needs "
                         "room for reasoning tokens on top of ~4.8k of JSON")
    ap.add_argument("--out", default="results/tokenrouter/model_finance.json")
    args = ap.parse_args()

    # results/ holds paid runs that cannot be regenerated; never clobber one.
    if os.path.exists(args.out):
        raise SystemExit(
            f"{args.out} already exists; refusing to overwrite an existing "
            "artifact. Choose a new --out path.")

    key = os.environ.get("TR_KEY")
    if not key:
        raise SystemExit("set TR_KEY in the environment (never in a file)")

    _JSON_MODE["ok"] = args.json_mode

    # Captured BEFORE the run, exactly as run_v0 does.
    code_hash_at_start = _code_hash()

    gen = dict(n_products=args.n_products, n_needs=args.n_needs,
               n_complements=args.n_complements, slope=args.slope)
    insts = [build_instance(seed, surface=args.surface, assist=args.assist,
                            cutoff=args.cutoff, horizon=args.horizon, **gen)
             for seed in range(args.universes)]
    rates = [np.mean(list(i.label.values())) for i in insts]
    print(f"universes: {len(insts)}  surface: {args.surface}  "
          f"assist: {args.assist}  candidates/universe: {len(insts[0].candidates)}")
    print(f"activation base rate: {np.mean(rates):.3f} "
          f"(min {min(rates):.3f}, max {max(rates):.3f})")

    # ---- baselines, over every universe, under the model's top-K contract ----
    rows: dict[str, list[dict]] = {}
    for name, fn in BASELINES.items():
        rows[name] = []
        for i, inst in enumerate(insts):
            raw = fn(inst, seed=i)
            sc, pr = apply_top_k_protocol(raw, raw, args.top_k)
            kind = SCORE_KIND.get(name, "rank")
            ev, cap = _value_orders(inst, raw, kind, args.top_k)
            row = evaluate(inst, sc, probs=pr if name in PROB_SCORERS else None,
                           ev_scores=ev, cap_scores=cap, probe_scores=raw)
            if kind == "rank":
                row["ndcg_ev@10"] = row["capture_auc"] = float("nan")
            rows[name].append(row)

    # ---- one call per universe, bounded thread pool ----
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=args.base_url, max_retries=0,
                    timeout=args.timeout)

    subset = insts[: args.llm_universes]
    resolved = {"model": None}
    latencies: list[float] = []

    def work(i_inst):
        i, inst = i_inst
        try:
            raw, model_id, lat, finish = call_one(
                client, args.model, build_system(inst.surface, inst.horizon),
                build_task(inst, args.top_k), args)
            with _USAGE_LOCK:
                resolved["model"] = model_id
                latencies.append(lat)
            print(f"  seed {i}: {lat:.0f}s  finish={finish}  "
                  f"{len(raw)} chars json")
            return i, raw, None
        except Exception as e:  # noqa: BLE001
            return i, None, f"{type(e).__name__}: {str(e)[:300]}"

    print(f"dispatching {len(subset)} prompts to {args.base_url} "
          f"({args.model}, {args.workers} workers, temperature 0)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        out = list(ex.map(work, list(enumerate(subset))))
    wall = time.time() - t0
    by_id = {i: (raw, err) for i, raw, err in out}
    print(f"returned in {wall:.1f}s wall clock; usage {USAGE}")

    # ---- score each completion exactly as llm_rank would ----
    failures, results_per_universe = [], []
    for i, inst in enumerate(subset):
        raw, err = by_id[i]
        if err or not raw:
            print(f"  ! seed {inst.universe.seed}: {err}")
            failures.append({"seed": inst.universe.seed,
                             "error": err or "no completion"})
            results_per_universe.append(_failure_row(inst))
            continue
        try:
            results_per_universe.append(rank_from_raw(inst, raw, args.top_k))
        except Exception as e:  # noqa: BLE001 - a completion that will not parse
            print(f"  ! seed {inst.universe.seed}: {type(e).__name__}: "
                  f"{str(e)[:200]}")
            failures.append({"seed": inst.universe.seed,
                             "error": f"{type(e).__name__}: {str(e)[:200]}"})
            results_per_universe.append(_failure_row(inst))

    llm_label = f"tokenrouter:{args.model}"
    if args.assist != "none":
        llm_label += f"+{args.assist}"

    model_outputs: dict = {}
    rows[llm_label] = []
    for idx, (inst, (s, p, m)) in enumerate(zip(subset, results_per_universe)):
        m = dict(m)
        ev = m.pop("ev_scores", None)
        cap = m.pop("cap_scores", None)
        probe_scores = m.pop("probe_scores", None)
        model_outputs[str(idx)] = {"predictions": m.pop("predictions", []),
                                   "raw_response": m.pop("raw_response", None)}
        rows[llm_label].append({**evaluate(inst, s, probs=p, ev_scores=ev, cap_scores=cap,
                                        probe_scores=probe_scores), **m})
    llm_universes = list(range(len(subset)))

    if failures:
        print(f"  {len(failures)}/{len(subset)} completions unusable; "
              f"scored as failures (not dropped)")
    if len(failures) == len(subset):
        raise SystemExit(
            "every model call failed; refusing to write an artifact that would "
            "look like a legitimate chance-level result")

    results = {name: aggregate(r) for name, r in rows.items()}
    print("\n" + f"{'baseline':<22}" + "".join(f"{c:>22}" for c in COLS))
    for name, r in results.items():
        print(f"{name:<22}" +
              "".join(fmt(r[c]) if c in r else f"{'-':>22}" for c in COLS))
    r = results[llm_label]
    for title, cols in (("elicited outputs", ELICITED_COLS),
                        ("evidence-path validity", PATH_COLS)):
        shown = [c for c in cols if c in r]
        if shown:
            print(f"\n{title} (model rows only):")
            for c in shown:
                print(f"  {c:20s} {fmt(r[c]).strip()}")

    print("\npaired differences on AUC (95% bootstrap CI over universes):")
    for a, b, why in list(CONTRASTS) + [
            (llm_label, "recognition", "model beats commentary-count control"),
            ("structured_evidence", llm_label, "headroom remains")]:
        idx = llm_universes if llm_label in (a, b) else range(len(insts))
        ra = rows[a] if a == llm_label else [rows[a][i] for i in idx]
        rb = rows[b] if b == llm_label else [rows[b][i] for i in idx]
        m, lo, hi = paired_diff(ra, rb, "auc")
        sig = "" if np.isnan(lo) else ("  significant" if lo > 0
                                       else "  NOT significant")
        ci = "" if np.isnan(lo) else f" [{lo:+.3f},{hi:+.3f}]"
        print(f"  {a} - {b}: {m:+.3f}{ci}{sig}   ({why})")

    if _code_hash() != code_hash_at_start:
        raise SystemExit(
            f"source changed during the run ({code_hash_at_start} -> "
            f"{_code_hash()}); refusing to write an artifact that would certify "
            "code which did not produce these results. Re-run.")

    provenance = {
        "backend": "tokenrouter",
        # the id the API resolved to, which need not equal what we requested
        "model": resolved["model"] or args.model,
        "requested_model": args.model,
        "base_url": args.base_url,
        "effort": "none", "tools": "none",
        # a stateless HTTP endpoint per call, no shared state, no network tools
        "isolated": True,
        "temperature": 0.0, "seed": None,
        "json_mode": _JSON_MODE["ok"],
        "max_tokens": args.max_tokens or None,
        "workers": args.workers, "retries": args.retries,
        "timeout_s": args.timeout,
        "usage": dict(USAGE),
        "latency_s": {"mean": float(np.mean(latencies)) if latencies else None,
                      "min": float(min(latencies)) if latencies else None,
                      "max": float(max(latencies)) if latencies else None,
                      "n": len(latencies)},
        "wall_clock_s": round(wall, 1),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(_finite({
            "n_universes": len(insts),
            "base_rate": float(np.mean(rates)),
            "config": {
                "surface": args.surface, "assist": args.assist,
                "cutoff": args.cutoff, "horizon": args.horizon,
                "top_k": args.top_k, "replicates": 1,
                "n_products": args.n_products, "n_needs": args.n_needs,
                "n_complements": args.n_complements, "slope": args.slope,
                "universe_seeds": list(range(args.universes)),
                "llm_seeds": llm_universes,
                "backend": provenance, "failures": failures,
                "code_hash": code_hash_at_start,
            },
            "results": results, "per_universe": rows,
            "model_outputs": model_outputs}), f, indent=2, allow_nan=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

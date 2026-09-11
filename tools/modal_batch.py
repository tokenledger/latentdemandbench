"""Offline batch inference for the open-weight arm, on Modal GPUs.

Not a served endpoint: one container boots vLLM, generates every prompt in the
run as a single batch, and exits. That is the cheap shape for this benchmark —
each universe is one long prompt with one long structured completion, there is
no interactivity, and a persistent endpoint would bill idle GPU time between
universes.

Lives under `tools/` deliberately: `run_v0._code_hash` hashes every .py under
`ldb/**` and at the repo root, so a new backend module there would invalidate
every artifact already in results/. This file is therefore NOT a `ldb.backends`
backend; `tools/run_open_weight.py` reimplements `ldb.llm.llm_rank`'s scoring
against it instead.

Usage is via `tools/run_open_weight.py`, which imports `app` and `generate`
and calls them inside an ephemeral `app.run()`.
"""

import json
import os

import modal

APP_NAME = "ldb-openweight-batch"

# Pinned: guided-decoding APIs moved between vLLM releases, and an artifact that
# cannot say which decoder produced it is not reproducible.
VLLM_VERSION = os.environ.get("LDB_VLLM_VERSION", "0.10.1.1")
# vLLM 0.10.x declares `transformers>=4.55` with no upper bound, so a fresh
# build resolves transformers 5.x, whose tokenizer base class dropped
# `all_special_tokens_extended` — vLLM crashes loading any tokenizer. Pinned.
TRANSFORMERS_VERSION = os.environ.get("LDB_TRANSFORMERS_VERSION", "4.55.2")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
        "huggingface_hub[hf_transfer]>=0.34",
        "hf_transfer>=0.1.8",
    )
    .env({
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # xgrammar is the fast structured-output backend; V1 engine picks it up
        "VLLM_USE_V1": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    })
)

# Weights survive between runs, so only the first run pays the download.
hf_cache = modal.Volume.from_name("ldb-hf-cache", create_if_missing=True)

app = modal.App(APP_NAME)

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_OUTPUT_TOKENS = 16000  # ~40 ranked + 16 probe JSON objects, budgeted long


def _strip(schema, keys):
    """Drop schema keywords a guided-decoding backend refuses to compile.

    Nothing is lost by dropping them here: the top-K cap is re-enforced in
    `ldb.llm.score_predictions` and numeric bounds are clamped there too, for
    every backend, so correctness never depended on the decoder honouring them.
    """
    if isinstance(schema, dict):
        return {k: _strip(v, keys) for k, v in schema.items() if k not in keys}
    if isinstance(schema, list):
        return [_strip(v, keys) for v in schema]
    return schema


@app.function(
    image=image,
    gpu="H100",
    volumes={"/cache": hf_cache},
    timeout=60 * 60 * 4,          # cold boot + weight download + long batch
    scaledown_window=60,
)
def generate(
    items: list[dict],
    model: str = DEFAULT_MODEL,
    schema: dict | None = None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    max_model_len: int = 32768,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    enable_thinking: bool | None = None,
) -> list[dict]:
    """[{id, system, task}] -> [{id, text, finish_reason, n_tokens}].

    Order of the returned list matches `items`. A generation that fails to
    produce anything still returns a row, with an `error` key, so the caller
    can score it as a failure rather than silently dropping a universe.
    """
    import time

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=model,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=False,
    )
    load_s = time.time() - t0
    print(f"[modal_batch] loaded {model} in {load_s:.1f}s", flush=True)

    guided = None
    if schema is not None:
        # vLLM >= 0.8 exposes GuidedDecodingParams; older builds took a raw
        # `guided_json` kwarg on SamplingParams. Try the modern path first.
        try:
            from vllm.sampling_params import GuidedDecodingParams
            try:
                guided = GuidedDecodingParams(json=schema)
            except Exception:
                guided = GuidedDecodingParams(
                    json=_strip(schema, {"maxItems", "additionalProperties"}))
        except ImportError:
            guided = None

    def _params(g):
        kw = dict(temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=0)
        if g is not None:
            kw["guided_decoding"] = g
        return SamplingParams(**kw)

    convs = [[{"role": "system", "content": it["system"]},
              {"role": "user", "content": it["task"]}] for it in items]

    chat_kw = {}
    if enable_thinking is not None and hasattr(llm, "chat"):
        # Qwen3 emits <think>...</think> before the answer, which guided JSON
        # would forbid at token 0; the run must pick one mode explicitly.
        chat_kw["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    def _run(g):
        if hasattr(llm, "chat"):
            return llm.chat(convs, _params(g), **chat_kw)
        tok = llm.get_tokenizer()
        prompts = [tok.apply_chat_template(c, tokenize=False,
                                           add_generation_prompt=True)
                   for c in convs]
        return llm.generate(prompts, _params(g))

    t1 = time.time()
    try:
        outs = _run(guided)
    except Exception as e:  # noqa: BLE001 - a schema the decoder cannot compile
        if guided is None:
            raise
        print(f"[modal_batch] guided decoding failed ({type(e).__name__}: "
              f"{str(e)[:200]}); retrying with a relaxed schema", flush=True)
        from vllm.sampling_params import GuidedDecodingParams
        guided = GuidedDecodingParams(
            json=_strip(schema, {"maxItems", "additionalProperties",
                                 "minimum", "maximum"}))
        outs = _run(guided)
    gen_s = time.time() - t1

    rows, total_out = [], 0
    for it, o in zip(items, outs):
        try:
            c = o.outputs[0]
            total_out += len(c.token_ids)
            rows.append({"id": it["id"], "text": c.text,
                         "finish_reason": c.finish_reason,
                         "n_tokens": len(c.token_ids)})
        except Exception as e:  # noqa: BLE001
            rows.append({"id": it["id"], "text": "",
                         "error": f"{type(e).__name__}: {e}"})

    print(f"[modal_batch] {len(rows)} completions, {total_out} output tokens, "
          f"load {load_s:.1f}s + generate {gen_s:.1f}s", flush=True)
    rows.append({"id": "__meta__", "text": json.dumps({
        "model": model, "vllm": VLLM_VERSION,
        "transformers": TRANSFORMERS_VERSION, "load_s": load_s,
        "generate_s": gen_s, "output_tokens": total_out,
        "guided": guided is not None, "max_tokens": max_tokens,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tensor_parallel_size,
        "enable_thinking": enable_thinking})})
    return rows


@app.local_entrypoint()
def smoke(model: str = DEFAULT_MODEL):
    """`modal run tools/modal_batch.py` — proves the image and GPU work."""
    out = generate.remote(
        [{"id": "0", "system": "Answer as JSON.",
          "task": "Return {\"predictions\": [], \"probes\": []}."}],
        model=model, schema=None, max_tokens=64)
    print(out)

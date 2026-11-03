#!/usr/bin/env python3
"""Benchmark Qwen3.6 through OpenVINO GenAI's stateful continuous-batching API.

This harness complements tools/llm_bench/benchmark.py and
tools/continuous_batching/benchmark/continuous_batching_benchmark.cpp: it uses fixed
input/output token counts, a fixed maximum of 1 or 4 active requests, and replaces completed
requests until the requested total has finished. Input IDs are built from MMLU question text,
wrapped with the model's chat template, and repeated/truncated to the exact requested length.

The model artifact used here is stateful IR (ReadValue/Assign cache state). GenAI's public LLM
pipelines do not expose a stateless KV/recurrent-cache execution path for this artifact, so this
script intentionally reports stateful GenAI results only. Compare those with the stateless
llama.cpp OpenVINO results in QWEN36-UD-MULTISLOT-BENCHMARK.md. OpenVINO's C++
`StatefulToStateless` pass can convert the language graph into stateless IR, but it does not add a
GenAI stateless serving API. A stateless comparison needs a runner that feeds KV and linear-cache
outputs back as inputs; keep MTP off because the current stateless GDN path does not provide the
recurrent-state snapshots required for MTP rollback. Simply selecting SDPA does not make the model
stateless.

Example (after installing compatible OpenVINO GenAI Python wheels and benchmark dependencies):

  python qwen36_genai_bench.py --device GPU --context-tokens 1024,8192,32768 \
      --concurrency 1,4 --mtp both --requests 8 --output-tokens 512

MTP uses the model directory's openvino_mtp_model.xml via GenAI's draft_model property, with two
assistant tokens and greedy decoding. Prefix caching is disabled to avoid reusing repeated MMLU
prefixes. The script does not install packages or download model/data files.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MODEL = Path.home() / "models/Qwen3.6-35B-A3B-int4-ov"
DEFAULT_PARQUET = REPO_ROOT / "tmp/mmlu-dataset/all/test-00000-of-00001.parquet"


def parse_positive_csv(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="OpenVINO GenAI model directory")
    parser.add_argument("--device", default="GPU", help="OpenVINO device (default: GPU)")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PARQUET,
        help="Parquet with a 'question' column; default is the local MMLU test split",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Optional JSONL or text file. JSONL accepts a string or an object with 'question'/'prompt'.",
    )
    parser.add_argument("--source-prompts", type=int, default=64, help="Questions used to form each context")
    parser.add_argument("--context-tokens", type=parse_positive_csv, default=[1024, 8192, 32768])
    parser.add_argument("--concurrency", type=parse_positive_csv, default=[1, 4], help="Active requests per run")
    parser.add_argument("--requests", type=int, default=8, help="Total requests per measured run")
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--max-batched-tokens", type=int, default=2048)
    parser.add_argument("--cache-size-gb", type=int, default=0, help="GenAI cache budget; 0 uses dynamic allocation")
    parser.add_argument("--assistant-tokens", type=int, default=2)
    parser.add_argument("--mtp", choices=("off", "on", "both"), default="both")
    parser.add_argument("--warmup-requests", type=int, default=0, help="Unmeasured requests per matrix row")
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "tmp/qwen36-genai-benchmark.json")
    args = parser.parse_args()

    if args.requests <= 0 or args.output_tokens <= 0:
        parser.error("--requests and --output-tokens must be positive")
    if args.source_prompts <= 0:
        parser.error("--source-prompts must be positive")
    if args.warmup_requests < 0 or args.warmup_tokens <= 0:
        parser.error("--warmup-requests must be nonnegative and --warmup-tokens positive")
    if args.max_batched_tokens <= 0 or args.cache_size_gb < 0:
        parser.error("--max-batched-tokens must be positive and --cache-size-gb nonnegative")
    if any(item not in (1, 4) for item in args.concurrency):
        parser.error("--concurrency supports only 1 and 4")
    if args.requests < max(args.concurrency):
        parser.error("--requests must be at least the largest requested concurrency")
    if args.assistant_tokens <= 0:
        parser.error("--assistant-tokens must be positive")
    return args


def load_source_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file is not None:
        if not args.prompt_file.is_file():
            raise FileNotFoundError(args.prompt_file)
        if args.prompt_file.suffix.lower() == ".jsonl":
            prompts = []
            with args.prompt_file.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if isinstance(row, str):
                        prompt = row
                    elif isinstance(row, dict):
                        prompt = row.get("question", row.get("prompt", row.get("text", "")))
                    else:
                        prompt = ""
                    if prompt:
                        prompts.append(str(prompt))
        else:
            prompts = [line.strip() for line in args.prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        if not args.parquet.is_file():
            raise FileNotFoundError(
                f"MMLU parquet not found: {args.parquet}. Pass --prompt-file to use local text instead."
            )
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("Reading the default MMLU parquet requires pyarrow; use --prompt-file otherwise") from exc
        prompts = parquet.read_table(args.parquet, columns=["question"])["question"].to_pylist()
        prompts = [str(prompt) for prompt in prompts if prompt]

    if not prompts:
        raise RuntimeError("No nonempty prompts were loaded")
    return prompts[: args.source_prompts]


def tensor_to_ids(tensor: Any) -> list[int]:
    import numpy as np

    return np.asarray(tensor.data, dtype=np.int64).reshape(-1).tolist()


def make_context_ids(tokenizer: Any, questions: list[str], token_count: int, request_index: int) -> list[int]:
    """Create an exact-length chat prompt, preserving its assistant-generation suffix."""
    import numpy as np

    marker = "QWEN36_CONTEXT_CONTENT_MARKER_9f40e8d1"
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}], add_generation_prompt=True
    )
    if marker not in templated:
        raise RuntimeError("Model chat template did not preserve the content marker")
    prefix_text, suffix_text = templated.split(marker, 1)
    prefix_ids = tensor_to_ids(tokenizer.encode(prefix_text, add_special_tokens=False).input_ids)
    suffix_ids = tensor_to_ids(tokenizer.encode(suffix_text, add_special_tokens=False).input_ids)
    body_len = token_count - len(prefix_ids) - len(suffix_ids)
    if body_len <= 0:
        raise ValueError(f"context length {token_count} is too short for the model chat template")

    rotated = questions[request_index % len(questions) :] + questions[: request_index % len(questions)]
    separator_ids = tensor_to_ids(tokenizer.encode("\n\n", add_special_tokens=False).input_ids)
    body_parts: list[int] = []
    for question in rotated:
        body_parts.extend(tensor_to_ids(tokenizer.encode(question, add_special_tokens=False).input_ids))
        body_parts.extend(separator_ids)
    if not body_parts:
        raise RuntimeError("Tokenization produced an empty context body")

    # Repeat the natural-text MMLU token pool to reach long contexts, then keep exactly the
    # requested number of body tokens. The prefix/suffix remain intact, so every prompt ends
    # at the assistant generation point. This is a throughput input, not an accuracy prompt.
    body = np.resize(np.asarray(body_parts, dtype=np.int64), body_len).tolist()
    return prefix_ids + body + suffix_ids


def build_generation_config(genai: Any, output_tokens: int, assistant_tokens: int | None) -> Any:
    config = genai.GenerationConfig()
    config.max_new_tokens = int(output_tokens)
    config.ignore_eos = True
    config.do_sample = False
    config.num_beams = 1
    config.apply_chat_template = False
    if assistant_tokens is not None:
        config.num_assistant_tokens = int(assistant_tokens)
    return config


def pair_mean(metrics: Any, method: str) -> float | None:
    value = getattr(metrics, method)()
    result = float(value.mean)
    return result if math.isfinite(result) and result >= 0 else None


def run_requests(
    pipeline: Any,
    ov: Any,
    genai: Any,
    prompts: list[str],
    expected_input_tokens: list[int],
    request_count: int,
    concurrency: int,
    generation_config: Any,
    request_id_start: int,
) -> dict[str, Any]:
    """Run a fixed-depth request stream, replacing each completed request immediately."""
    active: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    submitted = 0
    wall_start = time.perf_counter()

    def submit_one() -> None:
        nonlocal submitted
        local_index = submitted
        request_id = request_id_start + local_index
        prompt_index = local_index % len(prompts)
        start = time.perf_counter()
        # Qwen3.6's GenAI language model consumes inputs_embeds. The string overload performs
        # tokenization and embedding; the tensor overload expects embeddings for this model.
        handle = pipeline.add_request(request_id, prompts[prompt_index], generation_config=generation_config)
        active[request_id] = {"handle": handle, "start": start, "streamed_tokens": 0}
        active[request_id]["expected_input_tokens"] = expected_input_tokens[prompt_index]
        submitted += 1

    while submitted < min(request_count, concurrency):
        submit_one()

    while active:
        if pipeline.has_non_finished_requests():
            pipeline.step()

        completed_ids: list[int] = []
        for request_id, record in list(active.items()):
            handle = record["handle"]
            while handle.can_read():
                output_map = handle.read()
                for output in output_map.values():
                    record["streamed_tokens"] += len(output.generated_ids)

            status = handle.get_status()
            if status == genai.GenerationStatus.RUNNING:
                continue

            metrics = handle.get_perf_metrics()
            input_count = int(metrics.get_num_input_tokens())
            output_count = int(metrics.get_num_generated_tokens())
            tpot_ms = pair_mean(metrics, "get_tpot")
            ttft_ms = pair_mean(metrics, "get_ttft")
            genai_tps = pair_mean(metrics, "get_throughput")
            rows.append(
                {
                    "request_id": request_id,
                    "input_tokens": input_count,
                    "expected_input_tokens": record["expected_input_tokens"],
                    "input_length_matches": input_count == record["expected_input_tokens"],
                    "generated_tokens": output_count,
                    "expected_output_tokens": int(generation_config.max_new_tokens),
                    "output_length_matches": output_count == int(generation_config.max_new_tokens),
                    "streamed_tokens": record["streamed_tokens"],
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                    "decode_tokens_per_second": 1000.0 / tpot_ms if tpot_ms else None,
                    "genai_reported_tokens_per_second": genai_tps,
                    "request_wall_ms": (time.perf_counter() - record["start"]) * 1000.0,
                    "finish_status": str(status),
                }
            )
            completed_ids.append(request_id)

        for request_id in completed_ids:
            del active[request_id]
            if submitted < request_count:
                submit_one()

        if active and not pipeline.has_non_finished_requests() and not completed_ids:
            raise RuntimeError("GenAI pipeline reported no active work while requests remained unfinished")

    wall_seconds = time.perf_counter() - wall_start
    rows.sort(key=lambda row: row["request_id"])
    invalid_rows = [
        row for row in rows
        if not row["input_length_matches"] or not row["output_length_matches"]
    ]
    if invalid_rows:
        raise RuntimeError(
            "GenAI did not honor the requested token lengths: "
            + json.dumps(invalid_rows, sort_keys=True)
        )
    output_total = sum(row["generated_tokens"] for row in rows)
    ttfts = [row["ttft_ms"] for row in rows if row["ttft_ms"] is not None]
    tpot_values = [row["tpot_ms"] for row in rows if row["tpot_ms"] is not None]
    decode_tps = [row["decode_tokens_per_second"] for row in rows if row["decode_tokens_per_second"] is not None]
    return {
        "request_count": len(rows),
        "wall_seconds": wall_seconds,
        "input_tokens_total": sum(row["input_tokens"] for row in rows),
        "generated_tokens_total": output_total,
        "end_to_end_aggregate_tokens_per_second": output_total / wall_seconds if wall_seconds else None,
        "ttft_mean_ms": statistics.mean(ttfts) if ttfts else None,
        "ttft_median_ms": statistics.median(ttfts) if ttfts else None,
        "tpot_mean_ms": statistics.mean(tpot_values) if tpot_values else None,
        "mean_per_request_decode_tokens_per_second": statistics.mean(decode_tps) if decode_tps else None,
        "requests": rows,
    }


def write_results(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def main() -> int:
    args = parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(f"OpenVINO model directory does not exist: {args.model}")
    for required_file in ("openvino_language_model.xml", "openvino_mtp_model.xml", "openvino_tokenizer.xml"):
        if not (args.model / required_file).is_file():
            raise FileNotFoundError(f"Missing {required_file} in {args.model}")

    try:
        import numpy
        import openvino as ov
        import openvino_genai as genai
    except ImportError as exc:
        raise RuntimeError(
            "Install compatible OpenVINO, openvino-tokenizers, and openvino-genai wheels in a Python environment first."
        ) from exc

    source_prompts = load_source_prompts(args)
    mtp_modes = [False, True] if args.mtp == "both" else [args.mtp == "on"]
    result: dict[str, Any] = {
        "model": str(args.model.resolve()),
        "device": args.device,
        "openvino_version": ov.get_version(),
        "openvino_genai_version": getattr(genai, "__version__", "unknown"),
        "execution_mode": "stateful GenAI ContinuousBatchingPipeline / PagedAttention",
        "cache_api": "GenAI-internal KV and linear-attention state; prefix caching disabled",
        "source": {
            "parquet": str(args.parquet) if args.prompt_file is None else None,
            "prompt_file": str(args.prompt_file) if args.prompt_file is not None else None,
            "source_prompt_count": len(source_prompts),
            "context_construction": "MMLU text token pool repeated to exact length inside model chat template",
        },
        "scheduler": {
            "max_num_seqs": 4,
            "max_num_batched_tokens": args.max_batched_tokens,
            "dynamic_split_fuse": True,
            "cache_size_gb": args.cache_size_gb,
            "enable_prefix_caching": False,
        },
        "requested_context_tokens": args.context_tokens,
        "requested_concurrency": args.concurrency,
        "requests_per_run": args.requests,
        "output_tokens": args.output_tokens,
        "warmup_requests_per_row": args.warmup_requests,
        "mtp_assistant_tokens": args.assistant_tokens,
        "runs": [],
    }

    model_path = args.model.resolve()
    for mtp_enabled in mtp_modes:
        scheduler = genai.SchedulerConfig()
        scheduler.max_num_seqs = 4
        scheduler.max_num_batched_tokens = args.max_batched_tokens
        scheduler.dynamic_split_fuse = True
        scheduler.cache_size = args.cache_size_gb
        scheduler.enable_prefix_caching = False

        properties: dict[str, Any] = {}
        if mtp_enabled:
            properties["draft_model"] = genai.draft_model(str(model_path), args.device)
        init_start = time.perf_counter()
        pipeline = genai.ContinuousBatchingPipeline(
            str(model_path), scheduler, args.device.upper(), properties=properties
        )
        init_seconds = time.perf_counter() - init_start
        print(
            f"Loaded GenAI stateful pipeline: MTP={'on' if mtp_enabled else 'off'}, "
            f"device={args.device}, init={init_seconds:.2f}s",
            flush=True,
        )
        tokenizer = pipeline.get_tokenizer()
        assistant_tokens = args.assistant_tokens if mtp_enabled else None
        generation_config = build_generation_config(genai, args.output_tokens, assistant_tokens)
        next_request_id = 0

        for context_tokens in args.context_tokens:
            for concurrency in args.concurrency:
                num_prompt_sets = max(args.requests, args.warmup_requests, concurrency)
                prompt_ids = [
                    make_context_ids(tokenizer, source_prompts, context_tokens, index)
                    for index in range(num_prompt_sets)
                ]
                actual_sizes = {len(ids) for ids in prompt_ids}
                if actual_sizes != {context_tokens}:
                    raise RuntimeError(f"context builder produced unexpected lengths: {actual_sizes}")
                prompt_texts = [tokenizer.decode(ids, skip_special_tokens=False) for ids in prompt_ids]
                for index, (prompt_text, expected_ids) in enumerate(zip(prompt_texts, prompt_ids)):
                    roundtrip_ids = tensor_to_ids(tokenizer.encode(prompt_text, add_special_tokens=False).input_ids)
                    if roundtrip_ids != expected_ids:
                        raise RuntimeError(f"tokenizer decode/encode changed prompt {index} at context {context_tokens}")
                expected_input_tokens = [len(ids) for ids in prompt_ids]

                if args.warmup_requests:
                    warmup_config = build_generation_config(
                        genai, min(args.warmup_tokens, args.output_tokens), assistant_tokens
                    )
                    run_requests(
                        pipeline,
                        ov,
                        genai,
                        prompt_texts,
                        expected_input_tokens,
                        args.warmup_requests,
                        concurrency,
                        warmup_config,
                        next_request_id,
                    )
                    next_request_id += args.warmup_requests

                print(
                    f"Running MTP={'on' if mtp_enabled else 'off'}, context={context_tokens}, "
                    f"active={concurrency}, requests={args.requests}",
                    flush=True,
                )
                measured = run_requests(
                    pipeline,
                    ov,
                    genai,
                    prompt_texts,
                    expected_input_tokens,
                    args.requests,
                    concurrency,
                    generation_config,
                    next_request_id,
                )
                next_request_id += args.requests
                row = {
                    "mtp": mtp_enabled,
                    "context_tokens": context_tokens,
                    "concurrency": concurrency,
                    "pipeline_init_seconds": init_seconds,
                    **measured,
                }
                result["runs"].append(row)
                write_results(args.output, result)
                print(
                    f"  TTFT={row['ttft_mean_ms']:.1f} ms, "
                    f"decode={row['mean_per_request_decode_tokens_per_second']:.2f} tok/s/request, "
                    f"aggregate end-to-end={row['end_to_end_aggregate_tokens_per_second']:.2f} tok/s",
                    flush=True,
                )

        del pipeline

    write_results(args.output, result)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

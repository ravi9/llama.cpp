#!/usr/bin/env python3
"""Benchmark Qwen3.6 OpenVINO IR with explicit KV and linear-cache IO.

The C++ companion exports the stateful language model using StatefulToStateless. This runner
feeds every cache Result back to its matching Parameter and uses greedy decoding. It is MTP-off:
the current stateless GDN path does not return recurrent-state snapshots for MTP rollback.

Example:
  python qwen36_stateless_bench.py --device GPU --context-tokens 1024,8192,32768 \
      --concurrency 1,4 --requests 4 --output-tokens 512
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
import openvino as ov
import openvino_genai as genai

from qwen36_genai_bench import (
    DEFAULT_PARQUET,
    REPO_ROOT,
    load_source_prompts,
    make_context_ids,
)


DEFAULT_MODEL = Path.home() / "models/Qwen3.6-35B-A3B-int4-ov"
DEFAULT_IR = REPO_ROOT / "tmp/qwen36-stateless-model/openvino_language_model.xml"


def csv_positive_ints(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--stateless-ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--source-prompts", type=int, default=64)
    parser.add_argument("--context-tokens", type=csv_positive_ints, default=[1024, 8192, 32768])
    parser.add_argument("--concurrency", type=csv_positive_ints, default=[1, 4])
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--max-batched-tokens", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "tmp/qwen36-genai-stateless-benchmark.json")
    args = parser.parse_args()
    if args.requests <= 0 or args.output_tokens <= 0 or args.source_prompts <= 0:
        parser.error("requests, output tokens, and source prompts must be positive")
    if any(count not in (1, 4) for count in args.concurrency):
        parser.error("concurrency supports only 1 and 4")
    if args.requests < max(args.concurrency):
        parser.error("requests must be at least the largest requested concurrency")
    if args.max_batched_tokens <= 0:
        parser.error("max-batched-tokens must be positive")
    return args


def encode_ids(tokenizer: Any, text: str) -> list[int]:
    tensor = tokenizer.encode(text, add_special_tokens=False).input_ids
    return np.asarray(tensor.data, dtype=np.int64).reshape(-1).tolist()


def cache_output_name(input_name: str) -> str:
    if input_name.startswith("past_key_values."):
        return input_name.replace("past_key_values.", "present.", 1)
    if input_name.startswith("cache_params.past."):
        return input_name.replace("cache_params.past.", "cache_params.present.", 1)
    raise ValueError(f"unrecognized cache input: {input_name}")


def cache_shape(name: str, batch: int, context_tokens: int = 0) -> tuple[int, ...]:
    if name.startswith("past_key_values."):
        return (batch, 2, context_tokens, 256)
    elif ".conv." in name:
        return (batch, 8192, 4)
    elif ".ssm." in name:
        return (batch, 32, 128, 128)
    raise ValueError(f"unrecognized cache input: {name}")


class StatelessRunner:
    def __init__(self, model_dir: Path, stateless_ir: Path, device: str):
        self.core = ov.Core()
        self.context = self.core.get_default_context(device)
        self.tokenizer = genai.Tokenizer(str(model_dir))

        embedding_model = self.core.read_model(model_dir / "openvino_text_embeddings_model.xml")
        self.embedding_model = self.core.compile_model(embedding_model, self.context)
        self.embedding_request = self.embedding_model.create_infer_request()
        self.embedding_input = self.embedding_model.input(0)
        self.embedding_output = self.embedding_model.output(0)
        self.embedding_buffers: dict[tuple[int, ...], ov.Tensor] = {}

        decoder_model = self.core.read_model(stateless_ir)
        self.model = self.core.compile_model(decoder_model, self.context)
        self.request = self.model.create_infer_request()
        self.inputs = {port.get_any_name(): port for port in self.model.inputs}
        self.outputs = {port.get_any_name(): port for port in self.model.outputs}
        if "attention_mask" not in self.inputs or "inputs_embeds" not in self.inputs or "position_ids" not in self.inputs:
            raise RuntimeError(f"unexpected stateless model inputs: {list(self.inputs)}")
        # OpenVINO IR keeps the original Result name "logits" even though its payload was
        # replaced with the TopK token index output by qwen36_stateless_export.cpp.
        self.token_output = self.outputs.get("next_token_ids", self.outputs.get("logits"))
        if self.token_output is None:
            raise RuntimeError(f"stateless model lacks greedy token output: {list(self.outputs)}")

        self.cache_pairs: dict[str, str] = {}
        for name in self.inputs:
            if name.startswith("past_key_values.") or name.startswith("cache_params.past."):
                output_name = cache_output_name(name)
                if output_name not in self.outputs:
                    raise RuntimeError(f"cache output {output_name} is missing")
                self.cache_pairs[name] = output_name
        if len(self.cache_pairs) != 80:
            raise RuntimeError(f"expected 80 cache inputs, found {len(self.cache_pairs)}")

        if self.token_output.get_partial_shape().rank.get_length() != 2:
            raise RuntimeError(
                "greedy token output must have shape [batch, 1] so it can feed the embedding model on-device"
            )
        self.token_buffers: dict[int, ov.Tensor] = {}
        self.fixed_cache_names = [name for name in self.cache_pairs if not name.startswith("past_key_values.")]
        self.usm_properties = {"SHARED_MEM_TYPE": "USM_DEVICE_BUFFER"}

    def device_tensor(
        self,
        element_type: ov.Type,
        shape: tuple[int, ...],
        host_data: np.ndarray | None = None,
    ) -> ov.Tensor:
        tensor = self.context.create_tensor(element_type, ov.Shape(list(shape)), self.usm_properties)
        if not isinstance(tensor, ov.RemoteTensor):
            raise RuntimeError("GPU USM tensor allocation did not return a RemoteTensor")
        if host_data is not None:
            array = np.ascontiguousarray(host_data)
            if array.nbytes:
                tensor.copy_from(ov.Tensor(array))
        return tensor

    def embed(self, token_ids: np.ndarray | ov.Tensor) -> ov.Tensor:
        if isinstance(token_ids, ov.RemoteTensor):
            input_tensor = token_ids
            input_shape = tuple(int(dim) for dim in token_ids.get_shape())
        else:
            ids = np.ascontiguousarray(token_ids, dtype=np.int64)
            input_shape = tuple(int(dim) for dim in ids.shape)
            input_tensor = self.device_tensor(ov.Type.i64, input_shape, ids)

        self.embedding_request.set_tensor(self.embedding_input.get_any_name(), input_tensor)
        output_shape = input_shape + (2048,)
        if output_shape not in self.embedding_buffers:
            self.embedding_buffers[output_shape] = self.device_tensor(ov.Type.f32, output_shape)
        embedding_tensor = self.embedding_buffers[output_shape]
        self.embedding_request.set_tensor(self.embedding_output.get_any_name(), embedding_tensor)
        self.embedding_request.start_async()
        self.embedding_request.wait()
        return embedding_tensor

    def infer_step(
        self,
        caches: dict[str, ov.Tensor],
        recyclable: dict[str, ov.Tensor],
        embeddings: ov.Tensor,
        position_start: int,
        position_end: int,
        total_context: int,
        batch: int,
    ) -> tuple[ov.Tensor, dict[str, ov.Tensor], dict[str, ov.Tensor]]:
        count = position_end - position_start
        positions = np.arange(position_start, position_end, dtype=np.int64)
        position_ids = np.broadcast_to(positions[None, None, :], (4, batch, count)).copy()
        attention_mask = np.ones((batch, total_context), dtype=np.int64)
        self.request.set_tensor("inputs_embeds", embeddings)
        self.request.set_tensor("position_ids", self.device_tensor(ov.Type.i64, position_ids.shape, position_ids))
        self.request.set_tensor(
            "attention_mask", self.device_tensor(ov.Type.i64, attention_mask.shape, attention_mask)
        )
        for input_name, tensor in caches.items():
            self.request.set_tensor(input_name, tensor)

        # Bind all Results to explicit GPU USM allocations. Fixed-size recurrent state alternates
        # between two device buffers; growing KV Results get a correctly-sized USM buffer.
        next_caches: dict[str, ov.Tensor] = {}
        next_recyclable: dict[str, ov.Tensor] = {}
        output_tensors: dict[str, ov.Tensor] = {}
        token_shape = (batch, 1)
        token_tensor = self.token_buffers.get(batch)
        if token_tensor is None:
            token_tensor = self.device_tensor(self.token_output.get_element_type(), token_shape)
            self.token_buffers[batch] = token_tensor
        output_tensors[self.token_output.get_any_name()] = token_tensor
        self.request.set_tensor(self.token_output.get_any_name(), token_tensor)

        for input_name, output_name in self.cache_pairs.items():
            if input_name.startswith("past_key_values."):
                output_tensor = self.device_tensor(
                    self.outputs[output_name].get_element_type(),
                    cache_shape(input_name, batch, total_context),
                )
            else:
                output_tensor = recyclable[input_name]
                next_recyclable[input_name] = caches[input_name]
            output_tensors[output_name] = output_tensor
            self.request.set_tensor(output_name, output_tensor)

        self.request.start_async()
        self.request.wait()
        for input_name, output_name in self.cache_pairs.items():
            next_caches[input_name] = output_tensors[output_name]
        return token_tensor, next_caches, next_recyclable

    def run_group(
        self,
        prompt_ids: list[list[int]],
        output_tokens: int,
        max_batched_tokens: int,
    ) -> dict[str, float]:
        batch = len(prompt_ids)
        context = len(prompt_ids[0])
        if any(len(ids) != context for ids in prompt_ids):
            raise ValueError("all prompts in a batch must have the same token count")
        chunk_tokens = max(1, max_batched_tokens // batch)
        caches: dict[str, ov.Tensor] = {}
        recyclable: dict[str, ov.Tensor] = {}
        for name in self.cache_pairs:
            shape = cache_shape(name, batch)
            if any(dim == 0 for dim in shape):
                # The empty KV prefix has no payload to transfer; keep its zero-length tensor on host.
                caches[name] = ov.Tensor(np.zeros(shape, dtype=np.float32))
            else:
                caches[name] = self.device_tensor(ov.Type.f32, shape, np.zeros(shape, dtype=np.float32))
            if not name.startswith("past_key_values."):
                recyclable[name] = self.device_tensor(ov.Type.f32, shape)

        prompt_start = time.perf_counter()
        next_ids: ov.Tensor | None = None
        for start in range(0, context, chunk_tokens):
            end = min(start + chunk_tokens, context)
            chunk_ids = np.asarray([ids[start:end] for ids in prompt_ids], dtype=np.int64)
            embeddings = self.embed(chunk_ids)
            next_ids, caches, recyclable = self.infer_step(
                caches, recyclable, embeddings, start, end, end, batch
            )
        ttft_ms = (time.perf_counter() - prompt_start) * 1000.0
        assert next_ids is not None

        decode_start = time.perf_counter()
        for generated in range(1, output_tokens):
            embeddings = self.embed(next_ids)
            position = context + generated - 1
            next_ids, caches, recyclable = self.infer_step(
                caches,
                recyclable,
                embeddings,
                position,
                position + 1,
                position + 1,
                batch,
            )
        decode_seconds = time.perf_counter() - decode_start
        decode_tps = (output_tokens - 1) / decode_seconds if decode_seconds > 0 else math.nan
        total_seconds = ttft_ms / 1000.0 + decode_seconds
        return {
            "ttft_ms": ttft_ms,
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second_per_request": decode_tps,
            "end_to_end_aggregate_tokens_per_second": batch * output_tokens / total_seconds,
            "chunk_tokens_per_request": chunk_tokens,
        }


def save_results(path: Path, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**metadata, "runs": rows}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    if not args.stateless_ir.is_file():
        raise FileNotFoundError(
            f"{args.stateless_ir} not found; run qwen36_stateless_export.cpp first"
        )

    questions = load_source_prompts(args)
    tokenizer = genai.Tokenizer(str(args.model))
    prompt_ids_by_length: dict[int, list[list[int]]] = {}
    for context in args.context_tokens:
        prompt_ids_by_length[context] = [
            make_context_ids(tokenizer, questions, context, index) for index in range(args.requests)
        ]
        if any(len(ids) != context for ids in prompt_ids_by_length[context]):
            raise RuntimeError(f"failed to build exact {context}-token prompts")

    init_start = time.perf_counter()
    runner = StatelessRunner(args.model, args.stateless_ir, args.device)
    init_seconds = time.perf_counter() - init_start
    print(f"Loaded explicit-cache OpenVINO model on {args.device} in {init_seconds:.2f}s", flush=True)

    metadata: dict[str, Any] = {
        "model": str(args.model),
        "stateless_ir": str(args.stateless_ir),
        "device": args.device,
        "openvino_version": ov.__version__,
        "openvino_genai_version": genai.get_version(),
        "execution_mode": "stateless OpenVINO infer request; explicit KV and linear-cache feedback",
        "cache_api": "80 cache Parameters fed from their matching Results; MTP off",
        "prompt_source": str(args.prompt_file or args.parquet),
        "source_prompt_count": len(questions),
        "requested_context_tokens": args.context_tokens,
        "requested_concurrency": args.concurrency,
        "requests_per_row": args.requests,
        "output_tokens": args.output_tokens,
        "max_batched_tokens": args.max_batched_tokens,
        "model_init_seconds": init_seconds,
        "mtp": False,
        "notes": [
            "Greedy token selection is a TopK(k=1) output node to avoid copying full vocabulary logits to the host.",
            "Prefill is chunked so aggregate tokens per model call do not exceed max_batched_tokens.",
            "Each concurrency group uses a dense batch with equal prompt lengths; no MTP state snapshots are used.",
        ],
    }
    rows: list[dict[str, Any]] = []
    save_results(args.output, metadata, rows)

    for context in args.context_tokens:
        prompt_pool = prompt_ids_by_length[context]
        for concurrency in args.concurrency:
            request_results: list[dict[str, float]] = []
            row_start = time.perf_counter()
            for start in range(0, args.requests, concurrency):
                ids = prompt_pool[start : min(start + concurrency, args.requests)]
                # The final group can be short when requests is not divisible by concurrency.
                result = runner.run_group(ids, args.output_tokens, args.max_batched_tokens)
                request_results.extend([result] * len(ids))
                print(
                    f"ctx={context} active={len(ids)} group {start // concurrency + 1}: "
                    f"TTFT={result['ttft_ms']:.1f} ms, "
                    f"decode={result['decode_tokens_per_second_per_request']:.2f} tok/s/request",
                    flush=True,
                )
            wall_seconds = time.perf_counter() - row_start
            tps = [x["decode_tokens_per_second_per_request"] for x in request_results]
            ttft = [x["ttft_ms"] for x in request_results]
            row = {
                "mtp": False,
                "context_tokens": context,
                "concurrency": concurrency,
                "request_count": len(request_results),
                "output_tokens": args.output_tokens,
                "ttft_mean_ms": statistics.fmean(ttft),
                "ttft_median_ms": statistics.median(ttft),
                "mean_per_request_decode_tokens_per_second": statistics.fmean(tps),
                "end_to_end_aggregate_tokens_per_second": (
                    len(request_results) * args.output_tokens / wall_seconds
                ),
                "wall_seconds": wall_seconds,
                "max_batched_tokens": args.max_batched_tokens,
                "chunk_tokens_per_request": max(1, args.max_batched_tokens // concurrency),
                "requests": [
                    {
                        "request_id": i,
                        "input_tokens": context,
                        "generated_tokens": args.output_tokens,
                        **request_results[i],
                    }
                    for i in range(len(request_results))
                ],
            }
            rows.append(row)
            save_results(args.output, metadata, rows)
            print(
                f"Completed ctx={context} active={concurrency}: "
                f"TTFT={row['ttft_mean_ms']:.1f} ms, "
                f"decode={row['mean_per_request_decode_tokens_per_second']:.2f} tok/s/request, "
                f"aggregate end-to-end={row['end_to_end_aggregate_tokens_per_second']:.2f} tok/s",
                flush=True,
            )
            gc.collect()

    print(f"Wrote {len(rows)} stateless rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()

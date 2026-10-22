#!/usr/bin/env python3
"""Benchmark saved natural-text prompts on an already running llama-server.

The corpus is local Markdown, rather than random token IDs. Prompts are sized
approximately at paragraph boundaries, saved once, and reused across server
configurations. Native completion timings provide the actual token counts.
"""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request


HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(url, body=None, timeout=7200):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    with HTTP.open(req, timeout=timeout) as response:
        return json.load(response)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def render(url, content):
    return request(url + "/apply-template", {
        "messages": [{"role": "user", "content": content}],
    })["prompt"]


def token_count(url, prompt):
    return len(request(url + "/tokenize", {
        "content": prompt, "add_special": True, "parse_special": True,
    })["tokens"])


def prepare_prompts(args):
    files = sorted(args.text_dir.rglob("*.md"))
    if not files:
        raise ValueError(f"No Markdown corpus found in {args.text_dir}")
    documents = [(str(p.relative_to(args.text_dir)), p.read_text()) for p in files]
    corpus_hash = hashlib.sha256(json.dumps(documents).encode()).hexdigest()
    prompts = {}
    for target in args.prompt_sizes:
        prompts[target] = []
        for variant in range(args.server_slots):
            path = args.prompts_dir / f"p{target}-v{variant}.json"
            if path.exists():
                item = json.loads(path.read_text())
                prompt = render(args.url, item["content"])
                if prompt != item["prompt"] or token_count(args.url, prompt) != item["prompt_n"]:
                    raise ValueError(f"Saved prompt/template/tokenizer mismatch: {path}")
                if item["corpus_sha256"] != corpus_hash:
                    raise ValueError(f"Corpus changed since {path} was saved")
            else:
                # Rotate documents for distinct long requests; never repeat a
                # short passage or pad with random IDs to reach a target length.
                offset = variant * len(documents) // args.server_slots
                ordered = documents[offset:] + documents[:offset]
                paragraphs = []
                for name, text in ordered:
                    paragraphs.extend(p for p in (f"Source: {name}\n\n" + text).split("\n\n") if p.strip())
                prefix = (
                    f"Read reference collection {variant + 1} below. Explain its main technical "
                    "ideas, practical recommendations, and tradeoffs in a detailed summary.\n\n"
                    "Reference material:\n\n"
                )
                # Find the nearest complete paragraph prefix. Exact lengths are
                # deliberately relaxed; the measured length is recorded below.
                cache = {}

                def candidate(n):
                    if n not in cache:
                        content = prefix + "\n\n".join(paragraphs[:n])
                        prompt = render(args.url, content)
                        cache[n] = (content, prompt, token_count(args.url, prompt))
                    return cache[n]

                if candidate(len(paragraphs))[2] < target:
                    raise ValueError(f"Corpus is too short for a {target}-token prompt")
                low, high = 1, len(paragraphs)
                while low < high:
                    middle = (low + high) // 2
                    if candidate(middle)[2] < target:
                        low = middle + 1
                    else:
                        high = middle
                choices = [candidate(low)]
                if low > 1:
                    choices.append(candidate(low - 1))
                content, prompt, count = min(choices, key=lambda x: abs(x[2] - target))
                if abs(count - target) > max(128, target * 0.05):
                    raise ValueError(f"Paragraph boundary gives {count} tokens for target {target}")
                item = {
                    "target_tokens": target, "variant": variant,
                    "corpus_sha256": corpus_hash, "source_dir": str(args.text_dir.resolve()),
                    "source_files": [name for name, _ in ordered],
                    "content": content, "prompt": prompt, "prompt_n": count,
                }
                save(path, item)
            if item["prompt_n"] + args.n_predict + 8 >= args.ctx_per_seq:
                raise ValueError(f"Insufficient sequence context for {path}")
            prompts[target].append(item)
            print(f"target={target} variant={variant} actual_prompt_tokens={item['prompt_n']}", flush=True)
    return prompts


def run_wave(args, items, concurrency, n_predict):
    barrier = threading.Barrier(concurrency)

    def one(slot):
        item = items[slot]
        body = {
            "prompt": item["prompt"], "id_slot": slot,
            "n_predict": n_predict, "temperature": 0.0,
            "ignore_eos": True, "cache_prompt": False, "stream": False,
        }
        barrier.wait(timeout=60)
        start = time.perf_counter()
        result = request(args.url + "/completion", body)
        duration = time.perf_counter() - start
        timing = result["timings"]
        if (timing["prompt_n"] != item["prompt_n"] or timing["cache_n"] != 0
                or timing["predicted_n"] != n_predict or result.get("truncated")):
            raise ValueError(f"Unexpected token counts or truncation in slot {slot}: {timing}")
        if result["id_slot"] != slot:
            raise ValueError(f"Requested slot {slot}, received {result['id_slot']}")
        if args.expect_mtp == "on" and n_predict > 8 and timing.get("draft_n", 0) == 0:
            raise ValueError("MTP was requested, but no draft tokens were reported")
        if args.expect_mtp == "off" and timing.get("draft_n", 0) > 0:
            raise ValueError("Non-MTP control unexpectedly produced draft tokens")
        return {"variant": item["variant"], "request_seconds": duration, "response": result}

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, range(concurrency)))
    wall = time.perf_counter() - start
    return {"wall_seconds": wall, "generated_tokens_per_wall_second_including_prefill":
            sum(r["response"]["timings"]["predicted_n"] for r in results) / wall,
            "requests": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18925")
    parser.add_argument("--text-dir", type=Path, default=Path("docs"))
    parser.add_argument("--prompts-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--server-slots", type=int, choices=[1, 4], required=True)
    parser.add_argument("--ctx-per-seq", type=int, default=262144)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1])
    parser.add_argument("--prompt-sizes", type=int, nargs="+", default=[1024, 8192, 32768, 65536])
    parser.add_argument("--n-predict", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--expect-mtp", choices=["on", "off"], required=True)
    args = parser.parse_args()
    args.url = args.url.rstrip("/")
    if any(n < 1 or n > args.server_slots for n in args.concurrency):
        parser.error("Concurrency must be between one and the configured server slot count")
    if args.repetitions < 1 or args.n_predict < 1 or any(n < 1 for n in args.prompt_sizes):
        parser.error("Repetitions and token lengths must be positive")
    if request(args.url + "/health").get("status") != "ok":
        raise ValueError("Server is not ready")
    slots = request(args.url + "/slots")
    if len(slots) != args.server_slots or any(s["n_ctx"] != args.ctx_per_seq for s in slots):
        raise ValueError(f"Server slot count/context does not match the benchmark: {slots}")
    expected = args.expect_mtp == "on"
    if any(bool(s["speculative"]) != expected for s in slots):
        raise ValueError("Server speculative mode does not match --expect-mtp")
    save(args.out_dir / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    prompts = prepare_prompts(args)
    summary = []
    for size in args.prompt_sizes:
        for concurrency in args.concurrency:
            directory = args.out_dir / f"p{size}-clients{concurrency}"
            save(directory / "warmup.json", run_wave(args, prompts[size], concurrency, 8))
            for repetition in range(args.repetitions):
                wave = run_wave(args, prompts[size], concurrency, args.n_predict)
                save(directory / f"run{repetition + 1}.json", wave)
                timings = [r["response"]["timings"] for r in wave["requests"]]
                row = {
                    "target_prompt_tokens": size, "concurrency": concurrency,
                    "repetition": repetition + 1,
                    "actual_prompt_tokens": [t["prompt_n"] for t in timings],
                    "decode_tokens_per_second_per_request": [t["predicted_per_second"] for t in timings],
                    "prompt_tokens_per_second_per_request": [t["prompt_per_second"] for t in timings],
                    "draft_tokens": [t.get("draft_n", 0) for t in timings],
                    "accepted_draft_tokens": [t.get("draft_n_accepted", 0) for t in timings],
                    "generated_tokens_per_wall_second_including_prefill": wave[
                        "generated_tokens_per_wall_second_including_prefill"],
                }
                summary.append(row)
                save(args.out_dir / "summary.json", summary)
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

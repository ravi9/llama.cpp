#!/usr/bin/env python3
"""Populate caches, then run the bounded in-place convolution comparison serially."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def memory_available_kib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])
    return None


def run_case(binary, model, output, mode, layout, prompt, decode, cache_only, deadline=None, probe=None,
             default_batch=False, cache_dir=None):
    name = f'{"timed" if cache_only else "prime"}-{mode}-{layout}-{prompt}'
    cache = cache_dir if cache_dir is not None else output / 'model-cache'
    spill = output / 'spill'
    cache.mkdir(exist_ok=True)
    spill.mkdir(exist_ok=True)
    env = os.environ.copy()
    if default_batch:
        env.pop('LLAMA_ARG_BATCH', None)
        env.pop('LLAMA_ARG_UBATCH', None)
    for key in ('GGML_OPENVINO_DISABLE_KV_SLICE', 'GGML_OPENVINO_RECURRENT_CACHE_INPLACE',
                'GGML_OPENVINO_SEPARATE_RECURRENT_CONV', 'GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY'):
        env.pop(key, None)
    env.update(GGML_OPENVINO_DEVICE='GPU', GGML_OPENVINO_NATIVE_SOFTPLUS='1',
               GGML_OPENVINO_REQUANT_KQUANT='q4_asym64_all', GGML_OPENVINO_ENABLE_KV_4D='1',
               GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR=str(cache), GGML_OPENVINO_SPILL_DIR=str(spill))
    if mode == 'separate':
        env['GGML_OPENVINO_SEPARATE_RECURRENT_CONV'] = '1'
    if cache_only:
        env['GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY'] = '1'
    if probe:
        env['LD_PRELOAD'] = str(probe)
    cmd = [str(binary), '-m', str(model), '-c', '131072' if layout == 'one' else '524288',
           '-npp', str(prompt), '-ntg', str(decode), '-npl', '1' if layout == 'one' else '1,4']
    if not default_batch:
        cmd.extend(['-b', '8192', '-ub', '8192'])
    cmd.extend(['--output-format', 'jsonl'])
    if layout == 'unified':
        cmd.append('--kv-unified')
    meta = dict(name=name, mode=mode, layout=layout, prompt=prompt, decode=decode, command=cmd,
                default_batch=default_batch, requested_parallel=cmd[cmd.index('-npl') + 1],
                environment={k: v for k, v in env.items() if k.startswith('GGML_OPENVINO_')},
                initial_available_kib=memory_available_kib())
    (output / f'{name}.command.json').write_text(json.dumps(meta, indent=2) + '\n')
    started = time.monotonic()
    peak_rss = 0
    minimum_available = meta['initial_available_kib']
    timed_out = False
    with (output / f'{name}.log').open('w') as log:
        proc = subprocess.Popen(cmd, env=env, cwd=output, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while proc.poll() is None:
            try:
                for line in Path(f'/proc/{proc.pid}/status').read_text().splitlines():
                    if line.startswith(('VmRSS:', 'VmHWM:')):
                        peak_rss = max(peak_rss, int(line.split()[1]))
            except FileNotFoundError:
                pass
            available = memory_available_kib()
            minimum_available = min(minimum_available, available)
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                break
            time.sleep(0.5)
    rows = []
    text = (output / f'{name}.log').read_text(errors='replace')
    for line in text.splitlines():
        if line.startswith('{"n_kv_max"'):
            rows.append(json.loads(line))
    meta.update(exit_code=proc.returncode, elapsed_seconds=time.monotonic() - started,
                peak_process_rss_kib=peak_rss, minimum_available_kib=minimum_available,
                timed_out=timed_out, rows=rows)
    (output / f'{name}.result.json').write_text(json.dumps(meta, indent=2) + '\n')
    with (output / 'results.jsonl').open('a') as file:
        file.write(json.dumps(meta) + '\n')
    print(f'{name}: rc={proc.returncode}, {len(rows)} rows, {meta["elapsed_seconds"]:.1f}s, RSS {peak_rss / 1048576:.2f} GiB', flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--phase', choices=('prime', 'timed'), required=True)
    parser.add_argument('--minutes', type=float, default=30)
    parser.add_argument('--mode', choices=('fused', 'separate'))
    parser.add_argument('--layout', choices=('one', 'split', 'unified'))
    parser.add_argument('--default-batch', action='store_true', help='Omit batch flags and use the binary defaults.')
    parser.add_argument('--prompts', type=int, nargs='+', help='Prompt lengths to run instead of the phase defaults.')
    parser.add_argument('--cache-dir', type=Path, help='Reuse an existing compiled-model cache directory.')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    modes = [args.mode] if args.mode else ['fused', 'separate']
    layouts = [args.layout] if args.layout else ['one', 'split', 'unified']
    deadline = time.monotonic() + args.minutes * 60 if args.phase == 'timed' else None
    cache_dir = args.cache_dir.resolve() if args.cache_dir else None
    for prompt in (args.prompts or ([16] if args.phase == 'prime' else [1024, 8192, 32768])):
        for layout in layouts:
            for mode in modes:
                if deadline is not None and time.monotonic() >= deadline:
                    print('Benchmark deadline reached; remaining cases are skipped.', flush=True)
                    return
                result = run_case(args.binary.resolve(), args.model.resolve(), output, mode, layout, prompt,
                                  1 if args.phase == 'prime' else 512, args.phase == 'timed', deadline,
                                  default_batch=args.default_batch, cache_dir=cache_dir)
                if result['exit_code'] != 0 and not result['timed_out']:
                    raise SystemExit(f'{result["name"]} failed; inspect its raw log before continuing.')


if __name__ == '__main__':
    main()

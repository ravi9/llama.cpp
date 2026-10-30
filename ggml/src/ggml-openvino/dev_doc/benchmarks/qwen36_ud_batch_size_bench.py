#!/usr/bin/env python3
"""Run matched default/large-batch OV and Vulkan benchmarks, resuming completed cases."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from qwen36_ud_inplace_batched import run_case
from qwen36_ud_inplace_summary import summarize_batch_comparison


def validate_result(result, binary, model, layout, prompt, decode, default_batch, backend, allow_canceled=False):
    expected = [1] if layout == 'one' else [1, 4]
    canceled = allow_canceled and result.get('canceled', False)
    if (result['exit_code'] != 0 or result['timed_out']) and not canceled:
        raise RuntimeError(f'{result["name"]} did not finish successfully; inspect its log.')
    if result['command'][0] != str(binary) or result['command'][2] != str(model):
        raise RuntimeError('Saved result uses another binary path or model.')
    if result.get('backend', 'openvino') != backend or result['default_batch'] != default_batch:
        raise RuntimeError('Saved result uses another backend or batch configuration.')
    parallel = sorted(row['pl'] for row in result['rows'])
    if (not canceled and parallel != expected) or (canceled and not set(parallel).issubset(expected)):
        raise RuntimeError(f'{result["name"]} does not contain all requested parallel rows.')
    for row in result['rows']:
        if (row['pp'], row['tg']) != (prompt, decode):
            raise RuntimeError('Saved prompt/decode counts do not match the campaign.')
        batch = (2048, 512) if default_batch else (8192, 8192)
        if (row['n_batch'], row['n_ubatch']) != batch:
            raise RuntimeError('Effective batch sizes do not match the campaign.')


def record_binary(directory, backend, binary):
    paths = [binary]
    for name in ('libllama.so', 'libggml-openvino.so', 'libggml-vulkan.so'):
        path = binary.parent / name
        if path.exists():
            paths.append(path)
    identities = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open('rb') as file:
            for data in iter(lambda: file.read(1048576), b''):
                digest.update(data)
        identities[str(path)] = digest.hexdigest()
    path = directory / f'{backend}-binary.json'
    if path.exists() and json.loads(path.read_text()) != identities:
        raise RuntimeError('Binary identities changed; use a new campaign output directory.')
    path.write_text(json.dumps(identities, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--openvino-binary', required=True, type=Path)
    parser.add_argument('--vulkan-binary', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--openvino-source', type=Path)
    parser.add_argument('--skip-canceled', action='store_true', help='Retain explicitly canceled result records and continue remaining cases.')
    cases = [f'{backend}/{batch}/{layout}/{prompt}' for backend in ('openvino', 'vulkan')
             for batch in ('default', 'large') for layout in ('one', 'split', 'unified')
             for prompt in (1024, 8192, 32768)]
    parser.add_argument('--skip-case', action='append', choices=cases, default=[],
                        help='Skip a backend/batch/layout/prompt process; the selection is saved for resumption.')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = args.model.resolve()
    metadata = output / 'metadata'
    metadata.mkdir(exist_ok=True)
    selection = metadata / 'skipped-cases.json'
    skipped = set(args.skip_case)
    if selection.exists():
        skipped.update(json.loads(selection.read_text()))
    selection.write_text(json.dumps(sorted(skipped), indent=2) + '\n')
    source = Path(__file__).resolve().parents[5]
    for name, directory in [('llamacpp', source), ('openvino', args.openvino_source)]:
        if directory:
            path = metadata / f'{name}-source.json'
            if not path.exists():
                base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=directory, text=True).strip()
                patch = subprocess.check_output(['git', 'diff'], cwd=directory)
                path.write_text(json.dumps({'directory': str(directory), 'base': base}, indent=2) + '\n')
                (metadata / f'{name}.patch').write_bytes(patch)
    cache = output / 'model-cache'
    prime = output / 'prime'
    prime.mkdir(exist_ok=True)
    ov_binary = args.openvino_binary.resolve()
    record_binary(metadata, 'openvino', ov_binary)
    summarize_batch_comparison(output, args.report)
    for layout in ('one', 'split', 'unified'):
        path = prime / f'prime-fused-{layout}-16.result.json'
        result = json.loads(path.read_text()) if path.exists() else run_case(
            ov_binary, model, prime, layout, 16, 1, False, cache_dir=cache)
        validate_result(result, ov_binary, model, layout, 16, 1, False, 'openvino')
    for backend, binary in [('openvino', ov_binary), ('vulkan', args.vulkan_binary.resolve())]:
        record_binary(metadata, backend, binary)
        for prompt in (1024, 8192, 32768):
            for layout in ('one', 'split', 'unified'):
                for batch in ('default', 'large'):
                    case = f'{backend}/{batch}/{layout}/{prompt}'
                    if case in skipped:
                        print(f'Skipped by selection: {case}', flush=True)
                        continue
                    directory = output / backend / batch
                    directory.mkdir(parents=True, exist_ok=True)
                    name = f'timed-{"fused" if backend == "openvino" else "vulkan"}-{layout}-{prompt}'
                    path = directory / f'{name}.result.json'
                    if path.exists():
                        result = json.loads(path.read_text())
                        print(f'Resuming: retained {backend}/{batch}/{name}', flush=True)
                    else:
                        print(f'Running {backend}/{batch}/{name}', flush=True)
                        result = run_case(binary, model, directory, layout, prompt, 512,
                                          backend == 'openvino', default_batch=batch == 'default',
                                          cache_dir=cache, backend=backend, phase='timed')
                    summarize_batch_comparison(output, args.report)
                    validate_result(result, binary, model, layout, prompt, 512, batch == 'default', backend,
                                    allow_canceled=args.skip_canceled)
    summarize_batch_comparison(output, args.report, finished=True)


if __name__ == '__main__':
    main()

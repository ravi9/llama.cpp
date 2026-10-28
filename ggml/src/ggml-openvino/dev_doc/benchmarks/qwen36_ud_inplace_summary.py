#!/usr/bin/env python3
"""Summarize fused benchmarks and archived convolution comparisons."""
import argparse
import json
from pathlib import Path
import re


def summarize_batch_comparison(output, report=None, finished=False):
    observations = {}
    complete = 0
    selection = output / 'metadata' / 'skipped-cases.json'
    skipped = set(json.loads(selection.read_text())) if selection.exists() else set()
    for backend in ('openvino', 'vulkan'):
        for batch in ('default', 'large'):
            directory = output / backend / batch
            for layout in ('one', 'split', 'unified'):
                for prompt in (1024, 8192, 32768):
                    name = f'timed-{"fused" if backend == "openvino" else "vulkan"}-{layout}-{prompt}'
                    path = directory / f'{name}.result.json'
                    log = directory / f'{name}.log'
                    result = json.loads(path.read_text()) if path.exists() else None
                    rows = result['rows'] if result else []
                    if result is None and log.exists():
                        rows = [json.loads(line) for line in log.read_text(errors='replace').splitlines()
                                if line.startswith('{"n_kv_max"')]
                    rows = {row['pl']: row for row in rows if row['tg'] == 512}
                    complete += len(rows)
                    state = ('Skipped' if f'{backend}/{batch}/{layout}/{prompt}' in skipped else
                             'Canceled' if result and result.get('canceled') else
                             'Failed' if result and result['exit_code'] else
                             'Incomplete' if finished else 'Running' if log.exists() else 'Pending')
                    observations[backend, batch, layout, prompt] = (rows, result, state)
    lines = [f'Completed {complete}/60 originally planned timing rows. ' + ('Selected campaign finished.' if finished else 'Campaign in progress.'), '',
             'This campaign compares default `-b 2048 -ub 512` with large `-b 8192 -ub 8192` on the same llama.cpp source. Default batch flags are omitted. OV uses fused in-place recurrent caches, GPU, native softplus, `q4_asym64_all`, `ENABLE_KV_4D=1`, enabled KV slicing and compiled-model cache-only loading. Vulkan uses the stored GGUF weights. GPU workloads run serially, with no elapsed-time cap.', '',
             'Each cell is **prefill seconds @ aggregate decode tokens/s**; four-slot rows show **one active / four active sequences**. Prompt lengths are 1024/8192/32768 tokens per sequence, with 512 generated tokens each. One-slot context is 131072 and four-slot context is 524288 total. MTP and shared-prompt processing are off; batched-bench uses random token IDs. Each timing row is one pass. Prefill includes first-use runtime shape/kernel preparation, excludes model loading and is not streaming TTFT. Decode excludes prefill.', '']
    if finished:
        missing = {}
        for (_, _, layout, _), (rows, _, state) in observations.items():
            count = sum(pl not in rows for pl in ((1,) if layout == 'one' else (1, 4)))
            if count:
                missing[state] = missing.get(state, 0) + count
        if missing:
            lines[0] += ' Missing rows: ' + ', '.join(f'{count} {state.lower()}' for state, count in missing.items()) + '.'
    labels = {'one': '`-npl 1`', 'split': '`-npl 1,4` split', 'unified': '`-npl 1,4 --kv-unified`'}
    for batch, settings in [('default', '2048 / 512'), ('large', '8192 / 8192')]:
        lines.extend([f'### {batch.capitalize()} batch: `-b / -ub = {settings}`', '',
                      '| Backend/configuration | 1K prompt | 8K prompt | 32K prompt |',
                      '|---|---:|---:|---:|'])
        for backend in ('openvino', 'vulkan'):
            for layout in labels:
                cells = []
                for prompt in (1024, 8192, 32768):
                    rows, _, state = observations[backend, batch, layout, prompt]
                    cells.append(' / '.join(f'{rows[pl]["t_pp"]:.3f} @ {rows[pl]["speed_tg"]:.2f}'
                                            if pl in rows else state
                                            for pl in ((1,) if layout == 'one' else (1, 4))))
                lines.append(f'| {"OV fused" if backend == "openvino" else "Vulkan"} {labels[layout]} | ' + ' | '.join(cells) + ' |')
        lines.append('')
        for backend in ('openvino', 'vulkan'):
            results = [result for (name, size, _, _), (_, result, _) in observations.items()
                       if name == backend and size == batch and result]
            if results:
                rss = [result['peak_process_rss_kib'] / 1048576 for result in results]
                available = [result['minimum_available_kib'] / 1048576 for result in results
                             if result['minimum_available_kib'] is not None]
                memory = f'{"OpenVINO" if backend == "openvino" else "Vulkan"} recorded-process peak RSS: {min(rss):.2f}-{max(rss):.2f} GiB.'
                if available:
                    memory += f' Lowest sampled system MemAvailable: {min(available):.2f} GiB.'
                lines.extend([memory, ''])
    for (backend, batch, _, _), (_, result, _) in observations.items():
        if result and result.get('canceled'):
            lines.extend([f'Canceled `{backend}/{batch}/{result["name"]}`: {result["cancel_reason"]} '
                          'Completed timing rows and sampled memory are retained; missing timings are not extrapolated.', ''])
    lines.extend(['Process RSS does not include every GPU/system allocation. Per-process command, timing and memory records are stored with the raw results.', '',
                  f'Raw artifacts: `{output}`. Source patches, base commits and binary identities are recorded in the campaign metadata.', ''])
    section = '\n'.join(lines)
    (output / 'matrix.md').write_text(section + '\n')
    if report:
        text = report.read_text()
        pattern = r'(?s)(<!-- batch-comparison:start -->\n).*?(<!-- batch-comparison:end -->)'
        text, count = re.subn(pattern, lambda match: match[1] + section + '\n' + match[2], text)
        if count != 1:
            raise SystemExit('Report must contain exactly one batch-comparison marker pair.')
        report.write_text(text)
    print(f'Updated batch-size comparison: {complete}/60 rows.', flush=True)


def summarize_continuation(args):
    cases = [('fused', 'unified', (1, 4)), ('separate', 'unified', (1, 4))]
    observations = []
    complete = 0
    for mode, layout, parallel in cases:
        name = f'timed-{mode}-{layout}-32768'
        path = args.output / f'{name}.result.json'
        result = json.loads(path.read_text()) if path.exists() else None
        rows = result['rows'] if result else []
        log = args.output / f'{name}.log'
        if result is None and log.exists():
            rows = [json.loads(line) for line in log.read_text(errors='replace').splitlines()
                    if line.startswith('{"n_kv_max"')]
        by_parallel = {row['pl']: row for row in rows}
        for pl in parallel:
            row = by_parallel.get(pl)
            complete += row is not None
            status = 'Complete' if row else ('Incomplete' if args.finished else ('Running' if log.exists() else 'Pending'))
            if result and result['exit_code'] != 0 and row is None:
                status = f'Failed ({result["exit_code"]})'
            observations.append((mode, layout, pl, row, result, status))
    status = 'Finished' if args.finished else 'In progress'
    lines = [f'{status}: {complete}/4 previously unfinished OV unified-KV 32K rows measured with default batch sizes. The three split-KV continuation rows were skipped at user request.', '',
             'These are separate continuation measurements with `-b 2048 -ub 512` defaults; neither flag is passed. They do not replace the historical `-b 8192 -ub 8192` rows. Each sequence has 32,768 prompt tokens and 512 decode tokens; four slots and 524,288 total context tokens are allocated in every case. MTP is off. All rates below are aggregate; prefill seconds exclude startup and are not streaming TTFT.', '',
             '| Convolution | KV | Active sequences | Prefill seconds | Prefill t/s | Decode t/s | Peak process RSS GiB | Status |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for mode, layout, pl, row, result, state in observations:
        values = [f'{row[key]:.{digits}f}' if row else '-' for key, digits in [('t_pp', 3), ('speed_pp', 2), ('speed_tg', 2)]]
        rss = f'{result["peak_process_rss_kib"] / 1048576:.2f}' if result else '-'
        lines.append(f'| {mode.capitalize()} | {layout.capitalize()} | {pl} | ' + ' | '.join(values) + f' | {rss} | {state} |')
    lines.extend(['', 'Peak process RSS is shared by both rows from a process. It does not include all GPU/system allocations.', ''])
    section = '\n'.join(lines)
    (args.output / 'matrix.md').write_text(section + '\n')
    if args.report:
        text = args.report.read_text()
        pattern = r'(?s)(<!-- default-continuation:start -->\n).*?(<!-- default-continuation:end -->)'
        text, count = re.subn(pattern, lambda match: match[1] + section + '\n' + match[2], text)
        if count != 1:
            raise SystemExit('Report must contain exactly one default-continuation marker pair.')
        args.report.write_text(text)
    print(f'Updated default-batch continuation: {complete}/4 rows, {status.lower()}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--finished', action='store_true')
    parser.add_argument('--batch-comparison', action='store_true', help='Summarize the matched default/large OV and Vulkan campaign.')
    parser.add_argument('--default-continuation', action='store_true', help='Summarize the four archived default-batch OV unified-KV 32K continuation rows.')
    args = parser.parse_args()
    if args.batch_comparison:
        summarize_batch_comparison(args.output, args.report, args.finished)
        return
    if args.default_continuation:
        summarize_continuation(args)
        return
    results = [json.loads(path.read_text()) for path in args.output.glob('timed-*.result.json')]
    results = [result for result in results if result['prompt'] in (1024, 8192, 32768)]
    lookup = {(r['mode'], r['layout'], r['prompt']): r for r in results}
    modes = ['fused']
    if any(result['mode'] == 'separate' for result in results):
        modes.append('separate')
    total = 15 * len(modes)
    complete = sum(len(result['rows']) for result in results)
    status = 'Finished' if args.finished else 'In progress'
    lines = [f'{status}: {complete}/{total} timing rows retained. A row is retained only after its prefill and all 512 decode steps finish.', '',
             'Decode throughput below excludes prefill. Values are aggregate tokens/s, using `speed_tg = pl * tg / t_tg`. In four-slot configurations, `A / B` means one active sequence / four active sequences; divide B by four for the simultaneous per-sequence rate.', '']
    labels = {'one': '`-npl 1`', 'split': '`-npl 1,4` split', 'unified': '`-npl 1,4 --kv-unified`'}

    def table(metric, fmt):
        lines.extend(['| OV convolution / KV configuration | 1K prompt | 8K prompt | 32K prompt |', '|---|---:|---:|---:|'])
        for mode in modes:
            for layout in labels:
                cells = []
                for prompt in (1024, 8192, 32768):
                    result = lookup.get((mode, layout, prompt))
                    rows = {row['pl']: row for row in result['rows']} if result else {}
                    missing = 'Skipped' if args.finished or result else 'Pending'
                    cells.append(' / '.join(format(rows[pl][metric], fmt) if pl in rows else missing
                                            for pl in ((1,) if layout == 'one' else (1, 4))))
                lines.append(f'| {mode.capitalize()} conv, {labels[layout]} | ' + ' | '.join(cells) + ' |')
        lines.append('')

    table('speed_tg', '.2f')
    lines.extend(['Prefill duration in seconds (not streaming TTFT). `t_pp` ends after all active prompts finish and synchronization completes. It is a prefill-latency proxy, and includes first-use runtime shape/kernel preparation; it does not include process startup or loading compiled blobs.', ''])
    table('t_pp', '.3f')
    lines.extend(['Aggregate prefill throughput in tokens/s. Prompt tokens are independently evaluated for each sequence; no shared-prompt optimization is enabled.', ''])
    table('speed_pp', '.2f')
    lines.extend(['Peak process RSS in GiB for each process (both active-sequence rows together). This is a host process measurement, not dedicated GPU memory or total system consumption.', '',
                  '| OV convolution / KV configuration | 1K prompt | 8K prompt | 32K prompt |', '|---|---:|---:|---:|'])
    for mode in modes:
        for layout in labels:
            cells = []
            for prompt in (1024, 8192, 32768):
                result = lookup.get((mode, layout, prompt))
                cells.append(f'{result["peak_process_rss_kib"] / 1048576:.2f}' if result else ('Skipped' if args.finished else 'Pending'))
            lines.append(f'| {mode.capitalize()} conv, {labels[layout]} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    for result in results:
        if result['timed_out']:
            lines.extend([f'The 30-minute limit interrupted `{result["name"]}` after {result["elapsed_seconds"]:.1f} seconds; {len(result["rows"])} completed row(s) from that process are retained. Remaining cells are marked Skipped.', ''])
    section = '\n'.join(lines) + '\n'
    (args.output / 'matrix.md').write_text(section)
    if args.report:
        text = args.report.read_text()
        pattern = r'(?s)(<!-- inplace-results:start -->\n).*?(<!-- inplace-results:end -->)'
        text, count = re.subn(pattern, lambda match: match[1] + section + '\n' + match[2], text)
        if count != 1:
            raise SystemExit('Report must contain exactly one inplace-results marker pair.')
        args.report.write_text(text)
    print(f'Updated matrix: {complete}/{total} rows, {status.lower()}.')


if __name__ == '__main__':
    main()

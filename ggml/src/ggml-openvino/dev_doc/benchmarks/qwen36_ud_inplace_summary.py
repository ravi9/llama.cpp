#!/usr/bin/env python3
"""Summarize completed rows from qwen36_ud_inplace_batched.py."""
import argparse
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--finished', action='store_true')
    args = parser.parse_args()
    results = [json.loads(path.read_text()) for path in args.output.glob('timed-*.result.json')]
    results = [result for result in results if result['prompt'] in (1024, 8192, 32768)]
    lookup = {(r['mode'], r['layout'], r['prompt']): r for r in results}
    complete = sum(len(result['rows']) for result in results)
    status = 'Finished' if args.finished else 'In progress'
    lines = [f'{status}: {complete}/30 timing rows retained. A row is retained only after its prefill and all 512 decode steps finish.', '',
             'Decode throughput below excludes prefill. Values are aggregate tokens/s, using `speed_tg = pl * tg / t_tg`. In four-slot configurations, `A / B` means one active sequence / four active sequences; divide B by four for the simultaneous per-sequence rate.', '']
    labels = {'one': '`-npl 1`', 'split': '`-npl 1,4` split', 'unified': '`-npl 1,4 --kv-unified`'}

    def table(metric, fmt):
        lines.extend(['| OV convolution / KV configuration | 1K prompt | 8K prompt | 32K prompt |', '|---|---:|---:|---:|'])
        for mode in ('fused', 'separate'):
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
    for mode in ('fused', 'separate'):
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
    print(f'Updated matrix: {complete}/30 rows, {status.lower()}.')


if __name__ == '__main__':
    main()

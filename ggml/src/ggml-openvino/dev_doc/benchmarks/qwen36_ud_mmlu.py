"""Matched native server-bench MMLU runs; server slots and traffic concurrency are independent."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import logging
import pyarrow.parquet as pq
import requests
from concurrent.futures import ThreadPoolExecutor
import itertools
import threading
import time

if len(sys.argv) != 5:
    raise SystemExit('Usage: qwen36_ud_mmlu.py ov|vulkan split|unified off|mtp 1|4')

root = Path(__file__).resolve().parents[5]
backend, layout, mode, concurrency_text = sys.argv[1:5]
concurrency = int(concurrency_text)
assert backend in ('ov', 'vulkan') and layout in ('split', 'unified')
assert mode in ('off', 'mtp') and concurrency in (1, 4)
base = Path(os.environ.get('UD_BENCH_DIR', str(root / 'tmp/ud-backend-matrix-20261023')))
label = f'{backend}-{layout}-{mode}-c{concurrency}'
out = base / os.environ.get('UD_RUN_LABEL', label)
out.mkdir(parents=True, exist_ok=True)
(out / 'complete.json').unlink(missing_ok=True)
for response in (out / 'responses').glob('request-*.json'):
    response.unlink()
spec = importlib.util.spec_from_file_location('server_bench', root / 'scripts/server-bench.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
logging.getLogger('httpx').setLevel(logging.WARNING)
parquet = Path(os.environ.get('UD_MMLU_PARQUET', str(root / 'tmp/mmlu-dataset/all/test-00000-of-00001.parquet')))
questions = pq.read_table(parquet, columns=['question'])['question'].to_pylist()
b.get_prompts_text = lambda name, n: questions[:n] if n >= 0 else questions
original_server = b.get_server

def get_server(path, log):
    os.environ['LLAMA_ARG_N_PARALLEL'] = '4'
    try:
        server = original_server(path, log)
        if os.environ.get('UD_PRIME_SUBSETS') == '1':
            address = server['address']
            prompt = requests.post(address + '/apply-template', json={'messages': [{'role': 'user', 'content': questions[0]}]}).json()['prompt']
            def prime_slot(slot):
                response = requests.post(address + '/completion', json={'prompt': prompt, 'id_slot': slot, 'n_predict': 16 + slot, 'temperature': 0, 'ignore_eos': True, 'cache_prompt': False}, timeout=600)
                response.raise_for_status()
                assert response.json()['timings']['predicted_n'] == 16 + slot
            for count in (1, 2, 3, 4):
                for subset in itertools.combinations(range(4), count):
                    print('Priming active slots', subset, flush=True)
                    with ThreadPoolExecutor(max_workers=count) as pool:
                        list(pool.map(prime_slot, subset))
        if backend == 'ov' and os.environ.get('UD_CACHE_ONLY', '1') == '1':
            def reject_missing_signature():
                process = server['process']
                while process.poll() is None:
                    if Path(log).exists() and 'missing or incompatible compiled model' in Path(log).read_text():
                        process.kill()
                        return
                    time.sleep(1)
            threading.Thread(target=reject_missing_signature, daemon=True).start()
        return server
    except BaseException:
        if 'server' in locals():
            server['process'].terminate()
            server['process'].wait()
        raise
    finally:
        os.environ['LLAMA_ARG_N_PARALLEL'] = str(concurrency)

b.get_server = get_server
for key in list(os.environ):
    if key.startswith(('LLAMA_ARG_', 'GGML_OPENVINO_', 'GGML_VULKAN_')) or key == 'LD_PRELOAD':
        os.environ.pop(key)
os.environ.update({
    'LLAMA_ARG_MODEL': os.environ.get('UD_MODEL', str(Path.home() / 'models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf')),
    'LLAMA_ARG_N_PARALLEL': str(concurrency),
    'LLAMA_ARG_CTX_SIZE': os.environ.get('UD_TEST_CTX', '524288'),
    'LLAMA_ARG_KV_UNIFIED': '1' if layout == 'unified' else '0',
    'LLAMA_ARG_PORT': '18931',
    'LLAMA_ARG_TEMPERATURE': '0',
})
if os.environ.get('UD_FORCE_GPU') == '1':
    os.environ['LLAMA_ARG_N_GPU_LAYERS'] = '99'
    os.environ['LLAMA_ARG_FIT'] = '0'
if os.environ.get('UD_LOG_VERBOSITY'):
    os.environ['LLAMA_ARG_LOG_VERBOSITY'] = os.environ['UD_LOG_VERBOSITY']
if backend == 'ov':
    binary = Path(os.environ.get('UD_OV_BIN_DIR', str(root / 'build/MtpOVPerf/bin'))) / 'llama-server'
    os.environ.update({
        'GGML_OPENVINO_DEVICE': 'GPU',
        'GGML_OPENVINO_NATIVE_SOFTPLUS': '1',
        'GGML_OPENVINO_REQUANT_KQUANT': 'q4_asym64_all',
        'GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR': os.environ.get('UD_COMPILED_CACHE', str(root / 'tmp/mtp-regression-20261022/rotation-cache')),
        'GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY': os.environ.get('UD_CACHE_ONLY', '1'),
    })
else:
    binary = Path(os.environ.get('UD_VULKAN_BIN_DIR', str(root / 'build/vulkan/bin'))) / 'llama-server'
if mode == 'mtp':
    os.environ['LLAMA_ARG_SPEC_TYPE'] = 'draft-mtp'
    os.environ['LLAMA_ARG_SPEC_DRAFT_N_MAX'] = '2'
n_prompts = int(os.environ.get('UD_N_PROMPTS', '8'))
n_predict = int(os.environ.get('UD_N_PREDICT', '512'))
(out / 'effective-env.json').write_text(json.dumps({k:v for k,v in os.environ.items() if k.startswith(('LLAMA_ARG_', 'GGML_OPENVINO_'))}, indent=2))
(out / 'provenance.json').write_text(json.dumps({
    'dataset': f'cais/mmlu all test, first {n_prompts} question texts',
    'llama_commit': os.environ.get('UD_LLAMA_COMMIT', subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'], text=True).strip()),
    'harness_commit': subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'], text=True).strip(),
    'openvino_commit': os.environ.get('UD_OPENVINO_COMMIT', 'unspecified'),
    'server_slots': 4, 'request_concurrency': concurrency, 'binary': str(binary),
    'backend': backend, 'layout': layout, 'mode': mode, 'n_predict': n_predict,
    'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
    'dataset_parquet': str(parquet),
}, indent=2))
os.chdir(out)
b.benchmark(str(binary), str(out / 'server.log'), None, label, 'mmlu', n_prompts,
            n_predict, n_predict, 0, str(out / 'responses'))
responses = [json.loads(p.read_text()) for p in sorted((out / 'responses').glob('request-*.json'))]
assert len(responses) == n_prompts, len(responses)
assert all(r['timings']['predicted_n'] == n_predict for r in responses)
assert all(r['events'][-1].get('stop') and r['events'][-1]['tokens_predicted'] == n_predict for r in responses)
assert all(len(r['token_arrival_times']) == n_predict for r in responses)
server_log = (out / 'server.log').read_text()
assert 'missing or incompatible compiled model' not in server_log, 'Missing draft signatures invalidate MTP timing'
assert 'returned -3' not in server_log, 'Draft computation failed'
(out / 'complete.json').write_text(json.dumps({'requests': n_prompts, 'generated_tokens': n_prompts*n_predict}, indent=2))

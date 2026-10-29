# OpenVINO Qwen development documentation

The default supported Qwen3.5/Qwen3.6 GPU path is **dynamic stateless execution with in-place fused recurrent ops and stable MTP sampling**. Unified KV is the recommended multi-slot layout because it enables native in-place attention KV updates. No new inference switch is needed for the recurrent/sampling default.

- [Backend setup and cache usage](../README.md): runtime requirements, environment settings and restrictions.
- [Implementation and validation](OPENVINO-RECURRENT-CACHE.md): buffer ownership, fusion, rollback, fallback and regression commands.
- [Multislot MTP graph](MTP-SAMPLING-GRAPH.md): why active-slot variants occurred and how stable translation works.
- [Benchmark results](QWEN36-UD-MULTISLOT-BENCHMARK.md): batch-size matrices, current MTP results and labeled historical references.
- [Optimization history](OPENVINO-RECURRENT-FUSION-PERF.md): retained evidence behind the design decisions, with source/artifact references.
- `benchmarks/`: reproducible runners and summary tools. Use `--help` and a fresh output directory for a new campaign.

Working checkouts are `/home/zijun/llama.cpp` and `/home/zijun/openvino`. Default local build paths are `build/MtpOVPerf` and `/home/zijun/openvino-build-mtp`. Historical logs keep their original worktree paths as provenance; those worktrees need not remain present.

Superseded per-step plans, old operator contracts, interrupted-run diaries and duplicate result matrices were consolidated on 2026-10-29. Full earlier reports remain in Git at `71c0bbc4d`; the final uncommitted measurements and pre-cleanup documents are also saved locally under `tmp/worktree-cleanup-20261029/dev_doc-before`. Raw benchmark caches, profiles, graph examples and the user's work note remain under `tmp/` and were not deleted.

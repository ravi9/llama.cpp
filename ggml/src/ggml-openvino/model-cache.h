#pragma once

// Frontend-level compiled-model cache (GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR).
//
// The OpenVINO plugin's own ov::cache_dir caches the compiled blob keyed by the
// *OV model*, but producing that model still runs the full frontend every time:
// weight requantization (incl. the large token_embd F32 transient) and the
// ggml->OV graph conversion. This cache keys off a fingerprint computed directly
// from the ggml cgraph, so a hit skips requant + convert + compile entirely and
// instead imports a previously exported CompiledModel blob.
//
// Opt-in and independent from GGML_OPENVINO_CACHE_DIR. Default off.

#include "ggml.h"

#include <cstdint>
#include <map>
#include <openvino/runtime/core.hpp>
#include <string>
#include <vector>

// Returns the compiled-model cache directory from GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR,
// or empty if unset/disabled. When empty, callers must not use the cache.
std::string ggml_openvino_model_cache_dir();

// Compute a stable 64-bit fingerprint identifying the model+config that a cgraph
// would compile to. Combines graph topology, a sampled hash of every weight
// tensor (name/shape/dtype + bounded byte sample), and the config that changes
// the produced blob (device, flash-attention, rope params, the compile-memory
// flags, stateful, and the OpenVINO version). `device` is the resolved device
// string; `fa` is the flash-attention flag; `rope_params`/`rope_len` cover the
// model's rope configuration; `extra_cfg` folds in any other blob-affecting bits.
uint64_t ggml_openvino_model_fingerprint(const ggml_cgraph * cgraph,
                                         const std::string & device,
                                         bool fa,
                                         const int32_t * rope_params,
                                         int rope_len,
                                         uint64_t extra_cfg);

// Fold a value into a fingerprint seed, for callers that must add blob-affecting
// numbers to `extra_cfg` (the static path folds in its baked-in shapes this way).
uint64_t ggml_openvino_fingerprint_mix(uint64_t h, int64_t v);

// Path to a cache file for a fingerprint (<dir>/<hex><ext>), e.g. ".bin" for the
// shared weights file or ".xml" for the IR used to produce a weightless blob.
std::string ggml_openvino_model_cache_file_path(const std::string & dir, uint64_t fingerprint, const char * ext);

// Path to the compiled-blob file for a fingerprint (<dir>/<hex>.blob).
std::string ggml_openvino_model_cache_blob_path(const std::string & dir, uint64_t fingerprint);

// Path to the sidecar manifest (<dir>/<hex>.manifest) holding the per-weight
// fingerprints, used to re-verify a hit before trusting the blob.
std::string ggml_openvino_model_cache_manifest_path(const std::string & dir, uint64_t fingerprint);

// Write/read the manifest. Layout: a "fingerprint"/"ov_version" header, then the
// Parameter and Result name lists ("inputs <n>" / "outputs <n>" followed by one
// name per line), then one "name ne0 ne1 ne2 ne3 type sample_hash" line per
// weight. Returns false on I/O error.
//
// The name lists exist because a blob does not carry usable ones: the NPU plugin
// regenerates Result friendly names on import ("Result_1042"), and the names the
// Parameters do keep embed a cgraph hash-set slot that depends on tensor addresses
// and so differs in the next process. The port *order* survives, so recording the
// names here (suffix-stripped, see ggml_openvino_resolve_cached_names) is what lets
// an importer map each port back to a ggml tensor.
bool ggml_openvino_model_cache_write_manifest(const std::string & path,
                                              const ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              const std::vector<std::string> & input_names = {},
                                              const std::vector<std::string> & output_names = {});

// Read back the Parameter/Result name lists. Returns false if the manifest is
// unreadable or records no names (e.g. written by a path that does not need them).
bool ggml_openvino_model_cache_read_io_names(const std::string & path,
                                             std::vector<std::string> & input_names,
                                             std::vector<std::string> & output_names);

// Verify that the cgraph's weights still match the stored manifest (guards the
// sampled-hash collision risk: a blob is only trusted if every weight's
// name/shape/type/sample-hash matches what was cached). Returns true on match.
bool ggml_openvino_model_cache_verify_manifest(const std::string & path,
                                               const ggml_cgraph * cgraph,
                                               uint64_t fingerprint);

// Hash the config that changes the produced blob but is not visible in the cgraph,
// for ggml_openvino_model_fingerprint()'s `extra_cfg`.
uint64_t ggml_openvino_model_cache_extra_cfg(const std::string & device,
                                             bool stateful,
                                             bool is_static,
                                             int prefill_chunk_size);

// Does a cache file exist and open for reading?
bool ggml_openvino_cache_file_exists(const std::string & path);

// Is a usable cache entry on disk for this fingerprint (blob present, manifest still
// matching the cgraph's weights)?
bool ggml_openvino_model_cache_ready(const std::string & blob_path,
                                     const std::string & manifest_path,
                                     const ggml_cgraph * cgraph,
                                     uint64_t fingerprint);

// Hash of the options selecting how weights are requantized (the token-embedding quant
// choices). Folded into both fingerprints below, since they change the cached bytes without
// changing any ggml tensor.
uint64_t ggml_openvino_requant_cfg();

// Fingerprint of just the weight set, with no graph topology or geometry folded in, so that
// every graph built over the same weights names the same canonical bin.
uint64_t ggml_openvino_weights_fingerprint(const ggml_cgraph * cgraph);

// Round-trip the weights through a weights-only IR so their Constants carry the offsets a
// weightless blob needs, independent of any graph. `weights` is replaced in place by the
// round-tripped nodes, which can then be built into any graph. Returns false on failure,
// leaving `weights` untouched.
bool ggml_openvino_canonical_weights(ov::Core & core,
                                     const std::string & xml_path,
                                     const std::string & bin_path,
                                     std::map<std::string, std::shared_ptr<ov::Node>> & weights);

// Strip / re-attach the per-process "#<slot>" suffix on graph-tensor names, so the names
// recorded in a manifest can be mapped back onto a later run's tensors.
std::string ggml_openvino_strip_name_suffix(const std::string & name);
bool ggml_openvino_resolve_cached_names(const std::vector<std::string> & cached,
                                        const std::vector<std::string> & live,
                                        std::vector<std::string> & out);

// Path OpenVINO's cache manager is expected to use for a CACHE_BLOB_ID under CACHE_PATH.
// Only ever used as a presence test (see the .cpp for why a wrong guess is harmless).
std::string ggml_openvino_model_cache_ov_blob_path(const std::string & dir, uint64_t fingerprint);

// Load the blob stored under `cfg`'s CACHE_BLOB_ID/CACHE_PATH. The caller must already have
// verified the manifest (via ggml_openvino_model_cache_ready): this performs no check of its
// own. Returns false on miss or import error.
bool ggml_openvino_model_cache_load(ov::Core & core,
                                    const std::string & device,
                                    const ov::AnyMap & cfg,
                                    ov::CompiledModel & out);

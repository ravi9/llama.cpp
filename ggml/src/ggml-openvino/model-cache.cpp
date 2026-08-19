#include "model-cache.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-openvino-extra.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <cctype>
#include <map>
#include <mutex>
#include <set>
#include <openvino/core/graph_util.hpp>
#include <openvino/core/version.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/result.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#if defined(_WIN32)
#    include <direct.h>
#endif

namespace {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
inline uint64_t fnv1a(uint64_t h, const void * data, size_t n) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

inline uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    return fnv1a(h, &v, sizeof(v));
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

// Bytes sampled from each end of a weight tensor for the sampled hash. The whole
// model is never hashed (that would cost seconds every run); instead we sample a
// bounded window from the head and tail of each weight's bytes. The manifest
// re-verify (same sample) guards the residual collision risk.
constexpr size_t WEIGHT_SAMPLE_BYTES = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
bool is_weight_src(const ggml_tensor * src) {
    if (src == nullptr || src->view_src != nullptr || src->buffer == nullptr) {
        return false;
    }
    return src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Per-weight sampled fingerprint: identity (name/shape/type) + a bounded byte
// sample. Returns FNV offset basis if data is unavailable (kept deterministic).
uint64_t weight_fingerprint(const ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, strlen(t->name));
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    const size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    if (t->data != nullptr && nbytes > 0) {
        const size_t head = nbytes < WEIGHT_SAMPLE_BYTES ? nbytes : WEIGHT_SAMPLE_BYTES;
        h = fnv1a(h, t->data, head);
        if (nbytes > WEIGHT_SAMPLE_BYTES) {
            const size_t tail = nbytes < 2 * WEIGHT_SAMPLE_BYTES ? nbytes - WEIGHT_SAMPLE_BYTES : WEIGHT_SAMPLE_BYTES;
            h = fnv1a(h, static_cast<const uint8_t *>(t->data) + (nbytes - tail), tail);
        }
    }
    return h;
}

// Identity of a weight, excluding its bytes. Used to detect a recycled tensor
// address in the memo below.
uint64_t weight_identity(const ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, strlen(t->name));
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    return h;
}

std::mutex g_weight_fp_mutex;
// tensor -> (identity, sampled fingerprint)
std::unordered_map<const ggml_tensor *, std::pair<uint64_t, uint64_t>> g_weight_fp_cache;

// Sample each weight's bytes once per process. This is what lets
// GGML_OPENVINO_RELEASE_WEIGHTS free the weight buffers after a cache hit: a later
// graph's fingerprint or manifest verify would otherwise re-read t->data, which by
// then is gone. Keyed by tensor pointer, with the identity folded in so a recycled
// address cannot return another weight's hash.
uint64_t weight_fingerprint_cached(const ggml_tensor * t) {
    const uint64_t id = weight_identity(t);
    std::lock_guard<std::mutex> lock(g_weight_fp_mutex);
    auto it = g_weight_fp_cache.find(t);
    if (it != g_weight_fp_cache.end() && it->second.first == id) {
        return it->second.second;
    }
    const uint64_t fp = weight_fingerprint(t);
    g_weight_fp_cache[t] = std::make_pair(id, fp);
    return fp;
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
template <typename F>
void for_each_weight(const ggml_cgraph * cgraph, F && fn) {
    std::vector<const ggml_tensor *> seen;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = node->src[s];
            if (!is_weight_src(src)) {
                continue;
            }
            bool dup = false;
            for (const auto * p : seen) {
                if (p == src) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                continue;
            }
            seen.push_back(src);
            fn(src);
        }
    }
}

std::string ov_version_string() {
    const ov::Version v = ov::get_openvino_version();
    return std::string(v.buildNumber ? v.buildNumber : "unknown");
}

std::string hex64(uint64_t v) {
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
    return std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
bool make_dir(const std::string & path) {
#if defined(_WIN32)
    int rc = _mkdir(path.c_str());
#else
    int rc = ::mkdir(path.c_str(), 0755);
#endif
    if (rc == 0 || errno == EEXIST) {
        return true;
    }
    return false;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
bool make_dirs(const std::string & path) {
    if (path.empty()) {
        return false;
    }
    std::string acc;
    for (size_t i = 0; i < path.size(); ++i) {
        const char c = path[i];
        acc.push_back(c);
        const bool sep = (c == '/'
#if defined(_WIN32)
                          || c == '\\'
#endif
        );
        // Create each intermediate component (skip a leading "/" root).
        if (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            if (!make_dir(component)) {
                return false;
            }
        }
    }
    return make_dir(path);
}

}  // namespace

std::string ggml_openvino_model_cache_dir() {
    const char * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    if (!dir || strlen(dir) == 0) {
        return std::string();
    }
    std::string path(dir);
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    if (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        return std::string();
    }
    return path;
}

uint64_t ggml_openvino_model_fingerprint(const ggml_cgraph * cgraph,
                                         const std::string & device,
                                         bool fa,
                                         const int32_t * rope_params,
                                         int rope_len,
                                         uint64_t extra_cfg) {
    uint64_t h = FNV_OFFSET;

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(node->op));
        h = fnv1a(h, node->name, strlen(node->name));
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](const ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint_cached(t)); });

    // Config that changes the produced blob.
    h = fnv1a(h, device.data(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    if (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, sizeof(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    const std::string ver = ov_version_string();
    h = fnv1a(h, ver.data(), ver.size());

    return h;
}

uint64_t ggml_openvino_fingerprint_mix(uint64_t h, int64_t v) {
    return fnv1a_u64(h, static_cast<uint64_t>(v));
}

std::string ggml_openvino_model_cache_file_path(const std::string & dir, uint64_t fingerprint, const char * ext) {
    return dir + "/" + hex64(fingerprint) + ext;
}

std::string ggml_openvino_model_cache_blob_path(const std::string & dir, uint64_t fingerprint) {
    return dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_path(const std::string & dir, uint64_t fingerprint) {
    return dir + "/" + hex64(fingerprint) + ".manifest";
}

bool ggml_openvino_model_cache_write_manifest(const std::string & path,
                                              const ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              const std::vector<std::string> & input_names,
                                              const std::vector<std::string> & output_names) {
    std::ofstream f(path, std::ios::trunc);
    if (!f.is_open()) {
        return false;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_version_string() << "\n";
    f << "inputs " << input_names.size() << "\n";
    for (const auto & n : input_names) {
        f << n << "\n";
    }
    f << "outputs " << output_names.size() << "\n";
    for (const auto & n : output_names) {
        f << n << "\n";
    }
    for_each_weight(cgraph, [&](const ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<int>(t->type) << " " << hex64(weight_fingerprint_cached(t)) << "\n";
    });
    return f.good();
}

namespace {

// Consume the header and the two name lists, leaving `f` at the first weight line.
// Fills the name vectors. Returns false if the header does not match.
bool read_manifest_header(std::ifstream & f,
                          uint64_t fingerprint,
                          std::vector<std::string> & input_names,
                          std::vector<std::string> & output_names) {
    auto header = [&f](const char * expect_tag, std::string & val) {
        std::string line;
        if (!std::getline(f, line)) {
            return false;
        }
        const size_t sp = line.find(' ');
        if (sp == std::string::npos || line.compare(0, sp, expect_tag) != 0) {
            return false;
        }
        val = line.substr(sp + 1);
        return true;
    };
    auto name_list = [&f, &header](const char * expect_tag, std::vector<std::string> & out) {
        std::string val;
        if (!header(expect_tag, val)) {
            return false;
        }
        const long n = std::strtol(val.c_str(), nullptr, 10);
        if (n < 0) {
            return false;
        }
        out.clear();
        out.reserve(static_cast<size_t>(n));
        for (long i = 0; i < n; ++i) {
            std::string line;
            if (!std::getline(f, line)) {
                return false;
            }
            out.push_back(line);
        }
        return true;
    };

    std::string val;
    if (!header("fingerprint", val) || val != hex64(fingerprint)) {
        return false;
    }
    if (!header("ov_version", val) || val != ov_version_string()) {
        return false;
    }
    return name_list("inputs", input_names) && name_list("outputs", output_names);
}

}  // namespace

bool ggml_openvino_model_cache_read_io_names(const std::string & path,
                                             std::vector<std::string> & input_names,
                                             std::vector<std::string> & output_names) {
    std::ifstream f(path);
    if (!f.is_open()) {
        return false;
    }
    // The fingerprint is already known to match (the caller verifies the manifest
    // first), so re-derive it from the file rather than requiring it as an argument.
    std::string line;
    if (!std::getline(f, line) || line.rfind("fingerprint ", 0) != 0) {
        return false;
    }
    f.seekg(0);
    const uint64_t fp = std::strtoull(line.substr(strlen("fingerprint ")).c_str(), nullptr, 16);
    if (!read_manifest_header(f, fp, input_names, output_names)) {
        return false;
    }
    return !input_names.empty() && !output_names.empty();
}

bool ggml_openvino_model_cache_verify_manifest(const std::string & path,
                                               const ggml_cgraph * cgraph,
                                               uint64_t fingerprint) {
    std::ifstream f(path);
    if (!f.is_open()) {
        return false;
    }
    std::vector<std::string> input_names, output_names;
    if (!read_manifest_header(f, fingerprint, input_names, output_names)) {
        return false;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_weight(cgraph, [&](const ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<int>(t->type)) + " " + hex64(weight_fingerprint_cached(t)));
    });

    size_t idx = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) {
            continue;
        }
        if (idx >= expected.size() || line != expected[idx]) {
            return false;
        }
        ++idx;
    }
    return idx == expected.size();
}

static uint64_t ggml_openvino_hash_env_str(uint64_t seed, const char * name) {
    const char * v = ggml_openvino_getenv_str(name);
    if (v == nullptr) {
        return seed * 131 + 0xFFu;
    }
    for (const char * p = v; *p != '\0'; p++) {
        seed = seed * 131 + (unsigned char) *p;
    }
    return seed * 131 + 1u;
}

// Every input here must cover something that changes the compiled blob, or a cached blob
// gets reused for a graph it does not match. is_static/prefill_chunk_size change the shapes
// the graph is built for; the token-embedding quant options and ROPE_PRECOMPUTE change its
// topology; the NPU
// compile knobs change what the compiler emits for it.
uint64_t ggml_openvino_model_cache_extra_cfg(const std::string & device,
                                                    bool stateful,
                                                    bool is_static,
                                                    int prefill_chunk_size) {
    const char * manual_gqa_env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
    const bool manual_gqa_enabled = manual_gqa_env != nullptr ?
                                        ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0 :
                                        device == "GPU";

    uint64_t extra_cfg = 0;
    extra_cfg = extra_cfg * 131 + (stateful ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_reduce_compile_mem_enabled() ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE") ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (manual_gqa_enabled ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (is_static ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (uint64_t) (prefill_chunk_size + 1);
    extra_cfg = extra_cfg * 131 + ggml_openvino_requant_cfg();
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_ROPE_PRECOMPUTE") ? 1u : 0u);
    extra_cfg = ggml_openvino_hash_env_str(extra_cfg, "GGML_OPENVINO_NPU_COMPILER_TYPE");
    extra_cfg = ggml_openvino_hash_env_str(extra_cfg, "GGML_OPENVINO_NPUW_FUNCALL_FOR_ALL");
    extra_cfg = ggml_openvino_hash_env_str(extra_cfg, "GGML_OPENVINO_NPUW_UNFOLD_IREQS");
    return extra_cfg;
}

bool ggml_openvino_cache_file_exists(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    return f.is_open();
}

// Is a usable cache entry on disk for this fingerprint? Separated from the import so the
// answer is known before any weight pages are dropped: this is the last step that reads
// weight bytes, and after it the hashes are memoized.
bool ggml_openvino_model_cache_ready(const std::string & blob_path,
                                            const std::string & manifest_path,
                                            const ggml_cgraph * cgraph,
                                            uint64_t fingerprint) {
    return ggml_openvino_cache_file_exists(blob_path) &&
           ggml_openvino_model_cache_verify_manifest(manifest_path, cgraph, fingerprint);
}

// Strip the "#<slot>" suffix the decoder appends to graph-tensor names. The slot is an
// index into the cgraph hash set, so it follows tensor addresses and differs between
// processes; the base name is stable and unique within a graph. Names without the suffix
// are returned unchanged.
std::string ggml_openvino_strip_name_suffix(const std::string & name) {
    const size_t pos = name.rfind('#');
    if (pos == std::string::npos || pos + 1 == name.size()) {
        return name;
    }
    for (size_t i = pos + 1; i < name.size(); ++i) {
        if (!isdigit(static_cast<unsigned char>(name[i]))) {
            return name;
        }
    }
    return name.substr(0, pos);
}

// Re-attach this run's suffixes to the names recorded when the blob was compiled, keeping
// the recorded (port) order. Returns false if a recorded name has no counterpart among the
// live ones, or if two live names strip to the same base -- either makes the mapping
// unsound, so the caller must fall back to compiling.
bool ggml_openvino_resolve_cached_names(const std::vector<std::string> & cached,
                                               const std::vector<std::string> & live,
                                               std::vector<std::string> & out) {
    std::map<std::string, std::string> by_base;
    for (const auto & name : live) {
        if (!by_base.emplace(ggml_openvino_strip_name_suffix(name), name).second) {
            return false;
        }
    }
    out.clear();
    out.reserve(cached.size());
    for (const auto & name : cached) {
        auto it = by_base.find(name);
        if (it == by_base.end()) {
            return false;
        }
        out.push_back(it->second);
    }
    return true;
}

// Everything that changes which ExtraQuantType a weight is requantized to, and so changes
// the bytes in the weights bin. Shared by the weights and model fingerprints.
uint64_t ggml_openvino_requant_cfg() {
    uint64_t h = 0;
    h = h * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_TOKEN_EMBD_I4") ? 1u : 0u);
    h = h * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_TOKEN_EMBD_I8") ? 1u : 0u);
    return h;
}

uint64_t ggml_openvino_weights_fingerprint(const ggml_cgraph * cgraph) {
    uint64_t h = FNV_OFFSET;
    for_each_weight(cgraph, [&](const ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint_cached(t)); });
    h = fnv1a_u64(h, 1);  // discriminator: this is a weights-set hash, not a model hash
    // The per-weight hashes above sample the *ggml* tensor, which is the same however the
    // frontend requantizes it. The bin holds the requantized bytes, so anything selecting a
    // different ExtraQuantType has to be part of its name or one setting's bin would be
    // handed to a run using another.
    h = fnv1a_u64(h, ggml_openvino_requant_cfg());
    const std::string ver = ov_version_string();
    h = fnv1a(h, ver.data(), ver.size());
    return h;
}

// Give the weights a graph-independent home.
//
// A weightless blob stores per-weight offsets into a weights file, and those offsets only
// exist on Constants carrying OpenVINO's WeightlessCacheAttribute, which nothing but the IR
// reader attaches. Serializing each compiled graph to get them makes the bin a function of
// the graph: prefill and decode disagree on the token-length constants, that changes what
// the serializer dedupes, and every later offset shifts -- so each graph, and each context
// geometry, ends up with its own near-identical 2 GB copy of the same weights.
//
// Round-tripping a model that holds *only* the weights removes the graph from the equation.
// The attribute lives on the node, so the Constants read back here can be grafted into any
// graph and still resolve against this one bin. Same weights -> same bin, whatever is built
// on top of them.
//
// `weights` is updated in place to the round-tripped nodes. On failure it is left untouched
// and the caller compiles normally, without caching.
bool ggml_openvino_canonical_weights(ov::Core & core,
                                     const std::string & xml_path,
                                     const std::string & bin_path,
                                     std::map<std::string, std::shared_ptr<ov::Node>> & weights) {
    try {
        // Reuse the bin a previous run (or a different geometry) already produced.
        if (!ggml_openvino_cache_file_exists(xml_path) || !ggml_openvino_cache_file_exists(bin_path)) {
            ov::ResultVector results;
            results.reserve(weights.size());
            for (const auto & kv : weights) {
                auto r = std::make_shared<ov::op::v0::Result>(kv.second);
                r->set_friendly_name(kv.first);
                results.push_back(r);
            }
            auto wm = std::make_shared<ov::Model>(results, ov::ParameterVector{}, "ggml_weights");
            // The serializer requires a ".xml" extension, so the temporaries keep it.
            ov::serialize(wm, xml_path + ".tmp.xml", bin_path + ".tmp.bin");
            // Publish bin first: a reader needs both, and the xml is what it opens.
            if (std::rename((bin_path + ".tmp.bin").c_str(), bin_path.c_str()) != 0 ||
                std::rename((xml_path + ".tmp.xml").c_str(), xml_path.c_str()) != 0) {
                std::remove((xml_path + ".tmp.xml").c_str());
                std::remove((bin_path + ".tmp.bin").c_str());
                return false;
            }
        }

        auto wm = core.read_model(xml_path, bin_path);
        std::map<std::string, std::shared_ptr<ov::Node>> out;
        for (const auto & r : wm->get_results()) {
            auto src = r->input_value(0).get_node_shared_ptr();
            // A weight is not always a bare Constant: a sub-byte one arrives wrapped in the
            // Convert (and sometimes a scale multiply) that widens it. Take whatever subgraph
            // produced the Result -- the offsets live on the Constants inside it either way --
            // but require that at least one such Constant came back, so a weight that somehow
            // got folded away cannot be silently cached as nothing.
            bool has_constant = false;
            std::vector<ov::Node *> stack{ src.get() };
            std::set<ov::Node *> seen;
            while (!stack.empty() && !has_constant) {
                ov::Node * n = stack.back();
                stack.pop_back();
                if (!seen.insert(n).second) {
                    continue;
                }
                if (ov::as_type<ov::op::v0::Constant>(n)) {
                    has_constant = true;
                    break;
                }
                for (const auto & in : n->inputs()) {
                    stack.push_back(in.get_source_output().get_node());
                }
            }
            if (!has_constant) {
                return false;
            }
            // Results keep the friendly name they were serialized with; the node behind one
            // does not reliably, so key off the Result.
            out[r->get_friendly_name()] = src;
        }
        if (out.size() != weights.size()) {
            return false;
        }
        for (const auto & kv : weights) {
            if (out.find(kv.first) == out.end()) {
                return false;
            }
        }
        weights.swap(out);
    } catch (const std::exception & e) {
        GGML_LOG_WARN("ggml-openvino: canonical weights round-trip failed (%s), not caching\n", e.what());
        return false;
    }
    return true;
}

// Where OpenVINO's cache manager puts the blob for a CACHE_BLOB_ID: the id in decimal,
// with a .blob suffix, under CACHE_PATH. OpenVINO does not promise this layout, so the
// path is only ever used as a presence test, never opened: guessing wrong costs a
// recompile (and leaves an orphan blob), never a bad import.
std::string ggml_openvino_model_cache_ov_blob_path(const std::string & dir, uint64_t fingerprint) {
    return dir + "/" + std::to_string(fingerprint) + ".blob";
}

// Load the blob OpenVINO stored under `cfg`'s CACHE_BLOB_ID. Compiling an empty model is
// how the cache manager is asked for an entry without the original ov::Model -- which is
// the whole point here, since rebuilding that model is the cost being skipped. The caller
// owns the key, so it must have verified the manifest first: OpenVINO checks nothing
// beyond the id.
bool ggml_openvino_model_cache_load(ov::Core & core,
                                    const std::string & device,
                                    const ov::AnyMap & cfg,
                                    ov::CompiledModel & out) {
    try {
        auto empty = std::make_shared<ov::Model>(ov::ResultVector{}, ov::ParameterVector{}, "empty");
        auto remote_context = ggml_openvino_get_remote_context();
        if (remote_context.has_value()) {
            out = core.compile_model(empty, remote_context.value(), cfg);
        } else {
            out = core.compile_model(empty, device, cfg);
        }
    } catch (const std::exception & e) {
        GGML_LOG_WARN("ggml-openvino: model cache load failed (%s), recompiling\n", e.what());
        return false;
    }
    return true;
}

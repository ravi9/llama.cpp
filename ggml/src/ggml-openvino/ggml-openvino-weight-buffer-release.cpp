#include "ggml-openvino-weight-buffer-release.h"

#include "ggml-impl.h"

#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#    include <sys/mman.h>
#    include <unistd.h>
#endif

// The OpenVINO weight Constants are zero-copy views into the host buffers
// allocated by the OpenVINO backend. On GPU the plugin holds its own device copy
// after compile_model, so the host pages are dead weight for inference and can
// be dropped to reclaim RSS.
//
// We do NOT free the buffer (ggml owns its lifetime and tensors still point into
// it); instead madvise(MADV_DONTNEED) drops the resident pages while keeping the
// mapping valid. A later recompile would re-read these Constants from now-zeroed
// memory and produce garbage, so once released we fail fast if a cache-miss
// compile branch is reached again.
namespace {
struct ov_weight_buffer_registry {
    std::mutex mutex;
    std::vector<std::pair<void *, size_t>> buffers;
    bool released = false;
};

ov_weight_buffer_registry & ov_weight_registry() {
    static ov_weight_buffer_registry reg;
    return reg;
}
}  // namespace

void ggml_openvino_register_weight_buffer(void * data, size_t size) {
    if (data == nullptr || size == 0) {
        return;
    }
    auto & reg = ov_weight_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    for (const auto & buffer : reg.buffers) {
        if (buffer.first == data) {
            return;
        }
    }
    reg.buffers.emplace_back(data, size);
}

bool ggml_openvino_weight_buffers_released() {
    auto & reg = ov_weight_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    return reg.released;
}

void ggml_openvino_release_weight_buffers() {
    auto & reg = ov_weight_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    if (reg.released) {
        return;
    }
    size_t total = 0;
#if !defined(_WIN32)
    for (const auto & buffer : reg.buffers) {
        const long page = sysconf(_SC_PAGESIZE);
        uintptr_t start = reinterpret_cast<uintptr_t>(buffer.first);
        uintptr_t end = start + buffer.second;
        uintptr_t aligned_start = (start + page - 1) & ~(uintptr_t) (page - 1);
        uintptr_t aligned_end = end & ~(uintptr_t) (page - 1);
        if (aligned_end > aligned_start) {
            if (madvise(reinterpret_cast<void *>(aligned_start), aligned_end - aligned_start, MADV_DONTNEED) == 0) {
                total += aligned_end - aligned_start;
            }
        }
    }
#endif
    reg.released = true;
    GGML_LOG_INFO("%s: released %zu MB of host weight buffers (%zu buffers)\n", __func__, total / 1024 / 1024,
                  reg.buffers.size());
}

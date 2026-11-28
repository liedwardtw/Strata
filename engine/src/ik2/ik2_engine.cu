// Based on Project Maya (https://github.com/mw00/project-maya)
// Copyright (c) 2026 Peasant Smith, OscarSheng, alesha-pro, cobra91 and the Project Maya contributors
// Licensed under the MIT License.

// src/ik2/ik2_engine.cu - strata-ik2: Inkling 2 (thinkingmachines/Inkling-2-Small, 594B / ~17B active) on one or two
// GPUs, from an `inkling2` GGUF in llama.cpp PR #32187's layout (docs/INKLING2.md; unsloth/Inkling-2-Small-GGUF's
// files: NVFP4, IQ3_XXS / IQ4_XS and Q*_K experts).
//
//   * The dense weights (attention, router, head) decoded once to FP16 in VRAM (the router in F32); cuBLAS' FP16 GEMMs
//     with F32 sums.  The attention (relative bias, sinks, sliding window, L2 clips) fused: an mma.sync kernel, and a
//     wgmma one on Hopper (sm_90a); the KV cache in FP16 (IK2_KV=fp8: E4M3).
//   * The routed experts as the GGUF holds them, all in pinned host memory (~323 GiB for UD-Q4_K_XL), and a cache of
//     them in VRAM (--cache-gb, at most what the GPU has room for).  A prompt goes through layer by layer in passes of
//     up to 16384 tokens, so each pass streams every expert it misses once, the next layer's prefetched while this one
//     runs; a scan-resistant eviction keeps a pass from flushing the cache.  IK2_WARM fills the cache at load with the
//     experts tokens pick most (a profile written with IK2_POP_OUT).
//   * Decode a token at a time; with IK2_CPU=1 a missing expert is computed on the CPU (AVX-512) when that is sooner
//     than its copy.  Two GPUs (--gpus 2): IK2_EP=1 deals each layer's experts to both (expert parallel - each with its
//     own cache and PCIe link), else the layers are split between them.
//
//   strata-ik2 --model FIRST-SHARD.gguf --serve [--max-context 262144] [--cache-gb 60] [--gpus 1] [--threads 8]
//              [--eos a,b]
//   strata-ik2 --model FIRST-SHARD.gguf --tokens 1,2,3|@FILE [--gen N] [--chunk 2048] [--dump DIR]
// --serve: the engine protocol on stdin / stdout (serve/server.py's StrataEngine: INFO / READY, GEN, PP, T, DONE,
// STOP); --serve-ik2 the older @-line one.  --dump: each layer's output (blk.L.out) and the log-probabilities of every
// prompt position, F32 raw files ([T][C], [T][n_vocab]) - what tools/ik2_compare.py holds against ref/inkling2.py's
// seams.  Settings (IK2_*): docs/SETTINGS.md.
#include "ik2_cpu.hpp"
#include "ik2_draft.hpp"
#include "ik2_kernels.cuh"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/ik2/engine.hpp"
#if defined(STRATA_USE_HIP)
#include "wmma_gemm.h"   // src/prefill: the gfx11 WMMA FP16 GEMM (IK2_WMMA_GEMM opt-in)
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <io.h>
#ifdef O_RDONLY
#undef O_RDONLY
#endif
#define O_RDONLY (_O_RDONLY | _O_BINARY)
#define open _open
#define close _close
typedef SSIZE_T ssize_t;
#define off_t int64_t
static inline ssize_t pread(int fd, void* buf, size_t count, int64_t offset) {
    if (_lseeki64(fd, offset, SEEK_SET) == -1) return -1;
    return _read(fd, buf, (unsigned int)count);
}
#define posix_fadvise(fd, off, len, adv) ((void)0)
#define MADV_WILLNEED 0
static inline int madvise(void*, size_t, int) { return 0; }
static inline int munmap(void*, size_t) { return 0; }
#ifndef SIGUSR1
#define SIGUSR1 30
#endif
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
static inline cudaError_t cudaMemPrefetchAsync_compat(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream = 0) {
    cudaMemLocation loc{};
    if (dstDevice < 0) {
        loc.type = cudaMemLocationTypeHost;
        loc.id = 0;
    } else {
        loc.type = cudaMemLocationTypeDevice;
        loc.id = dstDevice;
    }
    return cudaMemPrefetchAsync(devPtr, count, loc, 0, stream);
}
#define cudaMemPrefetchAsync cudaMemPrefetchAsync_compat

static inline cudaError_t cudaMemAdvise_compat(const void* devPtr, size_t count, cudaMemoryAdvise advice, int device) {
    cudaMemLocation loc{};
    if (device < 0) {
        loc.type = cudaMemLocationTypeHost;
        loc.id = 0;
    } else {
        loc.type = cudaMemLocationTypeDevice;
        loc.id = device;
    }
    return cudaMemAdvise(devPtr, count, advice, loc);
}
#define cudaMemAdvise cudaMemAdvise_compat
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <fstream>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <csignal>
#include <iostream>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace {

namespace k = ik2::k;

// IK2_VPROF2: per-visit GPU-event pair (attn start -> MoE start) shared across the two namespaces that host
// forward() and the speculative loop.  Events accumulate per step; the loop prints the mean split.
cudaEvent_t vpA[64], vpB[64];
// IK2_VPROF3: extra per-visit boundaries to split attention into proj / prep+conv / flash / wo.
// IK2_VPROF4 splits the proj phase further: vpR before rms/gates, vpS before the wqkvrg matmul.
cudaEvent_t vpP[64], vpK[64], vpF[64], vpR[64], vpS[64];
int vp_attn_idx = -1;   // set by forward() to the current visit index before run_layer_attn
bool vp_init = false;
int vp_n = 0;

#define CU(x)                                                                                              \
    do {                                                                                                   \
        cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess) {                                                                           \
            std::fprintf(stderr, "strata-ik2: %s: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                  \
        }                                                                                                  \
    } while (0)
#define CB(x)                                                                                              \
    do {                                                                                                   \
        cublasStatus_t e_ = (x);                                                                           \
        if (e_ != CUBLAS_STATUS_SUCCESS) {                                                                 \
            std::fprintf(stderr, "strata-ik2: %s: cuBLAS status %d (line %d)\n", #x, (int) e_, __LINE__);    \
            std::exit(1);                                                                                  \
        }                                                                                                  \
    } while (0)

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "strata-ik2: %s\n", m.c_str());
    std::exit(1);
}

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- the files

struct Shards {
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    std::vector<std::string> paths;
    std::map<std::string, std::pair<int, const strata::TensorInfo*>> by_name;

    explicit Shards(const std::string& first) {
        std::vector<std::string> ps{first};
        const size_t at = first.find("-00001-of-");
        if (at != std::string::npos) {
            const int n = std::atoi(first.c_str() + at + 10);
            for (int i = 2; i <= n; ++i) {
                char num[16];
                std::snprintf(num, sizeof num, "%05d", i);
                std::string p = first;
                p.replace(at + 1, 5, num);
                ps.push_back(p);
            }
        }
        for (const auto& p : ps) {
            files.emplace_back(new strata::GgufFile(p));
            paths.push_back(p);
            for (const auto& t : files.back()->tensors()) by_name[t.name] = {(int) files.size() - 1, &t};
        }
    }
    const strata::TensorInfo* info(const std::string& n) const {
        auto it = by_name.find(n);
        return it == by_name.end() ? nullptr : it->second.second;
    }
    const uint8_t* data(const std::string& n) const {
        auto it = by_name.find(n);
        if (it == by_name.end()) die("no tensor " + n);
        return files[it->second.first]->tensor_data(*it->second.second);
    }
    uint64_t file_offset(const std::string& n) const {
        auto it = by_name.find(n);
        return files[it->second.first]->data_start() + it->second.second->offset;
    }
    int file_of(const std::string& n) const { return by_name.at(n).first; }
    double num(const std::string& key) const {
        const auto* v = files[0]->get(key);
        if (!v) die("no key " + key);
        return v->num();
    }
    const strata::MetaValue* get(const std::string& key) const { return files[0]->get(key); }
};

size_t tensor_bytes(const strata::TensorInfo& t) {
    // (the blocks decoded here first: strata's block_geometry names NVFP4 but has no geometry for it)
    if (k::block_values((int) t.type)) return (size_t) (t.elements() / k::block_values((int) t.type)) * k::block_bytes((int) t.type);
    int e = 0, b = 0;
    if (!strata::block_geometry(t.type, e, b)) die(std::string("type ") + t.type_name() + " of " + t.name);
    return (size_t) (t.elements() / e) * b;
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ---- the model

struct Config {
    int n_layer, C, H, KV, D, E, K, F, window, d_rel, extent, extent_swa, block, n_vocab, n_rows;
    // A looped model (Inkling-2-Mini): the layers core0 .. core1 - 1 (whole blocks) run n_pass times a token, so a
    // token is n_visit layer visits (n_layer without a loop).  rank: the rank of the loop's message gates.
    int n_visit = 0, n_pass = 1, core0 = 0, core1 = 0, rank = 0;
    float eps, route_scale, qk_clip, rel_clip, alpha, n_floor, gate_cap, up_cap, logit_cap;
    std::vector<bool> sliding;   // by layer
};

// a dense matrix as the GGUF holds it, in VRAM: its blocks [N][K]
struct Mat {
    // (IK2_DENSE=q8 with Q8_0 blocks and K a multiple of 128: the blocks taken apart - the int8 values [N][K] and the
    //  blocks' FP16 scales [N][K / 32] - which a warp reads 128 values an instruction; raw is then null)
    int8_t* q8 = nullptr;
    uint16_t* q8d = nullptr;
    uint8_t* raw = nullptr;     // IK2_DENSE=q8: the blocks, one token's products straight over them
    uint16_t* f16 = nullptr;    // else (the default): an FP16 copy, every product a cuBLAS GEMM - measured faster on an
    int type = 0, N = 0, K = 0; // RTX PRO 6000 (16 against 21 ms a token), for 4 GiB more VRAM
    // IK2_INT4_DENSE: a rotated-symmetric-int4 sidecar of this whole matrix (the expert k_repack layout), loaded only
    // when the gate is on and the bins exist. Decode matvec reads these; q8/f16 stay resident for prefill. Null = off.
    const uint8_t* i4Wp = nullptr;
    const uint8_t* i4Wh = nullptr;
    const float* i4WS = nullptr;
    // IK2_INT4_HEAD: the sidecar is reserved for run_head's T=2..8 verify GEMM; the generic decode matvec (mv1/mv1c)
    // and matmul's IK2_INT4_DENSE_MULTI branch skip it, so T=1 decode and prefill stay byte-identical even when on.
    bool i4_head_only = false;
    // IK2_INT4_MSG: mat_rows' row slice carries ft0-shifted sidecar pointers. The repack is 16-row-block local, so
    // the slice reads the same bytes a standalone conversion of its rows would. mv1/mv1c skip it (T=1 stays Q8).
    bool i4_row_slice = false;
    // IK2_BATCH_DENSE_MV_MATS: which dense projection this is (1=wqkvrg, 2=wqrg/wq, 3=wk/wv, 4=wo, 5=wr/wgate).
    // Lets the A/B bisect gate individual mats; 0 (default) = excluded from the batched INT4 matvec.
    int batch_mv_cat = 0;
};

struct Layer {
    int dev = 0;   // the GPU it runs on (--gpus: the layers in equal runs)
    bool sliding;
    // FP16 [out][in] (fused projections on !g_dense_q8, or when quant types match on g_dense_q8)
    Mat wq, wk, wv, wr, wgate, wo;
    Mat wqkvrg, wqrg, wk_dyn, wv_dyn;
    uint16_t *sink16 = nullptr;                    // F16 [H][C] (used in g_dense_q8; fused in !g_dense_q8)
    uint16_t *router16 = nullptr;                  // F16 [E][C]
    float *probs_b;                                // F32 [E]
    float *attn_norm, *q_in_norm, *k_in_norm, *v_in_norm, *ffn_norm;
    float *q_clip, *r_clip, *rel_proj;             // [H][D], [H][d_rel], [extent][d_rel]
    float *res_score, *ffn_res_score;              // [nq][C], [C]
    int n_res_q;
    float *sc_k, *sc_v, *sc_attn, *sc_mlp;         // [C'][4]
    uint16_t *sc_k_dyn16 = nullptr, *sc_v_dyn16 = nullptr;   // F16 [KV * 4][C] (used in g_dense_q8; fused in !g_dense_q8)
    // the experts: types, bytes an expert, host blocks, scales (host)
    int ty[3];
    size_t eb[3];            // bytes of a role's part of an expert
    size_t stride[3];        // bytes between two experts' parts in host[r] (eb, or 2 eb in a fused gate_up, or slot_bytes)
    const uint8_t* host[3];
    bool host_contig = false;
    size_t slot_bytes = 0;
    std::vector<float> scale[3];                   // empty: 1
    // a looped core's layer: its later passes' message gates by module (0 q's, 1 k's, 2 v's, 3 the MoE's) -
    // down [rank][2 C], forget and gain [rank][C] - and the four messages' queries [4][C] in that order (F32)
    float *lp_down[4] = {}, *lp_forget[4] = {}, *lp_gain[4] = {}, *lp_query = nullptr;
    // IK2_INT4_EXPERTS: rotated-int4 expert weights, device-resident (replace NVFP4 residency).
    // Per layer, per role [0]=gate_up, [2]=down: one contiguous Wp/Wh/WS, and [E] pointer tables.
    bool i4_loaded = false;
    uint8_t *i4_Wp[3] = {}, *i4_Wh[3] = {};
    float *i4_WS[3] = {};
    const uint8_t** i4_tab_Wp[3] = {}, **i4_tab_Wh[3] = {};
    const float** i4_tab_WS[3] = {};
    size_t i4_wp_stride[3] = {}, i4_wh_stride[3] = {};
    int i4_Kpad[3] = {};
    // the resident token path (every expert in VRAM): where each expert's blocks are, and the experts' scales, on the
    // device ([E] each; a scale's null: 1)
    const uint8_t** tab_blob = nullptr;
    float* tab_sc[3] = {};
    std::vector<const uint8_t*> tab_h;             // (tab_blob on the host: a short prompt's groups)
    // multi-GPU EP resident tables: local expert pointers on device d (null if owned by peer)
    const uint8_t** ep_tab_blob[2] = {nullptr, nullptr};
    float* ep_tab_sc[2][3] = {};
    // the state: of the visit being run (bind_visit)
    uint16_t *kc = nullptr, *vc = nullptr;         // FP16 [kv_cap][KV * D] (IK2_KV=f16)
    uint8_t *kc8 = nullptr, *vc8 = nullptr;        // FP8 E4M3 [kv_cap][KV * D] (IK2_KV=fp8)
    uint8_t *kci = nullptr, *vci = nullptr;        // INT8 / INT4 codes [kv_cap][KV * D] / [kv_cap][KV * D / 2] (IK2_KV=int8/int4)
    uint16_t *ks = nullptr, *vs = nullptr;         // FP16 per-(token, KV-head) scales [kv_cap][KV] (int8/int4)
    // IK2_KV_HOST_CAP: host-mapped overflow for the int KV of full visits. The last kv_host_cap token rows stay in
    // VRAM (kci/vci/ks/vs above, sized kv_host_cap); older rows live in host-mapped RAM (hkci/hvci/hks/hvs, device
    // pointers from cudaHostGetDevicePointer) and the attention kernels read them over PCIe. kv_base = first VRAM row
    // (= max(0, newest_pos - kv_host_cap)); a row at position p < kv_base reads host[p], else VRAM[p - kv_base].
    uint8_t *hkci = nullptr, *hvci = nullptr;
    uint16_t *hks = nullptr, *hvs = nullptr;
    int kv_host_cap = 0, kv_base = 0;
    int kv_cap = 0;
    float *h_k, *h_v, *h_attn, *h_mlp;             // conv histories [3][C']
    float *rec_old[4] = {}, *rec_x[4] = {};        // (speculative) the convs' histories and inputs of a verify window
};

// A layer visit and its own state.  Without a loop a visit is a layer; a looped core's layer is visited once a pass,
// each time with that visit's K / V cache and conv histories (the weights, and the experts in the cache, are the
// layer's).  bind_visit() points the layer's state at a visit's before it runs.
struct Visit {
    int l = 0;        // the layer
    int pass = 0;     // the core's pass (0: its first, or a layer outside the core)
    uint16_t *kc = nullptr, *vc = nullptr;
    uint8_t *kc8 = nullptr, *vc8 = nullptr;
    uint8_t *kci = nullptr, *vci = nullptr;
    uint16_t *ks = nullptr, *vs = nullptr;
    uint8_t *hkci = nullptr, *hvci = nullptr;
    uint16_t *hks = nullptr, *hvs = nullptr;
    int kv_host_cap = 0, kv_base = 0;
    int kv_cap = 0;
    float *h_k = nullptr, *h_v = nullptr, *h_attn = nullptr, *h_mlp = nullptr;
    float *rec_old[4] = {}, *rec_x[4] = {};
};

// IK2_BATCH_ATTN: one independent KV/conv state per (batch slot, visit).  The pointers are VRAM-resident and
// immutable; a batch forwards n_rows slots at once by copying the selected slots' pointers into Work's stable
// device pointer tables.  This is real extra VRAM: 2 slots * n_visit * (2 * kv_cap * KD * bits/8 + 2 * kv_cap *
// KV * 2 + 4 * 3 * max(KD,C) * 4) bytes.
struct BatchSlots {
    static constexpr int kMaxRows = 8;
    int n_slot = 0, n_visit = 0;
    size_t bytes = 0;
    std::vector<uint8_t*> kci, vci;
    std::vector<uint16_t*> ks, vs;
    std::vector<float*> h_k, h_v, h_attn, h_mlp;
    inline size_t idx(int slot, int vis) const { return (size_t) slot * n_visit + vis; }
};
BatchSlots g_batch_slots;
static bool g_batch_active = false;
static const int* g_batch_pos = nullptr;
static const int* g_batch_slot_ids = nullptr;
static int g_batch_rows = 0;

struct Model {
    Config c;
    std::vector<Layer> L;
    std::vector<Visit> V;   // the visits of a token, in order
    Mat out_w;            // [n_rows][C]
    float *out_norm, *out_res_score, *embd_norm;
    float* lp_query_all = nullptr;   // a looped core's message queries, layer by layer ([n_core][4][C]: Layer::lp_query's)
    const uint8_t* embd;  // host Q8_0 [n_rows][C]
    int embd_type;
    // IK2_AR_KSTEP / IK2_EMBD_GPU: the same table on the device, for device-side gather in the K-step graph.
    const uint8_t* embd_d = nullptr;
    int embd_rows = 0;
    size_t embd_row_bytes = 0;
};

inline Layer& bind_visit(Model& M, int v) {
    const Visit& Vi = M.V[v];
    Layer& Ly = M.L[Vi.l];
    Ly.kc = Vi.kc, Ly.vc = Vi.vc, Ly.kc8 = Vi.kc8, Ly.vc8 = Vi.vc8, Ly.kv_cap = Vi.kv_cap;
    Ly.kci = Vi.kci, Ly.vci = Vi.vci, Ly.ks = Vi.ks, Ly.vs = Vi.vs;
    Ly.hkci = Vi.hkci, Ly.hvci = Vi.hvci, Ly.hks = Vi.hks, Ly.hvs = Vi.hvs;
    Ly.kv_host_cap = Vi.kv_host_cap, Ly.kv_base = Vi.kv_base;
    Ly.h_k = Vi.h_k, Ly.h_v = Vi.h_v, Ly.h_attn = Vi.h_attn, Ly.h_mlp = Vi.h_mlp;
    for (int i = 0; i < 4; ++i) Ly.rec_old[i] = Vi.rec_old[i], Ly.rec_x[i] = Vi.rec_x[i];
    return Ly;
}

// ---- the device helpers

cublasHandle_t g_blas;
cudaStream_t g_s;
// IK2_AR_STREAM2: second stream for AR slot 1 (non-blocking, created only when the env gate is on).
cudaStream_t g_s1 = nullptr;
cudaEvent_t g_ar2_ev[4] = {};
bool g_ar2_stats = false;
cudaStream_t g_cs;          // the experts' copies of the token path, beside the compute stream
cudaEvent_t g_ev;
bool g_overlap = true;      // IK2_OVERLAP=0: the copies on the compute stream, before the experts
// The CPU lane (IK2_CPU=1: on): a missing NVFP4 expert computed in host memory by the pool while the others' blocks
// are copied; its own blocks follow on the copy stream without being waited for ("lazy"), so it is in VRAM next time.
ik2::cpu::Pool* g_pool = nullptr;
double g_t_cpu = 0.5e-3, g_t_up = 0.36e-3;   // seconds an expert: the pool's (measured as it runs), the link's
uint64_t g_n_cpu = 0, g_n_wait = 0;
bool g_cpu_lazy = true;   // IK2_CPU_LAZY=0: a CPU-computed expert is not copied in afterwards (the link carries less)
constexpr int kRing = 256;
cudaEvent_t g_ring[kRing];                   // a lazy batch's end on the copy stream, by sequence number
uint64_t g_seq_next = 1, g_seq_done = 0;
void lazy_poll() {
    while (g_seq_done + 1 < g_seq_next && cudaEventQuery(g_ring[(g_seq_done + 1) % kRing]) == cudaSuccess) ++g_seq_done;
}

// IK2_PROF=1: the stream waited for at each phase's end, its time added to the phase (slower, but where it goes)
bool g_prof = false;
// IK2_PROF_FAST=1: event-based phase timers INSIDE the resident fast path (fast_token_launches), which IK2_PROF
// itself disables. Run with IK2_GRAPH=0 (events are not captured). Non-blocking cudaEventRecord at phase boundaries,
// elapsed summed after the token's sync - no per-phase stream wait, so the resident path's speed is preserved.
bool g_prof_fast = false;
namespace pf_fast {
// 0 emb_norm, 1 loop_bank, 2 input_norm, 3 proj, 4 prep_kv, 5 attn, 6 wo, 7 moe_norm, 8 router_route, 9 experts, 10 combine, 11 head, 12 tail
// (13, not 12: one_token_launches marks 12 at its end, and acc[] is indexed by the mark.)
constexpr int NPH = 13;
const char* const names[NPH] = {"emb_norm", "loop_bank", "input_norm", "proj", "prep_kv", "attn",
                                "wo", "moe_norm", "router_route", "experts", "combine", "head", "tail"};
// One event per mark, recorded in sequence across the token's 48 visits (a fixed-handle scheme reused the
// same 13 events every visit and kept only the LAST timestamp, so every phase but one read as ~0 or garbage).
constexpr int kPfMaxMarks = 1100;
cudaEvent_t evs[kPfMaxMarks];
int ev_phase[kPfMaxMarks];
int nmark = 0;
double acc[NPH];
int ntok = 0;
bool init = false;
void setup() {
    if (init) return;
    init = true;
    for (int i = 0; i < kPfMaxMarks; ++i) cudaEventCreate(&evs[i]);
}
inline void mark(int i, cudaStream_t s) {
    if (!g_prof_fast || nmark >= kPfMaxMarks) return;
    cudaEventRecord(evs[nmark], s);
    ev_phase[nmark] = i;
    ++nmark;
}
void finish(cudaStream_t s) {   // (after the token's stream sync: every recorded event is complete)
    if (!g_prof_fast) return;
    for (int k = 0; k + 1 < nmark; ++k) {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, evs[k], evs[k + 1]) == cudaSuccess) acc[ev_phase[k]] += ms;
    }
    nmark = 0;
    ++ntok;
}
void report(FILE* o = stdout) {
    if (!g_prof_fast || !ntok) return;
    double tot = 0.0;
    for (int i = 0; i < NPH; ++i) tot += acc[i];
    std::fprintf(o, "\nRESIDENT DECODE PROFILE (%d tokens, %.3f ms/token total):\n", ntok, tot / ntok);
    for (int i = 0; i < NPH; ++i)
        std::fprintf(o, "  %-13s %7.3f ms/token  %5.1f%%\n", names[i], acc[i] / ntok, 100.0 * acc[i] / tot);
    std::fflush(o);
    // (zeroed so a serve loop reports one request's split at a time; the CLI reports once and exits)
    for (int i = 0; i < NPH; ++i) acc[i] = 0.0;
    ntok = 0;
}
}  // namespace pf_fast
bool g_mv = true;   // IK2_MV=0: a few tokens' experts through the FP16 GEMMs too
// How often each expert is picked by a token (prefill and decode), [n_layer * E]: IK2_POP_OUT=FILE writes it at
// the end (int32 n_layer, int32 E, then uint64 counts); IK2_WARM=FILE fills the expert cache with the most picked
// experts at load, so the first requests do not start from an empty cache.
std::vector<uint64_t> g_pop;
int g_pop_E = 0;
void pop_count(int l, const int* top, int S) {
    if (g_pop.empty()) return;
    for (int i = 0; i < S; ++i) ++g_pop[(size_t) l * g_pop_E + top[i]];
}
void pop_dump() {
    const char* f = getenv("IK2_POP_OUT");
    if (!f || g_pop.empty()) return;
    FILE* o = std::fopen(f, "wb");
    if (!o) return;
    const int32_t hdr[2] = {(int32_t) (g_pop.size() / g_pop_E), (int32_t) g_pop_E};
    std::fwrite(hdr, 4, 2, o);
    std::fwrite(g_pop.data(), 8, g_pop.size(), o);
    std::fclose(o);
}
int g_pop_E_ep = 0;
// IK2_EP=1 with --gpus 2: expert parallel. Every layer's attention and dense weights (and the KV cache) on GPU 0;
// expert e of every layer belongs to GPU e & 1, each GPU with its own cache over all layers and its own PCIe link.
// In decode a layer's picks go to their GPUs at once, both links copying the misses side by side.
bool g_ep = false;
bool g_ep_fast = false;
cudaEvent_t g_ep_x, g_ep_cp[2], g_ep_done;
struct EpStaging {
    void* h_xq = nullptr;
    void* h_route = nullptr;
    void* h_y = nullptr;
    float* d_xq[2] = {};
    int* d_route[2] = {};
    float* d_y[2] = {};
};
EpStaging g_ep_stage;
// IK2_TP=1 with --gpus 2: the experts tensor-parallel instead. GPU d holds half of every expert: rows [d F/2, ..) of
// gate and up and rows [d C/2, ..) of down, so every miss is half on each link. A layer: each GPU its half of h
// (gate/up/SwiGLU), the halves swapped (P2P), each GPU its half of the output rows (down), concatenated on GPU 0.
// Attention, dense weights and the KV cache on GPU 0 as with IK2_EP.
bool g_tp = false;
cudaEvent_t g_tp_h[2];
int g_tp_F = 0, g_tp_C = 0;
inline bool dual() { return g_ep || g_tp; }
// which GPU an expert belongs to, [n_layer * E]: IK2_EP_SPLIT = GPU 1's share of the picks (0.5; GPU 0 also holds the
// attention, the dense weights and the KV cache). With IK2_WARM's profile each layer's experts are dealt by how often
// they are picked, so the two GPUs' shares come out as asked; without it by a hash of the expert.
std::vector<uint8_t> g_ep_own;
inline int ep_owner(int l, int e) { return g_ep_own[(size_t) l * g_pop_E_ep + e]; }
bool g_flash = true;   // IK2_FLASH=0: the prefill's attention (FP16 KV) through the logits in memory and cuBLAS, not fused
bool g_flash_wmma = true;   // IK2_FLASH_WMMA=0: gfx1100 uses the scalar/cuBLAS attention even when fused is on
bool g_gate_fuse = true;    // IK2_GATE_FUSE=0: the scalar decode's output gate as its own gate_mul launch, not the combine's epilogue
bool g_rms_fuse = true;     // IK2_RMS_FUSE=0: layer-0 full attention's q/k/v norms as three rms launches, not one rms_multi
bool g_onepass = false;     // IK2_ATTN_ONEPASS=1: sliding decode attention split+combine as one launch (env-gated experiment)
int g_parallel = 1;         // IK2_PARALLEL=1..8: serve batch slots (1 = legacy single stream, byte-identical)
int g_attn_dpp = 0;         // IK2_ATTN_DPP=1: int4-only bit-exact DPP warp sum, same split grid (default off)
bool g_attn_gqa = false;    // IK2_ATTN_GQA=1: int decode K/V shares R=3 Q heads per KV head (default off)
bool g_loopbank_fuse = false;  // IK2_LOOPBANK_FUSE=1: loop-core pass>0 bank copies as one kernel, not n_bank+1 D2D memcpys
bool g_int4_experts = true;  // IK2_INT4_EXPERTS=0: NVFP4 experts; default ON: rotated-int4 device-resident experts (l<20), falls back to NVFP4 when int4-bin absent
int g_int4_layers = 20;      // IK2_INT4_LAYERS: number of low layers served by rotated-int4 sidecars (24 = also convert the Q8_0 l>=20 experts, ~2 GiB smaller for 1M context)
bool g_i4_mid_fuse = false;  // IK2_I4_MID_FUSE=1: fused swiglu+prep_down+rot (SLOWER on gfx1100: 92.0 vs 93.8; kept for reference)
int g_i4_rot_in = 0;         // IK2_I4_ROT_IN: phase D, 1 = gate_up rot-in-matvec, 2 = also down prep+rot-in (default off)
bool g_int4_pf_off = false;   // IK2_INT4_PF=0: disable INT4 prefill experts (isolate decode vs prefill bug)
// IK2_FULL_WINDOW=N (default 0 = off, byte-identical): the 8 full-attention layers read only the last N
// keys when decoding (T<=8), so a 640K conversation stops paying O(ctx) KV traffic on them. The cold KV is
// left in VRAM (it is still written during prefill, and the sliding layers keep their own 1024 window), it is
// simply not read again. This is the "discard the cold KV" lever and it DOES change the output: measure it
// with quality.py and a long-context needle before trusting it.
int g_full_window = 0;
bool g_verify_fast = false;   // IK2_VERIFY_FAST=1: device-only INT4 verify MoE (g_rec, T=2..8): the T=1 resident
                              // sequence at S=T*K slots, no route D2H / stream sync (graph-capturable). Default OFF.
bool g_verify_graph = false;  // IK2_VERIFY_GRAPH=1: capture the T=8 verify forward once and replay it. Needs
                              // IK2_VERIFY_FAST=1 and the single-GPU resident path. Default OFF.
bool g_batch_decode = false;  // IK2_BATCH_ATTN=1: true multi-slot batched decode (one row per slot) through the
                               // 48 visits. Reuses the verify-fast T=2..8 path with per-row KV/conv pointer tables.
                               // Default OFF; IK2_PARALLEL=1 stays byte-identical.
bool g_ar_dual = false;        // IK2_AR_DUAL=1 (needs IK2_PARALLEL=2): two AR decode slots, each running the
                               // proven T=1 resident fast_token path from its OWN CUDA graph with that slot's
                               // KV/conv pointers baked in. One stream, round-robin. Default OFF: byte-identical.
bool g_ar_stream2 = false;      // IK2_AR_STREAM2=1 (needs IK2_AR_DUAL=1, IK2_PARALLEL=2): replay slot 0 on g_s and
                                // slot 1 on a second non-blocking stream g_s1, with a second isolated Work. Default
                                // OFF: byte-identical to the original single-stream AR dual path.
bool g_ar_kstep = false;        // IK2_AR_KSTEP=1 (implies IK2_AR_STREAM2 and IK2_EMBD_GPU): pre-captured K-token
                                 // graphs per AR slot, with device embed gather + device pos advance. Default OFF.
int g_ar_K = 4;                 // IK2_AR_KSTEP_K, clamped to [2,8].
bool g_ar_kstep_ready = false;  // set only after BOTH slot graphs instantiate successfully.
bool g_embd_gpu = false;        // IK2_EMBD_GPU=1: upload the embedding table once, for device gather.
bool g_batch_exact = true;     // IK2_BATCH_EXACT (default 1): route every batched greedy step through the serial
                               // T=1 forward()/fast_token path, which is byte-identical to solo. IK2_BATCH_EXACT=0
                               // restores the original true-batched T=2..8 general path (faster but not bit-exact).
bool g_batch_moe_device = false;  // IK2_BATCH_MOE_DEVICE=1 (default OFF): device-only INT4 MoE for the true batched
                                  // decode. Reproduces run_prefill_moe's rot_pack8 + gemm8 + expert-major scatter
                                  // numerics without the per-visit route D2H/sync. Applies only when W.batch.
bool g_batch_moe_compact = false; // IK2_BATCH_MOE_COMPACT=1 (default OFF): launch int4_gemm8_moe_masked on only the
                                  // experts with routed rows (grid.z <= S instead of E), preserving pack8/gemm8 math.
bool g_batch_graph = false;        // IK2_BATCH_GRAPH=1 (default OFF): capture the true batched T=2..8 forward into one
                                   // CUDA graph and replay it. The graph reads per-row state through Work's stable
                                   // device pointer tables (stage_batch_tables), so changing slot state needs only a
                                   // pinned-host table refresh before replay. Implies g_batch_moe_device: the general
                                   // MoE path's route D2H + cudaStreamSynchronize cannot be captured.
bool g_batch_graph_exact = false;  // IK2_BATCH_GRAPH_EXACT=1 (default OFF): while a true-batched CUDA graph is
                                   // captured/replayed, use the T=1 solo-equivalent numeric kernels for the stages
                                   // that differ in the general T=2..8 path (dense projections, sink, dynamic KV,
                                   // attention splits, router/MoE, head and argmax). No host sync inside capture.
bool g_fast_graph_capturing = false; bool g_batch_graph_capturing = false;  // true only inside the batch graph's cudaStreamBeginCapture window; makes
                                       // stage_batch_tables() skip its event-ring sync and use the fixed slot 0.
bool g_serve_dspec = false;  // IK2_SERVE_DSPEC=1: --serve greedy (temp<=0) uses DSpark spec when a drafter is
                             // loaded and ctx has room; temp>0/short ctx fall back to byte-identical AR. Default OFF.
bool g_batch_dense_mv = false;  // IK2_BATCH_DENSE_MV=1: W.batch T>1 dense projections take the row-aware INT4
                                // matvec (i4Wp VRAM sidecar read once per weight row for all T rows) instead of the
                                // Q8->f16 decode + M=4 cublas GEMM. Default OFF -> binary byte-identical.
int g_batch_dense_mv_mask = 0x3F;  // IK2_BATCH_DENSE_MV_MATS: bitmask of Mat::batch_mv_cat to enable (bisect aid).
bool g_batch_solo_fast = false;  // IK2_BATCH_SOLO_FAST=1: when exactly one slot is decoding (no prefill pending),
                                 // run the existing solo fast_token/forward T=1 path for it instead of the batch
                                 // scheduler's general forward. Recovers ~107 t/s/slot for the 1-active case.
                                 // Default OFF -> batch path byte-identical.
inline bool batch_dense_mv_on(int T) {
    return g_batch_dense_mv && g_batch_active && T > 1 && T <= 8;
}
inline bool batch_dense_mv_mat_ok(const Mat& m) {
    return m.batch_mv_cat >= 1 && m.batch_mv_cat <= 5 && ((g_batch_dense_mv_mask >> m.batch_mv_cat) & 1);
}
bool g_batch_rotmv = false;   // IK2_BATCH_ROTMV=1 (default OFF): batched T=2..8 dense projections run k::int4_dense_rot_rows,
                              // the multi-row generalization of the T=1 int4_dense_rot path (rotate+quantize+sdot+reduction
                              // verbatim per row, one weight-row read shared by all T rows). Unlike IK2_BATCH_DENSE_MV it
                              // does NOT go through int4_rot_q8x/int4_matvec (a different activation quantizer), so it stays
                              // bit-exact with the solo T=1 path. Default OFF -> byte-identical.
inline bool batch_rotmv_on(int T) {
    return (g_batch_rotmv || (g_batch_graph_exact && g_batch_graph)) && g_batch_active && T > 1 && T <= 8;
}
inline bool batch_graph_exact_on(int T) {
    return g_batch_graph_exact && g_batch_graph && g_batch_active && T > 1 && T <= 8;
}
bool g_moe_group_exp = false; // IK2_MOE_GROUP_EXPERT=1: verify-fast MoE grouped BY EXPERT -- each expert's
                              // weights read once for all its slots (bit-exact vs k_mv_grouped). Needs
                              // IK2_VERIFY_FAST=1. Default OFF.
enum VGStage { VG_NORMAL = 0, VG_CAPTURE = 1, VG_REPLAY = 2 };
VGStage g_vg_stage = VG_NORMAL;
bool g_int4_dense = true;     // IK2_INT4_DENSE=0: disable rotated-int4 sidecars for the dense projections' DECODE matvecs (byte-identical Q8 path)
                              // (wqkvrg/wqrg/wk/wv/wo); default OFF, the Q8_0 path byte-identical. Prefill stays Q8.
bool g_int4_dense_multi = false;  // IK2_INT4_DENSE_MULTI=1: int4 W4A8 GEMM for T=2..8 verify dense projections (default OFF)
bool g_int4_msg = false;      // IK2_INT4_MSG=1: int4 W4A8 GEMM for the looped layers' msg row slices of wqkvrg
bool g_int4_dense_mv = true;  // IK2_INT4_DENSE_MV=0: verify projections via 256-tiled WMMA GEMM (7/8 waves idle at T=8)
bool g_int4_prefill = false;  // IK2_INT4_PREFILL=1: int4 W4A8 WMMA GEMM for ALL prefill dense projections (T>8),
                              // reading the VRAM int4 sidecar instead of decoding host/Q8 to f16. Removes the
                              // per-chunk Q8->f16 decode (the PCIe-read bottleneck when dense is host-mapped at 1M).
                              // (T=2..8; reuses the whole-matrix sidecar with ft0-shifted pointers). Default OFF.
bool g_int4_rot_dedup = false;  // IK2_INT4_ROT_DEDUP=1: rotate+quant each activation once per run_layer_attn call
                                // (the msg branch projects in_q twice). Epoch-gated per call, never across visits.
                                // Default OFF: the shared-scratch rotate-every-call path stays byte-identical.
bool g_int4_mv_alt = false;   // IK2_INT4_MV_ALT=1: fuse the sliding msg branch's 4 wqkvrg row-slice matvecs into ONE
                              // k_mv_alt launch over the full fused sidecar (per-tile activation select by row range).
                              // Needs IK2_INT4_MSG + the dense MV path; default OFF keeps the 4-call path byte-identical.
bool g_mv_lds = false;        // IK2_MV_LDS=1: dense verify matvec via k_mv_lds -- the NC-token activation staged into
                              // LDS once per CTA, dot loop reads per-group slices from LDS (fewer live activation
                              // VGPRs at NC=8). Bit-exact vs int4_matvec; default OFF keeps that path byte-identical.
bool g_int4_head = false;     // IK2_INT4_HEAD=1: int4 W4A8 GEMM for the verify HEAD projection out_w (T=2..8). Default
                              // OFF: the head stays f16/cuBLAS byte-identical. The drafter's fold reads M.out_w.f16
                              // (the f16 WEIGHT copy), which this never touches; verify logits stay f32 in W.logits.
bool g_int4_head_decode = false;  // IK2_INT4_HEAD_DECODE=1: T=1 decode head over the out_w INT4 sidecar (VRAM) instead
                              // of the Q8 blocks. With IK2_DENSE_HOST=1 the Q8 head is host-mapped, so every token
                              // read ~490 MiB over PCIe (60 ms = 81.7% of the 1M-ctx decode); the int4 sidecar is
                              // ~240 MiB in VRAM (~1 ms). Needs IK2_INT4_HEAD=1 (sidecar loaded). Greedy may differ
                              // from the Q8 head (int4 vs Q8 quantization) - measure parity.
bool g_draft_head_int4 = true;    // IK2_DRAFT_HEAD_INT4=0 to disable: drafter base-logits head over out_w INT4 sidecar
                              // INT4 sidecar (NC=7 matvec) instead of the FP16 head copy. Needs the sidecar loaded
                              // (IK2_INT4_HEAD=1 + IK2_DENSE=q8); without it propose() falls back to gemm_h. Default
                              // OFF keeps the draft byte-identical. fold still reads M.out_w.f16 either way.
bool g_exp_prefill_wmma = false;   // IK2_EXPERTS_PREFILL_WMMA=1: NVFP4-direct multi-token WMMA prefill experts (skip decode_f16)
int g_n_sm = 0;
bool g_group = true;   // IK2_GROUP=0: 2 .. 8 tokens' experts a slot at a time, their attention a head at a time
bool g_group_exp = true, g_group_attn = true;   // (IK2_GROUP=1: the experts grouped only, 2: the attention only)
FILE* g_trace = nullptr;   // IK2_TRACE=FILE: each layer's picks, records of int32 (layer, T, K, T x K experts)
struct Prof {
    double attn = 0, route = 0, upload = 0, experts = 0, rest = 0;
    double pf_host = 0, pf_gpu = 0, pf_sync = 0;   // run_prefill_moe: host bookkeeping / GPU expert work / cache waits
    double pf_dec = 0, pf_gemm = 0, pf_gather = 0, pf_scatter = 0, pf_swiglu = 0;   // run_exp phase split
    double a_proj = 0, a_conv = 0, a_qk = 0, a_probs = 0, a_pv = 0, a_out = 0;   // run_layer_attn sub-phase split
    double pf_loopgate = 0;   // loop_gates_chunk (FP32 cuBLAS rank-64 products)
    int pf_calls = 0;
    long br_rg = 0, br_mv = 0, br_pf = 0;   // IK2_BRANCH: which MoE path fires (resident-grouped / token / run_prefill_moe)
    double t = 0;
    void start() {
        if (g_prof) cudaStreamSynchronize(g_s), t = now();
    }
    void lap(double& acc) {
        if (!g_prof) return;
        cudaStreamSynchronize(g_s);
        const double n = now();
        acc += n - t;
        t = n;
    }
} g_pf;

template <class T>
T* dalloc(size_t n) {
    T* p = nullptr;
    size_t fr_b = 0, fr_a = 0, tt = 0;
    const bool tr = getenv("IK2_VRAM_TRACE") != nullptr;
    if (tr) CU(cudaMemGetInfo(&fr_b, &tt));
    CU(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)));
    if (tr) {
        CU(cudaMemGetInfo(&fr_a, &tt));
        static size_t tot = 0; static int cnt = 0;
        const size_t b = std::max<size_t>(1, n) * sizeof(T);
        tot += b; ++cnt;
        const size_t drop = fr_b > fr_a ? fr_b - fr_a : 0;
        if (drop >= (64ull << 20) || cnt % 40 == 0)
            std::printf("DALLOC #%d req %zu MiB drop %zu MiB free %.2f GiB (cum %zu MiB)\n", cnt, b >> 20, drop >> 20, fr_a / 1073741824.0, tot >> 20);
    }
    return p;
}
// IK2_VRAM_TRACE: checkpoint free VRAM at a named load phase (finds the ctx-scaled hog outside dalloc).
static void vram_mark(const char* tag) {
    if (!getenv("IK2_VRAM_TRACE")) return;
    size_t f = 0, t = 0; CU(cudaMemGetInfo(&f, &t));
    std::printf("VRAM-MARK %-22s free %.2f GiB\n", tag, f / 1073741824.0);
    std::fflush(stdout);
}
// IK2_KV_HOST_CAP: host-mapped (zero-copy) allocation for the long-context int KV. The device pointer reads/writes
// host RAM over PCIe, so a 1M-token KV (8.4 GiB) costs ~0 VRAM and the model stays resident. Slower per token than
// VRAM-resident KV (the attention reads the whole cache over PCIe) but it is the only way 1M fits beside the experts.
// Returns a DEVICE pointer into pinned mapped host memory; *host_out is the host pointer (for free / serve).
template <class T>
T* halloc_mapped(size_t n, void** host_out) {
    void* h = nullptr;
    CU(cudaHostAlloc(&h, std::max<size_t>(1, n) * sizeof(T), cudaHostAllocMapped));
    void* d = nullptr;
    CU(cudaHostGetDevicePointer(&d, h, 0));
    if (host_out) *host_out = h;
    if (getenv("IK2_VRAM_TRACE"))
        std::printf("HALLOC-MAPPED req %zu MiB (host RAM, device-visible)\n", (std::max<size_t>(1, n) * sizeof(T)) >> 20);
    return (T*) d;
}

// Y [T][N] (ldy) = X [T][K] FP16 . W [N][K]^T FP16, F32 sums
#if defined(STRATA_USE_HIP) && defined(IK2_WMMA_DEBUG)
// Debug: run BOTH the gfx11 WMMA GEMM and hipBLAS, compare, report the worst shape, keep hipBLAS's result (correct).
static void gemm16_wmma_debug(const uint16_t* W, int N, int Kd, const uint16_t* X, int T, float* Y, int ldy, float beta, bool use_wmma) {
    const size_t yn = (size_t) T * ldy;
    static float* yw = nullptr;
    static size_t yw_cap = 0;
    if (yn > yw_cap) { if (yw) cudaFree(yw); CU(cudaMalloc(&yw, yn * sizeof(float))); yw_cap = yn; }
    CU(cudaMemsetAsync(yw, 0, yn * sizeof(float), g_s));
    const bool ran = strata_wmma_gemm_f16(X, W, yw, T, N, Kd, ldy, beta, g_s);
    const float one = 1.0f;
    CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, Kd, &one, W, CUDA_R_16F, Kd, X, CUDA_R_16F, Kd, &beta, Y,
                    CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CU(cudaStreamSynchronize(g_s));
    if (!ran) { std::printf("WMMA_DBG declined N=%d K=%d T=%d ldy=%d beta=%.0f\n", N, Kd, T, ldy, beta); return; }
    std::vector<float> a(yn), b(yn);
    CU(cudaMemcpy(a.data(), yw, yn * sizeof(float), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(b.data(), Y, yn * sizeof(float), cudaMemcpyDeviceToHost));
    double maxd = 0; size_t wi = 0;
    for (size_t i = 0; i < yn; ++i) {
        const double d = std::fabs((double) a[i] - (double) b[i]);
        if (d > maxd) { maxd = d; wi = i; }
    }
    if (maxd > 1e-2) {
        std::printf("WMMA_DBG N=%d K=%d T=%d ldy=%d beta=%.0f maxd=%.4g at t=%zu n=%zu wmma=%.5g blas=%.5g\n",
                    N, Kd, T, ldy, beta, maxd, wi / ldy, wi % ldy, a[wi], b[wi]);
        std::fflush(stdout);
    }
    if (use_wmma && ran) CU(cudaMemcpyAsync(Y, yw, yn * sizeof(float), cudaMemcpyDeviceToDevice, g_s));
}
#endif
void gemm16(const uint16_t* W, int N, int Kd, const uint16_t* X, int T, float* Y, int ldy, float beta = 0.0f) {
#if defined(STRATA_USE_HIP)
#if defined(IK2_WMMA_DEBUG)
    {
        // IK2_WMMA_DEBUG=1 compares WMMA vs hipBLAS but keeps hipBLAS; =2 uses WMMA's result (test correctness).
        static const int dbg_use = getenv("IK2_WMMA_DEBUG") ? std::atoi(getenv("IK2_WMMA_DEBUG")) : 1;
        gemm16_wmma_debug(W, N, Kd, X, T, Y, ldy, beta, dbg_use == 2);
        return;
    }
#endif
    // IK2_WMMA_GEMM=1: the prompt projections on the gfx11 matrix cores instead of hipBLAS (opt-in; the default
    // path is byte-identical). The wide shapes take the tuned 128x256 BK64 kernel (T>=64, N>=1024, K%64==0);
    // everything else keeps hipBLAS. The 64x64 double-buffer kernel races on gfx1100, so it is never selected
    // here. IK2_WMMA_TILE=1 forces the small single-wave path (debug).
    static const int wmma_gemm = getenv("IK2_WMMA_GEMM") ? std::atoi(getenv("IK2_WMMA_GEMM")) : 0;
    if (wmma_gemm) {
        // The tuned 128x256 BK64 kernel wins on wide projections (T>=128, N>=1024); the small single-wave
        // kernel wins on medium ones (T>=32, N>=512). Tiny shapes keep hipBLAS. The 64x64 double-buffer
        // kernel races on gfx1100, so it is never selected. IK2_WMMA_TILE=1 forces the small path (debug).
        static const int force_small = getenv("IK2_WMMA_TILE") ? std::atoi(getenv("IK2_WMMA_TILE")) : 0;
        const bool wide = T >= 128 && N >= 1024 && Kd % 64 == 0;
        const bool medium = T >= 32 && N >= 512 && Kd % 16 == 0;
        bool ok = false;
        if (force_small) ok = strata_wmma_gemm_f16(X, W, Y, T, N, Kd, ldy, beta, g_s);
        else if (wide) ok = strata_pf_gemm_f16(X, W, Y, T, N, Kd, ldy, beta, g_s);
        else if (medium) ok = strata_wmma_gemm_f16(X, W, Y, T, N, Kd, ldy, beta, g_s);
        if (ok) return;
    }
#endif
    const float one = 1.0f;
    CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, Kd, &one, W, CUDA_R_16F, Kd, X, CUDA_R_16F, Kd, &beta, Y,
                    CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

uint8_t* g_stage = nullptr;   // a raw tensor on its way to its FP16 decode
size_t g_stage_bytes = 0;

bool g_dense_q8 = false;
// The KV cache in FP16 (IK2_KV=fp8: E4M3, half the bytes).  Against ref/inkling2.py over 48 tokens FP16 kept every
// top-1 token (KL mean 8.5e-4) and FP8 changed 5 (KL 8.3e-3): Inkling's router is full of near-ties, so FP8's 1.5% in
// layer 0's attention output already moves 7 of 48 tokens' experts.  The full layers' cache is 32 KB a token in FP16.
bool g_kv_fp8 = false;
// INT8 / INT4 KV (IK2_KV=int8 / int4): per-(token, KV-head) FP16 scale + symmetric int codes, software dequant.
// int8 ~1.016 B/elem (precision baseline), int4 ~0.516 B/elem (halves the KV read bandwidth of fp8).  Kept as
// separate flags so every existing g_kv_fp8 site keeps its exact FP8 semantics and the f16 default is untouched.
enum KvMode { KV_F16 = 0, KV_FP8 = 1, KV_INT8 = 2, KV_INT4 = 3 };
KvMode g_kv_mode = KV_F16;
bool g_kv_int8 = false, g_kv_int4 = false;
inline bool kv_is_int() { return g_kv_int8 || g_kv_int4; }
inline int kv_int_bits() { return g_kv_int4 ? 4 : 8; }
// IK2_KV_HOST: host-map the WHOLE int KV of the full-attention visits (the ctx-scaled hog). 0 = VRAM (default,
// byte-identical); 1 = host-mapped zero-copy (1M fits beside the experts, decode reads the cache over PCIe).
int g_kv_host = 0;
// IK2_DENSE_HOST: host-map the Q8_0 planar dense copies (prefill-only when int4 sidecars serve decode). Frees ~1.1 GiB
// VRAM so the 1M int4 KV (8.4 GiB) fits beside the experts ALL IN VRAM -> 1M decode reads KV at VRAM speed, not PCIe.
// Prefill decodes these host-mapped Q8 weights to f16 over PCIe (slower prefill, faster 1M decode). 0 = VRAM (default).
int g_dense_host = 0;
// Q8_0 blocks (2 bytes of FP16 scale, 32 int8 values) taken apart into m.q8 / m.q8d on the device; false if the
// matrix cannot be kept so (another type, or K not a multiple of 128)
bool upload_planar(Mat& m, const std::vector<const uint8_t*>& parts, const std::vector<size_t>& part_values,
                   bool force_vram = false) {
    if (m.type != k::kQ8_0 || m.K % 128) return false;
    size_t n = 0;
    for (size_t v : part_values) n += v;
    std::vector<int8_t> q(n);
    std::vector<uint16_t> d(n / 32);
    size_t b = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
        const uint8_t* p = parts[i];
        for (size_t j = 0; j < part_values[i] / 32; ++j, ++b, p += 34) {
            std::memcpy(&d[b], p, 2);
            std::memcpy(&q[b * 32], p + 2, 32);
        }
    }
    if (g_dense_host && !force_vram) {
        // host-mapped zero-copy: prefill decode_q8p_f16 reads these over PCIe; decode uses int4 sidecars (VRAM).
        m.q8 = halloc_mapped<int8_t>(n, nullptr);
        m.q8d = halloc_mapped<uint16_t>(n / 32, nullptr);
    } else {
        m.q8 = dalloc<int8_t>(n);
        m.q8d = dalloc<uint16_t>(n / 32);
    }
    CU(cudaMemcpy(m.q8, q.data(), n, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(m.q8d, d.data(), n / 32 * 2, cudaMemcpyHostToDevice));
    return true;
}

Mat load_mat(const Shards& S, const std::string& n, int N, int Kd, bool force_vram = false, bool skip_q8 = false) {
    const auto* t = S.info(n);
    if (!t) die("no tensor " + n);
    if (t->elements() != (uint64_t) N * Kd) die(n + ": " + std::to_string(t->elements()) + " values");
    if (!k::block_values((int) t->type) || Kd % k::block_values((int) t->type))
        die(n + ": type " + t->type_name() + " not read here");
    Mat m;
    m.type = (int) t->type, m.N = N, m.K = Kd;
    const size_t nb = tensor_bytes(*t);
    if (skip_q8) {
        // IK2_INT4_HEAD_DECODE: the head is loaded as its INT4 sidecar (below); no Q8/raw copy. The sidecar is
        // the only weight copy - decode and prefill both read it. m.type stays kQ8_0 for any block fallback.
        return m;
    }
    if (g_dense_q8) {
        if (upload_planar(m, {S.data(n)}, {(size_t) N * Kd}, force_vram)) return m;
        m.raw = dalloc<uint8_t>(nb);
        CU(cudaMemcpy(m.raw, S.data(n), nb, cudaMemcpyHostToDevice));
        return m;
    }
    if (nb > g_stage_bytes) {
        if (g_stage) cudaFree(g_stage);
        g_stage = dalloc<uint8_t>(g_stage_bytes = nb);
    }
    CU(cudaMemcpy(g_stage, S.data(n), nb, cudaMemcpyHostToDevice));
    m.f16 = dalloc<uint16_t>((size_t) N * Kd);
    k::decode_f16(m.type, g_stage, (size_t) N * Kd, m.f16, g_s);
    CU(cudaStreamSynchronize(g_s));
    return m;
}

struct SegPack {
    float* dst[6];
    int off[7];
    int n;
};
// (float4 a thread when every width is a multiple of 4 - the segments' lookup unrolled, so the parameters stay in
// the constant bank instead of being copied to local memory for a dynamic index)
__global__ void unpack_segments4_kernel(const float4* __restrict__ src, int T, SegPack p) {
    const int c4 = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    const int stride4 = p.off[p.n] >> 2;
    if (c4 >= stride4 || t >= T) return;
    const float4 v = src[(size_t) t * stride4 + c4];
    const int col = c4 << 2;
#pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < p.n && col >= p.off[s] && col < p.off[s + 1]) {
            const int w = p.off[s + 1] - p.off[s];
            *(float4*) (p.dst[s] + (size_t) t * w + (col - p.off[s])) = v;
        }
    }
}
__global__ void unpack_segments_kernel(const float* __restrict__ src, int T, SegPack p) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int stride = p.off[p.n];
    if (col >= stride || t >= T) return;
    const float v = src[(size_t) t * stride + col];
#pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < p.n && col >= p.off[s] && col < p.off[s + 1]) {
            const int w = p.off[s + 1] - p.off[s];
            p.dst[s][(size_t) t * w + (col - p.off[s])] = v;
        }
    }
}
void unpack_segments(const float* src, int T, const std::initializer_list<std::pair<float*, int>>& segs, cudaStream_t s) {
    SegPack p{};
    p.n = (int) segs.size();
    int off = 0, idx = 0;
    bool vec = ((uintptr_t) src & 15) == 0;
    p.off[0] = 0;
    for (const auto& sg : segs) {
        p.dst[idx] = sg.first;
        vec = vec && (sg.second & 3) == 0 && ((uintptr_t) sg.first & 15) == 0;
        off += sg.second;
        p.off[++idx] = off;
    }
    if (vec) {
        const dim3 grid((unsigned) ((off / 4 + 255) / 256), (unsigned) T);
        unpack_segments4_kernel<<<grid, 256, 0, s>>>((const float4*) src, T, p);
    } else {
        const dim3 grid((unsigned) ((off + 255) / 256), (unsigned) T);
        unpack_segments_kernel<<<grid, 256, 0, s>>>(src, T, p);
    }
}

struct MatPart {
    std::string name;
    int N = 0;
    std::vector<float> f32;
};

void ensure_stage(size_t nb) {
    if (nb > g_stage_bytes) {
        if (g_stage) cudaFree(g_stage);
        g_stage = dalloc<uint8_t>(g_stage_bytes = nb);
    }
}

Mat load_fused_f16(const Shards& S, const std::vector<MatPart>& parts, int Kd) {
    int N_tot = 0;
    for (const auto& pt : parts) N_tot += pt.N;
    Mat m;
    m.N = N_tot;
    m.K = Kd;
    m.f16 = dalloc<uint16_t>((size_t) N_tot * Kd);
    int row = 0;
    for (const auto& pt : parts) {
        if (pt.f32.empty()) {
            const auto* t = S.info(pt.name);
            if (!t) die("no tensor " + pt.name);
            if (t->elements() != (uint64_t) pt.N * Kd) die(pt.name + ": " + std::to_string(t->elements()) + " values");
            if (!k::block_values((int) t->type) || Kd % k::block_values((int) t->type))
                die(pt.name + ": type " + t->type_name() + " not read here");
            const size_t nb = tensor_bytes(*t);
            ensure_stage(nb);
            CU(cudaMemcpy(g_stage, S.data(pt.name), nb, cudaMemcpyHostToDevice));
            k::decode_f16((int) t->type, g_stage, (size_t) pt.N * Kd, m.f16 + (size_t) row * Kd, g_s);
        } else {
            const size_t nb = pt.f32.size() * 4;
            ensure_stage(nb);
            CU(cudaMemcpy(g_stage, pt.f32.data(), nb, cudaMemcpyHostToDevice));
            k::to_f16((const float*) g_stage, pt.f32.size(), m.f16 + (size_t) row * Kd, g_s);
        }
        row += pt.N;
    }
    CU(cudaStreamSynchronize(g_s));
    return m;
}

bool try_load_fused_raw(const Shards& S, const std::vector<MatPart>& parts, int Kd, Mat& out) {
    int ty = -1, N_tot = 0;
    size_t nb_tot = 0;
    for (const auto& pt : parts) {
        if (!pt.f32.empty()) return false;
        const auto* t = S.info(pt.name);
        if (!t) die("no tensor " + pt.name);
        if (t->elements() != (uint64_t) pt.N * Kd) die(pt.name + ": " + std::to_string(t->elements()) + " values");
        if (!k::block_values((int) t->type) || Kd % k::block_values((int) t->type))
            die(pt.name + ": type " + t->type_name() + " not read here");
        if (ty < 0) ty = (int) t->type;
        else if (ty != (int) t->type) return false;
        N_tot += pt.N;
        nb_tot += tensor_bytes(*t);
    }
    out = Mat{};
    out.type = ty;
    out.N = N_tot;
    out.K = Kd;
    {
        std::vector<const uint8_t*> ps;
        std::vector<size_t> vs;
        for (const auto& pt : parts) ps.push_back(S.data(pt.name)), vs.push_back((size_t) pt.N * Kd);
        if (upload_planar(out, ps, vs)) return true;
    }
    out.raw = dalloc<uint8_t>(nb_tot);
    size_t off = 0;
    for (const auto& pt : parts) {
        const size_t nb = tensor_bytes(*S.info(pt.name));
        CU(cudaMemcpy(out.raw + off, S.data(pt.name), nb, cudaMemcpyHostToDevice));
        off += nb;
    }
    return true;
}

// one token's product over a dense matrix kept as the GGUF's blocks, or planar
void mv1(const Mat& m, const float* x, float* y) {
    if (m.i4Wp && (!m.i4_head_only || g_int4_head_decode) && !m.i4_row_slice) {   // IK2_INT4_DENSE: rotated-int4 sidecar, rotation folded into the matvec prologue (head too when IK2_INT4_HEAD_DECODE)
        k::int4_dense_rot(m.i4Wp, m.i4Wh, m.i4WS, m.N, m.K, x, y, k::MvConv{}, k::MvConv{}, 1.f / (1 << k::kInt4Sh8), g_s);
        return;
    }
    if (m.q8) k::mv_q8p(m.q8, m.q8d, m.N, m.K, x, y, g_s);
    else k::mv_blocks(m.type, m.raw, m.N, m.K, x, y, g_s);
}
// ... with up to two short convs on its rows: the planar Q8 product has them as its epilogue (a launch less each;
// those rows of y are then not written), any other matrix gets the product and then the convs
void mv1c(const Mat& m, const float* x, float* y, const k::MvConv& c0, const k::MvConv& c1 = k::MvConv{}) {
    if (m.i4Wp && !m.i4_head_only && !m.i4_row_slice) {   // IK2_INT4_DENSE: same fused conv epilogue as the planar Q8 path, over the int4 product
        k::int4_dense_rot(m.i4Wp, m.i4Wh, m.i4WS, m.N, m.K, x, y, c0, c1, 1.f / (1 << k::kInt4Sh8), g_s);
        return;
    }
    if (m.q8) {
        k::mv_q8p_conv(m.q8, m.q8d, m.N, m.K, x, y, c0, c1, g_s);
        return;
    }
    k::mv_blocks(m.type, m.raw, m.N, m.K, x, y, g_s);
    for (const k::MvConv* cv : {&c0, &c1})
        if (cv->n) k::conv4_res(y + cv->r0, cv->hist, cv->w, cv->dyn, cv->heads, 1, cv->n, cv->res, cv->y, g_s);
}

uint16_t* g_w16 = nullptr;   // a matrix's FP16 decode for the prompt path's GEMMs (the largest of a layer's)
size_t g_w16_n = 0;          // values it holds
// Y [T][N] = X [T][K] . W^T: one token straight over the blocks (x32), more through an FP16 decode and a GEMM (x16).
// A matrix larger than the decode buffer (the head: its few rows of logits) goes a token at a time over its blocks.
bool try_int4_dense_multi(const Mat& m, const float* x32, int T, float* Y, int ldy);
bool try_int4_dense_mv(const Mat& m, const float* x32, int T, float* Y, int ldy);
void matmul(const Mat& m, const float* x32, const uint16_t* x16, int T, float* Y, int ldy) {
    // IK2_INT4_DENSE_MULTI for the verify projections; IK2_INT4_HEAD for the head out_w (i4_head_only);
    // IK2_INT4_MSG for wqkvrg row slices (i4_row_slice). The head sidecar is reserved for this T=2..8 GEMM:
    // mv1/mv1c and the DENSE_MULTI gate skip it, so T=1 decode and the prompt path (chunks of 16) keep their
    // existing route even when IK2_INT4_HEAD=1.
    const bool i4_multi = m.i4_head_only ? g_int4_head : (m.i4_row_slice ? g_int4_msg : g_int4_dense_multi);
    // IK2_BATCH_DENSE_MV: W.batch T>1 dense projections (wqkvrg/wqrg/wk/wv/wo) use the SAME row-aware INT4
    // matvec the T=1 resident path uses, so one weight row is read from VRAM for all T rows. Head and msg row
    // slices keep their existing route (the sidecar is whole-matrix / head-only); a mat without an i4Wp sidecar
    // falls through to the existing Q8->f16 GEMM path unchanged.
    const bool batch_mv = batch_dense_mv_on(T) && g_int4_dense && !m.i4_head_only && !m.i4_row_slice &&
                          batch_dense_mv_mat_ok(m) && m.i4Wp && m.i4Wh && m.i4WS;
    // IK2_BATCH_ROTMV: the batched T=2..8 dense projections take k::int4_dense_rot_rows, the exact multi-row
    // generalization of the T=1 int4_dense_rot path (same rotate_block/quant_rot8/sdot4/swap_rows reduction
    // order per row, weight row read once for all T rows). No convs here: the batch path applies conv_rec_w /
    // conv4_w AFTER matmul, exactly as the T=1 path does for these matrices, so the raw product is what
    // matmul must return (nconv=0). Head and msg row slices keep their route; a mat without an i4Wp sidecar
    // falls through unchanged.
    const bool batch_rotmv = batch_rotmv_on(T) && g_int4_dense && !m.i4_head_only && !m.i4_row_slice &&
                             m.i4Wp && m.i4Wh && m.i4WS && x32 &&
                             m.K % 256 == 0 && m.N % 16 == 0 && ldy == m.N;
    if (batch_graph_exact_on(T) && g_batch_active && m.i4Wp && !m.i4_head_only && !m.i4_row_slice &&
        m.K % 256 == 0 && m.N % 16 == 0 && ldy == m.N) {
        if (getenv("IK2_GXE_HASH")) std::fprintf(stderr, "GXM matmul_exact N=%d K=%d T=%d ldy=%d x32=%d\n", m.N, m.K, T, ldy, x32 != nullptr);
        if (x32) {
            if (getenv("IK2_GXE_HASH")) std::fprintf(stderr, "GXM wo_int4rows N=%d K=%d T=%d ldy=%d\n", m.N, m.K, T, ldy);
            k::int4_dense_rot_rows(m.i4Wp, m.i4Wh, m.i4WS, m.N, m.K, x32, T, Y, ldy,
                                   1.f / (1 << k::kInt4Sh8), g_s);
        } else {
            if (getenv("IK2_GXE_HASH")) std::fprintf(stderr, "GXM wo_fallback N=%d K=%d T=%d ldy=%d\n", m.N, m.K, T, ldy);
            for (int t = 0; t < T; ++t) {
                float xbuf[8192];
                const uint16_t* src = x16 + (size_t) t * m.K;
                for (int i = 0; i < m.K; ++i) xbuf[i] = __half2float(__ushort_as_half(src[i]));
                mv1(m, xbuf, Y + (size_t) t * ldy);
            }
        }
        return;
    }
    if (batch_rotmv) {
        k::int4_dense_rot_rows(m.i4Wp, m.i4Wh, m.i4WS, m.N, m.K, x32, T, Y, ldy,
                               1.f / (1 << k::kInt4Sh8), g_s);
        return;
    }
    // IK2_INT4_DENSE_MV: verify projections over the decode-style dp4a matvec instead of the prefill WMMA GEMM.
    // The GEMM tiles the token dim to 256, so T=8 leaves 7 of 8 waves idle; the matvec reads each weight row once
    // and reuses it across the T tokens (NC). rot+quant once, then one launch per matrix (slices keep ldy).
    if ((i4_multi || batch_mv) && g_int4_dense_mv && m.i4Wp && m.i4Wh && m.i4WS && x32 && T > 1 && T <= 8 &&
        m.K % 256 == 0 && m.N % 16 == 0 && (m.i4_row_slice ? ldy >= m.N : ldy == m.N) &&
        try_int4_dense_mv(m, x32, T, Y, ldy))
        return;
    if (i4_multi && m.i4Wp && m.i4Wh && m.i4WS && x32 && T > 1 && T <= 8 &&
        m.K % 256 == 0 && m.N % 32 == 0 && (m.i4_row_slice ? ldy >= m.N : ldy == m.N) &&
        try_int4_dense_multi(m, x32, T, Y, ldy))
        return;
    // IK2_INT4_PREFILL: the int4 W4A8 WMMA GEMM for ANY prefill chunk (T>8). Reads the VRAM int4 sidecar
    // directly (no Q8->f16 decode), which is the lever when dense is host-mapped at long context. The GEMM
    // tiles N to 256, so it is efficient for large T (unlike T=8 where the dp4a matvec above wins).
    if (g_int4_prefill && !m.i4_head_only && m.i4Wp && m.i4Wh && m.i4WS && x32 && T > 8 &&
        m.K % 256 == 0 && m.N % 32 == 0 && (m.i4_row_slice ? ldy >= m.N : ldy == m.N) &&
        try_int4_dense_multi(m, x32, T, Y, ldy))
        return;
    if (m.f16) {
        gemm16(m.f16, m.N, m.K, x16, T, Y, ldy);
        return;
    }
    if (m.K <= 8192 && (T == 1 || (size_t) m.N * m.K > g_w16_n)) {
        for (int t = 0; t < T; ++t) mv1(m, x32 + (size_t) t * m.K, Y + (size_t) t * ldy);
        return;
    }
    if (m.q8) k::decode_q8p_f16(m.q8, m.q8d, (size_t) m.N * m.K, g_w16, g_s);
    else k::decode_f16(m.type, m.raw, (size_t) m.N * m.K, g_w16, g_s);
    gemm16(g_w16, m.N, m.K, x16, T, Y, ldy);
}

std::vector<float> host_f32(const Shards& S, const std::string& n, size_t expect) {
    const auto* t = S.info(n);
    if (!t) die("no tensor " + n);
    if (t->elements() != expect) die(n + ": " + std::to_string(t->elements()) + " values, expected " + std::to_string(expect));
    std::vector<float> v(expect);
    const uint8_t* p = S.data(n);
    if (t->type == 0) {
        std::memcpy(v.data(), p, expect * 4);
    } else if (t->type == 30) {
        for (size_t i = 0; i < expect; ++i) v[i] = bf16_to_f32((uint16_t) (p[2 * i] | (p[2 * i + 1] << 8)));
    } else if (k::block_values((int) t->type)) {
        k::decode_host((int) t->type, p, expect, v.data());
    } else {
        die(n + ": type " + t->type_name());
    }
    return v;
}
float* dev_f32(const std::vector<float>& v) {
    float* p = dalloc<float>(v.size());
    CU(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    return p;
}
uint16_t* dev_f16(const std::vector<float>& v) {
    const size_t nb = v.size() * 4;
    ensure_stage(nb);
    CU(cudaMemcpy(g_stage, v.data(), nb, cudaMemcpyHostToDevice));
    uint16_t* p = dalloc<uint16_t>(v.size());
    k::to_f16((const float*) g_stage, v.size(), p, g_s);
    CU(cudaStreamSynchronize(g_s));
    return p;
}
float* load_f32(const Shards& S, const std::string& n, size_t expect) { return dev_f32(host_f32(S, n, expect)); }
uint16_t* load_f16(const Shards& S, const std::string& n, size_t expect) { return dev_f16(host_f32(S, n, expect)); }

// ---- loading the experts into pinned host memory: pread from the files, several at once

struct Pinned {
    struct Block {
        void* ptr = nullptr;
        size_t bytes = 0;
        bool mapped = false;
        bool registered = false;
    };
    std::vector<Block> blocks;
    ~Pinned() {
        for (const auto& b : blocks) {
            if (b.mapped) {
                if (b.registered) cudaHostUnregister(b.ptr);
                ::munmap(b.ptr, b.bytes);
            } else if (b.ptr) {
                cudaFreeHost(b.ptr);
            }
        }
    }
};

void read_into_strided(const std::string& path, uint64_t off, int n_exp, size_t exp_bytes, size_t dst_stride,
                       uint8_t* dst) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) die("cannot open " + path);
    const size_t chunk = 64ull << 20;
    for (int e = 0; e < n_exp; ++e) {
        size_t done = 0;
        uint8_t* ep = dst + (size_t) e * dst_stride;
        const uint64_t eoff = off + (uint64_t) e * exp_bytes;
        while (done < exp_bytes) {
            const ssize_t r = ::pread(fd, ep + done, std::min(chunk, exp_bytes - done), (off_t) (eoff + done));
            if (r <= 0) die("read failed: " + path);
            done += (size_t) r;
        }
    }
    posix_fadvise(fd, (off_t) off, (off_t) ((uint64_t) n_exp * exp_bytes), POSIX_FADV_DONTNEED);
    ::close(fd);
}

// ---- the expert cache in VRAM: per-size-class pools, O(1) S3-FIFO with ghost history

// GPU d's half of role r (0 gate, 1 up, 2 down) of an expert: offset in the role's part, bytes (IK2_TP)
inline void tp_part(const Layer& Ly, int d, int r, size_t& off, size_t& bytes) {
    const size_t rows = r < 2 ? (size_t) g_tp_F : (size_t) g_tp_C, rb = Ly.eb[r] / rows, half = rows / 2;
    off = (size_t) d * half * rb, bytes = half * rb;
}
inline size_t tp_bytes(const Layer& Ly, int d) {
    size_t o, b, t = 0;
    for (int r = 0; r < 3; ++r) tp_part(Ly, d, r, o, b), t += b;
    return t;
}
struct ExpertCache {
    static constexpr uint8_t kMaxFreq = 7;
    struct Pool {
        size_t slot_bytes = 0;
        int n_slots = 0, prob_cap = 0, n_layers = 0;
        int n_active = 0;               // slots in use as cache; [n_active, n_slots) lent out (lend)
        uint8_t* data = nullptr;
        std::vector<int> owner;         // slot -> expert idx (l * E + e), -1 if free
        std::vector<uint64_t> used;     // slot -> last tick
        std::vector<uint64_t> sseq;     // slot -> lazy batch seq (0: none)
        std::vector<uint8_t> qid;       // 0=free, 1=prob, 2=main
        std::vector<uint8_t> freq;      // saturating access counter [0..kMaxFreq]
        std::vector<int> prev, next;
        int prob_head = -1, prob_tail = -1, prob_sz = 0;
        int main_head = -1, main_tail = -1, main_sz = 0;
        int free_head = -1;
        uint64_t ghost_seq = 1;
    };

    int n_layers = 0, E = 0;
    std::vector<Pool> pools;
    std::vector<int> layer_pool;       // layer l -> pool index (-1 if on another GPU)
    std::vector<int> loc;              // l * E + e -> slot in pools[layer_pool[l]], or -1
    std::vector<uint64_t> ghost_at;    // l * E + e -> ghost_seq when evicted in decode (0: none)
    int last_handle = -1;
    bool scan = false;   // a long prefill's pass (evict_scan; IK2_SCAN=0: off)
    int tp = -1;         // IK2_TP: this GPU's half of every expert (its index), not whole experts
    uint64_t tick = 0, hits = 0, misses = 0, bytes_up = 0;
    double t_up = 0;

    static void push_back(Pool& P, int q, int s) {
        P.qid[s] = (uint8_t) q;
        P.prev[s] = -1;
        P.next[s] = -1;
        int& hd = q == 1 ? P.prob_head : P.main_head;
        int& tl = q == 1 ? P.prob_tail : P.main_tail;
        int& sz = q == 1 ? P.prob_sz : P.main_sz;
        if (tl < 0) hd = tl = s;
        else P.next[tl] = s, P.prev[s] = tl, tl = s;
        ++sz;
    }
    static int pop_front(Pool& P, int q) {
        int& hd = q == 1 ? P.prob_head : P.main_head;
        int& tl = q == 1 ? P.prob_tail : P.main_tail;
        int& sz = q == 1 ? P.prob_sz : P.main_sz;
        const int s = hd;
        if (s < 0) return -1;
        hd = P.next[s];
        if (hd >= 0) P.prev[hd] = -1;
        else tl = -1;
        P.prev[s] = P.next[s] = -1;
        P.qid[s] = 0;
        --sz;
        return s;
    }
    static int evict_from_prob(Pool& P, uint64_t t, bool& saw_busy_lazy) {
        const int n = P.prob_sz;
        for (int i = 0; i < n; ++i) {
            const int s = pop_front(P, 1);
            if (P.used[s] >= t) {
                push_back(P, 1, s);
                continue;
            }
            if (P.sseq[s] > g_seq_done) {
                lazy_poll();
                if (P.sseq[s] > g_seq_done) {
                    saw_busy_lazy = true;
                    push_back(P, 1, s);
                    continue;
                }
            }
            if (P.freq[s] > 0) {
                P.freq[s] = std::min<uint8_t>(P.freq[s] - 1, 1);
                push_back(P, 2, s);
                continue;
            }
            return s;
        }
        return -1;
    }
    static int evict_from_main(Pool& P, uint64_t t, bool& saw_busy_lazy) {
        int pinned_run = 0;
        while (P.main_sz > 0 && pinned_run < P.main_sz) {
            const int s = pop_front(P, 2);
            if (P.used[s] >= t) {
                ++pinned_run;
                push_back(P, 2, s);
                continue;
            }
            if (P.sseq[s] > g_seq_done) {
                lazy_poll();
                if (P.sseq[s] > g_seq_done) {
                    saw_busy_lazy = true;
                    ++pinned_run;
                    push_back(P, 2, s);
                    continue;
                }
            }
            pinned_run = 0;
            if (P.freq[s] > 0) {
                --P.freq[s];
                push_back(P, 2, s);
                continue;
            }
            return s;
        }
        return -1;
    }
    // A pass that streams nearly every expert of every layer in order (a long prefill), once the cache is full:
    // the victim is the most recently taken probation slot that is free to go (in practice an expert of the layer
    // just computed), not the oldest. Under the usual order such a pass, larger than the cache, evicts the experts of
    // the layers it is about to reach, and the next pass finds none of them (32k after other prompts: 9% hits);
    // like this, the streamed experts take turns in a few slots and the rest of the cache stays for the next pass.
    static int evict_scan(Pool& P, uint64_t t) {
        for (int s = P.prob_tail; s >= 0; s = P.prev[s]) {
            if (P.used[s] >= t) continue;
            if (P.sseq[s] > g_seq_done) {
                lazy_poll();
                if (P.sseq[s] > g_seq_done) continue;
            }
            unlink(P, s);
            return s;
        }
        return -1;
    }
    static int alloc_slot(Pool& P, uint64_t t, bool scan = false) {
        if (P.free_head >= 0) {
            const int s = P.free_head;
            P.free_head = P.next[s];
            P.next[s] = P.prev[s] = -1;
            return s;
        }
        if (scan) {
            const int s = evict_scan(P, t);
            if (s >= 0) return s;
        }
        for (int attempt = 0; attempt < 2; ++attempt) {
            bool saw_busy_lazy = false;
            int s = -1;
            if (P.prob_sz > P.prob_cap || P.main_sz == 0) {
                s = evict_from_prob(P, t, saw_busy_lazy);
                if (s < 0) s = evict_from_main(P, t, saw_busy_lazy);
            } else {
                s = evict_from_main(P, t, saw_busy_lazy);
                if (s < 0) {
                    const int prev_main = P.main_sz;
                    s = evict_from_prob(P, t, saw_busy_lazy);
                    if (s < 0 && P.main_sz > prev_main) s = evict_from_main(P, t, saw_busy_lazy);
                }
            }
            if (s >= 0) return s;
            if (attempt == 0 && saw_busy_lazy) {
                CU(cudaStreamSynchronize(g_cs));
                g_seq_done = g_seq_next - 1;
                continue;
            }
            break;
        }
        die("expert cache: no slot free");
    }

    void init(const std::vector<Layer>& L, int dev, int n_layers_in, int E_in, size_t budget, int int4_below = 0) {
        n_layers = n_layers_in;
        E = E_in;
        layer_pool.assign(n_layers, -1);
        loc.assign((size_t) n_layers * E, -1);
        ghost_at.assign((size_t) n_layers * E, 0);
        struct Group {
            size_t sb;
            std::vector<int> layers;
        };
        std::map<size_t, std::vector<int>> by_sb;
        for (int l = 0; l < n_layers; ++l) {
            if (dev >= 0 && L[l].dev != dev) continue;   // (dev -1: every layer, an expert-parallel GPU)
            const size_t eb = tp >= 0 ? tp_bytes(L[l], tp) : L[l].eb[0] + L[l].eb[1] + L[l].eb[2];
            const size_t sb = (eb + 255) / 256 * 256;
            by_sb[sb].push_back(l);
        }
        std::vector<Group> grps;
        for (auto& kv : by_sb) grps.push_back({kv.first, std::move(kv.second)});
        auto min_req_bytes = [&]() {
            size_t s = 0;
            for (const auto& g : grps) {
                // INT4 pools serve prefill STREAMING only (decode reads the i4_* sidecars, already resident). A small
                // working set suffices; requiring a full 128-expert pool here would OOM at 1M ctx where the KV hog
                // leaves little VRAM. Non-INT4 (Q8 decode-critical) pools still need their full per-layer minimum.
                bool pool_all_int4 = int4_below > 0;
                for (int l : g.layers) if (l >= int4_below) pool_all_int4 = false;
                // An INT4 pool may be sized down to 0 slots (see the clamp below): it never has to exist.
                const int min_e = pool_all_int4 ? 0 : std::min(E, (int) g.layers.size() * E);
                s += (size_t) min_e * g.sb;
            }
            return s;
        };
        while (grps.size() > 1 && (min_req_bytes() > budget || grps.size() > 4)) {
            size_t best_waste = SIZE_MAX;
            size_t best_i = 0;
            for (size_t i = 0; i + 1 < grps.size(); ++i) {
                const size_t w = grps[i].layers.size() * (grps[i + 1].sb - grps[i].sb);
                if (w < best_waste) best_waste = w, best_i = i;
            }
            grps[best_i + 1].layers.insert(grps[best_i + 1].layers.end(), grps[best_i].layers.begin(),
                                           grps[best_i].layers.end());
            grps.erase(grps.begin() + (ptrdiff_t) best_i);
        }
        const size_t hard_min = min_req_bytes();
        if (hard_min > budget)
            die("--cache-gb too small: need at least " + std::to_string((hard_min + (1 << 20) - 1) >> 20) + " MiB");
        size_t soft_min = 0;
        for (const auto& g : grps) soft_min += (size_t) std::min(E + 16, (int) g.layers.size() * E) * g.sb;
        const int min_base = soft_min <= budget ? E + 16 : E;

        pools.resize(grps.size());
        std::vector<int> pmin(grps.size()), pmax(grps.size());
        std::vector<bool> is_int4_pool(grps.size(), false);
        for (size_t p = 0; p < grps.size(); ++p) {
            pools[p].slot_bytes = grps[p].sb;
            pools[p].n_layers = (int) grps[p].layers.size();
            for (int l : grps[p].layers) layer_pool[l] = (int) p;
            pmax[p] = pools[p].n_layers * E;
            // INT4 layers (l<int4_below) use i4_* buffers for decode; their cache slots serve prefill streaming
            // only. Cap them so the VRAM budget leaves the l>=20 Q8 pools fully resident (needed for decode).
            bool all_int4 = int4_below > 0;
            for (int l : grps[p].layers) if (l >= int4_below) all_int4 = false;
            is_int4_pool[p] = all_int4;
            if (all_int4) pmax[p] = std::min(pmax[p], E + 16);
            pmin[p] = std::min(min_base, pmax[p]);
        }
        // Guarantee full residency for the non-INT4 (l>=int4_below, Q8_0 decode-critical) pools FIRST: reserve
        // their pmax up front, then size the capped INT4 prefill-streaming pools from the remainder. Without this
        // the spl binary search + greedy grow can leave a Q8 pool a few slots short when dense INT4 sidecars
        // shrink the budget, dropping the resident path (and long-context decode) entirely.
        if (int4_below > 0) {
            size_t reserved = 0;
            for (size_t p = 0; p < pools.size(); ++p)
                if (!is_int4_pool[p]) reserved += (size_t) pmax[p] * pools[p].slot_bytes;
            if (reserved <= budget) {
                const size_t int4_budget = budget - reserved;
                size_t iused = 0;
                for (size_t p = 0; p < pools.size(); ++p)
                    if (is_int4_pool[p]) iused += (size_t) pmin[p] * pools[p].slot_bytes;
                if (iused > int4_budget) {
                    // The INT4 pools serve prefill STREAMING only (a resident int4 model reads its i4_* sidecars), so
                    // they may be as small as 0 slots. Clamp them to what the budget actually leaves instead of the
                    // E-slot floor: with a larger --prefill the work buffers come first, and the old floor made the
                    // engine allocate past the budget (or die in the guard above) for a cache nothing reads.
                    size_t left = int4_budget;
                    for (size_t p = 0; p < pools.size(); ++p) {
                        if (!is_int4_pool[p]) { pools[p].n_slots = pmax[p]; continue; }
                        const int ns = (int) std::min<size_t>((size_t) pmax[p], left / pools[p].slot_bytes);
                        pools[p].n_slots = ns;
                        left -= (size_t) ns * pools[p].slot_bytes;
                    }
                    goto pools_sized;
                }
                size_t irem = int4_budget > iused ? int4_budget - iused : 0;
                for (size_t p = 0; p < pools.size(); ++p) {
                    if (!is_int4_pool[p]) { pools[p].n_slots = pmax[p]; continue; }
                    int ns = pmin[p];
                    while (ns < pmax[p] && pools[p].slot_bytes <= irem) { ++ns; irem -= pools[p].slot_bytes; }
                    pools[p].n_slots = ns;
                }
                goto pools_sized;
            }
        }
        auto used_for = [&](double spl) {
            size_t u = 0;
            for (size_t p = 0; p < pools.size(); ++p) {
                const int ns = std::clamp((int) std::floor(pools[p].n_layers * spl), pmin[p], pmax[p]);
                u += (size_t) ns * pools[p].slot_bytes;
            }
            return u;
        };
        double lo = 0.0, hi = (double) E;
        for (int it = 0; it < 60; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (used_for(mid) <= budget) lo = mid;
            else hi = mid;
        }
        size_t used = 0;
        for (size_t p = 0; p < pools.size(); ++p) {
            pools[p].n_slots = std::clamp((int) std::floor(pools[p].n_layers * lo), pmin[p], pmax[p]);
            used += (size_t) pools[p].n_slots * pools[p].slot_bytes;
        }
        size_t rem = budget - used;
        bool grew = true;
        while (grew) {
            grew = false;
            for (size_t p = 0; p < pools.size(); ++p) {
                if (pools[p].n_slots < pmax[p] && pools[p].slot_bytes <= rem) {
                    ++pools[p].n_slots;
                    rem -= pools[p].slot_bytes;
                    grew = true;
                }
            }
        }
    pools_sized:
        for (size_t p = 0; p < pools.size(); ++p) {
            Pool& P = pools[p];
            P.prob_cap = std::max(1, P.n_slots / 5);
            P.data = dalloc<uint8_t>((size_t) P.n_slots * P.slot_bytes);
            P.owner.assign(P.n_slots, -1);
            P.used.assign(P.n_slots, 0);
            P.sseq.assign(P.n_slots, 0);
            P.qid.assign(P.n_slots, 0);
            P.freq.assign(P.n_slots, 0);
            P.prev.assign(P.n_slots, -1);
            P.next.assign(P.n_slots, -1);
            for (int s = 0; s < P.n_slots - 1; ++s) P.next[s] = s + 1;
            P.free_head = P.n_slots > 0 ? 0 : -1;
            P.n_active = P.n_slots;
        }
    }
    int total_slots() const {
        int s = 0;
        for (const auto& P : pools) s += P.n_slots;
        return s;
    }
    size_t total_bytes() const {
        size_t b = 0;
        for (const auto& P : pools) b += (size_t) P.n_slots * P.slot_bytes;
        return b;
    }
    uint64_t& slot_sseq(int handle) { return pools[handle >> 20].sseq[handle & 0xfffff]; }
    // the slot of expert e of layer l, its blocks copied in if missing; pinned for this tick (not evicted in it)
    // st: the stream the copy is queued on; *missed (given): whether it was copied in now
    // copy false: the slot is taken but its blocks are the caller's to copy in (copy_in)
    uint8_t* get(const Layer& Ly, int l, int e, uint64_t t, cudaStream_t st = nullptr, bool* missed = nullptr,
                 bool copy = true, bool is_prefill = false) {
        if (!st) st = g_s;
        if (missed) *missed = false;
        const int idx = l * E + e;
        const int p = layer_pool[l];
        Pool& P = pools[p];
        if (loc[idx] >= 0) {
            const int s = loc[idx];
            ++hits;
            P.used[s] = t;
            if (P.freq[s] < kMaxFreq) ++P.freq[s];
            last_handle = (p << 20) | s;
            return P.data + (size_t) s * P.slot_bytes;
        }
        ++misses;
        if (missed) *missed = true;
        const int best = alloc_slot(P, t, scan);
        if (P.owner[best] >= 0) {
            const int old_idx = P.owner[best];
            loc[old_idx] = -1;
            if (!is_prefill) ghost_at[old_idx] = P.ghost_seq;
        }
        if (!is_prefill) ++P.ghost_seq;
        const bool is_ghost = !is_prefill && ghost_at[idx] != 0 && (P.ghost_seq - ghost_at[idx]) <= (uint64_t) P.n_slots;
        P.owner[best] = idx;
        P.used[best] = t;
        P.sseq[best] = 0;
        P.freq[best] = is_ghost ? 1 : 0;
        push_back(P, is_ghost ? 2 : 1, best);
        loc[idx] = best;
        last_handle = (p << 20) | best;
        uint8_t* dst = P.data + (size_t) best * P.slot_bytes;
        if (copy) copy_in(Ly, e, dst, st);
        return dst;
    }
    static void unlink(Pool& P, int s) {
        const int q = P.qid[s];
        if (!q) return;
        int& hd = q == 1 ? P.prob_head : P.main_head;
        int& tl = q == 1 ? P.prob_tail : P.main_tail;
        int& sz = q == 1 ? P.prob_sz : P.main_sz;
        if (P.prev[s] >= 0) P.next[P.prev[s]] = P.next[s];
        else hd = P.next[s];
        if (P.next[s] >= 0) P.prev[P.next[s]] = P.prev[s];
        else tl = P.prev[s];
        P.prev[s] = P.next[s] = -1;
        P.qid[s] = 0;
        --sz;
    }
    static void rebuild_free(Pool& P) {
        P.free_head = -1;
        for (int s = P.n_active - 1; s >= 0; --s)
            if (P.owner[s] < 0 && P.qid[s] == 0) P.next[s] = P.free_head, P.prev[s] = -1, P.free_head = s;
    }
    // Lends the last k cache slots of pool p as one block of scratch (the experts in them are dropped), for a long
    // prefill whose passes stream the experts anyway; give_back() makes them cache slots again. Nothing may be in
    // flight on those slots (the caller synchronizes first).
    uint8_t* lend(int p, int k) {
        Pool& P = pools[p];
        if (k <= 0 || k > P.n_active) return nullptr;
        const int s0 = P.n_active - k;
        for (int s = s0; s < P.n_active; ++s) {
            unlink(P, s);
            if (P.owner[s] >= 0) loc[P.owner[s]] = -1, P.owner[s] = -1;
            P.used[s] = 0, P.sseq[s] = 0, P.freq[s] = 0;
        }
        P.n_active = s0;
        rebuild_free(P);
        return P.data + (size_t) s0 * P.slot_bytes;
    }
    void give_back(int p) {
        Pool& P = pools[p];
        P.n_active = P.n_slots;
        rebuild_free(P);
    }
    bool has(int l, int e) const { return loc[l * E + e] >= 0; }
    void copy_in(const Layer& Ly, int e, uint8_t* dst, cudaStream_t st) {
        if (tp >= 0) {   // (gate's half, up's half, down's half, packed)
            size_t off = 0;
            for (int r = 0; r < 3; ++r) {
                size_t o, b;
                tp_part(Ly, tp, r, o, b);
                CU(cudaMemcpyAsync(dst + off, Ly.host[r] + (size_t) e * Ly.stride[r] + o, b, cudaMemcpyHostToDevice, st));
                off += b, bytes_up += b;
            }
            return;
        }
        if (Ly.host_contig) {
            CU(cudaMemcpyAsync(dst, Ly.host[0] + (size_t) e * Ly.slot_bytes, Ly.slot_bytes, cudaMemcpyHostToDevice, st));
            bytes_up += Ly.slot_bytes;
            return;
        }
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            CU(cudaMemcpyAsync(dst + off, Ly.host[r] + (size_t) e * Ly.stride[r], Ly.eb[r], cudaMemcpyHostToDevice, st));
            off += Ly.eb[r];
            bytes_up += Ly.eb[r];
        }
    }
};

// ---- the forward

struct Work {
    static constexpr int kPackRing = 4;
    static constexpr int kMissBatch = 24;
    int Tm, ctx, nk_max, T_sup = 0;
    size_t blk_ld = 0, lg_cap = 0;
    float *h, *res, *r, *a, *tmp, *xq, *xk, *xv, *q, *kk, *vv, *rr, *gate, *sink, *dyn, *kconv, *vconv, *o, *lg;
    float *qkv;         // fused projection staging (aliases gu32)
    float *blocks;      // [n_blk][blk_ld][C]
    uint16_t *x16, *xk16, *xv16, *xq16_sup = nullptr;
    uint16_t *kv_k16 = nullptr, *kv_v16 = nullptr;
    int kv_tmp_cap = 0;
    int* pos;
    float* rlog;
    int* route_d;       // contiguous [S] top (int) followed by [S] weights (float)
    // the experts
    uint16_t *ex16, *w16, *h16;
    float *gu32, *y32, *sg, *su, *wt;
    int *gidx, *inv;
    const uint8_t** blob;
    int* tok;
    uint8_t *pack_h[kPackRing] = {}, *pack_d[kPackRing] = {};   // ring-buffered token path tables
    uint16_t* tp_h = nullptr;   // IK2_TP: the whole h, [S][F] FP16 (each GPU fills its half, then the other's)
    uint8_t* tp_pk = nullptr;   // (the decode's tables of this layer on the device: canonical blobs at +1536)
    float *y_h[kPackRing] = {};                                 // ring-buffered CPU lane outputs [64][C]
    int pack_slot = 0;
    float *x_h = nullptr;                                       // CPU lane inputs [8][C], pinned
    float *emb_h = nullptr;                                     // pinned host staging
    int *pos_h = nullptr, *route_h = nullptr;
    int *gidx_h[2] = {}, *inv_h[2] = {};
    float *sg_h[2] = {}, *su_h[2] = {}, *wt_h[2] = {};
    int *argmax_d = nullptr, *argmax_h = nullptr;
    // IK2_AR_STREAM2: multi-block argmax partials must be per-Work when two slot graphs replay concurrently.
    float* argmax_part_v = nullptr;
    int* argmax_part_i = nullptr;
    // IK2_AR_KSTEP: K-step graph state, per slot (graphs bake these addresses, so slot 1 gets fresh copies).
    int kstep_K = 0;
    float* emb_dK = nullptr;        // [(K-1)*C] device gather output for steps 1..K-1
    int* argmax_dK = nullptr;       // [K] device argmax writes, all in graph order
    int* argmax_hK = nullptr;       // [K] pinned host, read only after the graph's stream sync
    std::vector<cudaEvent_t> miss_ev;
    // prefill: the next layer's experts copied in while this one runs (prefetch_layer)
    // the prefetches in flight, a ring by layer: their batches' events, expert -> its batch
    // (-1: not prefetched), the layer each slot is for
    static constexpr int kPfRing = 4;
    std::vector<cudaEvent_t> pf_evs[kPfRing];
    std::vector<int> pf_bs[kPfRing];
    int pf_layers[kPfRing] = {-1, -1, -1, -1};
    std::vector<int> miss_pos;   // expert -> its index among this layer's misses (-1: not one)
    std::vector<uint32_t> use;   // [layer * E + e] how many tokens each expert ran in prefills so far
    std::vector<uint64_t> use_tot;   // [layer] their sum
    float* wtr = nullptr;        // [S] the routing weight of each expert-sorted row
    float* wtr_h[2] = {nullptr, nullptr};
    // IK2_INT4_EXPERTS decode scratch
    int8_t *i4_xq = nullptr; float2 *i4_xd = nullptr;        // [C], [C/32]: quantized input activation (verify-fast: [Tm] rows)
    int i4_xq_rows = 1;
    float *i4_hf = nullptr; int8_t *i4_hq = nullptr; float2 *i4_hd = nullptr;   // [S][Kpad], [S][Kpad], [S][Kpad/32]
    const uint8_t **i4_Wp_s = nullptr, **i4_Wh_s = nullptr; const float **i4_WS_s = nullptr;  // [S] per-slot ptrs (gate_up)
    const uint8_t **i4_Wp_dn = nullptr, **i4_Wh_dn = nullptr; const float **i4_WS_dn = nullptr;  // [S] per-slot ptrs (down)
    // IK2_MOE_GROUP_EXPERT: device-side expert->slot grouping of this visit's routing (S<=64 slots).
    // gcnt/gexp [64], gslots [64*8], ngrp [1]. One build serves gate_up and down; fixed matvec grid.
    int *i4_g_cnt = nullptr, *i4_g_slots = nullptr, *i4_g_exp = nullptr, *i4_g_n = nullptr;
    int i4_Kpad_f = 0;
    // IK2_INT4_EXPERTS prefill scratch: one expert's rows at a time (tiled to i4_pf_cap tokens)
    float* i4_pf_x = nullptr;         // [cap][Kmax] f32 activations (gathered input / padded hidden)
    uint8_t* i4_pf_x4 = nullptr;      // two planes of cap * Kmax / 2 bytes each (hi nibbles, then lo)
    uint8_t* i4_pf_xm = nullptr;      // [cap/16][Kmax/256] block maxima, 16 bytes per block
    float* i4_pf_xr = nullptr;        // [cap] row residual scales
    int i4_pf_cap = 0;                // rows the scratch holds (T_buf rounded up to 256)
    int i4_pf_Kmax = 0;               // Kmax the i4_pf_x4/xm/xr scratch was allocated for
    int8_t* i4_mv_xq = nullptr;       // IK2_INT4_DENSE_MV: decode-matvec activation codes [T][Kmax]
    float2* i4_mv_xd = nullptr;       // IK2_INT4_DENSE_MV: per-32-group (d, d*sum) scales [T][Kmax/32]
    // IK2_INT4_MV_ALT: the fused msg matvec needs in_q/in_k/in_v rotated CONCURRENTLY (one launch reads all
    // three by row range). Slots 0/1/2 = in_q/in_k/in_v; slot 0 is reused for the r+gate rows. Same [T][Kmax]
    // layout as i4_mv_xq/xd, allocated only when the env is on. Never aliased with i4_mv_xq/xd or the dedup slots.
    int8_t* i4_ma_xq[3] = {};
    float2* i4_ma_xd[3] = {};
    // IK2_INT4_ROT_DEDUP: per-input rotation slots. Each slot owns its codes/scales, so in_q's rotation survives
    // the in_k/in_v rotations between the msg branch's two in_q projections. A slot is valid only for the
    // run_layer_attn call that filled it: the key includes the call epoch (W.xq/xk/xv are reused across visits
    // with DIFFERENT content, so the pointer alone is never a safe key) and hits are confined to the call by
    // i4_rot_in_attn (RAII in run_layer_attn). Eviction only costs a re-rotation, never correctness.
    static constexpr int kI4RotSlots = 4;   // msg visit: in_q, in_k, in_v + wo's W.o
    int8_t* i4_rd_xq[kI4RotSlots] = {};     // per-slot activation codes [T][Kmax]
    float2* i4_rd_xd[kI4RotSlots] = {};     // per-slot (d, d*sum) scales [T][Kmax/32]
    const float* i4_rd_src[kI4RotSlots] = {};   // slot key: input pointer
    int i4_rd_T[kI4RotSlots] = {};          // slot key: token rows
    int i4_rd_K[kI4RotSlots] = {};          // slot key: activation width
    uint64_t i4_rd_ep[kI4RotSlots] = {};    // slot key: run_layer_attn epoch the codes were rotated in
    uint64_t i4_rot_epoch = 0;              // bumped at every run_layer_attn entry AND exit (host-only)
    bool i4_rot_in_attn = false;            // true only while a run_layer_attn call is on the stack
    int i4_rd_next = 0;                     // round-robin slot for the next miss
    // prefill attention: the relative bias of up to kRelbRows queries x heads, [rows * H][extent] (IK2_RELB=0: off)
    static constexpr int kRelbRows = 512;
    float* relb = nullptr;
    int relb_ext = 0;
    uint16_t* lg16 = nullptr;   // the prefill attention's probabilities (FP16, lg_cap of them; aliases gu32)
    size_t gu32_bytes = 0;   // (gu32's size: a lent scratch only replaces it when it needs more)
    int T_host = 0;          // tokens the pinned host staging (pos_h, route_h, gidx_h, ..) holds
    // IK2_BATCH_ATTN: stable device pointer tables of CURRENT visit's selected rows.  Only the pointees change
    // between visits/batches, so a CUDA graph could capture these addresses later. n_rows <= 8.
    bool batch = false;
    int batch_rows = 0;
    uint8_t **bkci_d = nullptr, **bvci_d = nullptr;
    uint16_t **bks_d = nullptr, **bvs_d = nullptr;
    uint8_t **bkci_h = nullptr, **bvci_h = nullptr;
    uint16_t **bks_h = nullptr, **bvs_h = nullptr;
    float **bh_k_d = nullptr, **bh_v_d = nullptr, **bh_attn_d = nullptr, **bh_mlp_d = nullptr;
    float **bh_k_h = nullptr, **bh_v_h = nullptr, **bh_attn_h = nullptr, **bh_mlp_h = nullptr;
    // IK2_BATCH_MOE_DEVICE removes the general path's per-layer D2H sync, so the pinned host staging arrays above
    // must not be overwritten before their H2D copy has executed. Ring-buffer them and gate reuse on an event.
    static constexpr int kBatchStageSlots = 16;
    cudaEvent_t batch_stage_ev[kBatchStageSlots] = {};
    bool batch_stage_ev_ok[kBatchStageSlots] = {};
    int batch_stage_next = 0;
    // IK2_BATCH_MOE_COMPACT: device-built z->expert map (<= S entries) and active count. No D2H required:
    // the masked GEMM launches min(E,S) z-planes and early-exits beyond this count.
    int* moe_active_map = nullptr;
    int* moe_active_count = nullptr;
    std::vector<int> cnt, start, fill, hit_exps, miss_exps;
    std::vector<uint8_t*> slot_ptrs;
    std::vector<uint32_t> seen_stamp, gi_stamp;
    std::vector<char> seen_late;
    std::vector<int> gi_idx;
    uint32_t stamp = 0;
    float* logits;
    // A looped core's passes ([Tm][C] each): the stream the core was first given; the bank of the pass before (the
    // blocks it added after its first, then its final stream); a layer's four messages (q's, k's, v's, the MoE's);
    // its three attention inputs before the messages.
    float *lp_pre = nullptr, *lp_bank = nullptr, *lp_msg = nullptr, *lp_in = nullptr;
    float* lp_msg_all = nullptr;   // a token's messages of a pass, every core layer's ([n_core][4][C]): one launch a pass
    float* lp_g = nullptr;   // the message gates' scratch [3][Tm][2 rank + 2]
    float *lp_a = nullptr, *lp_f = nullptr;   // ... of a chunk's: [3][Tm][rank] and [2][Tm][C]
    float* logits_h = nullptr;   // the resident token path's logits, pinned [n_vocab]
    int *cand_d = nullptr, *cand_h = nullptr, *cand_part = nullptr;   // ... its candidates (k::cand_topk's out), scratch
    // a short prompt's experts by groups, every expert in VRAM (forward): the tables, pinned and on the device -
    // [rg_groups] pointers | n [rg_groups] | tok, row [rg_groups][kGroupMax] | sg, su, wt, id [rg_slots]
    uint8_t *rg_h = nullptr, *rg_d = nullptr;
    int rg_slots = 0, rg_groups = 0;
    size_t rg_bytes = 0;
};

// IK2_AR_STREAM2: a second Work for slot 1.  The AR graph bakes W's device addresses into every kernel, so a
// concurrent replay on g_s1 needs its own scratch and its own pinned emb/pos/argmax staging.  Only the fast T=1
// path uses this Work; prefill and every other path stay on g_dev[0].W.
Work g_ar_work1;

// Deep-copy the scratch buffers used by fast_token_launches() into dst.  Scalar fields and pointer tables are
// copied by assignment first; every writable device/pinned pointer is then replaced by a fresh allocation.
void clone_ar_work(Work& dst, const Work& src, const Config& c) {
    dst = src;
    const size_t C = c.C, HD = (size_t) c.H * c.D, KD = (size_t) c.KV * c.D;
    const size_t Tm = src.Tm, T_buf = std::max(Tm, (size_t) src.T_sup);
    const size_t S8 = T_buf * c.K, S_buf = S8 * 2;
    const size_t n_core = (size_t) std::max(0, c.core1 - c.core0);

    for (float** p : {&dst.h, &dst.r, &dst.tmp}) *p = dalloc<float>(T_buf * C);
    for (float** p : {&dst.res, &dst.a, &dst.xq, &dst.xk, &dst.xv}) *p = dalloc<float>(Tm * C);
    dst.q = dalloc<float>(Tm * HD); dst.gate = dalloc<float>(Tm * HD); dst.o = dalloc<float>(Tm * HD);
    dst.kk = dalloc<float>(Tm * KD); dst.vv = dalloc<float>(Tm * KD);
    dst.kconv = dalloc<float>(Tm * KD); dst.vconv = dalloc<float>(Tm * KD);
    dst.rr = dalloc<float>(Tm * (size_t) c.H * c.d_rel);
    dst.sink = dalloc<float>(Tm * c.H);
    dst.dyn = dalloc<float>(2 * Tm * (size_t) c.KV * 4);
    dst.lg = dalloc<float>(src.lg_cap);
    dst.blocks = dalloc<float>((size_t) (c.n_layer / c.block + 1) * T_buf * C);
    dst.x16 = dalloc<uint16_t>(Tm * std::max(C, HD));
    dst.xk16 = dalloc<uint16_t>(Tm * C); dst.xv16 = dalloc<uint16_t>(Tm * C);
    if (src.xq16_sup) dst.xq16_sup = dalloc<uint16_t>(T_buf * C); else dst.xq16_sup = nullptr;
    if (src.relb) dst.relb = dalloc<float>((size_t) Work::kRelbRows * c.H * src.relb_ext); else dst.relb = nullptr;
    if (src.kv_k16) dst.kv_k16 = dalloc<uint16_t>((size_t) src.kv_tmp_cap * KD); else dst.kv_k16 = nullptr;
    if (src.kv_v16) dst.kv_v16 = dalloc<uint16_t>((size_t) src.kv_tmp_cap * KD); else dst.kv_v16 = nullptr;
    dst.pos = dalloc<int>(T_buf);
    dst.rlog = dalloc<float>(T_buf * c.E);
    dst.route_d = dalloc<int>(S_buf * 2);
    dst.ex16 = dalloc<uint16_t>(T_buf * C);
    dst.w16 = dalloc<uint16_t>((size_t) 2 * c.F * C);
    dst.h16 = dalloc<uint16_t>(std::max<size_t>(64, T_buf) * c.F);
    dst.gu32 = dalloc<float>(src.gu32_bytes / 4);
    dst.qkv = dst.gu32; dst.lg16 = (uint16_t*) dst.gu32;
    dst.y32 = dalloc<float>(std::max<size_t>(64, T_buf) * C);
    dst.wtr = dalloc<float>(S_buf);
    dst.sg = dalloc<float>(S_buf); dst.su = dalloc<float>(S_buf); dst.wt = dalloc<float>(S_buf);
    dst.gidx = dalloc<int>(S_buf); dst.inv = dalloc<int>(S_buf);
    dst.blob = dalloc<const uint8_t*>(S8); dst.tok = dalloc<int>(S8);

    if (src.i4_xq) {
        dst.i4_xq = dalloc<int8_t>((size_t) src.i4_xq_rows * C);
        dst.i4_xd = dalloc<float2>((size_t) src.i4_xq_rows * (C / 32 + 2));
        dst.i4_hf = dalloc<float>(S8 * src.i4_Kpad_f);
        dst.i4_hq = dalloc<int8_t>(S8 * src.i4_Kpad_f);
        dst.i4_hd = dalloc<float2>(S8 * (src.i4_Kpad_f / 32 + 2));
        dst.i4_Wp_s = dalloc<const uint8_t*>(S8); dst.i4_Wh_s = dalloc<const uint8_t*>(S8);
        dst.i4_WS_s = dalloc<const float*>(S8);
        dst.i4_Wp_dn = dalloc<const uint8_t*>(S8); dst.i4_Wh_dn = dalloc<const uint8_t*>(S8);
        dst.i4_WS_dn = dalloc<const float*>(S8);
    }
    if (src.i4_g_cnt) dst.i4_g_cnt = dalloc<int>(64);
    if (src.i4_g_slots) dst.i4_g_slots = dalloc<int>(64 * 8);
    if (src.i4_g_exp) dst.i4_g_exp = dalloc<int>(64);
    if (src.i4_g_n) dst.i4_g_n = dalloc<int>(1);
    if (src.i4_pf_x) dst.i4_pf_x = dalloc<float>((size_t) src.i4_pf_cap * src.i4_pf_Kmax);
    if (src.i4_pf_x4) dst.i4_pf_x4 = dalloc<uint8_t>((size_t) src.i4_pf_cap * src.i4_pf_Kmax);
    if (src.i4_pf_xm) dst.i4_pf_xm = dalloc<uint8_t>((size_t) (src.i4_pf_cap / 16) * (src.i4_pf_Kmax / 256) * 16);
    if (src.i4_pf_xr) dst.i4_pf_xr = dalloc<float>(src.i4_pf_cap);
    if (src.i4_mv_xq) dst.i4_mv_xq = dalloc<int8_t>((size_t) src.i4_xq_rows * src.i4_pf_Kmax);
    if (src.i4_mv_xd) dst.i4_mv_xd = dalloc<float2>((size_t) src.i4_xq_rows * (src.i4_pf_Kmax / 32 + 2));
    for (int i = 0; i < Work::kI4RotSlots; ++i) {
        if (src.i4_rd_xq[i]) dst.i4_rd_xq[i] = dalloc<int8_t>((size_t) src.i4_xq_rows * src.i4_pf_Kmax);
        if (src.i4_rd_xd[i]) dst.i4_rd_xd[i] = dalloc<float2>((size_t) src.i4_xq_rows * (src.i4_pf_Kmax / 32 + 2));
    }
    for (int i = 0; i < 3; ++i) {
        if (src.i4_ma_xq[i]) dst.i4_ma_xq[i] = dalloc<int8_t>((size_t) src.i4_xq_rows * src.i4_pf_Kmax);
        if (src.i4_ma_xd[i]) dst.i4_ma_xd[i] = dalloc<float2>((size_t) src.i4_xq_rows * (src.i4_pf_Kmax / 32 + 2));
    }

    if (c.n_pass > 1) {
        dst.lp_pre = dalloc<float>(Tm * C);
        dst.lp_bank = dalloc<float>((c.block ? n_core / c.block : 0) * Tm * C);
        dst.lp_msg = dalloc<float>(4 * Tm * C);
        dst.lp_msg_all = dalloc<float>(n_core * 4 * C);
        dst.lp_in = dalloc<float>(3 * Tm * C);
        dst.lp_g = dalloc<float>(3 * Tm * (2 * (size_t) c.rank + 2));
        dst.lp_a = dalloc<float>(3 * Tm * c.rank);
        dst.lp_f = dalloc<float>(2 * Tm * C);
    }
    dst.logits = dalloc<float>((size_t) 16 * c.n_rows);
    dst.argmax_d = dalloc<int>(Tm);
    dst.argmax_part_v = dalloc<float>(256);
    dst.argmax_part_i = dalloc<int>(256);
    // The K-step graph bakes these addresses too: slot 1 must NOT inherit slot 0's buffers.
    dst.kstep_K = 0; dst.emb_dK = nullptr; dst.argmax_dK = nullptr; dst.argmax_hK = nullptr;
    if (src.kstep_K > 0) {
        dst.kstep_K = src.kstep_K;
        dst.emb_dK = dalloc<float>((size_t) (src.kstep_K - 1) * C);
        dst.argmax_dK = dalloc<int>(src.kstep_K);
        CU(cudaHostAlloc((void**) &dst.argmax_hK, (size_t) src.kstep_K * 4, cudaHostAllocPortable));
    }
    dst.cand_d = dalloc<int>(2 + 2 * 4096);
    dst.cand_part = dalloc<int>(k::cand_scratch(c.n_vocab));

    const size_t S_host = (size_t) dst.T_host * c.K;
    CU(cudaHostAlloc((void**) &dst.emb_h, Tm * C * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc((void**) &dst.pos_h, (size_t) dst.T_host * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc((void**) &dst.route_h, S_host * 8, cudaHostAllocPortable));
    CU(cudaHostAlloc((void**) &dst.argmax_h, Tm * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc((void**) &dst.logits_h, (size_t) c.n_rows * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc((void**) &dst.cand_h, (size_t) (2 + 2 * 4096) * 4, cudaHostAllocPortable));
    // These pinned mirrors are not touched by the AR graph; clear the inherited pointers so nothing can use
    // the slot-0 buffers by accident.
    for (int i = 0; i < 2; ++i) {
        dst.gidx_h[i] = nullptr; dst.inv_h[i] = nullptr;
        dst.sg_h[i] = nullptr; dst.su_h[i] = nullptr; dst.wt_h[i] = nullptr; dst.wtr_h[i] = nullptr;
    }
    dst.pack_h[0] = nullptr; dst.x_h = nullptr; dst.rg_h = nullptr;
    std::printf("ik2: AR stream2 isolated Work allocated (%.2f GiB scratch)\n",
                (double) (src.gu32_bytes + src.lg_cap * 4) / 1073741824.0);
    std::fflush(stdout);
}

// IK2_BATCH_ATTN helpers.  load_batch_slot()/reset_batch_slot() are used by P1 (prefill runs one request at a
// time through M.V); stage_batch_tables() publishes the selected slots of one visit into Work's stable device
// pointer tables.  Only the table CONTENTS change, never their addresses.  Defined after Work/g_s/g_batch_decode.
inline void load_batch_slot(Model& M, int slot) {
    if (!g_batch_decode || slot < 0 || slot >= g_batch_slots.n_slot) return;
    const BatchSlots& B = g_batch_slots;
    for (int v = 0; v < B.n_visit; ++v) {
        const size_t i = B.idx(slot, v);
        Visit& Vi = M.V[v];
        Vi.kci = B.kci[i]; Vi.vci = B.vci[i]; Vi.ks = B.ks[i]; Vi.vs = B.vs[i];
        Vi.h_k = B.h_k[i]; Vi.h_v = B.h_v[i]; Vi.h_attn = B.h_attn[i]; Vi.h_mlp = B.h_mlp[i];
    }
}
inline void reset_batch_slot(int slot, int kd, int Cc) {
    if (!g_batch_decode || slot < 0 || slot >= g_batch_slots.n_slot) return;
    const BatchSlots& B = g_batch_slots;
    for (int v = 0; v < B.n_visit; ++v) {
        const size_t i = B.idx(slot, v);
        CU(cudaMemsetAsync(B.h_k[i], 0, (size_t) 3 * kd * 4, g_s));
        CU(cudaMemsetAsync(B.h_v[i], 0, (size_t) 3 * kd * 4, g_s));
        CU(cudaMemsetAsync(B.h_attn[i], 0, (size_t) 3 * Cc * 4, g_s));
        CU(cudaMemsetAsync(B.h_mlp[i], 0, (size_t) 3 * Cc * 4, g_s));
    }
}
// Fill one visit's pinned host pointer-table slice.  The device tables' ADDRESSES never change; only their contents
// (which slot's KV/conv blocks this visit reads) do.  During graph replay all 48 visits copy from off=0, so refreshing
// that one slice per visit before cudaGraphLaunch is enough to retarget the whole replayed batch.
inline void fill_batch_table_host(Work& W, int vis, int n_rows, const int* slot_ids, int off) {
    const BatchSlots& B = g_batch_slots;
    for (int i = 0; i < n_rows; ++i) {
        const size_t j = B.idx(slot_ids[i], vis);
        W.bkci_h[off + i] = B.kci[j]; W.bvci_h[off + i] = B.vci[j];
        W.bks_h[off + i] = B.ks[j];  W.bvs_h[off + i] = B.vs[j];
        W.bh_k_h[off + i] = B.h_k[j]; W.bh_v_h[off + i] = B.h_v[j];
        W.bh_attn_h[off + i] = B.h_attn[j]; W.bh_mlp_h[off + i] = B.h_mlp[j];
    }
}
inline void stage_batch_tables(Work& W, int vis, int n_rows, const int* slot_ids) {
    const BatchSlots& B = g_batch_slots;
    int si = 0;
    // A capture cannot contain cudaEventQuery/cudaEventSynchronize; use the fixed slot 0 and skip the ring entirely.
    if (g_batch_moe_device && !g_batch_graph_capturing) {
        si = W.batch_stage_next++ % Work::kBatchStageSlots;
        if (W.batch_stage_ev_ok[si]) {
            const cudaError_t q = cudaEventQuery(W.batch_stage_ev[si]);
            if (q == cudaErrorNotReady) CU(cudaEventSynchronize(W.batch_stage_ev[si]));
            else CU(q);
        }
    }
    const int off = si * BatchSlots::kMaxRows;
    fill_batch_table_host(W, vis, n_rows, slot_ids, off);
    CU(cudaMemcpyAsync(W.bkci_d, W.bkci_h + off, n_rows * sizeof(uint8_t*), cudaMemcpyHostToDevice, g_s));
    CU(cudaMemcpyAsync(W.bvci_d, W.bvci_h + off, n_rows * sizeof(uint8_t*), cudaMemcpyHostToDevice, g_s));
    CU(cudaMemcpyAsync(W.bks_d, W.bks_h + off, n_rows * sizeof(uint16_t*), cudaMemcpyHostToDevice, g_s));
    CU(cudaMemcpyAsync(W.bvs_d, W.bvs_h + off, n_rows * sizeof(uint16_t*), cudaMemcpyHostToDevice, g_s));
    CU(cudaMemcpyAsync(W.bh_k_d, W.bh_k_h + off, n_rows * sizeof(float*), cudaMemcpyHostToDevice, g_s));
    CU(cudaMemcpyAsync(W.bh_v_d, W.bh_v_h + off, n_rows * sizeof(float*), cudaMemcpyHostToDevice, g_s));
    CU(cudaMemcpyAsync(W.bh_attn_d, W.bh_attn_h + off, n_rows * sizeof(float*), cudaMemcpyHostToDevice, g_s));
    CU(cudaMemcpyAsync(W.bh_mlp_d, W.bh_mlp_h + off, n_rows * sizeof(float*), cudaMemcpyHostToDevice, g_s));
    if (g_batch_moe_device && !g_batch_graph_capturing) {
        CU(cudaEventRecord(W.batch_stage_ev[si], g_s));
        W.batch_stage_ev_ok[si] = true;
    }
}

// --serve: SIGUSR1 asks the request running to stop (its client went away) at its next token or prompt chunk
volatile std::sig_atomic_t g_cancel = 0;
// IK2_STATS=1: per request, how many tokens the prefill's experts each ran ([0] 1, [1] 2, [2] 3-7, [3] 8+)
uint64_t g_cnt_hist[4] = {0, 0, 0, 0};
// IK2_STATS: a request's experts by where they came from: used and already cached / used and prefetched for this
// pass / used and copied after the routing / prefetched and not used
uint64_t g_src_hist[4] = {0, 0, 0, 0};
extern "C" void on_usr1(int) { g_cancel = 1; }

// Speculative decoding: g_taps [g_taps_n][g_taps_ld][C] gets the drafter's taps of every position a forward call runs (the
// stream at the start of layers 5, 11, .., 41, the final normed hidden); with g_rec each short conv's history and
// inputs are kept, so that a verify window's rejected part can be taken back (rollback()).
float* g_taps = nullptr;
int g_taps_ld = 0;
int g_taps_n = 0;
bool g_rec = false;
void conv_rec(Layer& Ly, int which, const float* x, const float* hist, int T, int Cc) {
    if (!g_rec) return;
    CU(cudaMemcpyAsync(Ly.rec_old[which], hist, (size_t) 3 * Cc * 4, cudaMemcpyDeviceToDevice, g_s));
    CU(cudaMemcpyAsync(Ly.rec_x[which], x, (size_t) T * Cc * 4, cudaMemcpyDeviceToDevice, g_s));
}

// IK2_BATCH_ATTN conv wrappers.  When W.batch is false these call the original scalar functions with identical
// arguments, so the default path is unchanged.  In batch mode T is the number of SLOTS (= rows), each row has one
// token, and the history pointers come from Work's per-row tables.  conv_rec is intentionally skipped: batched
// decode is a plain one-token-per-step path and does not use speculation rollback.
inline float** batch_hist_tab(Work& W, int which) {
    return which == 0 ? W.bh_k_d : which == 1 ? W.bh_v_d : which == 2 ? W.bh_attn_d : W.bh_mlp_d;
}
inline void conv_rec_w(Layer& Ly, Work& W, int which, const float* x, int T, int Cc) {
    if (!W.batch) conv_rec(Ly, which, x, which == 0 ? Ly.h_k : which == 1 ? Ly.h_v :
                           which == 2 ? Ly.h_attn : Ly.h_mlp, T, Cc);
}
inline void conv4_w(Layer& Ly, Work& W, const float* x, const float* w, const float* dyn, int heads, int T, int C,
                    float* y) {
    if (!W.batch) { k::conv4(x, Ly.h_k, w, dyn, heads, T, C, y, g_s); return; }
    k::conv4_rows(x, W.bh_k_d, w, dyn, heads, 1, C, y, W.batch_rows, g_s);
}
inline void conv4_v_w(Layer& Ly, Work& W, const float* x, const float* w, const float* dyn, int heads, int T, int C,
                      float* y) {
    if (!W.batch) { k::conv4(x, Ly.h_v, w, dyn, heads, T, C, y, g_s); return; }
    k::conv4_rows(x, W.bh_v_d, w, dyn, heads, 1, C, y, W.batch_rows, g_s);
}
inline void conv4_attn_w(Layer& Ly, Work& W, bool block_start, const float* x, const float* res, float* y, int T,
                         int C) {
    if (!W.batch) {
        if (block_start) k::conv4(x, Ly.h_attn, Ly.sc_attn, nullptr, 0, T, C, y, g_s);
        else k::conv4_res(x, Ly.h_attn, Ly.sc_attn, nullptr, 0, T, C, res, y, g_s);
    } else {
        if (block_start) k::conv4_rows(x, W.bh_attn_d, Ly.sc_attn, nullptr, 0, 1, C, y, W.batch_rows, g_s);
        else k::conv4_res_rows(x, W.bh_attn_d, Ly.sc_attn, nullptr, 0, 1, C, res, y, W.batch_rows, g_s);
    }
}
inline void conv4_mlp_w(Layer& Ly, Work& W, const float* x, const float* res, float* y, int T, int C) {
    if (!W.batch) k::conv4_res(x, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, T, C, res, y, g_s);
    else k::conv4_res_rows(x, W.bh_mlp_d, Ly.sc_mlp, nullptr, 0, 1, C, res, y, W.batch_rows, g_s);
}

// The pictures of the request running (GENI): the image encoder's rows (strata-vision's SVE1 records, [n][C] F32), each
// in place of one image token of the prompt - the model reads them as they are, without the embedding's norm.
struct Pictures {
    std::vector<float> rows;
    std::vector<int> row_at;   // a prompt position's row (-1: a token)
    int n = 0;
    void clear() { rows.clear(), row_at.clear(), n = 0; }
    int at(int pos) const { return n && pos < (int) row_at.size() ? row_at[pos] : -1; }
};
Pictures g_img;
// the rows of `path` for the image tokens of ids in order; sig[i]: a hash of position i's row (0: a token), the
// kept state's key beside the ids.  -> an error's text, or empty
std::string load_pictures(const std::string& path, const std::vector<int>& ids, int image_token, int C, Pictures& P,
                          std::vector<uint64_t>& sig) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot read the pictures' file";
    for (;;) {
        int32_t h[5];
        if (std::fread(h, 4, 5, f) != 5) break;
        if (h[0] != 0x31455653 || h[1] <= 0 || h[4] != C) {
            std::fclose(f);
            return "the pictures' file is not the image encoder's for this model";
        }
        const size_t at = P.rows.size(), n = (size_t) h[1] * C;
        P.rows.resize(at + n);
        if (std::fread(P.rows.data() + at, 4, n, f) != n) {
            std::fclose(f);
            return "the pictures' file is cut short";
        }
    }
    std::fclose(f);
    const int n_rows = (int) (P.rows.size() / C);
    P.row_at.assign(ids.size(), -1);
    int k = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] != image_token) continue;
        if (k >= n_rows) return "the prompt has more image tokens than its pictures have rows";
        P.row_at[i] = k;
        uint64_t hsh = 1469598103934665603ull;   // FNV-1a over the row's bytes
        const uint8_t* b = (const uint8_t*) (P.rows.data() + (size_t) k * C);
        for (size_t j = 0; j < (size_t) C * 4; ++j) hsh = (hsh ^ b[j]) * 1099511628211ull;
        sig[i] = hsh | 1;
        ++k;
    }
    if (k != n_rows) return "the prompt's image tokens and its pictures' rows do not match";
    P.n = n_rows;
    return "";
}
// the pictures' rows of positions pos0 .. pos0 + T - 1 over their tokens' embeddings in h [T][C]
void place_pictures(float* h, int T, int pos0, int C, cudaStream_t s) {
    if (!g_img.n) return;
    for (int t = 0; t < T;) {
        const int r = g_img.at(pos0 + t);
        if (r < 0) {
            ++t;
            continue;
        }
        int n = 1;
        while (t + n < T && g_img.at(pos0 + t + n) == r + n) ++n;
        CU(cudaMemcpyAsync(h + (size_t) t * C, g_img.rows.data() + (size_t) r * C, (size_t) n * C * 4,
                           cudaMemcpyHostToDevice, s));
        t += n;
    }
}

// a GPU's own: its streams and cuBLAS handle, work buffers and expert cache; use_dev() makes it the current one (the
// globals g_s, g_cs, g_ev, g_blas the kernels and copies go to)
struct DevCtx {
    cudaStream_t s = nullptr, cs = nullptr;
    cudaEvent_t ev = nullptr;
    cublasHandle_t blas = nullptr;
    Work W{};
    ExpertCache EC;
};
std::vector<DevCtx> g_dev;
int g_cur = -1;
Work& dev_work(int d) { return g_dev[d].W; }
ExpertCache& dev_cache(int d) { return g_dev[d].EC; }
// IK2_INT4_DENSE_MULTI / IK2_INT4_MSG: pack T float rows and run the W4A8 GEMM on the current device. The scratch
// is the prefill-expert rot_pack8 buffer, unused by this multi-token dense path in the same forward call.
bool try_int4_dense_multi(const Mat& m, const float* x32, int T, float* Y, int ldy) {
    if (g_cur < 0 || g_cur >= (int) g_dev.size()) return false;
    Work& W = dev_work(g_cur);
    if (!W.i4_pf_x4 || !W.i4_pf_xm || !W.i4_pf_xr || W.i4_pf_cap < 256 || m.K > W.i4_pf_Kmax) return false;
    k::int4_rot_pack8(x32, T, m.K, W.i4_pf_x4, W.i4_pf_xm, W.i4_pf_xr, g_s);
    k::int4_gemm8(m.i4Wp, m.i4Wh, m.i4WS, m.N, m.K, W.i4_pf_x4, W.i4_pf_xm, W.i4_pf_xr,
                  T, 4, 8, Y, nullptr, k::kInt4Sh8, g_s, ldy);
    return true;
}
// IK2_INT4_DENSE_MV: same projection over the decode dp4a matvec (NC=T). Reads each weight row once and reuses
// it across the T tokens, avoiding the GEMM's 256-token tiling (7/8 waves idle at T=8). One rot+quant + one launch.
bool try_int4_dense_mv(const Mat& m, const float* x32, int T, float* Y, int ldy) {
    if (g_cur < 0 || g_cur >= (int) g_dev.size()) return false;
    Work& W = dev_work(g_cur);
    if (!W.i4_mv_xq || !W.i4_mv_xd || m.K > W.i4_pf_Kmax ||
        T > ((g_verify_fast || batch_dense_mv_on(T)) ? 8 : 1)) return false;
    // IK2_INT4_ROT_DEDUP: the rotation is a pure function of (x32, T, K), so within ONE run_layer_attn call an
    // input already rotated into its own slot needs no second rotation (the msg branch projects in_q twice).
    // The epoch matches only the current call and the flag is only set inside it, so codes from a previous
    // visit (same W.xq/xk/xv pointers, new content) or an outside matmul (the head rewrites W.xq) never hit.
    if (g_int4_rot_dedup && W.i4_rot_in_attn && W.i4_rd_xq[0]) {
        int hit = -1;
        for (int i = 0; i < Work::kI4RotSlots; ++i)
            if (W.i4_rd_ep[i] == W.i4_rot_epoch && W.i4_rd_src[i] == x32 && W.i4_rd_T[i] == T && W.i4_rd_K[i] == m.K) {
                hit = i;
                break;
            }
        if (hit < 0) {
            hit = W.i4_rd_next;
            W.i4_rd_next = (W.i4_rd_next + 1) % Work::kI4RotSlots;
            k::int4_rot_q8x(x32, T, m.K, W.i4_rd_xq[hit], W.i4_rd_xd[hit], g_s);
            W.i4_rd_src[hit] = x32; W.i4_rd_T[hit] = T; W.i4_rd_K[hit] = m.K; W.i4_rd_ep[hit] = W.i4_rot_epoch;
        }
        if (g_mv_lds)   // IK2_MV_LDS: LDS-staged activation variant, bit-exact vs int4_matvec
            k::int4_matvec_lds(m.i4Wp, m.i4Wh, m.i4WS, m.N, m.K, W.i4_rd_xq[hit], W.i4_rd_xd[hit], T, Y, ldy, nullptr,
                               1.f / (1 << k::kInt4Sh8), g_s);
        else
            k::int4_matvec(m.i4Wp, m.i4Wh, m.i4WS, m.N, m.K, W.i4_rd_xq[hit], W.i4_rd_xd[hit], T, Y, ldy, nullptr,
                           1.f / (1 << k::kInt4Sh8), g_s);
        return true;
    }
    k::int4_rot_q8x(x32, T, m.K, W.i4_mv_xq, W.i4_mv_xd, g_s);
    if (g_mv_lds)   // IK2_MV_LDS: LDS-staged activation variant, bit-exact vs int4_matvec
        k::int4_matvec_lds(m.i4Wp, m.i4Wh, m.i4WS, m.N, m.K, W.i4_mv_xq, W.i4_mv_xd, T, Y, ldy, nullptr,
                           1.f / (1 << k::kInt4Sh8), g_s);
    else
        k::int4_matvec(m.i4Wp, m.i4Wh, m.i4WS, m.N, m.K, W.i4_mv_xq, W.i4_mv_xd, T, Y, ldy, nullptr,
                       1.f / (1 << k::kInt4Sh8), g_s);
    return true;
}
void use_dev(int d) {
    if (d == g_cur) return;
    if (g_stage) cudaFree(g_stage), g_stage = nullptr, g_stage_bytes = 0;   // (the previous device's)
    CU(cudaSetDevice(d));
    g_cur = d;
    g_s = g_dev[d].s, g_cs = g_dev[d].cs, g_ev = g_dev[d].ev, g_blas = g_dev[d].blas;
}

// (kCand: sampling's candidates - the resident token path finds the largest logits on the device, g_cand; any other
// path gives the logits as kLogits does and leaves g_cand.n at -1)
enum class OutMode { kNone, kArgmax, kLogits, kCand };
constexpr int kCandK = 1000, kCandCap = 4096;   // the largest logits sampling looks at; the candidates a token keeps
struct Cand {
    int n = -1;                    // candidates of the last kCand token (-1: its logits were given instead)
    const int* idx = nullptr;      // [n] token ids, in no order
    const float* val = nullptr;    // [n] their logits
} g_cand;

__global__ void gather_f16_16_kernel(const uint16_t* __restrict__ x16, const int* __restrict__ idx, int n, int C,
                                     uint16_t* __restrict__ y16) {
    const int row = blockIdx.x;
    if (row >= n) return;
    const uint4* src = (const uint4*) (x16 + (size_t) idx[row] * C);
    uint4* dst = (uint4*) (y16 + (size_t) row * C);
    const int n4 = C >> 3;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) dst[i] = src[i];
}

#if defined(STRATA_USE_HIP) && defined(STRATA_WMMA_GFX11)
// FP16 rows gathered to F32 (the INT4 prefill's activation input): y [n][C] = x[idx[i]][C].
// The row copy is the largest single memory move of a prefill chunk's expert loop, and reading it a half at
// a time reached only 89 GB/s: with C a multiple of 8 the halves come in 16-byte loads and go out as float4.
__global__ void gather_f32_16_kernel(const uint16_t* __restrict__ x16, const int* __restrict__ idx, int n, int C,
                                     float* __restrict__ y) {
    const int row = blockIdx.x;
    if (row >= n) return;
    const uint16_t* src16 = x16 + (size_t) idx[row] * C;
    float* dst = y + (size_t) row * C;
    if ((C & 7) == 0) {
        const uint4* src = (const uint4*) src16;
        float4* d4 = (float4*) dst;
        const int n8 = C >> 3;   // uint4 = 8 halves
        for (int i = threadIdx.x; i < n8; i += blockDim.x) {
            const uint4 v = src[i];
            const uint16_t* h = (const uint16_t*) &v;
            float4 a, b;
            a.x = __half2float(__ushort_as_half(h[0])); a.y = __half2float(__ushort_as_half(h[1]));
            a.z = __half2float(__ushort_as_half(h[2])); a.w = __half2float(__ushort_as_half(h[3]));
            b.x = __half2float(__ushort_as_half(h[4])); b.y = __half2float(__ushort_as_half(h[5]));
            b.z = __half2float(__ushort_as_half(h[6])); b.w = __half2float(__ushort_as_half(h[7]));
            d4[2 * i] = a;
            d4[2 * i + 1] = b;
        }
        return;
    }
    for (int i = threadIdx.x; i < C; i += blockDim.x) dst[i] = __half2float(__ushort_as_half(src16[i]));
}

// h [n][F] FP16 -> x [n][Kpad] F32, zero-padded (INT4 down-projection input). One block per row.
__global__ void i4_pad_h_kernel(const uint16_t* __restrict__ h, int n, int F, int Kpad, float* __restrict__ x) {
    const int row = blockIdx.x;
    if (row >= n) return;
    const uint16_t* src = h + (size_t) row * F;
    float* dst = x + (size_t) row * Kpad;
    for (int i = threadIdx.x; i < Kpad; i += blockDim.x)
        dst[i] = i < F ? __half2float(__ushort_as_half(src[i])) : 0.0f;
}
#endif

// out[idx[row]] += wr[row] * y[row] over rows (one expert's tokens: each token at most once, so no two rows collide)
__global__ void scatter_add_rows_kernel(const float* __restrict__ y, const int* __restrict__ idx,
                                        const float* __restrict__ wr, int C, float* __restrict__ out, int ld) {
    const int row = blockIdx.x;
    const float w = wr[row];
    const float4* src = (const float4*) (y + (size_t) row * C);
    float4* dst = (float4*) (out + (size_t) idx[row] * ld);
    for (int i = threadIdx.x; i < (C >> 2); i += blockDim.x) {
        const float4 a = src[i];
        float4 b = dst[i];
        b.x += w * a.x, b.y += w * a.y, b.z += w * a.z, b.w += w * a.w;
        dst[i] = b;
    }
}

// IK2_BATCH_MOE_DEVICE: the exact expert-major scatter, without a host D2H.  One block per token
// walks the expert-major rows in order, so each token's float additions happen in exactly the same
// order as run_prefill_moe's per-expert scatter_add_rows_kernel launches (and no two blocks race).
__global__ void moe_batch_scatter_kernel(const float* __restrict__ y, const int* __restrict__ tok,
                                         const float* __restrict__ wr, int S, int C,
                                         float* __restrict__ out, int ld) {
    const int t = blockIdx.x;
    for (int row = 0; row < S; ++row) {
        if (tok[row] != t) continue;
        const float w = wr[row];
        const float4* src = (const float4*) (y + (size_t) row * C);
        float4* dst = (float4*) (out + (size_t) t * ld);
        for (int i = threadIdx.x; i < (C >> 2); i += blockDim.x) {
            const float4 a = src[i];
            float4 b = dst[i];
            b.x += w * a.x, b.y += w * a.y, b.z += w * a.z, b.w += w * a.w;
            dst[i] = b;
        }
    }
}

// IK2_BATCH_MOE_DEVICE: build the general MoE path's expert-major row order entirely on the device.
// Row r of the returned tables is the r-th row of the host grouping run_prefill_moe would build:
// experts in ascending order, slots in ascending order within each expert (rank = # earlier slots
// with the same expert). One launch, no D2H; S <= 64. Dynamic shared memory = 2 * E ints.
#if defined(STRATA_USE_HIP) && defined(STRATA_WMMA_GFX11)
__global__ void moe_batch_perm_kernel(const int* __restrict__ top, const float* __restrict__ w,
                                      const uint8_t* const* __restrict__ tabWp0,
                                      const uint8_t* const* __restrict__ tabWh0,
                                      const float* const* __restrict__ tabWS0,
                                      const uint8_t* const* __restrict__ tabWp2,
                                      const uint8_t* const* __restrict__ tabWh2,
                                      const float* const* __restrict__ tabWS2,
                                      int E, int K, int S, int* __restrict__ slots, int* __restrict__ tok,
                                      float* __restrict__ wt, float* __restrict__ sg, float* __restrict__ su,
                                      const uint8_t** __restrict__ Wp0, const uint8_t** __restrict__ Wh0,
                                      const float** __restrict__ WS0, const uint8_t** __restrict__ Wp2,
                                      const uint8_t** __restrict__ Wh2, const float** __restrict__ WS2,
                                       int* __restrict__ rexp,
                                       int* __restrict__ active_map,
                                       int* __restrict__ active_count) {
    extern __shared__ int sh[];
    int* cnt = sh;
    int* start = sh + E;
    const int tid = threadIdx.x;
    if (tid < E) cnt[tid] = 0;
    __syncthreads();
    if (tid < S) atomicAdd(&cnt[top[tid]], 1);
    __syncthreads();
    if (tid == 0) {
        int sum = 0;
        for (int e = 0; e < E; ++e) { start[e] = sum; sum += cnt[e]; }
        if (active_map && active_count) {
            int z = 0;
            for (int e = 0; e < E; ++e) if (cnt[e] > 0) active_map[z++] = e;
            *active_count = z;
        }
    }
    __syncthreads();
    if (tid < S) {
        const int e = top[tid];
        int rank = 0;
        for (int j = 0; j < tid; ++j) if (top[j] == e) ++rank;   // stable, same as host fill[e]++
        const int row = start[e] + rank;
        slots[row] = tid;
        tok[row] = tid / K;
        wt[row] = w[tid];       // i4_pf folds Ly.scale into WS: wtr is route weight only
        sg[row] = su[row] = 1.0f;
        rexp[row] = e;          // row_expert[] for int4_gemm8_moe_masked
        Wp0[row] = tabWp0[e]; Wh0[row] = tabWh0[e]; WS0[row] = tabWS0[e];
        Wp2[row] = tabWp2[e]; Wh2[row] = tabWh2[e]; WS2[row] = tabWS2[e];
    }
}
#endif

// Y [T] rows (ldy) = rows [r0, r0 + n) of a dense matrix over X: a fused projection's part that takes its own input
Mat mat_rows(const Mat& m, int r0, int n) {   // the rows [r0, r0 + n) of a matrix
    Mat p = m;
    p.N = n;
    p.i4Wp = p.i4Wh = nullptr;   // the int4 sidecar is whole-matrix; a row slice falls back to q8/raw/f16
    p.i4WS = nullptr;
    // IK2_INT4_MSG: the repack is 16-row-block local, so rows [r0, r0+n) read the whole-matrix sidecar from the
    // block ft0 = r0/16 on: Wp ft0*nsb*2048 bytes, Wh ft0*nsb*64 bytes, WS r0 floats (nsb = K/256).
    if (g_int4_msg && m.i4Wp && !m.i4_head_only && r0 % 16 == 0 && n % 16 == 0 && m.K % 256 == 0) {
        const size_t nsb = m.K / 256, ft0 = r0 / 16;
        p.i4Wp = m.i4Wp + ft0 * nsb * 2048;
        p.i4Wh = m.i4Wh + ft0 * nsb * 64;
        p.i4WS = m.i4WS + r0;
        p.i4_row_slice = true;
    }
    if (m.f16) p.f16 = m.f16 + (size_t) r0 * m.K;
    else if (m.q8) p.q8 = m.q8 + (size_t) r0 * m.K, p.q8d = m.q8d + (size_t) r0 * (m.K >> 5);
    else p.raw = m.raw + (size_t) r0 * (m.K / k::block_values(m.type)) * k::block_bytes(m.type);
    return p;
}
void matmul_rows(const Mat& m, int r0, int n, const float* x32, const uint16_t* x16, int T, float* Y, int ldy) {
    if (batch_graph_exact_on(T) && m.i4Wp && !m.i4_head_only && x32 && r0 % 16 == 0 && n % 16 == 0 &&
        m.K % 256 == 0 && (m.K == 2304 || m.K == 3072)) {
        const size_t nsb = m.K / 256, ft0 = r0 / 16;
        k::int4_dense_rot_rows(m.i4Wp + ft0 * nsb * 2048, m.i4Wh + ft0 * nsb * 64, m.i4WS + r0, n, m.K,
                               x32, T, Y, ldy, 1.f / (1 << k::kInt4Sh8), g_s, nullptr, 0);
        return;
    }
    matmul(mat_rows(m, r0, n), x32, x16, T, Y, ldy);
}

// A looped layer's message gates over a chunk of tokens.  A gate is two matrix products - g = D [rms(x); rms(m)],
// then F g and G g - and k::loop_gates' kernels take them a token at a time: over a prefill chunk that was a quarter
// of the prefill.  Here the products go through cuBLAS and the kernels are what is left around them.
constexpr int kLoopGemmMin = 32;   // tokens from which the products are cuBLAS's
void loop_gates_chunk(Work& W, const k::LoopGate* gt, int n, int T, int C, int R, float eps) {
    const double tlg = g_prof && T >= kLoopGemmMin ? now() : 0;
    if (T < kLoopGemmMin) {
        k::loop_gates(gt, n, T, C, R, eps, W.lp_g, g_s);
        return;
    }
    const float one = 1.0f, zero = 0.0f;
    float *a1 = W.lp_a, *a2 = a1 + (size_t) T * R, *g = a2 + (size_t) T * R;
    float *f = W.lp_f, *gn = f + (size_t) T * C;
    for (int j = 0; j < n; ++j) {
        // a1 [T][R] = x D_x^T, a2 = m D_m^T (D [R][2 C]: its rows' halves)
        CB(cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, R, T, C, &one, gt[j].D, 2 * C, gt[j].x, C, &zero, a1, R));
        CB(cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, R, T, C, &one, gt[j].D + C, 2 * C, gt[j].m, C, &zero, a2, R));
        k::loop_gate_mix(gt[j].x, gt[j].m, a1, a2, T, C, R, eps, g, g_s);
        // f [T][C] = g F, gn = g G (F, G [R][C])
        CB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, C, T, R, &one, gt[j].F, C, g, R, &zero, f, C));
        CB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, C, T, R, &one, gt[j].G, C, g, R, &zero, gn, C));
        k::loop_gate_blend(gt[j].x, gt[j].m, f, gn, (size_t) T * C, gt[j].y, g_s);
    }
    if (tlg) { cudaStreamSynchronize(g_s); g_pf.pf_loopgate += now() - tlg; }
}

// One layer's attention + post-attention conv + residual into r_out ([T][C]).
// h_in ([T][C]) is unmodified and acts as the residual input; when l % c.block == 0, it is saved to blk(nb).
// msg: a looped core's later pass - the layer's messages [4][T][C] (q's, k's, v's, the MoE's), which gate its q, k and
// v inputs before their norms (so a sliding layer too projects each from its own input).
// IK2_INT4_ROT_DEDUP scope: the rotation cache in try_int4_dense_mv may only hit inside ONE run_layer_attn
// call. Entry bumps the epoch (the norms rewrite W.xq/xk/xv every visit, so the pointer alone is not a key);
// the RAII destructor closes the scope on EVERY exit and bumps again, so later matmuls reading a rewritten
// W.xq (verify MoE norm, the head's out_w) always take the original rotate-every-call path. Host-only state:
// graph capture bakes each visit's own rotation kernels into the replayed stream sequence, unchanged.
struct RotDedupScope {
    Work& W;
    bool on;
    explicit RotDedupScope(Work& w) : W(w), on(g_int4_rot_dedup && w.i4_rd_xq[0] != nullptr) {
        if (on) { ++W.i4_rot_epoch; W.i4_rot_in_attn = true; }
    }
    ~RotDedupScope() { if (on) { W.i4_rot_in_attn = false; ++W.i4_rot_epoch; } }
};
extern int g_decode_pfull;   // defined with the solo decode path below; read here for IK2_BATCH_GRAPH_EXACT P
static void gx_hash_dev(const char* tag, const float* d, int n, int T, int vis) {
    if (!getenv("IK2_GXE_HASH") || g_batch_graph_capturing || g_fast_graph_capturing) return;
    std::vector<float> hv((size_t) n);
    cudaMemcpy(hv.data(), d, (size_t) n * 4, cudaMemcpyDeviceToHost);
    unsigned long long hv64 = 1469598103934665603ull;
    for (int i = 0; i < n; ++i) {
        unsigned int u; std::memcpy(&u, &hv[(size_t) i], 4); hv64 = (hv64 ^ u) * 1099511628211ull;
    }
    std::fprintf(stderr, "GXH %s T=%d vis=%d h=%016llx\\n", tag, T, vis, hv64);
}
void run_layer_attn(const Model& M, Layer& Ly, int l, int T, int pos0, const float* h_in, const int* pos_p, int nb,
                    size_t blk_off, float* r_out, Work& W, const float* msg = nullptr) {
    RotDedupScope rot_dedup_scope(W);   // (no-op unless IK2_INT4_ROT_DEDUP=1)
    const Config& c = M.c;
    const int C = c.C;
    const size_t TC = (size_t) T * C;
    auto blk = [&](int i) { return W.blocks + ((size_t) i * W.blk_ld + blk_off) * C; };

    const bool need_x32 = g_dense_q8 && (T == 1 || g_int4_dense_multi || g_int4_msg || g_int4_prefill ||
                                         batch_dense_mv_on(T) || batch_rotmv_on(T));
    const bool apr = g_prof && T > 1;   // profile prefill attention sub-phases (sync per phase)
    const double ap0 = apr ? now() : 0;
    const uint16_t* xk16 = W.x16;
    const uint16_t* xv16 = W.x16;
    const float *in_q = h_in, *in_k = h_in, *in_v = h_in;
    if (vp_attn_idx >= 0) cudaEventRecord(vpR[vp_attn_idx], g_s);
    if (msg) {
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
        st.st[nb] = h_in;
        k::attn_res(st, nb + 1, Ly.res_score, Ly.n_res_q, T, C, 1e-6f, W.lp_in, g_s);
        const float* nw[3] = {Ly.sliding ? Ly.attn_norm : Ly.q_in_norm, Ly.sliding ? Ly.attn_norm : Ly.k_in_norm,
                              Ly.sliding ? Ly.attn_norm : Ly.v_in_norm};
        uint16_t* y16[3] = {W.x16, W.xk16, W.xv16};
        float* y32[3] = {W.xq, W.xk, W.xv};
        k::LoopGate gt[3];
        for (int j = 0; j < 3; ++j)
            gt[j] = {W.lp_in + (size_t) (Ly.n_res_q == 3 ? j : 0) * TC, msg + (size_t) j * TC, Ly.lp_down[j],
                     Ly.lp_forget[j], Ly.lp_gain[j], y32[j]};
        loop_gates_chunk(W, gt, 3, T, C, c.rank, 1e-6f);
        for (int j = 0; j < 3; ++j)
            k::rms_f16(y32[j], nw[j], T, C, c.eps, y16[j], need_x32 ? y32[j] : nullptr, g_s);
        in_q = W.xq, in_k = W.xk, in_v = W.xv;
        xk16 = W.xk16, xv16 = W.xv16;
    } else if (l > 0) {
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
        st.st[nb] = h_in;
        k::RmsOuts outs{};
        if (Ly.sliding) {
            outs.w[0] = Ly.attn_norm;
            outs.y16[0] = W.x16;
            outs.y32[0] = need_x32 ? W.xq : nullptr;
            k::attn_res_rms_f16(st, nb + 1, Ly.res_score, Ly.n_res_q, T, C, 1e-6f, c.eps, outs, g_s);
            in_q = in_k = in_v = W.xq;
        } else {
            outs.w[0] = Ly.q_in_norm, outs.w[1] = Ly.k_in_norm, outs.w[2] = Ly.v_in_norm;
            outs.y16[0] = W.x16, outs.y16[1] = W.xk16, outs.y16[2] = W.xv16;
            if (need_x32) outs.y32[0] = W.xq, outs.y32[1] = W.xk, outs.y32[2] = W.xv;
            k::attn_res_rms_f16(st, nb + 1, Ly.res_score, Ly.n_res_q, T, C, 1e-6f, c.eps, outs, g_s);
            in_q = W.xq, in_k = W.xk, in_v = W.xv;
            xk16 = W.xk16, xv16 = W.xv16;
        }
    } else if (Ly.sliding) {
        k::rms_f16(in_q, Ly.attn_norm, T, C, c.eps, W.x16, need_x32 ? W.xq : nullptr, g_s);
        in_q = in_k = in_v = W.xq;
    } else {
        k::rms_f16(in_q, Ly.q_in_norm, T, C, c.eps, W.x16, need_x32 ? W.xq : nullptr, g_s);
        k::rms_f16(in_k, Ly.k_in_norm, T, C, c.eps, W.xk16, need_x32 ? W.xk : nullptr, g_s);
        k::rms_f16(in_v, Ly.v_in_norm, T, C, c.eps, W.xv16, need_x32 ? W.xv : nullptr, g_s);
        in_q = W.xq, in_k = W.xk, in_v = W.xv;
        xk16 = W.xk16, xv16 = W.xv16;
    }
    const int HD = c.H * c.D, KD = c.KV * c.D, HR = c.H * c.d_rel;
    float *q_p = W.q, *k_p = W.kk, *v_p = W.vv, *r_p = W.rr, *gate_p = W.gate, *sink_p = W.sink;
    float *kd_p = W.dyn, *vd_p = W.dyn;
    if (vp_attn_idx >= 0) cudaEventRecord(vpS[vp_attn_idx], g_s);
    if (Ly.sliding) {
        if (Ly.wqkvrg.N > 0) {
            if (msg) {   // the fused rows by the input they take: q | k | v | r, gate (and the sink)
                const int N = Ly.wqkvrg.N;
                // IK2_INT4_MV_ALT: one k_mv_alt launch over the WHOLE fused sidecar instead of four
                // small-grid slice launches (k/v slices are 64 blocks on 96 CUs - latency-bound).
                // Guards mirror matmul's DENSE_MV gate + mat_rows' slice conditions + try_int4_dense_mv,
                // so this fires exactly where the OFF path took four k_mv launches. Same weights, same
                // rotated codes per input (in_q rotated once, reused for r+gate), same per-row dp4a
                // chain and reduction -> bit-identical W.qkv. Boundaries HD/HD+KD/HD+2KD are 16-aligned.
                const int K = Ly.wqkvrg.K;
                const bool mv_alt = g_int4_mv_alt && g_int4_msg && g_int4_dense_mv &&
                                    Ly.wqkvrg.i4Wp && Ly.wqkvrg.i4Wh && Ly.wqkvrg.i4WS && !Ly.wqkvrg.i4_head_only &&
                                    K % 256 == 0 && N % 16 == 0 && HD % 16 == 0 && KD % 16 == 0 &&
                                    in_q && in_k && in_v && T > 1 && T <= (g_verify_fast ? 8 : 1) &&
                                    K <= W.i4_pf_Kmax && W.i4_ma_xq[0] && W.i4_ma_xd[0];
                if (mv_alt) {
                    k::int4_rot_q8x(in_q, T, K, W.i4_ma_xq[0], W.i4_ma_xd[0], g_s);
                    k::int4_rot_q8x(in_k, T, K, W.i4_ma_xq[1], W.i4_ma_xd[1], g_s);
                    k::int4_rot_q8x(in_v, T, K, W.i4_ma_xq[2], W.i4_ma_xd[2], g_s);
                    k::int4_matvec_alt(Ly.wqkvrg.i4Wp, Ly.wqkvrg.i4Wh, Ly.wqkvrg.i4WS, N, K,
                                       W.i4_ma_xq[0], W.i4_ma_xd[0], W.i4_ma_xq[1], W.i4_ma_xd[1],
                                       W.i4_ma_xq[2], W.i4_ma_xd[2], HD, HD + KD, HD + 2 * KD,
                                       T, W.qkv, N, 1.f / (1 << k::kInt4Sh8), g_s);
                } else {
                    matmul_rows(Ly.wqkvrg, 0, HD, in_q, W.x16, T, W.qkv, N);
                    matmul_rows(Ly.wqkvrg, HD, KD, in_k, xk16, T, W.qkv + HD, N);
                    matmul_rows(Ly.wqkvrg, HD + KD, KD, in_v, xv16, T, W.qkv + HD + KD, N);
                    matmul_rows(Ly.wqkvrg, HD + 2 * KD, N - HD - 2 * KD, in_q, W.x16, T, W.qkv + HD + 2 * KD, N);
                }
            } else {
                matmul(Ly.wqkvrg, in_q, W.x16, T, W.qkv, Ly.wqkvrg.N);
            }
            const bool has_sink = !g_dense_q8;
            if (T == 1) {
                q_p = W.qkv;
                k_p = q_p + HD;
                v_p = k_p + KD;
                r_p = v_p + KD;
                gate_p = r_p + HR;
                if (has_sink) sink_p = gate_p + HD;
            } else {
                if (has_sink)
                    unpack_segments(W.qkv, T,
                                    {{W.q, HD}, {W.kk, KD}, {W.vv, KD}, {W.rr, HR}, {W.gate, HD}, {W.sink, c.H}}, g_s);
                else
                    unpack_segments(W.qkv, T, {{W.q, HD}, {W.kk, KD}, {W.vv, KD}, {W.rr, HR}, {W.gate, HD}}, g_s);
            }
            if (!has_sink) {
                if (batch_graph_exact_on(T) && W.batch && Ly.sink16)
                    k::sink_dot_rows(Ly.sink16, in_q, T, c.H, C, C, W.sink, g_s);
                else
                    gemm16(Ly.sink16, c.H, C, W.x16, T, sink_p, c.H);
            }
        } else {
            matmul(Ly.wq, in_q, W.x16, T, q_p, HD);
            matmul(Ly.wk, in_k, xk16, T, k_p, KD);
            matmul(Ly.wv, in_v, xv16, T, v_p, KD);
            matmul(Ly.wr, in_q, W.x16, T, r_p, HR);
            matmul(Ly.wgate, in_q, W.x16, T, gate_p, HD);
            if (batch_graph_exact_on(T) && W.batch && Ly.sink16)
                k::sink_dot_rows(Ly.sink16, in_q, T, c.H, C, C, W.sink, g_s);
            else
                gemm16(Ly.sink16, c.H, C, W.x16, T, sink_p, c.H);
        }
        conv_rec_w(Ly, W, 0, k_p, T, KD);
        conv_rec_w(Ly, W, 1, v_p, T, KD);
        conv4_w(Ly, W, k_p, Ly.sc_k, nullptr, 0, T, KD, W.kconv);
        conv4_v_w(Ly, W, v_p, Ly.sc_v, nullptr, 0, T, KD, W.vconv);
    } else {
        if (Ly.wqrg.N > 0) {
            matmul(Ly.wqrg, in_q, W.x16, T, W.qkv, Ly.wqrg.N);
            const bool has_sink = !g_dense_q8;
            if (T == 1) {
                q_p = W.qkv;
                r_p = q_p + HD;
                gate_p = r_p + HR;
                if (has_sink) sink_p = gate_p + HD;
            } else {
                if (has_sink)
                    unpack_segments(W.qkv, T, {{W.q, HD}, {W.rr, HR}, {W.gate, HD}, {W.sink, c.H}}, g_s);
                else
                    unpack_segments(W.qkv, T, {{W.q, HD}, {W.rr, HR}, {W.gate, HD}}, g_s);
            }
            if (!has_sink) {
                if (batch_graph_exact_on(T) && W.batch && Ly.sink16)
                    k::sink_dot_rows(Ly.sink16, in_q, T, c.H, C, C, W.sink, g_s);
                else
                    gemm16(Ly.sink16, c.H, C, W.x16, T, sink_p, c.H);
            }
        } else {
            matmul(Ly.wq, in_q, W.x16, T, q_p, HD);
            matmul(Ly.wr, in_q, W.x16, T, r_p, HR);
            matmul(Ly.wgate, in_q, W.x16, T, gate_p, HD);
            if (batch_graph_exact_on(T) && W.batch && Ly.sink16)
                k::sink_dot_rows(Ly.sink16, in_q, T, c.H, C, C, W.sink, g_s);
            else
                gemm16(Ly.sink16, c.H, C, W.x16, T, sink_p, c.H);
        }
        if (!g_dense_q8) {
            const int off_k = T == 1 ? Ly.wqrg.N : 0;
            const int off_v = T == 1 ? Ly.wqrg.N + Ly.wk_dyn.N : 0;
            matmul(Ly.wk_dyn, in_k, xk16, T, W.qkv + off_k, Ly.wk_dyn.N);
            if (T == 1) {
                k_p = W.qkv + off_k;
                kd_p = k_p + KD;
            } else {
                unpack_segments(W.qkv + off_k, T, {{W.kk, KD}, {W.dyn, c.KV * 4}}, g_s);
            }
            conv_rec_w(Ly, W, 0, k_p, T, KD);
            conv4_w(Ly, W, k_p, Ly.sc_k, kd_p, c.KV, T, KD, W.kconv);

            matmul(Ly.wv_dyn, in_v, xv16, T, W.qkv + off_v, Ly.wv_dyn.N);
            if (T == 1) {
                v_p = W.qkv + off_v;
                vd_p = v_p + KD;
            } else {
                unpack_segments(W.qkv + off_v, T, {{W.vv, KD}, {W.dyn, c.KV * 4}}, g_s);
            }
            conv_rec_w(Ly, W, 1, v_p, T, KD);
            conv4_v_w(Ly, W, v_p, Ly.sc_v, vd_p, c.KV, T, KD, W.vconv);
        } else {
            matmul(Ly.wk, in_k, xk16, T, k_p, KD);
            if (batch_graph_exact_on(T) && W.batch)
                for (int t = 0; t < T; ++t)
                    k::mv_f16(Ly.sc_k_dyn16, c.KV * 4, C, in_k + (size_t) t * C, kd_p + (size_t) t * c.KV * 4, g_s);
            else
                gemm16(Ly.sc_k_dyn16, c.KV * 4, C, xk16, T, kd_p, c.KV * 4);
            conv_rec_w(Ly, W, 0, k_p, T, KD);
            conv4_w(Ly, W, k_p, Ly.sc_k, kd_p, c.KV, T, KD, W.kconv);

            matmul(Ly.wv, in_v, xv16, T, v_p, KD);
            if (batch_graph_exact_on(T) && W.batch)
                for (int t = 0; t < T; ++t)
                    k::mv_f16(Ly.sc_v_dyn16, c.KV * 4, C, in_v + (size_t) t * C, vd_p + (size_t) t * c.KV * 4, g_s);
            else
                gemm16(Ly.sc_v_dyn16, c.KV * 4, C, xv16, T, vd_p, c.KV * 4);
            conv_rec_w(Ly, W, 1, v_p, T, KD);
            conv4_v_w(Ly, W, v_p, Ly.sc_v, vd_p, c.KV, T, KD, W.vconv);
        }
    }
    const float k_scale = c.qk_clip <= 28.0f ? 16.0f : 1.0f;
    const float v_scale = 32.0f;
    const float k_inv = 1.0f / k_scale, v_inv = 1.0f / v_scale;
    if (vp_attn_idx >= 0) cudaEventRecord(vpP[vp_attn_idx], g_s);
    double ap1 = ap0;
    if (apr) { cudaStreamSynchronize(g_s); ap1 = now(); g_pf.a_proj += ap1 - ap0; }
    if (kv_is_int()) {
        const int bits = kv_int_bits();
        if (W.batch && T > 1 && T <= BatchSlots::kMaxRows) {
            k::prep_qkv_r_int_rows(bits, q_p, Ly.q_clip, W.kconv, W.bkci_d, W.bks_d, W.vconv, W.bvci_d, W.bvs_d,
                                   r_p, Ly.r_clip, T, c.H, c.KV, c.D, c.d_rel, c.qk_clip, c.rel_clip, c.eps,
                                   pos_p, c.alpha, c.n_floor, Ly.kv_cap, nullptr, nullptr,
                                   nullptr, 0, nullptr, g_s, Ly.sliding ? nullptr : pos_p);
        } else if (g_vg_stage != VG_NORMAL) {
            k::prep_qkv_r_int_g(bits, q_p, Ly.q_clip, W.kconv, Ly.kci, Ly.ks, W.vconv, Ly.vci, Ly.vs, r_p, Ly.r_clip,
                                T, c.H, c.KV, c.D, c.d_rel, c.qk_clip, c.rel_clip, c.eps,
                                Ly.sliding ? nullptr : pos_p, c.alpha, c.n_floor, pos_p, Ly.kv_cap, g_s);
        } else {
            k::prep_qkv_r_int(bits, q_p, Ly.q_clip, W.kconv, Ly.kci, Ly.ks, W.vconv, Ly.vci, Ly.vs, r_p, Ly.r_clip,
                              T, c.H, c.KV, c.D, c.d_rel, c.qk_clip, c.rel_clip, c.eps,
                              Ly.sliding ? nullptr : pos_p, c.alpha, c.n_floor, pos0, Ly.kv_cap, g_s);
        }
    } else if (g_kv_fp8) {
        if (g_vg_stage != VG_NORMAL) {
            k::prep_qkv_r_fp8_g(q_p, Ly.q_clip, W.kconv, Ly.kc8, W.vconv, Ly.vc8, r_p, Ly.r_clip, T, c.H, c.KV, c.D,
                                c.d_rel, c.qk_clip, c.rel_clip, c.eps, Ly.sliding ? nullptr : pos_p, c.alpha,
                                c.n_floor, pos_p, Ly.kv_cap, k_scale, v_scale, g_s);
        } else {
        k::prep_qkv_r_fp8(q_p, Ly.q_clip, W.kconv, Ly.kc8, W.vconv, Ly.vc8, r_p, Ly.r_clip, T, c.H, c.KV, c.D, c.d_rel,
                          c.qk_clip, c.rel_clip, c.eps, Ly.sliding ? nullptr : pos_p, c.alpha, c.n_floor, pos0,
                          Ly.kv_cap, k_scale, v_scale, g_s);
        }
    } else {
        if (g_vg_stage != VG_NORMAL) {
            k::prep_qkv_r_f16_g(q_p, Ly.q_clip, W.kconv, Ly.kc, W.vconv, Ly.vc, r_p, Ly.r_clip, T, c.H, c.KV, c.D,
                                c.d_rel, c.qk_clip, c.rel_clip, c.eps, Ly.sliding ? nullptr : pos_p, c.alpha,
                                c.n_floor, pos_p, Ly.kv_cap, g_s);
        } else {
        k::prep_qkv_r_f16(q_p, Ly.q_clip, W.kconv, Ly.kc, W.vconv, Ly.vc, r_p, Ly.r_clip, T, c.H, c.KV, c.D, c.d_rel,
                          c.qk_clip, c.rel_clip, c.eps, Ly.sliding ? nullptr : pos_p, c.alpha, c.n_floor, pos0,
                          Ly.kv_cap, g_s);
        }
    }
    gx_hash_dev("qprep", q_p, T * HD, T, l);
    gx_hash_dev("kprep", k_p, T * KD, T, l);
    gx_hash_dev("vprep", v_p, T * KD, T, l);
    gx_hash_dev("sink", W.sink, T * c.H, T, l);

    const int extent = Ly.sliding ? c.extent_swa : c.extent;
    // The kernels recompute their own k0 from `window` (attn_split_int_kernel: k0 = qpos0 - window + 1), so the
    // window argument is what actually bounds the walk; the host k0 below must agree with it.
    const int window = Ly.sliding ? c.window
                                 : ((g_full_window > 0) ? g_full_window : (1 << 30));
    int k0 = Ly.sliding ? std::max(0, pos0 - c.window + 1) : 0;
    // IK2_FULL_WINDOW: a full-attention layer reads only the last g_full_window keys. Decode (T<=8) and the
    // prompt pass (T>8) both honour it, so a long prompt no longer walks the whole KV once per chunk - at a
    // 600K context that walk was 4.9 GB a chunk and the prompt did not finish in ten minutes.
    if (g_full_window > 0 && !Ly.sliding && pos0 + 1 > g_full_window) k0 = pos0 - g_full_window + 1;
    const int nk = pos0 + T - k0;
    const float one = 1.0f, zero = 0.0f;
    const float qk_scale = 1.0f / std::sqrt((float) c.D);
    const int attn_kv_cap = Ly.sliding ? Ly.kv_cap : 0;
    if (apr) { cudaStreamSynchronize(g_s); const double n = now(); g_pf.a_conv += n - ap1; ap1 = n; }
    if (vp_attn_idx >= 0) cudaEventRecord(vpK[vp_attn_idx], g_s);
    if (T == 1) {
        if (kv_is_int()) {
            if (g_attn_gqa && c.H % c.KV == 0 && c.H / c.KV == 3)
                k::attn_decode_int_gqa(kv_int_bits(), q_p, r_p, sink_p, Ly.kci, Ly.vci, Ly.ks, Ly.vs, attn_kv_cap,
                                       c.H, c.KV, c.D, k0, nk, pos0, qk_scale, Ly.rel_proj, c.d_rel, extent, window,
                                       W.lg, W.o, g_s, g_attn_dpp ? 1 : 0);
            else
                k::attn_decode_int(kv_int_bits(), q_p, r_p, sink_p, Ly.kci, Ly.vci, Ly.ks, Ly.vs, attn_kv_cap, c.H, c.KV,
                                   c.D, k0, nk, pos0, qk_scale, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        } else if (g_kv_fp8) {
            k::attn_decode_fp8(q_p, r_p, sink_p, Ly.kc8, Ly.vc8, attn_kv_cap, c.H, c.KV, c.D, k0, nk, pos0,
                               qk_scale * k_inv, v_inv, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        } else {
            k::attn_decode_f16(q_p, r_p, sink_p, Ly.kc, Ly.vc, attn_kv_cap, c.H, c.KV, c.D, k0, nk, pos0, qk_scale,
                               Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        }
    } else if (W.batch && T > 1 && T <= BatchSlots::kMaxRows && kv_is_int()) {
        const int P = batch_graph_exact_on(T)
                          ? (Ly.sliding ? std::min(128, std::max(1, (c.window + 63) / 64)) : g_decode_pfull)
                          : 128;
        const uint8_t* const* kci_tab = (const uint8_t* const*) W.bkci_d;
        const uint8_t* const* vci_tab = (const uint8_t* const*) W.bvci_d;
        const uint16_t* const* ks_tab = (const uint16_t* const*) W.bks_d;
        const uint16_t* const* vs_tab = (const uint16_t* const*) W.bvs_d;
        if (g_attn_gqa && c.H % c.KV == 0 && c.H / c.KV == 3)
            k::attn_decode_int_rows_gqa(kv_int_bits(), q_p, r_p, sink_p, kci_tab, vci_tab, ks_tab, vs_tab,
                                        attn_kv_cap, c.H, c.KV, pos_p, P, qk_scale, Ly.rel_proj, c.d_rel, extent,
                                        window, T, W.lg, W.o, g_s,
                                        batch_graph_exact_on(T) ? gate_p : nullptr, g_attn_dpp ? 1 : 0);
        else
            k::attn_decode_int_rows(kv_int_bits(), q_p, r_p, sink_p, kci_tab, vci_tab, ks_tab, vs_tab, attn_kv_cap,
                                    c.H, c.KV, pos_p, P, qk_scale, Ly.rel_proj, c.d_rel, extent, window, T, W.lg,
                                    W.o, g_s, batch_graph_exact_on(T) ? gate_p : nullptr, g_attn_dpp ? 1 : 0);
    } else if (T <= 8 && g_group && g_group_attn) {
        if (g_vg_stage != VG_NORMAL) {
            // P fixed at capture; W.lg is already sized for 8 queries x 128 splits.
            if (kv_is_int())
                k::attn_multi_int_g(kv_int_bits(), q_p, r_p, sink_p, Ly.kci, Ly.vci, Ly.ks, Ly.vs, attn_kv_cap, T,
                                    c.H, c.KV, pos_p, 128, qk_scale, Ly.rel_proj, c.d_rel, extent, window, W.lg,
                                    W.o, g_s);
            else if (g_kv_fp8)
                k::attn_multi_fp8_g(q_p, r_p, sink_p, Ly.kc8, Ly.vc8, attn_kv_cap, T, c.H, c.KV, pos_p, 128,
                                    qk_scale * k_inv, v_inv, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
            else
                k::attn_multi_f16_g(q_p, r_p, sink_p, Ly.kc, Ly.vc, attn_kv_cap, T, c.H, c.KV, pos_p, 128, qk_scale,
                                    Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        } else if (kv_is_int()) {
            k::attn_multi_int(kv_int_bits(), q_p, r_p, sink_p, Ly.kci, Ly.vci, Ly.ks, Ly.vs, attn_kv_cap, T, c.H,
                              c.KV, k0, nk, pos0, qk_scale, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        } else if (g_kv_fp8) {
            k::attn_multi_fp8(q_p, r_p, sink_p, Ly.kc8, Ly.vc8, attn_kv_cap, T, c.H, c.KV, k0, nk, pos0,
                              qk_scale * k_inv, v_inv, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        } else {
            k::attn_multi_f16(q_p, r_p, sink_p, Ly.kc, Ly.vc, attn_kv_cap, T, c.H, c.KV, k0, nk, pos0, qk_scale,
                              Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        }
    } else {
        k::to_f16(q_p, (size_t) T * HD, W.x16, g_s);
        uint16_t* lg16 = W.lg16;
        const int GQA = c.H / c.KV;
        // softmax rows (with the relative bias and the sink) -> lg16: the bias as one small GEMM (K = d_rel) first
        auto attn_probs = [&](int Tq, int nk_s, int ts, int k0_s) {
            const double pb0 = apr ? now() : 0;
            if (W.relb && extent <= W.relb_ext) {
                const float one_f = 1.0f, zero_f = 0.0f;
                CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, extent, Tq * c.H, c.d_rel, &one_f, Ly.rel_proj,
                                CUDA_R_32F, c.d_rel, r_p + (size_t) ts * HR, CUDA_R_32F, c.d_rel, &zero_f, W.relb,
                                CUDA_R_32F, extent, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
                k::attn_probs_bias_f16(W.lg, lg16, c.H, Tq, nk_s, pos0 + ts, k0_s, qk_scale, W.relb, extent, window,
                                       sink_p + (size_t) ts * c.H, g_s);
            } else {
                k::attn_probs_f16(W.lg, lg16, c.H, Tq, nk_s, pos0 + ts, k0_s, qk_scale, r_p + (size_t) ts * HR,
                                  c.d_rel, Ly.rel_proj, extent, window, sink_p + (size_t) ts * c.H, g_s);
            }
            if (apr) { cudaStreamSynchronize(g_s); g_pf.a_probs += now() - pb0; }
        };
        // int8/int4 prefill: the WMMA flash kernel dequantizes int codes to FP16 as staged (kv_intx8_h8), so int KV
        // keeps the fast flash path; only a non-WMMA build falls back to unpack -> FP16 staging + the GEMM path below.
        const bool int_flash_ok = kv_is_int()
#if defined(STRATA_USE_HIP) && defined(STRATA_WMMA_GFX11)
            && g_flash_wmma
#endif
            ;
        if (g_flash && c.D == 128 && W.relb && extent <= W.relb_ext && (!kv_is_int() || int_flash_ok)) {
            if (!g_n_sm) CU(cudaDeviceGetAttribute(&g_n_sm, cudaDevAttrMultiProcessorCount, Ly.dev));
            for (int ts = 0; ts < T; ts += Work::kRelbRows) {
                const int Tq = std::min(Work::kRelbRows, T - ts);
                const float one_f = 1.0f, zero_f = 0.0f;
                CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, extent, Tq * c.H, c.d_rel, &one_f, Ly.rel_proj,
                                CUDA_R_32F, c.d_rel, r_p + (size_t) ts * HR, CUDA_R_32F, c.d_rel, &zero_f, W.relb,
                                CUDA_R_32F, extent, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
                const int k_lo = Ly.sliding ? std::max(0, pos0 + ts - c.window + 1)
                                            : (g_full_window > 0 ? std::max(0, pos0 + ts - g_full_window + 1) : 0);
                const int ns = k::attn_flash_splits(Tq, c.H, pos0 + ts + Tq - k_lo, g_n_sm);
                if (kv_is_int())
#if defined(STRATA_USE_HIP) && defined(STRATA_WMMA_GFX11)
                    k::attn_flash_wmma_int(kv_int_bits(), W.x16 + (size_t) ts * HD, HD, Ly.kci, Ly.vci, Ly.ks, Ly.vs,
                                           attn_kv_cap, c.H, c.KV, c.D, Tq, pos0 + ts, qk_scale, W.relb, extent, window,
                                           sink_p + (size_t) ts * c.H, W.o + (size_t) ts * HD, HD, W.lg, W.lg_cap, ns, g_s);
#else
                    ;   // (int_flash_ok is false off a WMMA build; the unpack path below handles int)
#endif
                else if (g_kv_fp8)
#if defined(STRATA_USE_HIP) && defined(STRATA_WMMA_GFX11)
                if (g_flash_wmma)
                    k::attn_flash_wmma_fp8(W.x16 + (size_t) ts * HD, HD, Ly.kc8, Ly.vc8, attn_kv_cap, c.H, c.KV, c.D, Tq,
                                           pos0 + ts, qk_scale * k_inv, v_inv, W.relb, extent, window,
                                           sink_p + (size_t) ts * c.H, W.o + (size_t) ts * HD, HD, W.lg, W.lg_cap, ns, g_s);
                else
                    k::attn_flash_fp8(W.x16 + (size_t) ts * HD, HD, Ly.kc8, Ly.vc8, attn_kv_cap, c.H, c.KV, c.D, Tq,
                                      pos0 + ts, qk_scale * k_inv, v_inv, W.relb, extent, window,
                                      sink_p + (size_t) ts * c.H, W.o + (size_t) ts * HD, HD, W.lg, W.lg_cap, ns, g_s);
#else
                    k::attn_flash_fp8(W.x16 + (size_t) ts * HD, HD, Ly.kc8, Ly.vc8, attn_kv_cap, c.H, c.KV, c.D, Tq,
                                      pos0 + ts, qk_scale * k_inv, v_inv, W.relb, extent, window,
                                      sink_p + (size_t) ts * c.H, W.o + (size_t) ts * HD, HD, W.lg, W.lg_cap, ns, g_s);
#endif
                else
#if defined(STRATA_USE_HIP) && defined(STRATA_WMMA_GFX11)
                if (g_flash_wmma)
                    k::attn_flash_wmma_f16(W.x16 + (size_t) ts * HD, HD, Ly.kc, Ly.vc, attn_kv_cap, c.H, c.KV, c.D, Tq,
                                           pos0 + ts, qk_scale, W.relb, extent, window, sink_p + (size_t) ts * c.H,
                                           W.o + (size_t) ts * HD, HD, W.lg, W.lg_cap, ns, g_s);
                else
                    k::attn_flash_f16(W.x16 + (size_t) ts * HD, HD, Ly.kc, Ly.vc, attn_kv_cap, c.H, c.KV, c.D, Tq,
                                      pos0 + ts, qk_scale, W.relb, extent, window, sink_p + (size_t) ts * c.H,
                                      W.o + (size_t) ts * HD, HD, W.lg, W.lg_cap, ns, g_s);
#else
                    k::attn_flash_f16(W.x16 + (size_t) ts * HD, HD, Ly.kc, Ly.vc, attn_kv_cap, c.H, c.KV, c.D, Tq,
                                      pos0 + ts, qk_scale, W.relb, extent, window, sink_p + (size_t) ts * c.H,
                                      W.o + (size_t) ts * HD, HD, W.lg, W.lg_cap, ns, g_s);
#endif
            }
        } else
        for (int ts = 0; ts < T;) {
            const int k0_s = Ly.sliding ? std::max(0, pos0 + ts - c.window + 1) : 0;
            const int nk_ub = pos0 + T - k0_s;
            int Tq = std::max(1, std::min(T - ts, (int) (W.lg_cap / ((size_t) c.H * nk_ub))));
            if (W.relb) Tq = std::min(Tq, Work::kRelbRows);
            const int nk_s = pos0 + ts + Tq - k0_s;
            const long long qn = (long long) Tq * nk_s;
            if (g_kv_fp8 || kv_is_int()) {
                // int8/int4 rows unpack to the same FP16 staging (their per-row scale applied there); the
                // QK / PV GEMMs below are unchanged.
                auto unpack_kv = [&](int k_off, int nk_c) {
                    if (kv_is_int())
                        k::unpack_kv_int_to_f16(kv_int_bits(), Ly.kci, Ly.vci, Ly.ks, Ly.vs, attn_kv_cap,
                                                k0_s + k_off, nk_c, KD, W.kv_k16, W.kv_v16, g_s);
                    else
                        k::unpack_kv_fp8_to_f16(Ly.kc8, Ly.vc8, attn_kv_cap, k0_s + k_off, nk_c, KD, k_inv, v_inv,
                                                W.kv_k16, W.kv_v16, g_s);
                };
                for (int k_off = 0; k_off < nk_s; k_off += W.kv_tmp_cap) {
                    const int nk_c = std::min(W.kv_tmp_cap, nk_s - k_off);
                    unpack_kv(k_off, nk_c);
                    for (int g = 0; g < c.KV; ++g) {
                        const uint16_t* Kp = W.kv_k16 + (size_t) g * c.D;
                        CB(cublasGemmStridedBatchedEx(
                            g_blas, CUBLAS_OP_T, CUBLAS_OP_N, nk_c, Tq, c.D, &one, Kp, CUDA_R_16F, KD, 0,
                            W.x16 + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_16F, HD, (long long) c.D, &zero,
                            W.lg + (size_t) g * GQA * qn + k_off, CUDA_R_32F, nk_s, qn, GQA, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
                    }
                }
                attn_probs(Tq, nk_s, ts, k0_s);
                for (int k_off = 0; k_off < nk_s; k_off += W.kv_tmp_cap) {
                    const int nk_c = std::min(W.kv_tmp_cap, nk_s - k_off);
                    if (nk_s > W.kv_tmp_cap) {
                        unpack_kv(k_off, nk_c);
                    }
                    const float beta = k_off == 0 ? zero : one;
                    for (int g = 0; g < c.KV; ++g) {
                        const uint16_t* Vp = W.kv_v16 + (size_t) g * c.D;
                        CB(cublasGemmStridedBatchedEx(
                            g_blas, CUBLAS_OP_N, CUBLAS_OP_N, c.D, Tq, nk_c, &one, Vp, CUDA_R_16F, KD, 0,
                            lg16 + (size_t) g * GQA * qn + k_off, CUDA_R_16F, nk_s, qn, &beta,
                            W.o + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_32F, HD, (long long) c.D, GQA,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                    }
                }
            } else {
                const int start = attn_kv_cap > 0 ? (k0_s % attn_kv_cap) : k0_s;
                const int n1 = attn_kv_cap > 0 ? std::min(nk_s, attn_kv_cap - start) : nk_s;
                const int n2 = nk_s - n1;
                const double qk0 = apr ? now() : 0;
                for (int g = 0; g < c.KV; ++g) {
                    const uint16_t* Kp1 = Ly.kc + (size_t) start * KD + (size_t) g * c.D;
#if defined(STRATA_USE_HIP)
                    // QK^T on the gfx11 matrix cores: per query head scores[t,key] = sum_d K[key,d] Q[t,d].
                    // K rows stride KD, Q rows stride HD (a strided sub-view, no copy). Off by default: hipBLAS's
                    // tuned batched GEMM beats the single-wave WMMA kernel on these small GQA-batched shapes.
                    static const int wmma_qk = getenv("IK2_WMMA_QK") ? atoi(getenv("IK2_WMMA_QK")) : 0;
                    if (wmma_qk && n1 >= 16 && Tq >= 16) {
                        for (int qh = 0; qh < GQA; ++qh) {
                            strata_wmma_gemm_f16_ld(W.x16 + (size_t) ts * HD + (size_t) (g * GQA + qh) * c.D, HD,
                                                    Kp1, KD, W.lg + (size_t) (g * GQA + qh) * qn, Tq, n1, c.D,
                                                    nk_s, 0.0f, g_s);
                        }
                    } else
#endif
                    CB(cublasGemmStridedBatchedEx(
                        g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n1, Tq, c.D, &one, Kp1, CUDA_R_16F, KD, 0,
                        W.x16 + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_16F, HD, (long long) c.D, &zero,
                        W.lg + (size_t) g * GQA * qn, CUDA_R_32F, nk_s, qn, GQA, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT));
                    if (n2 > 0) {
                        const uint16_t* Kp2 = Ly.kc + (size_t) g * c.D;
#if defined(STRATA_USE_HIP)
                        if (wmma_qk && n2 >= 16 && Tq >= 16) {
                            for (int qh = 0; qh < GQA; ++qh) {
                                strata_wmma_gemm_f16_ld(W.x16 + (size_t) ts * HD + (size_t) (g * GQA + qh) * c.D, HD,
                                                        Kp2, KD, W.lg + (size_t) (g * GQA + qh) * qn + n1, Tq, n2, c.D,
                                                        nk_s, 0.0f, g_s);
                            }
                        } else
#endif
                        CB(cublasGemmStridedBatchedEx(
                            g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n2, Tq, c.D, &one, Kp2, CUDA_R_16F, KD, 0,
                            W.x16 + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_16F, HD, (long long) c.D, &zero,
                            W.lg + (size_t) g * GQA * qn + n1, CUDA_R_32F, nk_s, qn, GQA, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
                    }
                }
                if (apr) { cudaStreamSynchronize(g_s); g_pf.a_qk += now() - qk0; }
                attn_probs(Tq, nk_s, ts, k0_s);
                const double pv0 = apr ? now() : 0;
                for (int g = 0; g < c.KV; ++g) {
                    const uint16_t* Vp1 = Ly.vc + (size_t) start * KD + (size_t) g * c.D;
                    CB(cublasGemmStridedBatchedEx(
                        g_blas, CUBLAS_OP_N, CUBLAS_OP_N, c.D, Tq, n1, &one, Vp1, CUDA_R_16F, KD, 0,
                        lg16 + (size_t) g * GQA * qn, CUDA_R_16F, nk_s, qn, &zero,
                        W.o + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_32F, HD, (long long) c.D, GQA,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                    if (n2 > 0) {
                        const uint16_t* Vp2 = Ly.vc + (size_t) g * c.D;
                        CB(cublasGemmStridedBatchedEx(
                            g_blas, CUBLAS_OP_N, CUBLAS_OP_N, c.D, Tq, n2, &one, Vp2, CUDA_R_16F, KD, 0,
                            lg16 + (size_t) g * GQA * qn + n1, CUDA_R_16F, nk_s, qn, &one,
                            W.o + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_32F, HD, (long long) c.D, GQA,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                    }
                }
                if (apr) { cudaStreamSynchronize(g_s); g_pf.a_pv += now() - pv0; }
            }
            ts += Tq;
        }
        if (apr) { cudaStreamSynchronize(g_s); g_pf.a_out += now() - ap1; }
    }
    if (vp_attn_idx >= 0) cudaEventRecord(vpF[vp_attn_idx], g_s);
    gx_hash_dev("attnout", W.o, C, T, l);
    if (!batch_graph_exact_on(T))
        k::gate_mul_f16(W.o, gate_p, (size_t) T * HD, W.x16, g_dense_q8 && (T == 1 || g_int4_dense_multi || g_int4_msg || g_int4_prefill || batch_dense_mv_on(T) || batch_rotmv_on(T)), g_s);
    matmul(Ly.wo, W.o, W.x16, T, W.tmp, C);
    gx_hash_dev("wo", W.tmp, C, T, l);
    conv_rec_w(Ly, W, 2, W.tmp, T, C);
    if (l % c.block == 0) {
        CU(cudaMemcpyAsync(blk(nb), h_in, TC * 4, cudaMemcpyDeviceToDevice, g_s));
        conv4_attn_w(Ly, W, true, W.tmp, h_in, r_out, T, C);
    } else {
        conv4_attn_w(Ly, W, false, W.tmp, h_in, r_out, T, C);
    }
    gx_hash_dev("postattn", r_out, C, T, l);
}

// A layer-major prefill pass of many tokens uses nearly every expert of a layer, so the ones not in VRAM can be
// copied in before the layer's routing is known: while the previous layer's experts and this layer's attention run.
// The copy stream then has no idle gap between layers. Slots are taken at the current tick, which keeps the previous
// layer's slots (still being computed) from being evicted; only done when the pool holds both layers with room.
void prefetch_layer(Model& M, int l, Work& W, ExpertCache& EC, int T, const Work* Wpop = nullptr, int own = -1, int ahead = 1) {
    const int pr = ((l % Work::kPfRing) + Work::kPfRing) % Work::kPfRing;
    W.pf_layers[pr] = -1;
    if (getenv("IK2_PREFETCH") && !std::atoi(getenv("IK2_PREFETCH"))) return;
    if (!g_overlap || g_prof || l >= M.c.n_layer || (!dual() && M.L[l].dev != g_cur)) return;
#if defined(STRATA_USE_HIP) && defined(STRATA_WMMA_GFX11)
    // Resident INT4 layers need no NVFP4 slots: don't fill the small pool0 with weights nobody reads.
    if (g_int4_experts && M.L[l].i4_loaded && l < g_int4_layers && !g_ep && !g_tp) return;
#endif
    const Work& U = Wpop ? *Wpop : W;   // (whose picks so far: under EP, GPU 0's)
    const int E = M.c.E, pi = EC.layer_pool[l];
    if (pi < 0 || EC.pools[pi].n_active < (ahead + 1) * E + 64) return;
    Layer& Ly = M.L[l];
    const uint64_t t = EC.tick + (uint64_t) ahead;   // (the tick its layer will run in: kept till then)
    W.pf_bs[pr].assign(E, -1);
    // Only the experts this pass will likely use: P(used) = 1 - (1 - p_e)^(T K) from the prefills so far (p_e the
    // expert's share of the layer's tokens); with little history yet, all of them (IK2_PF_P: the threshold)
    const bool known = !U.use_tot.empty() && U.use_tot[l] >= (uint64_t) 16 * E;
    // (0.999, not 0.5: a request's tokens are alike, so the independent-picks estimate is far too high - at 0.5 a
    // fifth of the prefetched experts went unused; 500 / 800 new tokens 4.2 / 4.3 -> 3.5 / 3.6 s, 21.8k unchanged)
    const double thr = getenv("IK2_PF_P") ? std::atof(getenv("IK2_PF_P")) : 0.999;
    const double picks = (double) T * M.c.K;
    int n = 0;
    for (int e = 0; e < E; ++e) {
        if (EC.has(l, e) || (own >= 0 && ep_owner(l, e) != own)) continue;
        if (known) {
            const double pe = (U.use[(size_t) l * E + e] + 0.5) / (U.use_tot[l] + 0.5 * E);
            if (1.0 - std::pow(1.0 - pe, picks) < thr) continue;
        }
        uint8_t* dst = EC.get(Ly, l, e, t, g_cs, nullptr, false, true);
        EC.copy_in(Ly, e, dst, g_cs);
        W.pf_bs[pr][e] = n / Work::kMissBatch;
        if (++n % Work::kMissBatch == 0) CU(cudaEventRecord(W.pf_evs[pr][n / Work::kMissBatch - 1], g_cs));
    }
    if (n % Work::kMissBatch) CU(cudaEventRecord(W.pf_evs[pr][n / Work::kMissBatch], g_cs));
    W.pf_layers[pr] = l;
}

void run_prefill_moe(const Model& M, Layer& Ly, int l, int T, const uint16_t* xq16, const int* top, const float* w,
                     const float* r_in, float* h_out, Work& W, ExpertCache& EC, double& t_exp) {
    const Config& c = M.c;
    const int C = c.C, F = c.F, S = T * c.K;
    // Resident INT4 experts: the converter folded each expert's outer scale into WS. The prefill then uses
    // sg=su=1 and a route-only wtr (no scale[2]); the NVFP4 cache is not consulted for these layers.
    // EP/TP keep the old path: the converted weights are only resident on the loading device.
    bool i4_pf = false;
#if defined(STRATA_USE_HIP) && defined(STRATA_WMMA_GFX11)
    i4_pf = g_int4_experts && Ly.i4_loaded && l < g_int4_layers && !g_ep && !g_tp && !g_int4_pf_off;
#endif
    if (g_prof) ++g_pf.pf_calls;
    const double te0 = now();
    const double ts0 = now();
    if (g_seq_done + 1 < g_seq_next) {
        CU(cudaStreamSynchronize(g_cs));
        g_seq_done = g_seq_next - 1;
    }
    if (g_prof) g_pf.pf_sync += now() - ts0;
    const double th0 = now();   // (prof: host bookkeeping ends here)
    std::fill(W.cnt.begin(), W.cnt.end(), 0);
    for (int i = 0; i < S; ++i) ++W.cnt[top[i]];
    if (W.use.empty()) W.use.assign((size_t) c.n_layer * c.E, 0), W.use_tot.assign(c.n_layer, 0);
    const int prl = l % Work::kPfRing;
    const bool pf_here = W.pf_layers[prl] == l && (int) W.pf_bs[prl].size() == c.E;
    for (int e = 0; e < c.E; ++e) {
        W.use[(size_t) l * c.E + e] += W.cnt[e];
        const int k = W.cnt[e];
        if (k) ++g_cnt_hist[k == 1 ? 0 : k == 2 ? 1 : k < 8 ? 2 : 3];
        const bool pfd = pf_here && W.pf_bs[prl][e] >= 0;
        if (k) ++g_src_hist[pfd ? 1 : EC.has(l, e) ? 0 : 2];
        else if (pfd) ++g_src_hist[3];
    }
    W.use_tot[l] += S;
    W.start[0] = 0;
    for (int e = 0; e < c.E; ++e) W.start[e + 1] = W.start[e] + W.cnt[e];
    for (int e = 0; e < c.E; ++e) W.fill[e] = W.start[e];

    const int ps = l & 1;
    int* gidx = W.gidx_h[ps];
    int* inv = W.inv_h[ps];
    float* sg = W.sg_h[ps];
    float* su = W.su_h[ps];
    float* wt = W.wt_h[ps];
    float* wtr = W.wtr_h[ps];
    for (int i = 0; i < S; ++i) {
        const int e = top[i], row = W.fill[e]++;
        gidx[row] = i / c.K;
        inv[i] = row;
        // i4_pf: the scale is in WS, so the per-row gate/up scales are 1 and wtr carries only the route weight
        sg[row] = i4_pf ? 1.0f : (Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e]);
        su[row] = i4_pf ? 1.0f : (Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e]);
        wt[i] = w[i] * (i4_pf || Ly.scale[2].empty() ? 1.0f : Ly.scale[2][e]);
        wtr[row] = wt[i];
    }
    if (g_prof) g_pf.pf_host += now() - th0;

    // the experts on a GPU: d -1 all of them (one GPU), else the ones it owns (IK2_EP), into X.tmp
    auto exec = [&](int d, Work& X, ExpertCache& XC, const uint16_t* xin) {
        const uint64_t tick = ++XC.tick;
        // (the overlap stream stays on under IK2_PROF: the phase laps already serialise the boundaries, so forcing
        // the copies onto g_s here only inflated t_exp and hid the real expert cost behind the upload phase)
        const cudaStream_t cst = g_overlap ? g_cs : g_s;
        X.hit_exps.clear();
        X.miss_exps.clear();
        const double tu = now();
        if (i4_pf) {
            // INT4 weights are resident: no NVFP4 slots are gathered, copied into, or waited on.
            std::fill(X.slot_ptrs.begin(), X.slot_ptrs.end(), (uint8_t*) nullptr);
        } else {
        for (int e = 0; e < c.E; ++e) {
            if (!W.cnt[e] || (d >= 0 && ep_owner(l, e) != d)) {
                X.slot_ptrs[e] = nullptr;
                continue;
            }
            if (XC.has(l, e)) {
                X.slot_ptrs[e] = XC.get(Ly, l, e, tick, cst, nullptr, false, true);
                X.hit_exps.push_back(e);
            } else {
                X.slot_ptrs[e] = nullptr;
                X.miss_exps.push_back(e);
            }
        }
        for (int e : X.miss_exps) X.slot_ptrs[e] = XC.get(Ly, l, e, tick, cst, nullptr, false, true);
        }

        const int n_miss = (int) X.miss_exps.size();
        const int n_batches = (n_miss + Work::kMissBatch - 1) / Work::kMissBatch;
        for (int b = 0; b < n_batches; ++b) {
            const int b0 = b * Work::kMissBatch, b1 = std::min(n_miss, b0 + Work::kMissBatch);
            for (int m = b0; m < b1; ++m) {
                const int e = X.miss_exps[m];
                XC.copy_in(Ly, e, X.slot_ptrs[e], cst);
            }
            if (cst != g_s) CU(cudaEventRecord(X.miss_ev[b], cst));
        }
        XC.t_up += now() - tu;
        g_pf.lap(g_pf.upload);
        const double tg0 = now();   // (prof: GPU expert work + cache waits)

        CU(cudaMemcpyAsync(X.gidx, gidx, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemcpyAsync(X.inv, inv, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemcpyAsync(X.sg, sg, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemcpyAsync(X.su, su, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemcpyAsync(X.wtr, wtr, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemsetAsync(X.tmp, 0, (size_t) T * C * 4, g_s));

        // Each expert: its tokens gathered, gate/up, SwiGLU, down, then added into tmp with their weights. The experts run
        // in index order whatever is cached (each waiting for its copies), so the sums' order and the result do not
        // depend on the cache. Only [T][C] of scratch: no [T * K][C] copies of the inputs or the outputs.
        auto run_exp = [&](int e) {
            const int cnt_e = W.cnt[e], st_e = W.start[e];
            const uint8_t* sl = X.slot_ptrs[e];
            const bool pr = g_prof && cnt_e >= 128;   // profile only the big GEMM experts (sync per phase)
            const double tga = pr ? now() : 0;
#if defined(STRATA_USE_HIP) && defined(STRATA_WMMA_GFX11)
            if (i4_pf) {
                // Resident INT4: no NVFP4 decode, no gathered weight pointers. Rows are tiled to the
                // scratch capacity (a lent pass can carry more tokens than the buffers were sized for).
                // sg/su=1 and wtr=route weight only: the expert's outer scale is folded into WS.
                // (cap is the f32 scratch's row count; gu32/h16/y32 are sized for T rows, so clamp to both)
                const int cap = std::min(X.i4_pf_cap, std::max(1, T)), Kp = Ly.i4_Kpad[2];
                double a_gather = 0, a_gemm = 0, a_swiglu = 0, a_scatter = 0;
                for (int t0 = 0; t0 < cnt_e; t0 += cap) {
                    const int n = std::min(cap, cnt_e - t0), off = st_e + t0;
                    const double q0 = pr ? now() : 0;
                    gather_f32_16_kernel<<<n, 256, 0, g_s>>>(xin, X.gidx + off, n, C, X.i4_pf_x);
                    if (pr) cudaStreamSynchronize(g_s);
                    const double q1 = pr ? now() : 0;
                    k::int4_rot_pack8(X.i4_pf_x, n, C, X.i4_pf_x4, X.i4_pf_xm, X.i4_pf_xr, g_s);
                    k::int4_gemm8(Ly.i4_Wp[0] + (size_t) e * Ly.i4_wp_stride[0],
                                  Ly.i4_Wh[0] + (size_t) e * Ly.i4_wh_stride[0],
                                  Ly.i4_WS[0] + (size_t) e * 2 * F, 2 * F, C,
                                  X.i4_pf_x4, X.i4_pf_xm, X.i4_pf_xr, n, 4, 8, X.gu32, nullptr, k::kInt4Sh8, g_s);
                    if (pr) cudaStreamSynchronize(g_s);
                    const double q2 = pr ? now() : 0;
                    k::swiglu_cap_f32(X.gu32, n, F, X.sg + off, X.su + off, c.gate_cap, c.up_cap, X.h16, g_s);
                    i4_pad_h_kernel<<<n, 256, 0, g_s>>>(X.h16, n, F, Kp, X.i4_pf_x);
                    if (pr) cudaStreamSynchronize(g_s);
                    const double q3 = pr ? now() : 0;
                    k::int4_rot_pack8(X.i4_pf_x, n, Kp, X.i4_pf_x4, X.i4_pf_xm, X.i4_pf_xr, g_s);
                    k::int4_gemm8(Ly.i4_Wp[2] + (size_t) e * Ly.i4_wp_stride[2],
                                  Ly.i4_Wh[2] + (size_t) e * Ly.i4_wh_stride[2],
                                  Ly.i4_WS[2] + (size_t) e * C, C, Kp,
                                  X.i4_pf_x4, X.i4_pf_xm, X.i4_pf_xr, n, 4, 8, X.y32, nullptr, k::kInt4Sh8, g_s);
                    if (pr) cudaStreamSynchronize(g_s);
                    const double q4 = pr ? now() : 0;
                    scatter_add_rows_kernel<<<n, 256, 0, g_s>>>(X.y32, X.gidx + off, X.wtr + off, C, X.tmp, C);
                    if (pr) {
                        cudaStreamSynchronize(g_s);
                        const double q5 = now();
                        a_gather += q1 - q0; a_gemm += (q2 - q1) + (q4 - q3);
                        a_swiglu += q3 - q2; a_scatter += q5 - q4;
                    }
                }
                if (pr) {
                    g_pf.pf_gather += a_gather; g_pf.pf_dec += 0; g_pf.pf_gemm += a_gemm;
                    g_pf.pf_swiglu += a_swiglu; g_pf.pf_scatter += a_scatter;
                }
                return;
            }
#endif
            gather_f16_16_kernel<<<cnt_e, 256, 0, g_s>>>(xin, X.gidx + st_e, cnt_e, C, X.ex16);
            if (pr) cudaStreamSynchronize(g_s);
            const double tde = pr ? now() : 0;
#if defined(STRATA_USE_HIP) && defined(STRATA_WMMA_GFX11)
            if (g_exp_prefill_wmma && Ly.ty[0] == k::kNVFP4 && Ly.ty[1] == k::kNVFP4 && Ly.ty[2] == k::kNVFP4
                && cnt_e >= 16) {
                // NVFP4-direct multi-token WMMA: no decode_f16 whole-weight expand; fragments read in the GEMM.
                k::experts_prefill_gate_up_wmma(sl, Ly.eb[0], C, F, cnt_e, X.ex16, X.gu32, g_s);
                if (pr) cudaStreamSynchronize(g_s);
                const double tgm = pr ? now() : 0;
                k::swiglu_cap_f32(X.gu32, cnt_e, F, X.sg + st_e, X.su + st_e, c.gate_cap, c.up_cap, X.h16, g_s);
                if (pr) cudaStreamSynchronize(g_s);
                const double tsw = pr ? now() : 0;
                k::experts_prefill_down_wmma(sl, Ly.eb[0] + Ly.eb[1], C, F, cnt_e, X.h16, X.y32, g_s);
                if (pr) cudaStreamSynchronize(g_s);
                const double tdn = pr ? now() : 0;
                scatter_add_rows_kernel<<<cnt_e, 256, 0, g_s>>>(X.y32, X.gidx + st_e, X.wtr + st_e, C, X.tmp, C);
                if (pr) {
                    cudaStreamSynchronize(g_s);
                    const double te = now();
                    g_pf.pf_gather += tde - tga;
                    g_pf.pf_dec += 0;   // no decode_f16 in this path
                    g_pf.pf_gemm += (tgm - tde) + (tdn - tsw);
                    g_pf.pf_swiglu += tsw - tgm;
                    g_pf.pf_scatter += te - tdn;
                }
                return;
            }
#endif
            if (Ly.ty[0] == Ly.ty[1]) {
                k::decode_f16(Ly.ty[0], sl, (size_t) 2 * F * C, X.w16, g_s);
            } else {
                k::decode_f16(Ly.ty[0], sl, (size_t) F * C, X.w16, g_s);
                k::decode_f16(Ly.ty[1], sl + Ly.eb[0], (size_t) F * C, X.w16 + (size_t) F * C, g_s);
            }
            if (pr) cudaStreamSynchronize(g_s);
            const double tgm = pr ? now() : 0;
            gemm16(X.w16, 2 * F, C, X.ex16, cnt_e, X.gu32, 2 * F);
            if (pr) cudaStreamSynchronize(g_s);
            const double tsw = pr ? now() : 0;
            k::swiglu_cap_f32(X.gu32, cnt_e, F, X.sg + st_e, X.su + st_e, c.gate_cap, c.up_cap, X.h16, g_s);
            k::decode_f16(Ly.ty[2], sl + Ly.eb[0] + Ly.eb[1], (size_t) C * F, X.w16, g_s);
            if (pr) cudaStreamSynchronize(g_s);
            const double tdn = pr ? now() : 0;
            gemm16(X.w16, C, F, X.h16, cnt_e, X.y32, C);
            if (pr) cudaStreamSynchronize(g_s);
            const double tsc = pr ? now() : 0;
            scatter_add_rows_kernel<<<cnt_e, 256, 0, g_s>>>(X.y32, X.gidx + st_e, X.wtr + st_e, C, X.tmp, C);
            if (pr) {
                cudaStreamSynchronize(g_s);
                const double te = now();
                g_pf.pf_gather += tde - tga;
                g_pf.pf_dec += (tgm - tde) + (tdn - tsw);
                g_pf.pf_gemm += (tsw - tgm) + (tsc - tdn);
                g_pf.pf_scatter += te - tsc;
            }
        };
        const bool pf = X.pf_layers[l % Work::kPfRing] == l;
        X.miss_pos.assign(c.E, -1);
        for (int m = 0; m < n_miss; ++m) X.miss_pos[X.miss_exps[m]] = m;
        int pf_waited = -1, miss_waited = -1;
        for (int e = 0; e < c.E; ++e) {
            if (!W.cnt[e] || (d >= 0 && ep_owner(l, e) != d)) continue;
            if (pf && X.pf_bs[l % Work::kPfRing][e] >= 0 && X.pf_bs[l % Work::kPfRing][e] > pf_waited)
                CU(cudaStreamWaitEvent(g_s, X.pf_evs[l % Work::kPfRing][pf_waited = X.pf_bs[l % Work::kPfRing][e]], 0));
            if (X.miss_pos[e] >= 0 && cst != g_s && X.miss_pos[e] / Work::kMissBatch > miss_waited)
                CU(cudaStreamWaitEvent(g_s, X.miss_ev[miss_waited = X.miss_pos[e] / Work::kMissBatch], 0));
            run_exp(e);
        }
        X.pf_layers[l % Work::kPfRing] = -1;
        if (g_prof) { cudaStreamSynchronize(g_s); g_pf.pf_gpu += now() - tg0; }
    };
    if (g_ep) {   // GPU 1's experts on GPU 1 (the input over, its sums back), GPU 0's meanwhile
        CU(cudaEventRecord(g_ep_x, g_s));
        use_dev(1);
        Work& X1 = dev_work(1);
        CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
        CU(cudaMemcpyPeerAsync(X1.xq16_sup, 1, xq16, 0, (size_t) T * C * 2, g_s));
        exec(1, X1, dev_cache(1), X1.xq16_sup);
        CU(cudaEventRecord(g_ep_done, g_s));
        use_dev(0);
        exec(0, W, EC, xq16);
        CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
        CU(cudaMemcpyPeerAsync(W.y32, 0, X1.tmp, 1, (size_t) T * C * 4, g_s));
        k::add(W.tmp, W.y32, (size_t) T * C, W.tmp, g_s);
    } else if (g_tp) {
        // tensor-parallel experts: (1) each GPU its half of every expert's h into tp_h (GPU 1 with the input copied
        // over), (2) the halves swapped, (3) each GPU its half of the output rows, GPU 1's columns copied to GPU 0
        const int Fd = F / 2, Cd = C / 2;
        auto tp_gu = [&](int d, Work& X, ExpertCache& XC, const uint16_t* xin) {
            const uint64_t tick = ++XC.tick;
            const cudaStream_t cst = g_overlap && !g_prof ? g_cs : g_s;
            X.hit_exps.clear();
            X.miss_exps.clear();
            for (int e = 0; e < c.E; ++e) {
                if (!W.cnt[e]) {
                    X.slot_ptrs[e] = nullptr;
                    continue;
                }
                if (XC.has(l, e)) {
                    X.slot_ptrs[e] = XC.get(Ly, l, e, tick, cst, nullptr, false, true);
                    X.hit_exps.push_back(e);
                } else {
                    X.slot_ptrs[e] = nullptr;
                    X.miss_exps.push_back(e);
                }
            }
            for (int e : X.miss_exps) X.slot_ptrs[e] = XC.get(Ly, l, e, tick, cst, nullptr, false, true);
            const int n_miss = (int) X.miss_exps.size();
            const int n_batches = (n_miss + Work::kMissBatch - 1) / Work::kMissBatch;
            for (int b = 0; b < n_batches; ++b) {
                const int b0 = b * Work::kMissBatch, b1 = std::min(n_miss, b0 + Work::kMissBatch);
                for (int m = b0; m < b1; ++m) XC.copy_in(Ly, X.miss_exps[m], X.slot_ptrs[X.miss_exps[m]], cst);
                if (cst != g_s) CU(cudaEventRecord(X.miss_ev[b], cst));
            }
            CU(cudaMemcpyAsync(X.gidx, gidx, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
            CU(cudaMemcpyAsync(X.sg, sg, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
            CU(cudaMemcpyAsync(X.su, su, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
            CU(cudaMemcpyAsync(X.wtr, wtr, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
            size_t o0, b0g, o1, b1u;
            tp_part(Ly, d, 0, o0, b0g);
            tp_part(Ly, d, 1, o1, b1u);
            const bool pf = X.pf_layers[l % Work::kPfRing] == l;
            X.miss_pos.assign(c.E, -1);
            for (int m = 0; m < n_miss; ++m) X.miss_pos[X.miss_exps[m]] = m;
            int pf_waited = -1, miss_waited = -1;
            for (int e = 0; e < c.E; ++e) {
                if (!W.cnt[e]) continue;
                if (pf && X.pf_bs[l % Work::kPfRing][e] >= 0 && X.pf_bs[l % Work::kPfRing][e] > pf_waited)
                    CU(cudaStreamWaitEvent(g_s, X.pf_evs[l % Work::kPfRing][pf_waited = X.pf_bs[l % Work::kPfRing][e]], 0));
                if (X.miss_pos[e] >= 0 && cst != g_s && X.miss_pos[e] / Work::kMissBatch > miss_waited)
                    CU(cudaStreamWaitEvent(g_s, X.miss_ev[miss_waited = X.miss_pos[e] / Work::kMissBatch], 0));
                const int cnt_e = W.cnt[e], st_e = W.start[e];
                const uint8_t* sl = X.slot_ptrs[e];
                gather_f16_16_kernel<<<cnt_e, 256, 0, g_s>>>(xin, X.gidx + st_e, cnt_e, C, X.ex16);
                if (Ly.ty[0] == Ly.ty[1]) {
                    k::decode_f16(Ly.ty[0], sl, (size_t) 2 * Fd * C, X.w16, g_s);
                } else {
                    k::decode_f16(Ly.ty[0], sl, (size_t) Fd * C, X.w16, g_s);
                    k::decode_f16(Ly.ty[1], sl + b0g, (size_t) Fd * C, X.w16 + (size_t) Fd * C, g_s);
                }
                gemm16(X.w16, 2 * Fd, C, X.ex16, cnt_e, X.gu32, 2 * Fd);
                k::swiglu_cap_f32(X.gu32, cnt_e, Fd, X.sg + st_e, X.su + st_e, c.gate_cap, c.up_cap, X.h16, g_s);
                CU(cudaMemcpy2DAsync(X.tp_h + (size_t) st_e * F + (size_t) d * Fd, (size_t) F * 2, X.h16, (size_t) Fd * 2,
                                     (size_t) Fd * 2, cnt_e, cudaMemcpyDeviceToDevice, g_s));
            }
            X.pf_layers[l % Work::kPfRing] = -1;
            CU(cudaEventRecord(g_tp_h[d], g_s));
        };
        auto tp_dn = [&](int d, Work& X, const Work& O) {
            // the other GPU's half of h in, then this GPU's rows of down for every expert
            CU(cudaStreamWaitEvent(g_s, g_tp_h[1 - d], 0));
            CU(cudaMemcpy2DAsync(X.tp_h + (size_t) (1 - d) * Fd, (size_t) F * 2, O.tp_h + (size_t) (1 - d) * Fd,
                                 (size_t) F * 2, (size_t) Fd * 2, S, cudaMemcpyDefault, g_s));
            CU(cudaMemsetAsync(X.tmp, 0, (size_t) T * C * 4, g_s));
            size_t o, b0g, b1u;
            tp_part(Ly, d, 0, o, b0g);
            tp_part(Ly, d, 1, o, b1u);
            for (int e = 0; e < c.E; ++e) {
                if (!W.cnt[e]) continue;
                const int cnt_e = W.cnt[e], st_e = W.start[e];
                k::decode_f16(Ly.ty[2], X.slot_ptrs[e] + b0g + b1u, (size_t) Cd * F, X.w16, g_s);
                gemm16(X.w16, Cd, F, X.tp_h + (size_t) st_e * F, cnt_e, X.y32, Cd);
                scatter_add_rows_kernel<<<cnt_e, 256, 0, g_s>>>(X.y32, X.gidx + st_e, X.wtr + st_e, Cd,
                                                               X.tmp + (size_t) d * Cd, C);
            }
        };
        Work& X1 = dev_work(1);
        CU(cudaEventRecord(g_ep_x, g_s));
        use_dev(1);
        CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
        CU(cudaMemcpyPeerAsync(X1.xq16_sup, 1, xq16, 0, (size_t) T * C * 2, g_s));
        tp_gu(1, X1, dev_cache(1), X1.xq16_sup);
        use_dev(0);
        tp_gu(0, W, EC, xq16);
        use_dev(1);
        tp_dn(1, X1, W);
        CU(cudaEventRecord(g_ep_done, g_s));
        use_dev(0);
        tp_dn(0, W, X1);
        CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
        CU(cudaMemcpy2DAsync(W.tmp + Cd, (size_t) C * 4, X1.tmp + Cd, (size_t) C * 4, (size_t) Cd * 4, T,
                             cudaMemcpyDefault, g_s));
    } else {
        exec(-1, W, EC, xq16);
    }
    g_pf.lap(g_pf.experts);
    t_exp += now() - te0;
    conv_rec_w(Ly, W, 3, W.tmp, T, C);
    conv4_mlp_w(Ly, W, W.tmp, r_in, h_out, T, C);
}

void run_head(const Model& M, int T, bool all_logits, OutMode out_mode, int nb, const float* h_in, Work& W,
              std::vector<float>& logits_out, std::vector<int>* argmax_out) {
    const Config& c = M.c;
    const int C = c.C;
    const size_t TC = (size_t) T * C;
    g_pf.start();
    if (out_mode != OutMode::kNone || g_taps) {
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = W.blocks + (size_t) i * W.blk_ld * C;
        st.st[nb] = h_in;
        if (out_mode != OutMode::kNone && (all_logits || T == 1)) {
            k::RmsOuts outs{};
            outs.w[0] = M.out_norm;
            outs.y16[0] = W.x16;
            outs.y32[0] = (g_taps || g_dense_q8) ? W.xq : nullptr;
            k::attn_res_rms_f16(st, nb + 1, M.out_res_score, 1, T, C, 1e-6f, c.eps, outs, g_s);
            if (g_taps)
                CU(cudaMemcpyAsync(g_taps + (size_t) (g_taps_n - 1) * g_taps_ld * C, W.xq, TC * 4, cudaMemcpyDeviceToDevice, g_s));
        } else {
            k::attn_res(st, nb + 1, M.out_res_score, 1, T, C, 1e-6f, W.tmp, g_s);
            k::rms(W.tmp, M.out_norm, T, C, c.eps, W.xq, g_s);
            if (g_taps)
                CU(cudaMemcpyAsync(g_taps + (size_t) (g_taps_n - 1) * g_taps_ld * C, W.xq, TC * 4, cudaMemcpyDeviceToDevice, g_s));
            if (out_mode != OutMode::kNone) k::to_f16(W.xq + (size_t) (T - 1) * C, (size_t) C, W.x16, g_s);
        }
        if (out_mode != OutMode::kNone) {
            const int t0 = all_logits ? 0 : T - 1, nt = T - t0;
            // IK2_INT4_HEAD rides matmul's T=2..8 gate (i4_head_only): the verify GEMM (nt=2..8) takes the int4
            // sidecar; n=1 and the prompt chunks (n=16) keep their existing route, with matmul's fallback intact.
            if (out_mode == OutMode::kArgmax) {
                argmax_out->resize(nt);
                const bool exact = batch_graph_exact_on(T) && W.batch && M.out_w.i4Wp && M.out_w.i4Wh &&
                                   M.out_w.i4WS && C % 256 == 0 && M.out_w.N % 16 == 0 && T <= 8;
                for (int i = 0; i < nt; i += 16) {
                    const int n = std::min(16, nt - i);
                    const float* xrow = W.xq + (size_t) (t0 + i) * C;
                    if (exact) {
                        k::int4_dense_rot_rows(M.out_w.i4Wp, M.out_w.i4Wh, M.out_w.i4WS, M.out_w.N, C, xrow, n,
                                               W.logits, M.out_w.N, 1.f / (1 << k::kInt4Sh8), g_s, nullptr, 0);
                    } else {
                        matmul(M.out_w, xrow, W.x16 + (size_t) i * C, n, W.logits, c.n_rows);
                    }
                    if (exact) {
                        for (int r = 0; r < n; ++r)
                            k::argmax_sum(W.logits + (size_t) r * M.out_w.N, nullptr, nullptr, c.n_vocab,
                                          W.argmax_d + i + r, g_s);
                    } else if (n == 1) k::argmax_sum(W.logits, nullptr, nullptr, c.n_vocab, W.argmax_d + i, g_s);
                    else k::argmax_rows(W.logits, c.n_rows, n, c.n_vocab, W.argmax_d + i, g_s);
                }
                if (g_vg_stage == VG_NORMAL) { CU(cudaMemcpyAsync(W.argmax_h, W.argmax_d, (size_t) nt * 4, cudaMemcpyDeviceToHost, g_s));
                CU(cudaStreamSynchronize(g_s));
                for (int i = 0; i < nt; ++i) (*argmax_out)[i] = W.argmax_h[i]; }
            } else {
                for (int i = 0; i < nt; i += 16) {   // (W.logits holds 16 rows)
                    const int n = std::min(16, nt - i);
                    matmul(M.out_w, W.xq + (size_t) (t0 + i) * C, W.x16 + (size_t) i * C, n, W.logits, c.n_rows);
                    k::softcap(W.logits, (size_t) n * c.n_rows, c.logit_cap, g_s);
                    const size_t at = logits_out.size();
                    logits_out.resize(at + (size_t) n * c.n_vocab);
                    CU(cudaMemcpy2DAsync(logits_out.data() + at, (size_t) c.n_vocab * 4, W.logits,
                                         (size_t) c.n_rows * 4, (size_t) c.n_vocab * 4, n, cudaMemcpyDeviceToHost,
                                         g_s));
                }
                CU(cudaStreamSynchronize(g_s));
            }
        }
    }
    g_pf.lap(g_pf.rest);
}

// ---- the resident token path
// When the cache holds every expert of the model (Inkling-2-Mini on a 24 GB card) none is ever missing or evicted, so
// an expert's blocks are found on the device (Layer::tab_blob) and a token needs nothing from the host between its
// embedding and its logits: no routing round trip a layer, no table upload.  The position is read on the device too
// (W.pos), so a token's launches do not change from one token to the next and are replayed as one CUDA graph.  The
// dense weights are the GGUF's blocks (IK2_DENSE=q8, the default when the experts fit), one product a launch.
bool g_fast = false;        // set at load: every expert resident, one GPU, the dense weights as blocks (IK2_FAST=0: no)
bool g_fast_want = false;
bool g_graph = true;        // IK2_GRAPH=0: the resident path's launches one by one
int g_attn_mma = 1;         // a token's attention on the tensor cores (IK2_ATTN_MMA=0: the scalar kernels; 2: the
                            // layers without a window only)
int g_mma_split = 1;        // its blocks a K / V head (k::attn_decode_mma_blocks: the device filled once)
int g_decode_pfull = 128;   // IK2_DECODE_PFULL: key splits for full-context decode attention (1..512)
int g_attn_wideload = 0;    // IK2_ATTN_WIDELOAD=1: int4-only coalesced decode attention (default off)
struct FastGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool disabled = false;
};
FastGraph g_fgraph[4];      // by OutMode
// IK2_AR_DUAL: one graph per (slot, OutMode).  Each graph captures the SAME T=1 fast_token_launches sequence with
// that slot's per-visit KV/conv pointers baked in, so replay is bit-exact to a solo T=1 token.
FastGraph g_ar_fgraph[2][4];
// IK2_AR_KSTEP: one pre-captured graph per slot.  Greedy-only (kArgmax), so no OutMode dimension is needed.
FastGraph g_ar_kgraph[2];

// IK2_VERIFY_GRAPH: one graph for the whole T=8 g_rec verify forward (48 layer visits + head).  Captured on the
// first pure-argmax verify window after a warm-up normal pass, replayed at every later window.  Inputs (token rows,
// positions) arrive through pinned W.emb_h / W.pos_h, and the base position is read on the device by the prep /
// attention kernels, so a replay needs no host value between launches.  The head's argmaxes are read once after
// the replay; the accept comparison and rollback stay host-side, outside the graph.
struct VerifyGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool built = false;
    bool disabled = false;
    bool warmed = false;   // one normal pass first: warms hipBLAS shapes/workspace before capture
};
VerifyGraph g_vgraph;

// IK2_BATCH_GRAPH: one graph for a true batched greedy step (T rows, one per slot).  Captured after one direct warm-up
// pass; the token rows, positions and per-visit pointer tables are staged into the same pinned buffers the graph was
// captured with, so replay only needs to refresh those buffers and launch.  The head's argmaxes are read after the
// launch; host sampling stays outside the graph.
struct BatchGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool built = false;
    bool warmed = false;
    bool disabled = false;
    int rows = 0;
};
BatchGraph g_bgraph;

// one token's launches.  With kstep=false this is EXACTLY the original fast_token_launches sequence (the default
// path is byte-identical).  With kstep=true, step 0 stages the embedding/position from pinned host, steps 1..K-1
// gather the previous token's embedding on the device from from_device_token and advance W.pos there; none of them
// does a host sync, and argmax writes go to argmax_out_d (per-Work scratch overload) instead of W.argmax_d.
void one_token_launches(Model& M, Work& W, OutMode out_mode, bool kstep = false, int step = 0,
                        const int* from_device_token = nullptr, int* argmax_out_d = nullptr,
                        bool copy_argmax_h = true) {
    const Config& c = M.c;
    const int C = c.C, HD = c.H * c.D, KD = c.KV * c.D, HR = c.H * c.d_rel;
    const size_t Cb = (size_t) C * 4;
    if (!argmax_out_d) argmax_out_d = W.argmax_d;
    const float qk_scale = 1.0f / std::sqrt((float) c.D);
    const float k_scale = c.qk_clip <= 28.0f ? 16.0f : 1.0f, v_scale = 32.0f;
    // the attention's key splits: fixed, so that the launches are the same at every position
    int P_sl = std::min(128, std::max(1, (c.window + 63) / 64)), P_full = g_decode_pfull;
    if (const char* ps = getenv("IK2_DECODE_P")) { int v = std::atoi(ps); if (v >= 1 && v <= 128) P_sl = v; }
    if (g_prof_fast) pf_fast::setup();
    pf_fast::mark(0, g_s);
    if (kstep && step > 0) {
        // Device gather of the token emitted by step-1, into this Work's fresh intermediate buffer.  rms reads it
        // below; W.tmp is only the step-0 embedding staging and is reused by the visits anyway.
        float* emb_src = W.emb_dK + (size_t) (step - 1) * C;
        k::embed_gather_f32(M.embd_type, M.embd_d, M.embd_rows, M.embd_row_bytes, C, from_device_token, 1, emb_src, g_s);
        k::rms(emb_src, M.embd_norm, 1, C, c.eps, W.h, g_s);
        // W.pos already holds base+step from the device-side pos_inc between steps; no H2D here.
    } else {
        CU(cudaMemcpyAsync(W.tmp, W.emb_h, Cb, cudaMemcpyHostToDevice, g_s));
        k::rms(W.tmp, M.embd_norm, 1, C, c.eps, W.h, g_s);
        CU(cudaMemcpyAsync(W.pos, W.pos_h, 4, cudaMemcpyHostToDevice, g_s));
    }
    pf_fast::mark(1, g_s);
    auto blk = [&](int i) { return W.blocks + (size_t) i * W.blk_ld * C; };
    int nb = 0, nb_pre = 0, n_bank = 0;
    for (int vis = 0; vis < c.n_visit; ++vis) {
        const int l = M.V[vis].l;
        Layer& Ly = bind_visit(M, vis);
        // (profiling: ev1 is re-recorded at each visit's start so ev1->ev2 spans only this visit's loop_bank work,
        //  not every prior visit's compute; the 48 per-visit deltas sum to the whole token.)
        pf_fast::mark(1, g_s);
        // a looped core's passes (as forward()'s)
        const float* msg = nullptr;
        if (c.n_pass > 1 && l == c.core0) {
            if (M.V[vis].pass == 0) {
                nb_pre = nb;
                CU(cudaMemcpyAsync(W.lp_pre, W.h, Cb, cudaMemcpyDeviceToDevice, g_s));
            } else {
                if (g_loopbank_fuse && nb > nb_pre) {
                    // the same copies as one launch (row = C floats; h's bank copy and its restore share a thread)
                    k::loop_bank_copies(W.lp_pre, W.h, W.blocks, (size_t) W.blk_ld * C, nb_pre, nb, W.lp_bank, W.h,
                                        (size_t) C, g_s);
                    n_bank = nb - nb_pre;
                } else {
                    n_bank = 0;
                    for (int i = nb_pre + 1; i < nb; ++i, ++n_bank)
                        CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * C, blk(i), Cb, cudaMemcpyDeviceToDevice, g_s));
                    CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * C, W.h, Cb, cudaMemcpyDeviceToDevice, g_s));
                    ++n_bank;
                    CU(cudaMemcpyAsync(W.h, W.lp_pre, Cb, cudaMemcpyDeviceToDevice, g_s));
                }
                nb = nb_pre;
                // the pass's messages, every core layer's at once: the bank stays as it is until the pass ends, and a
                // layer's launch of its own was 7 us on one SM (a job a block: the same sums, 24 launches less a token)
                k::ResStates st{};
                for (int i = 0; i < n_bank; ++i) st.st[i] = W.lp_bank + (size_t) i * C;
                k::attn_res_jobs(st, n_bank, M.lp_query_all, 2, 2 * (c.core1 - c.core0), C, 1e-6f, W.lp_msg_all, g_s);
            }
        }
        if (M.V[vis].pass > 0) msg = W.lp_msg_all + (size_t) (l - c.core0) * 4 * C;   // (q, k | v, MoE)
        pf_fast::mark(2, g_s);
        // the attention's inputs, normed, in F32: W.xq, W.xk, W.xv (sep: q, k and v each have their own)
        float* x3[3] = {W.xq, W.xk, W.xv};
        const float* nw[3] = {Ly.sliding ? Ly.attn_norm : Ly.q_in_norm, Ly.sliding ? Ly.attn_norm : Ly.k_in_norm,
                              Ly.sliding ? Ly.attn_norm : Ly.v_in_norm};
        bool sep = !Ly.sliding;
        // (rms_alt experiment REVERTED: norming the fused projection's input in every block's prologue recomputes
        //  the 2304-norm 2144x and costs LDS/syncthreads - measured -3% decode vs one cheap rms launch. The kernel
        //  mv_q8p_rms_alt stays in the tree, unwired.)
        const bool rms_alt = false;
        {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
            st.st[nb] = W.h;
            if (msg) {
                k::attn_res(st, nb + 1, Ly.res_score, Ly.n_res_q, 1, C, 1e-6f, W.lp_in, g_s);
                k::LoopGate gt[3];
                for (int j = 0; j < 3; ++j)
                    gt[j] = {W.lp_in + (size_t) (Ly.n_res_q == 3 ? j : 0) * C, msg + (size_t) j * C, Ly.lp_down[j],
                             Ly.lp_forget[j], Ly.lp_gain[j], x3[j]};
                k::loop_gates(gt, 3, 1, C, c.rank, 1e-6f, W.lp_g, g_s);
                k::rms_multi(k::RmsJobs{{x3[0], x3[1], x3[2]}, {nw[0], nw[1], nw[2]}, {x3[0], x3[1], x3[2]}}, 3, C, c.eps,
                             g_s);
                sep = true;
            } else if (l > 0) {
                k::RmsOuts outs{};
                uint16_t* y16[3] = {W.x16, W.xk16, W.xv16};
                for (int j = 0; j < Ly.n_res_q; ++j) outs.w[j] = nw[j], outs.y16[j] = y16[j], outs.y32[j] = x3[j];
                k::attn_res_rms_f16(st, nb + 1, Ly.res_score, Ly.n_res_q, 1, C, 1e-6f, c.eps, outs, g_s);
            } else if (l == 0 && sep && g_rms_fuse) {
                // (a full layer 0: q/k/v share W.h, three norms in one launch instead of three)
                k::rms_multi(k::RmsJobs{{W.h, W.h, W.h}, {nw[0], nw[1], nw[2]}, {x3[0], x3[1], x3[2]}}, 3, C, c.eps, g_s);
            } else if (rms_alt) {
                // (a sliding layer: the fused product's prologue norms W.h; no separate input-norm launch here)
            } else {
                for (int j = 0; j < (sep ? 3 : 1); ++j) k::rms(W.h, nw[j], 1, C, c.eps, x3[j], g_s);
            }
        }
        // the projections, straight over the blocks
        pf_fast::mark(3, g_s);
        float *q_p, *k_p, *v_p, *r_p, *gate_p, *kd_p = nullptr, *vd_p = nullptr;
        // (k's and v's short convs ride their products' launches - mv1c - and the sinks the clips' launch below)
        if (Ly.sliding) {
            const Mat& m = Ly.wqkvrg;   // q | k | v | r | gate
            q_p = W.qkv, k_p = q_p + HD, v_p = k_p + KD, r_p = v_p + KD, gate_p = r_p + HR;
            if (!sep) {
                if (rms_alt) {
                    // (a sliding layer: q/k/v/r/gate share the normed input W.h; the rms rides the product launch)
                    k::mv_q8p_rms_alt(m.q8, m.q8d, m.N, m.K, W.h, nw[0], c.eps, W.qkv,
                                      k::MvConv{HD, KD, Ly.h_k, Ly.sc_k, nullptr, 0, nullptr, W.kconv},
                                      k::MvConv{HD + KD, KD, Ly.h_v, Ly.sc_v, nullptr, 0, nullptr, W.vconv}, g_s);
                } else {
                    mv1c(m, W.xq, W.qkv, k::MvConv{HD, KD, Ly.h_k, Ly.sc_k, nullptr, 0, nullptr, W.kconv},
                         k::MvConv{HD + KD, KD, Ly.h_v, Ly.sc_v, nullptr, 0, nullptr, W.vconv});
                }
            } else if (m.i4Wp) {   // IK2_INT4_DENSE: same one launch, the int4 product with the k/v inputs by tile
                k::int4_dense_rot(m.i4Wp, m.i4Wh, m.i4WS, m.N, m.K, W.xq, W.qkv,
                                  k::MvConv{HD, KD, Ly.h_k, Ly.sc_k, nullptr, 0, nullptr, W.kconv},
                                  k::MvConv{HD + KD, KD, Ly.h_v, Ly.sc_v, nullptr, 0, nullptr, W.vconv},
                                  1.f / (1 << k::kInt4Sh8), g_s, k::MvAlt{W.xk, HD, KD}, k::MvAlt{W.xv, HD + KD, KD});
            } else if (m.q8) {   // (q, k and v each with its input: k's and v's rows take theirs, one launch)
                k::mv_q8p_conv(m.q8, m.q8d, m.N, m.K, W.xq, W.qkv,
                               k::MvConv{HD, KD, Ly.h_k, Ly.sc_k, nullptr, 0, nullptr, W.kconv},
                               k::MvConv{HD + KD, KD, Ly.h_v, Ly.sc_v, nullptr, 0, nullptr, W.vconv}, g_s,
                               k::MvAlt{W.xk, HD, KD}, k::MvAlt{W.xv, HD + KD, KD});
            } else {
                mv1(mat_rows(m, 0, HD), W.xq, W.qkv);
                mv1c(mat_rows(m, HD, KD), W.xk, k_p, k::MvConv{0, KD, Ly.h_k, Ly.sc_k, nullptr, 0, nullptr, W.kconv});
                mv1c(mat_rows(m, HD + KD, KD), W.xv, v_p,
                     k::MvConv{0, KD, Ly.h_v, Ly.sc_v, nullptr, 0, nullptr, W.vconv});
                mv1(mat_rows(m, HD + 2 * KD, m.N - HD - 2 * KD), W.xq, W.qkv + HD + 2 * KD);
            }
        } else {
            const Mat& m = Ly.wqrg;     // q | r | gate
            mv1(m, W.xq, W.qkv);
            q_p = W.qkv, r_p = q_p + HD, gate_p = r_p + HR;
            k_p = W.kk, v_p = W.vv, kd_p = W.dyn, vd_p = W.dyn + (size_t) c.KV * 4;
            k::mv_f16(Ly.sc_k_dyn16, c.KV * 4, C, W.xk, kd_p, g_s);
            k::mv_f16(Ly.sc_v_dyn16, c.KV * 4, C, W.xv, vd_p, g_s);
            mv1c(Ly.wk, W.xk, k_p, k::MvConv{0, KD, Ly.h_k, Ly.sc_k, kd_p, c.KV, nullptr, W.kconv});
            mv1c(Ly.wv, W.xv, v_p, k::MvConv{0, KD, Ly.h_v, Ly.sc_v, vd_p, c.KV, nullptr, W.vconv});
        }
        // the clips, the token's K and V into the cache at its position, the attention
        pf_fast::mark(4, g_s);
        const int extent = Ly.sliding ? c.extent_swa : c.extent;
        // IK2_FULL_WINDOW: the kernels bound their own key walk from `window` (k0 = qpos - window + 1), so this is
        // the one argument that decides whether a full-attention layer re-reads the whole context or just its tail.
        const int window = Ly.sliding ? c.window : (g_full_window > 0 ? g_full_window : (1 << 30));
        const int P = Ly.sliding ? P_sl : P_full;
        // the attention on the tensor cores, from an FP16 copy of q (IK2_ATTN_MMA=0: the scalar kernels - a layer
        // without a window, its keys the whole context, is bound by their arithmetic).  int8/int4 KV use the
        // scalar dequant kernels: the tensor-core path reads raw f16 / fp8 codes and has no scale per head.
        const bool mma = (g_attn_mma == 1 || (g_attn_mma == 2 && !Ly.sliding)) && !kv_is_int();
        const int ring = Ly.sliding ? Ly.kv_cap : 0;
        const int ns = Ly.sliding ? std::min(g_mma_split, std::max(1, (c.window + 63) / 64)) : g_mma_split;
        uint16_t* q16 = mma ? W.x16 : nullptr;
        pf_fast::mark(5, g_s);
        if (kv_is_int()) {
            const int bits = kv_int_bits();
            k::prep_qkv_r_int_at(bits, q_p, Ly.q_clip, W.kconv, Ly.kci, Ly.ks, W.vconv, Ly.vci, Ly.vs, r_p,
                                 Ly.r_clip, c.H, c.KV, c.D, c.d_rel, c.qk_clip, c.rel_clip, c.eps, W.pos,
                                 !Ly.sliding, c.alpha, c.n_floor, Ly.kv_cap, q16, g_s, Ly.sink16, W.xq, C, W.sink);
            gx_hash_dev("qprep", q_p, HD, 1, l);
            gx_hash_dev("kprep", k_p, KD, 1, l);
            gx_hash_dev("vprep", v_p, KD, 1, l);
            gx_hash_dev("sink", W.sink, c.H, 1, l);
            if (g_onepass && Ly.sliding)
                k::attn_decode_int_onepass(bits, q_p, r_p, W.sink, Ly.kci, Ly.vci, Ly.ks, Ly.vs, Ly.kv_cap, c.H,
                                           c.KV, W.pos, qk_scale, Ly.rel_proj, c.d_rel, extent, window, W.o, g_s,
                                           g_gate_fuse ? gate_p : nullptr);
            else if (g_attn_gqa && c.H % c.KV == 0 && c.H / c.KV == 3)
                k::attn_decode_int_at_gqa(bits, q_p, r_p, W.sink, Ly.kci, Ly.vci, Ly.ks, Ly.vs,
                                          Ly.sliding ? Ly.kv_cap : 0, c.H, c.KV, W.pos, P, qk_scale, Ly.rel_proj,
                                          c.d_rel, extent, window, W.lg, W.o, g_s, g_gate_fuse ? gate_p : nullptr,
                                          g_attn_dpp ? 1 : 0);
            else if (g_attn_wideload && bits == 4)
                k::attn_decode_int_wideload(bits, q_p, r_p, W.sink, Ly.kci, Ly.vci, Ly.ks, Ly.vs,
                                            Ly.sliding ? Ly.kv_cap : 0, c.H, c.KV, W.pos, P, qk_scale, Ly.rel_proj,
                                            c.d_rel, extent, window, W.lg, W.o, g_s, g_gate_fuse ? gate_p : nullptr);
            else if (g_attn_dpp && bits == 4)
                k::attn_decode_int_at_dpp(bits, q_p, r_p, W.sink, Ly.kci, Ly.vci, Ly.ks, Ly.vs,
                                          Ly.sliding ? Ly.kv_cap : 0, c.H, c.KV, W.pos, P, qk_scale, Ly.rel_proj,
                                          c.d_rel, extent, window, W.lg, W.o, g_s, g_gate_fuse ? gate_p : nullptr);
            else
                k::attn_decode_int_at(bits, q_p, r_p, W.sink, Ly.kci, Ly.vci, Ly.ks, Ly.vs,
                                      Ly.sliding ? Ly.kv_cap : 0, c.H, c.KV, W.pos, P, qk_scale, Ly.rel_proj,
                                      c.d_rel, extent, window, W.lg, W.o, g_s, g_gate_fuse ? gate_p : nullptr);
        } else if (g_kv_fp8) {
            k::prep_qkv_r_fp8_at(q_p, Ly.q_clip, W.kconv, Ly.kc8, W.vconv, Ly.vc8, r_p, Ly.r_clip, c.H, c.KV, c.D, c.d_rel,
                                 c.qk_clip, c.rel_clip, c.eps, W.pos, !Ly.sliding, c.alpha, c.n_floor, Ly.kv_cap,
                                 k_scale, v_scale, q16, g_s, Ly.sink16, W.xq, C, W.sink);
            if (mma)
                k::attn_decode_mma_fp8(q16, r_p, W.sink, Ly.kc8, Ly.vc8, ring, c.H, c.KV, W.pos, 0, ns,
                                       qk_scale / k_scale, 1.0f / v_scale, Ly.rel_proj, c.d_rel, extent, window, W.lg,
                                       W.o, gate_p, W.x16, g_s);
            else if (g_onepass && Ly.sliding)
                k::attn_decode_fp8_onepass(q_p, r_p, W.sink, Ly.kc8, Ly.vc8, Ly.kv_cap, c.H, c.KV, W.pos,
                                           qk_scale / k_scale, 1.0f / v_scale, Ly.rel_proj, c.d_rel, extent, window,
                                           W.o, g_s, g_gate_fuse ? gate_p : nullptr);
            else
                k::attn_decode_fp8_at(q_p, r_p, W.sink, Ly.kc8, Ly.vc8, Ly.sliding ? Ly.kv_cap : 0, c.H, c.KV, W.pos, P,
                                      qk_scale / k_scale, 1.0f / v_scale, Ly.rel_proj, c.d_rel, extent, window, W.lg,
                                      W.o, g_s, g_gate_fuse ? gate_p : nullptr);
        } else {
            k::prep_qkv_r_f16_at(q_p, Ly.q_clip, W.kconv, Ly.kc, W.vconv, Ly.vc, r_p, Ly.r_clip, c.H, c.KV, c.D, c.d_rel,
                                 c.qk_clip, c.rel_clip, c.eps, W.pos, !Ly.sliding, c.alpha, c.n_floor, Ly.kv_cap, q16,
                                 g_s, Ly.sink16, W.xq, C, W.sink);
            if (mma)
                k::attn_decode_mma_f16(q16, r_p, W.sink, Ly.kc, Ly.vc, ring, c.H, c.KV, W.pos, 0, ns, qk_scale,
                                       Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, gate_p, W.x16, g_s);
            else if (g_onepass && Ly.sliding)
                k::attn_decode_f16_onepass(q_p, r_p, W.sink, Ly.kc, Ly.vc, Ly.kv_cap, c.H, c.KV, W.pos, qk_scale,
                                           Ly.rel_proj, c.d_rel, extent, window, W.o, g_s, g_gate_fuse ? gate_p : nullptr);
            else
                k::attn_decode_f16_at(q_p, r_p, W.sink, Ly.kc, Ly.vc, Ly.sliding ? Ly.kv_cap : 0, c.H, c.KV, W.pos, P,
                                      qk_scale, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s,
                                      g_gate_fuse ? gate_p : nullptr);
        }
        if (!mma && !g_gate_fuse) k::gate_mul_f16(W.o, gate_p, (size_t) HD, W.x16, true, g_s);   // (the tensor cores' merge has it)
        // the output projection with its short conv (and the stream added, but where a block starts: there the
        // stream joins the blocks and starts again)
        pf_fast::mark(6, g_s);
        const bool first = l % c.block == 0;
        if (first) CU(cudaMemcpyAsync(blk(nb), W.h, Cb, cudaMemcpyDeviceToHost, g_s));
        if (getenv("IK2_GXE_HASH") && !g_batch_graph_capturing && !g_fast_graph_capturing) {
            std::vector<float> hv(C);
            CU(cudaMemcpy(hv.data(), W.o, (size_t) C * 4, cudaMemcpyDeviceToHost));
            unsigned long long x = 1469598103934665603ull;
            for (int i = 0; i < C; ++i) {
                unsigned int u; std::memcpy(&u, &hv[(size_t) i], 4); x = (x ^ u) * 1099511628211ull;
            }
            std::fprintf(stderr, "GXH fastattnout vis=%d h=%016llx\\n", vis, x);
        }
        mv1c(Ly.wo, W.o, W.tmp, k::MvConv{0, C, Ly.h_attn, Ly.sc_attn, nullptr, 0, first ? nullptr : W.h, W.r});
        if (getenv("IK2_GXE_HASH") && !g_batch_graph_capturing && !g_fast_graph_capturing) {
            std::vector<float> hv(C);
            CU(cudaMemcpy(hv.data(), W.tmp, (size_t) C * 4, cudaMemcpyDeviceToHost));
            unsigned long long x = 1469598103934665603ull;
            for (int i = 0; i < C; ++i) {
                unsigned int u; std::memcpy(&u, &hv[(size_t) i], 4); x = (x ^ u) * 1099511628211ull;
            }
            std::fprintf(stderr, "GXH wo vis=%d h=%016llx\\n", vis, x);
        }
        if (first) ++nb;
        if (getenv("IK2_GXE_HASH") && !g_batch_graph_capturing && !g_fast_graph_capturing) {
            std::vector<float> hv(C);
            CU(cudaMemcpy(hv.data(), W.r, (size_t) C * 4, cudaMemcpyDeviceToHost));
            unsigned long long x = 1469598103934665603ull;
            for (int i = 0; i < C; ++i) {
                unsigned int u; std::memcpy(&u, &hv[(size_t) i], 4); x = (x ^ u) * 1099511628211ull;
            }
            std::fprintf(stderr, "GXH fastattn vis=%d h=%016llx\\n", vis, x);
        }
        // the MoE: its input, the router and the picks' tables on the device, the experts
        pf_fast::mark(7, g_s);
        {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
            st.st[nb] = W.r;
            if (msg) {
                k::attn_res(st, nb + 1, Ly.ffn_res_score, 1, 1, C, 1e-6f, W.xq, g_s);
                k::loop_gate(W.xq, msg + (size_t) 3 * C, Ly.lp_down[3], Ly.lp_forget[3], Ly.lp_gain[3], 1, C, c.rank,
                             1e-6f, W.lp_g, W.xq, g_s);
                k::rms(W.xq, Ly.ffn_norm, 1, C, c.eps, W.xq, g_s);
            } else {
                k::RmsOuts outs{};
                outs.w[0] = Ly.ffn_norm, outs.y16[0] = W.x16, outs.y32[0] = W.xq;
                k::attn_res_rms_f16(st, nb + 1, Ly.ffn_res_score, 1, 1, C, 1e-6f, c.eps, outs, g_s);
            }
        }
        pf_fast::mark(8, g_s);
        k::mv_f16(Ly.router16, c.E, C, W.xq, W.rlog, g_s);
        int* rtop = W.route_d;
        float* rw = (float*) (W.route_d + c.K);
        // (NVFP4 experts: the scalar GEMV is faster than the WMMA one on gfx1100 (single token wastes the
        // tensor core's 16 columns); default scalar there, tensor cores on CUDA. IK2_EXPERT_MMA overrides.)
#if defined(__HIPCC__)
        static const int exp_mma = getenv("IK2_EXPERT_MMA") ? atoi(getenv("IK2_EXPERT_MMA")) : 0;
#else
        static const int exp_mma = getenv("IK2_EXPERT_MMA") ? atoi(getenv("IK2_EXPERT_MMA")) : 1;
#endif
        const bool exp_tc = exp_mma && k::experts_mma_fits(Ly.ty[0], Ly.ty[1], Ly.ty[2], C, c.F);
        if (!g_ep_fast) {
            pf_fast::mark(9, g_s);
            const bool i4 = g_int4_experts && Ly.i4_loaded && l < g_int4_layers;
            k::route_token(W.rlog, Ly.probs_b, c.E, c.K, c.route_scale, Ly.tab_blob,
                           i4 ? nullptr : Ly.tab_sc[0], i4 ? nullptr : Ly.tab_sc[1],
                           i4 ? nullptr : Ly.tab_sc[2], rtop, rw, W.blob, W.tok, W.inv, W.sg, W.su,
                           W.wt, g_s, W.xq, C, exp_tc ? (uint16_t*) W.gu32 : nullptr,
                           i4 ? Ly.i4_tab_Wp[0] : nullptr, i4 ? Ly.i4_tab_Wh[0] : nullptr,
                           i4 ? Ly.i4_tab_WS[0] : nullptr, W.i4_Wp_s, W.i4_Wh_s, W.i4_WS_s);
            if (i4) {
                // Rotated-int4 grouped decode: scale folded in WS, so sg=su=1 and wt = route weight only.
                // Both gate_up and down slot pointers were gathered in route_token's epilogue (no extra launches).
                const int S = c.K, F = c.F, Kp = W.i4_Kpad_f;
                const float inv4 = 1.0f / (float) (1 << 12);   // sh=12 (kInt4Sh8)
                // Phase D (IK2_I4_ROT_IN): rot_q8x folded into the matvec prologue, one 256-block per
                // wave, full matvec parallelism kept. 1: gate_up only; 2: also down prep+rot.
                const bool rot_up = g_i4_rot_in >= 1 && C % 256 == 0 && C / 256 <= 10;
                const bool rot_dn = g_i4_rot_in >= 2 && !g_i4_mid_fuse && Kp % 256 == 0 && Kp / 256 <= 10;
                // gate_up: shared input activation (one token, W.xq), 8 expert weight sets.
                if (rot_up) {
                    k::int4_matvec_grouped_rot(W.i4_Wp_s, W.i4_Wh_s, W.i4_WS_s, 2 * F, C, W.xq, nullptr, 0,
                                               S, W.gu32, 2 * F, inv4, g_s);
                } else {
                    k::int4_rot_q8x(W.xq, 1, C, W.i4_xq, W.i4_xd, g_s);
                    k::int4_matvec_grouped(W.i4_Wp_s, W.i4_Wh_s, W.i4_WS_s, 2 * F, C, W.i4_xq, W.i4_xd,
                                           S, W.gu32, 2 * F, inv4, g_s);   // xq_stride=0: shared act
                }
                if (g_i4_mid_fuse) {
                    // Fuse swiglu + prep_down + rot_q8x into one launch. Bit-exact: gu values and the
                    // f16 round-trip are reproduced verbatim; zero-pad beyond F matches prep_down.
                    k::int4_mid(W.gu32, S, F, Kp, c.gate_cap, c.up_cap, W.i4_hq, W.i4_hd, g_s);
                } else {
                    k::swiglu_cap_f32(W.gu32, S, F, W.sg, W.su, c.gate_cap, c.up_cap, W.h16, g_s);   // sg=su=1
                    if (!rot_dn) {
                        k::int4_prep_down(W.h16, W.i4_hf, S, F, Kp, g_s);
                        k::int4_rot_q8x(W.i4_hf, S, Kp, W.i4_hq, W.i4_hd, g_s);
                    }
                }
                // down: per-slot activations (h16[S][F] -> padded float [S][Kp]).
                k::int4_gather_slots(rtop, Ly.i4_tab_Wp[2], Ly.i4_tab_Wh[2], Ly.i4_tab_WS[2], S,
                                     W.i4_Wp_dn, W.i4_Wh_dn, W.i4_WS_dn, g_s);
                if (rot_dn) {
                    k::int4_matvec_grouped_rot(W.i4_Wp_dn, W.i4_Wh_dn, W.i4_WS_dn, C, Kp, nullptr, W.h16, F,
                                               S, W.y32, C, inv4, g_s);
                } else {
                    k::int4_matvec_grouped(W.i4_Wp_dn, W.i4_Wh_dn, W.i4_WS_dn, C, Kp, W.i4_hq, W.i4_hd,
                                           S, W.y32, C, inv4, g_s, Kp, Kp / 32);
                }
            } else if (exp_tc) {
                k::experts_gate_up_swiglu_mma(W.blob, c.K, Ly.eb[0], (const uint16_t*) W.gu32, C, c.F, W.sg, W.su,
                                              c.gate_cap, c.up_cap, W.h16, g_s);
                k::experts_down_mma(W.blob, c.K, Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32, g_s);
            } else {
                k::experts_gate_up_swiglu(W.blob, W.tok, c.K, Ly.ty[0], Ly.ty[1], Ly.eb[0], W.xq, C, c.F, W.sg, W.su,
                                          c.gate_cap, c.up_cap, W.h16, g_s);
                k::experts_down(W.blob, c.K, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32, g_s);
            }
            pf_fast::mark(10, g_s);
            k::moe_combine_conv(W.y32, W.inv, W.wt, c.K, C, Ly.h_mlp, Ly.sc_mlp, W.r, W.h, g_s);
        } else {
            const int S = c.K;
            const int F = c.F;
            k::route(W.rlog, Ly.probs_b, 1, c.E, c.K, c.route_scale, rtop, rw, g_s);

            // 1. Send xq and route_d from GPU 0 to GPU 1
            if (g_ep_stage.d_xq[0]) {
                CU(cudaMemcpyAsync(g_ep_stage.d_xq[0], dev_work(0).xq, (size_t) C * 4, cudaMemcpyDeviceToDevice, g_s));
                CU(cudaMemcpyAsync(g_ep_stage.d_route[0], dev_work(0).route_d, (size_t) S * 8, cudaMemcpyDeviceToDevice, g_s));
            } else {
                CU(cudaMemcpyPeerAsync(dev_work(1).xq, 1, dev_work(0).xq, 0, (size_t) C * 4, g_s));
                CU(cudaMemcpyPeerAsync(dev_work(1).route_d, 1, dev_work(0).route_d, 0, (size_t) S * 8, g_s));
            }
            CU(cudaEventRecord(g_ep_x, g_s));

            // Helper to run experts on a device
            auto run_dev_exp = [&](int /*d*/, Work& Wd, cudaStream_t strm) {
                if (exp_tc) {
                    k::to_f16_limbs(Wd.xq, (size_t) C, (uint16_t*) Wd.gu32, strm);
                    k::experts_gate_up_swiglu_mma(Wd.blob, S, Ly.eb[0], (const uint16_t*) Wd.gu32, C, F, Wd.sg, Wd.su,
                                                  c.gate_cap, c.up_cap, Wd.h16, strm);
                    k::experts_down_mma(Wd.blob, S, Ly.eb[0] + Ly.eb[1], Wd.h16, C, F, Wd.y32, strm);
                } else {
                    k::experts_gate_up_swiglu(Wd.blob, Wd.tok, S, Ly.ty[0], Ly.ty[1], Ly.eb[0], Wd.xq, C, F, Wd.sg, Wd.su,
                                              c.gate_cap, c.up_cap, Wd.h16, strm);
                    k::experts_down(Wd.blob, S, Ly.ty[2], Ly.eb[0] + Ly.eb[1], Wd.h16, C, F, Wd.y32, strm);
                }
            };

            // 2. GPU 1: wait for xq & route_d, select resident experts, compute experts, copy result to GPU 0
            use_dev(1);
            CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
            if (g_ep_stage.d_xq[1]) {
                CU(cudaMemcpyAsync(dev_work(1).xq, g_ep_stage.d_xq[1], (size_t) C * 4, cudaMemcpyDeviceToDevice, g_s));
                k::ep_select_token(g_ep_stage.d_route[1], (const float*)(g_ep_stage.d_route[1] + S), Ly.ep_tab_blob[1],
                                   Ly.ep_tab_sc[1][0], Ly.ep_tab_sc[1][1], Ly.ep_tab_sc[1][2],
                                   S, dev_work(1).blob, dev_work(1).tok, dev_work(1).inv, dev_work(1).sg, dev_work(1).su, dev_work(1).wt, g_s);
            } else {
                k::ep_select_token(dev_work(1).route_d, (const float*)(dev_work(1).route_d + S), Ly.ep_tab_blob[1],
                                   Ly.ep_tab_sc[1][0], Ly.ep_tab_sc[1][1], Ly.ep_tab_sc[1][2],
                                   S, dev_work(1).blob, dev_work(1).tok, dev_work(1).inv, dev_work(1).sg, dev_work(1).su, dev_work(1).wt, g_s);
            }
            run_dev_exp(1, dev_work(1), g_s);
            if (g_ep_stage.d_y[1]) {
                CU(cudaMemcpyAsync(g_ep_stage.d_y[1], dev_work(1).y32, (size_t) S * C * 4, cudaMemcpyDeviceToDevice, g_s));
            } else {
                CU(cudaMemcpyPeerAsync(dev_work(0).y32 + (size_t) S * C, 0, dev_work(1).y32, 1, (size_t) S * C * 4, g_s));
            }
            CU(cudaEventRecord(g_ep_done, g_s));

            // 3. GPU 0: select resident experts, compute experts, wait for GPU 1, combine & conv
            use_dev(0);
            k::ep_select_token(dev_work(0).route_d, (const float*)(dev_work(0).route_d + S), Ly.ep_tab_blob[0],
                               Ly.ep_tab_sc[0][0], Ly.ep_tab_sc[0][1], Ly.ep_tab_sc[0][2],
                               S, dev_work(0).blob, dev_work(0).tok, dev_work(0).inv, dev_work(0).sg, dev_work(0).su, dev_work(0).wt, g_s);
            run_dev_exp(0, dev_work(0), g_s);
            CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
            const float* y_peer = g_ep_stage.d_y[0] ? g_ep_stage.d_y[0] : (dev_work(0).y32 + (size_t) S * C);
            k::ep_combine_conv(dev_work(0).y32, y_peer, dev_work(0).blob, dev_work(0).wt,
                               S, C, Ly.h_mlp, Ly.sc_mlp, dev_work(0).r, dev_work(0).h, g_s);
        }
        if (getenv("IK2_GXE_HASH") && !g_batch_graph_capturing && !g_fast_graph_capturing) {
            std::vector<float> hv(C);
            CU(cudaMemcpy(hv.data(), W.h, (size_t) C * 4, cudaMemcpyDeviceToHost));
            unsigned long long x = 1469598103934665603ull;
            for (int i = 0; i < C; ++i) {
                unsigned int u; std::memcpy(&u, &hv[(size_t) i], 4); x = (x ^ u) * 1099511628211ull;
            }
            std::fprintf(stderr, "GXH fast vis=%d h=%016llx\\n", vis, x);
        }
    }
    pf_fast::mark(11, g_s);
    if (out_mode == OutMode::kNone) { pf_fast::mark(12, g_s); return; }
    {   // the head
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
        st.st[nb] = W.h;
        k::RmsOuts outs{};
        outs.w[0] = M.out_norm, outs.y16[0] = W.x16, outs.y32[0] = W.xq;
        k::attn_res_rms_f16(st, nb + 1, M.out_res_score, 1, 1, C, 1e-6f, c.eps, outs, g_s);
        mv1(M.out_w, W.xq, W.logits);
        if (out_mode == OutMode::kArgmax) {
            k::argmax_sum(W.logits, nullptr, nullptr, c.n_vocab, argmax_out_d, g_s, W.argmax_part_v, W.argmax_part_i);
            if (!kstep && copy_argmax_h)
                CU(cudaMemcpyAsync(W.argmax_h, argmax_out_d, 4, cudaMemcpyDeviceToHost, g_s));
        } else if (out_mode == OutMode::kCand) {   // (the softcap keeps the logits within +- logit_cap)
            k::softcap(W.logits, (size_t) c.n_rows, c.logit_cap, g_s);
            k::cand_topk(W.logits, c.n_vocab, -c.logit_cap, c.logit_cap, kCandK, kCandCap, W.cand_part, W.cand_d, g_s);
            CU(cudaMemcpyAsync(W.cand_h, W.cand_d, (size_t) (2 + 2 * kCandCap) * 4, cudaMemcpyDeviceToHost, g_s));
        } else {
            k::softcap(W.logits, (size_t) c.n_rows, c.logit_cap, g_s);
            CU(cudaMemcpyAsync(W.logits_h, W.logits, (size_t) c.n_vocab * 4, cudaMemcpyDeviceToHost, g_s));
        }
    }
    pf_fast::mark(12, g_s);
}

// Default entry point.  Every existing caller uses this, with no new arguments: byte-identical to HEAD.
void fast_token_launches(Model& M, Work& W, OutMode out_mode) {
    one_token_launches(M, W, out_mode, false, 0, nullptr, W.argmax_d, true);
}

// IK2_AR_KSTEP: K tokens in ONE graph.  Step 0 takes the host-staged token/position; later steps read the previous
// token's id on the device and the device-advanced position.  Exactly ONE D2H at the end publishes all K ids.
void fast_token_launches_k(Model& M, Work& W, int K) {
    for (int s = 0; s < K; ++s) {
        const int* prev = s > 0 ? W.argmax_dK + (s - 1) : nullptr;
        one_token_launches(M, W, OutMode::kArgmax, true, s, prev, W.argmax_dK + s, false);
        if (s + 1 < K) k::pos_inc(W.pos, g_s);
    }
    CU(cudaMemcpyAsync(W.argmax_hK, W.argmax_dK, (size_t) K * 4, cudaMemcpyDeviceToHost, g_s));
}

// one token through the resident path: its logits appended to logits_out (kLogits) or its argmax (kArgmax)
void fast_token(Model& M, int tok, int pos, OutMode out_mode, std::vector<float>& logits_out,
                std::vector<int>* argmax_out, int slot = -1, bool submit_only = false) {
    const Config& c = M.c;
    use_dev(0);
    const bool ar_slot = slot >= 0 && slot < 2 && g_ar_dual && g_batch_decode;
    // IK2_AR_STREAM2: slot 1 replays on g_s1 with its OWN Work.  The graph bakes W's device addresses into
    // every kernel, so reusing slot 0's scratch or pinned staging would race with the concurrent g_s replay.
    const bool ar2 = ar_slot && g_ar_stream2 && g_s1 && slot == 1;
    Work& W = ar2 ? g_ar_work1 : dev_work(0);
    cudaStream_t stream = ar2 ? g_s1 : g_s;
    if (ar_slot) {
        // Host-only pointer binding: point M.V at this slot's persistent KV/conv storage.  fast_token_launches()
        // binds those pointers into M.L for every visit, so the graph captured below bakes THIS slot's addresses.
        // M.V is left pointing at the slot, which is also what prefill needs before its own launches.
        load_batch_slot(M, slot);
    }
    const size_t rb = (size_t) c.C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
    k::decode_host(M.embd_type, M.embd + (size_t) tok * rb, c.C, W.emb_h);
    W.pos_h[0] = pos;
    // The resident graph captures the DEVICE addresses of W.emb_h / W.pos_h / W.argmax_h.  Those addresses are
    // stable; each slot stages the host-side contents into its own Work immediately before its replay and reads
    // its own result after its stream sync.
    FastGraph* PG = nullptr;
    if (ar_slot) PG = &g_ar_fgraph[slot][(int) out_mode];
    else if (g_graph && !g_ar_dual) PG = &g_fgraph[(int) out_mode];
    const bool want_stats = ar_slot && g_ar2_stats;
    if (!PG) {
        cudaStream_t old_s = g_s;
        if (ar2) g_s = stream;   // fast_token_launches() queues on the global stream; capture must see g_s1
        fast_token_launches(M, W, out_mode);
        if (ar2) g_s = old_s;
    } else {
        FastGraph& G = *PG;
        if (!G.exec && !G.disabled) {
            if (ar_slot) CU(cudaStreamSynchronize(stream));   // no pending prefill/memset work inside the capture window
            cudaStream_t old_s = g_s;
            if (ar2) g_s = stream;
            cudaError_t ce = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            if (ce == cudaSuccess) {
                g_fast_graph_capturing = true; fast_token_launches(M, W, out_mode);
                ce = cudaStreamEndCapture(stream, &G.graph); g_fast_graph_capturing = false;
                if (ce == cudaSuccess) {
                    ce = cudaGraphInstantiate(&G.exec, G.graph, 0);
                    if (ce == cudaSuccess) {
                        std::printf("resident%s: CUDA graph captured and instantiated!\n",
                                    ar_slot ? " AR slot KV" : (g_ep_fast ? " EP" : ""));
                        if (ar_slot) std::printf("ik2: AR dual graph slot %d (out_mode %d) captured%s\n",
                                                 slot, (int) out_mode, ar2 ? " on g_s1" : "");
                        std::fflush(stdout);
                    } else {
                        std::printf("resident: graph instantiate failed (%s), fallback to direct launches\n",
                                    cudaGetErrorString(ce));
                        std::fflush(stdout);
                        cudaGraphDestroy(G.graph);
                        G.graph = nullptr;
                        G.disabled = true;
                        (void) cudaGetLastError();
                    }
                } else {
                    std::printf("resident: graph end capture failed (%s), fallback to direct launches\n",
                                cudaGetErrorString(ce));
                    std::fflush(stdout);
                    G.disabled = true;
                    (void) cudaGetLastError();
                }
            } else {
                G.disabled = true;
                (void) cudaGetLastError();
            }
            if (ar2) g_s = old_s;
        }
        if (want_stats) CU(cudaEventRecord(g_ar2_ev[(size_t) slot * 2], stream));
        if (G.exec) {
            CU(cudaGraphLaunch(G.exec, stream));
        } else {
            cudaStream_t old_s = g_s;
            if (ar2) g_s = stream;
            fast_token_launches(M, W, out_mode);
            if (ar2) g_s = old_s;
        }
        if (want_stats) CU(cudaEventRecord(g_ar2_ev[(size_t) slot * 2 + 1], stream));
    }
    if (submit_only) return;   // batch_step will sync BOTH streams, then read each slot's own argmax
    CU(cudaStreamSynchronize(stream));
    pf_fast::finish(stream);
    if (out_mode == OutMode::kArgmax) {
        argmax_out->assign(1, W.argmax_h[0]);
    } else if (out_mode == OutMode::kLogits) {
        logits_out.insert(logits_out.end(), W.logits_h, W.logits_h + c.n_vocab);
    } else if (out_mode == OutMode::kCand) {
        if (W.cand_h[0] <= kCandCap) {
            g_cand.n = W.cand_h[0], g_cand.idx = W.cand_h + 2, g_cand.val = (const float*) (W.cand_h + 2 + kCandCap);
        } else {   // (more than the buffer holds - logits flat past belief: the logits themselves, this once)
            CU(cudaMemcpy(W.logits_h, W.logits, (size_t) c.n_vocab * 4, cudaMemcpyDeviceToHost));
            logits_out.insert(logits_out.end(), W.logits_h, W.logits_h + c.n_vocab);
            g_cand.n = -1;
        }
    }
}

// IK2_AR_KSTEP: submit the PRE-BUILT K-token graph for one AR slot.  No lazy capture here: K graphs are built before
// the serve loop, because K bookkeeping consumes K ids from argmax_hK, not one id from argmax_h.
void fast_token_k(Model& M, int slot, int tok0, int pos0, bool submit_only,
                  std::vector<int>* argmax_out = nullptr) {
    const Config& c = M.c;
    use_dev(0);
    const bool ar_slot = slot >= 0 && slot < 2 && g_ar_dual && g_batch_decode;
    const bool ar2 = ar_slot && g_ar_stream2 && g_s1 && slot == 1;
    Work& W = ar2 ? g_ar_work1 : dev_work(0);
    cudaStream_t stream = ar2 ? g_s1 : g_s;
    FastGraph& G = g_ar_kgraph[slot];
    if (!ar_slot || !G.exec || G.disabled) {
        // Never replay a K graph under one-step bookkeeping.  Disable globally; batch_step falls through.
        std::fprintf(stderr, "ik2: AR K-step graph slot %d not ready; disabling KSTEP\n", slot);
        g_ar_kstep = false; g_ar_kstep_ready = false;
        return;
    }
    if (ar_slot) load_batch_slot(M, slot);
    const size_t rb = (size_t) c.C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
    k::decode_host(M.embd_type, M.embd + (size_t) tok0 * rb, c.C, W.emb_h);
    W.pos_h[0] = pos0;
    const bool want_stats = ar_slot && g_ar2_stats;
    cudaStream_t old_s = g_s;
    if (ar2) g_s = stream;   // every launch and the final D2H capture/use the same global stream variable
    if (want_stats) CU(cudaEventRecord(g_ar2_ev[(size_t) slot * 2], stream));
    CU(cudaGraphLaunch(G.exec, stream));
    if (want_stats) CU(cudaEventRecord(g_ar2_ev[(size_t) slot * 2 + 1], stream));
    if (ar2) g_s = old_s;
    if (submit_only) return;
    CU(cudaStreamSynchronize(stream));
    pf_fast::finish(stream);
    if (argmax_out) argmax_out->assign(W.argmax_hK, W.argmax_hK + W.kstep_K);
}

void forward(Model& M, const int* tok, int T, int pos0, bool all_logits, OutMode out_mode, const std::string& dump,
             std::vector<float>& logits_out, std::vector<int>* argmax_out, double& t_exp) {
    const Config& c = M.c;
    const int C = c.C;
    const bool vprof2 = getenv("IK2_VPROF2") != nullptr || getenv("IK2_VPROF3") != nullptr || getenv("IK2_VPROF4") != nullptr;
    if (vprof2 && !vp_init) { for (int q = 0; q < 64; ++q) { cudaEventCreate(&vpA[q]); cudaEventCreate(&vpB[q]); cudaEventCreate(&vpP[q]); cudaEventCreate(&vpK[q]); cudaEventCreate(&vpF[q]); cudaEventCreate(&vpR[q]); cudaEventCreate(&vpS[q]); } vp_init = true; }
    if (vprof2) vp_n = 0;
    if (g_prof && T > 1) std::printf("FWDDBG T=%d K=%d S=%d rg_slots=%d g_fast=%d g_mv=%d tab_empty=%d n_visit=%d\n", T, c.K, T * c.K, dev_work(g_cur).rg_slots, (int) g_fast, (int) g_mv, (int) M.L[0].tab_h.empty(), c.n_visit);
    if ((g_fast || g_ep_fast) && T == 1 && dump.empty() && !g_prof && !g_rec && !g_taps && !g_trace && g_pop.empty() &&
        g_img.at(pos0) < 0) {
        fast_token(M, tok[0], pos0, out_mode, logits_out, argmax_out);
        return;
    }
    if (out_mode == OutMode::kCand) out_mode = OutMode::kLogits, g_cand.n = -1;   // (this path: the logits)
    // Verify graph needs every layer visit to stay device-only: VERIFY_FAST's shape guards are model constants,
    // so require them up front or a visit falls through to the general MoE path (route D2H + stream sync),
    // which is not capturable and would abort the process via CU().
    const bool vg_shape_ok = C % 256 == 0 && (2 * (size_t) c.F) % 16 == 0 &&
                             dev_work(0).i4_Kpad_f % 256 == 0 && T <= dev_work(0).Tm &&
                             (size_t) T * c.K <= 64 && (size_t) T * c.K <= dev_work(0).i4_pf_cap;
    bool vg_layers_ok = true;
    for (int l = 0; l < c.n_layer && vg_layers_ok; ++l)
        if (!M.L[l].tab_blob || (l < g_int4_layers && !M.L[l].i4_loaded)) vg_layers_ok = false;
    const bool vg_try = g_verify_graph && g_verify_fast && g_rec && T == ik2::draft::kBlock && all_logits &&
                        out_mode == OutMode::kArgmax && argmax_out && !g_vgraph.disabled && !g_prof && !g_prof_fast &&
                        dump.empty() && !g_trace && g_pop.empty() && !dual() && !g_ep_fast && !g_ep && g_fast &&
                        g_group && g_group_attn && g_int4_experts && !g_int4_pf_off && !g_img.n &&
                        !g_exp_prefill_wmma && g_dev.size() == 1 && vg_shape_ok && vg_layers_ok && !g_batch_active;
    if (vg_try && g_vgraph.built && g_vgraph.exec) {
        // One replay.  The token rows and positions are staged into the same pinned buffers the graph was captured
        // with; the prep / attention kernels read the base position from W.pos on the device.  The head's eight
        // argmaxes are the only D2H, after the graph, before the host accept comparison / rollback.
        use_dev(0);
        Work& Wv = dev_work(0);
        const size_t rb = (size_t) C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
        for (int t = 0; t < T; ++t)
            k::decode_host(M.embd_type, M.embd + (size_t) tok[t] * rb, C, Wv.emb_h + (size_t) t * C);
        for (int t = 0; t < T; ++t) Wv.pos_h[t] = pos0 + t;
        CU(cudaGraphLaunch(g_vgraph.exec, g_s));
        CU(cudaMemcpyAsync(Wv.argmax_h, Wv.argmax_d, (size_t) T * 4, cudaMemcpyDeviceToHost, g_s));
        CU(cudaStreamSynchronize(g_s));
        argmax_out->resize(T);
        for (int t = 0; t < T; ++t) (*argmax_out)[t] = Wv.argmax_h[t];
        return;
    }

    // IK2_BATCH_GRAPH: the true batched T>1 path.  The graph is captured after ONE direct, launch-only warm-up pass
    // (hipBLAS shapes/workspace).  Its launch list reads token rows from W.emb_h, positions from W.pos_h, and per-row
    // KV/conv pointers from Work's stable device tables.  Before each replay we refresh only those pinned buffers;
    // no per-visit raw pointer in the captured kernels ever changes.  Host sampling stays outside forward().
    const bool bg_base = g_batch_graph && g_batch_active && g_batch_pos && g_batch_slot_ids && dev_work(0).batch &&
                         T > 1 && T <= BatchSlots::kMaxRows && out_mode == OutMode::kArgmax && argmax_out &&
                         !g_bgraph.disabled && !getenv("IK2_GXE_HASH") && !g_prof && !g_prof_fast && !vprof2 && dump.empty() &&
                         !g_trace && g_pop.empty() && !dual() && !g_ep && !g_tp && !g_taps && !g_rec &&
                         g_fast && g_group && g_group_attn && g_int4_experts && !g_int4_pf_off &&
                         g_dev.size() == 1 && vg_shape_ok && vg_layers_ok && (2 * (size_t) c.F) % 32 == 0;
    if (bg_base && g_bgraph.rows && g_bgraph.rows != T) {   // a shape change needs a fresh capture
        CU(cudaStreamSynchronize(g_s));
        if (g_bgraph.exec) cudaGraphExecDestroy(g_bgraph.exec);
        if (g_bgraph.graph) cudaGraphDestroy(g_bgraph.graph);
        g_bgraph.graph = nullptr; g_bgraph.exec = nullptr;
        g_bgraph.built = false; g_bgraph.warmed = false; g_bgraph.rows = 0;
    }
    const bool bg_try = bg_base && !g_bgraph.disabled;
    auto bg_stage = [&](Work& Wb) {
        const size_t rb = (size_t) C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
        for (int t = 0; t < T; ++t)
            k::decode_host(M.embd_type, M.embd + (size_t) tok[t] * rb, C, Wb.emb_h + (size_t) t * C);
        for (int t = 0; t < T; ++t) Wb.pos_h[t] = g_batch_pos[t];
        for (int vis = 0; vis < c.n_visit; ++vis)
            fill_batch_table_host(Wb, vis, T, g_batch_slot_ids, 0);
    };
    auto bg_read_argmax = [&](Work& Wb) {
        CU(cudaMemcpyAsync(Wb.argmax_h, Wb.argmax_d, (size_t) T * 4, cudaMemcpyDeviceToHost, g_s));
        CU(cudaStreamSynchronize(g_s));
        argmax_out->resize(T);
        for (int t = 0; t < T; ++t) (*argmax_out)[t] = Wb.argmax_h[t];
    };
    if (bg_try && g_bgraph.built && g_bgraph.exec && g_bgraph.rows == T) {
        use_dev(0);
        Work& Wb = dev_work(0);
        bg_stage(Wb);
        CU(cudaGraphLaunch(g_bgraph.exec, g_s));
        bg_read_argmax(Wb);
        return;
    }
    const bool bg_warm = bg_try && !g_bgraph.warmed;
    bool bg_cap = bg_try && !bg_warm && !g_bgraph.built;
    bool bg_began = false;
    bool bg_direct_read = bg_warm;
    if (bg_warm || bg_cap) {
        use_dev(0);
        g_vg_stage = bg_cap ? VG_CAPTURE : VG_REPLAY;   // launch-only MoE on every layer, including non-INT4 ones
        if (bg_cap) {
            const cudaError_t ce = cudaStreamBeginCapture(g_s, cudaStreamCaptureModeGlobal);
            if (ce == cudaSuccess) {
                bg_began = true;
                g_batch_graph_capturing = true;
            } else {
                g_bgraph.disabled = true;
                bg_cap = false;
                bg_direct_read = true;
                g_vg_stage = VG_REPLAY;
                (void) cudaGetLastError();
                std::printf("batch graph: begin capture failed (%s), fallback to direct launches\n",
                            cudaGetErrorString(ce));
                std::fflush(stdout);
            }
        }
    }
    use_dev(M.L[0].dev);
    Work* Wp = &dev_work(g_cur);
    ExpertCache* ECp = &dev_cache(g_cur);
#define W (*Wp)
#define EC (*ECp)
    // IK2_VERIFY_GRAPH: the first eligible window runs the ordinary device path to warm hipBLAS's GEMM shapes and
    // internal workspace; the second is captured; later ones replay at the top.  Capture begins after use_dev (so
    // g_s is the right stream) and before the pinned staging H2D, so staging is in the graph.
    bool vg_capture = false;
    if (vg_try && !g_vgraph.warmed) {
        g_vgraph.warmed = true;
    } else if (vg_try && !g_vgraph.disabled) {
        cudaError_t ce = cudaStreamBeginCapture(g_s, cudaStreamCaptureModeGlobal);
        if (ce == cudaSuccess) {
            vg_capture = true;
            g_vg_stage = VG_CAPTURE;
        } else {
            g_vgraph.disabled = true;
            (void) cudaGetLastError();
            std::printf("verify graph: begin capture failed (%s), fallback to direct launches\n", cudaGetErrorString(ce));
            std::fflush(stdout);
        }
    }
    // the embedding rows, normed (pinned host staging, no stream sync needed)
    {
        const size_t rb = (size_t) C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
        for (int t = 0; t < T; ++t)
            k::decode_host(M.embd_type, M.embd + (size_t) tok[t] * rb, C, W.emb_h + (size_t) t * C);
        for (int t = 0; t < T; ++t) W.pos_h[t] = g_batch_active ? g_batch_pos[t] : pos0 + t;
    }
    CU(cudaMemcpyAsync(W.tmp, W.emb_h, (size_t) T * C * 4, cudaMemcpyHostToDevice, g_s));
    k::rms(W.tmp, M.embd_norm, T, C, c.eps, W.h, g_s);
    if (!g_batch_active) place_pictures(W.h, T, pos0, C, g_s);
    CU(cudaMemcpyAsync(W.pos, W.pos_h, (size_t) T * 4, cudaMemcpyHostToDevice, g_s));
    int nb = 0;
    const size_t TC = (size_t) T * C;
    auto blk = [&](int i) { return W.blocks + (size_t) i * W.blk_ld * C; };
    auto dump_f = [&](const std::string& name, const float* d, size_t n) {
        if (dump.empty()) return;
        std::vector<float> h(n);
        CU(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
        FILE* f = std::fopen((dump + "/" + name + ".f32").c_str(), "wb");
        if (!f) die("cannot write in " + dump);
        std::fwrite(h.data(), 4, n, f);
        std::fclose(f);
    };
    auto gxe_hash = [&](const char* tag, const float* d, int v) {
        if (!getenv("IK2_GXE_HASH") || g_batch_graph_capturing) return;
        std::vector<float> h(C);
        CU(cudaMemcpy(h.data(), d, (size_t) C * 4, cudaMemcpyDeviceToHost));
        unsigned long long hv = 1469598103934665603ull;
        for (int i = 0; i < C; ++i) {
            unsigned int u;
            std::memcpy(&u, &h[(size_t) i], 4);
            hv = (hv ^ u) * 1099511628211ull;
        }
        std::fprintf(stderr, "GXH %s T=%d vis=%d h=%016llx\\n", tag, T, v, hv);
    };

    int nb_pre = 0, n_bank = 0;   // (a looped core: the blocks before it, the states of the pass before's bank)
    for (int vis = 0; vis < c.n_visit; ++vis) {
        const int l = M.V[vis].l;   // (the layer: its weights, its experts in the cache; the state is the visit's)
        Layer& Ly = bind_visit(M, vis);
        if (g_batch_active) stage_batch_tables(W, vis, g_batch_rows, g_batch_slot_ids);
        if (Ly.dev != g_cur) {   // the next GPU's layers: the stream and the block states go along
            const int from = g_cur;
            Work& F = dev_work(from);
            Work& Tn = dev_work(Ly.dev);
            CU(cudaStreamSynchronize(g_dev[from].s));
            CU(cudaMemcpyPeer(Tn.h, Ly.dev, F.h, from, TC * 4));
            for (int i = 0; i < nb; ++i)
                CU(cudaMemcpyPeer(Tn.blocks + (size_t) i * Tn.blk_ld * C, Ly.dev, F.blocks + (size_t) i * F.blk_ld * C,
                                  from, TC * 4));
            CU(cudaMemcpyPeer(Tn.pos, Ly.dev, F.pos, from, (size_t) T * 4));
            use_dev(Ly.dev);
            Wp = &dev_work(g_cur);
            ECp = &dev_cache(g_cur);
        }
        g_pf.start();
        vp_attn_idx = (vprof2 && vis < 64) ? vis : -1;
        if (vprof2 && vis < 64) cudaEventRecord(vpA[vis], g_s);
        // A looped core.  Its first pass: the stream it is given is kept.  A later pass starts again from that stream
        // and the blocks before the core, and reads the pass before through messages: that pass's bank is the blocks
        // it added after its first (the first is the kept stream) and its final stream.
        const float* msg = nullptr;
        if (c.n_pass > 1 && l == c.core0) {
            if (M.V[vis].pass == 0) {
                nb_pre = nb;
                CU(cudaMemcpyAsync(W.lp_pre, W.h, TC * 4, cudaMemcpyDeviceToDevice, g_s));
            } else {
                if (g_loopbank_fuse && nb > nb_pre) {
                    // the same copies as one launch (row = TC floats, one per token; same thread-order guarantee)
                    k::loop_bank_copies(W.lp_pre, W.h, W.blocks, (size_t) W.blk_ld * C, nb_pre, nb, W.lp_bank, W.h,
                                        TC, g_s);
                    n_bank = nb - nb_pre;
                } else {
                    n_bank = 0;
                    for (int i = nb_pre + 1; i < nb; ++i, ++n_bank)
                        CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * TC, blk(i), TC * 4, cudaMemcpyDeviceToDevice, g_s));
                    CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * TC, W.h, TC * 4, cudaMemcpyDeviceToDevice, g_s));
                    ++n_bank;
                    CU(cudaMemcpyAsync(W.h, W.lp_pre, TC * 4, cudaMemcpyDeviceToDevice, g_s));
                }
                nb = nb_pre;
            }
        }
        if (M.V[vis].pass > 0) {   // the layer's messages: q's, k's, v's, the MoE's
            k::ResStates st{};
            for (int i = 0; i < n_bank; ++i) st.st[i] = W.lp_bank + (size_t) i * TC;
            // (two queries a launch: the kernel takes at most three)
            k::attn_res(st, n_bank, Ly.lp_query, 2, T, C, 1e-6f, W.lp_msg, g_s);
            k::attn_res(st, n_bank, Ly.lp_query + (size_t) 2 * C, 2, T, C, 1e-6f, W.lp_msg + 2 * TC, g_s);
            msg = W.lp_msg;
        }
        // the drafter's taps
        if (g_taps) {
            int tap_idx = -1;
            if (c.n_pass > 1) {
                if (vis == 5) tap_idx = 0;
                else if (vis == 23) tap_idx = 1;
                else if (vis == 41) tap_idx = 2;
            } else {
                if (vis % 6 == 5 && vis < c.n_visit - 1) tap_idx = vis / 6;
            }
            if (tap_idx >= 0 && tap_idx < g_taps_n) {
                CU(cudaMemcpyAsync(g_taps + (size_t) tap_idx * g_taps_ld * C, W.h, TC * 4, cudaMemcpyDeviceToDevice, g_s));
            }
        }
        run_layer_attn(M, Ly, l, T, pos0, W.h, W.pos, nb, 0, W.r, W, msg);
        gxe_hash("attn", W.h, vis);
        dump_f("blk." + std::to_string(vis) + ".attn_out", W.r, TC);
        if (l % c.block == 0) ++nb;
        // the MoE
        {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
            st.st[nb] = W.r;
            if (msg) {   // (its message gates the input before the norm)
                k::attn_res(st, nb + 1, Ly.ffn_res_score, 1, T, C, 1e-6f, W.xq, g_s);
                const k::LoopGate gt{W.xq, msg + (size_t) 3 * TC, Ly.lp_down[3], Ly.lp_forget[3], Ly.lp_gain[3], W.xq};
                loop_gates_chunk(W, &gt, 1, T, C, c.rank, 1e-6f);
                k::rms_f16(W.xq, Ly.ffn_norm, T, C, c.eps, W.x16, W.xq, g_s);
            } else {
                k::RmsOuts outs{};
                outs.w[0] = Ly.ffn_norm;
                outs.y16[0] = W.x16;
                outs.y32[0] = W.xq;
                k::attn_res_rms_f16(st, nb + 1, Ly.ffn_res_score, 1, T, C, 1e-6f, c.eps, outs, g_s);
            }
        }
        g_pf.lap(g_pf.attn);
        vp_attn_idx = -1;
        if (vprof2 && vis < 64) { cudaEventRecord(vpB[vis], g_s); vp_n = vis + 1; }
        const bool graph_exact = batch_graph_exact_on(T) && W.batch;
        if (graph_exact) {
            for (int t = 0; t < T; ++t)
                k::mv_f16(Ly.router16, c.E, C, W.xq + (size_t) t * C, W.rlog + (size_t) t * c.E, g_s);
        } else {
            gemm16(Ly.router16, c.E, C, W.x16, T, W.rlog, c.E);
        }
        const int S = T * c.K;
        int* rtop = W.route_d;
        float* rw = (float*) (W.route_d + S);
        if (!graph_exact) k::route(W.rlog, Ly.probs_b, T, c.E, c.K, c.route_scale, rtop, rw, g_s);
#if defined(STRATA_USE_HIP) && defined(STRATA_WMMA_GFX11)
        // IK2_VERIFY_FAST (default OFF): a verify window's INT4 MoE entirely on the device, with no route D2H
        // and no stream sync. The resident T=1 sequence (above) at S=T*K slots; the slots' f32 input is gathered
        // from W.x16 (the same f16 rows run_prefill_moe gathers, so the numerics of the activation stay those
        // of the old verify path). Layers 20+ and g_rec=false keep the old general path.
        // IK2_BATCH_ATTN: the device-only q8x+matvec MoE below is bit-exact ONLY against the resident T=1
        // verify path (which uses the same family).  The general BATCH MoE path uses rot_pack8+gemm8 with a
        // different quantizer/reduction order, so enabling q8x+matvec for W.batch diverges.  Keep g_rec-only;
        // the batch device-only path must reproduce pack8+gemm8 numerics (IK2_BATCH_MOE_DEVICE work).
        // IK2_BATCH_MOE_DEVICE: W.batch's bit-exact device MoE (pack8+gemm8, expert-major scatter).
        // Only true batched decode (W.batch); the non-batch g_rec verify path keeps fast_moe_ok verbatim.
        // IK2_BATCH_GRAPH_EXACT: reproduce the solo T=1 INT4 expert sequence row by row. The old device batch
        // MoE uses rot_pack8+gemm8, a different quantizer/reduction from the resident q8x+matvec solo path.
        const bool graph_exact_moe = graph_exact && g_int4_experts && Ly.i4_loaded && l < g_int4_layers &&
            !g_int4_pf_off && !dual() && !g_ep_fast && !g_prof && dump.empty() && !g_trace && g_pop.empty() &&
            C % 256 == 0 && (2 * c.F) % 16 == 0 && W.i4_Kpad_f % 256 == 0 && Ly.tab_blob &&
            Ly.i4_tab_Wp[0] && Ly.i4_tab_Wh[0] && Ly.i4_tab_WS[0] && Ly.i4_tab_Wp[2] &&
            Ly.i4_tab_Wh[2] && Ly.i4_tab_WS[2] && T <= W.Tm && c.K <= 8 &&
            T <= W.i4_xq_rows;
        if (getenv("IK2_GRAPH_EXACT_DEBUG"))
            std::fprintf(stderr, "GXE l=%d vis=%d T=%d exact=%d moe=%d xqrows=%d K=%d Tm=%d\\n", l, vis, T,
                         (int) graph_exact, (int) graph_exact_moe, W.i4_xq_rows, c.K, W.Tm);
        if (graph_exact_moe) {
            const int F = c.F, Kp = W.i4_Kpad_f, K = c.K;
            const float inv4 = 1.0f / (float) (1 << k::kInt4Sh8);
#if defined(__HIPCC__)
            static const int exp_mma_x = getenv("IK2_EXPERT_MMA") ? atoi(getenv("IK2_EXPERT_MMA")) : 0;
#else
            static const int exp_mma_x = getenv("IK2_EXPERT_MMA") ? atoi(getenv("IK2_EXPERT_MMA")) : 1;
#endif
            const bool exp_tc_x = exp_mma_x && k::experts_mma_fits(Ly.ty[0], Ly.ty[1], Ly.ty[2], C, F);
            for (int t = 0; t < T; ++t) {
                int* top_t = rtop + t * K;
                float* w_t = rw + t * K;
                const uint8_t** wp_s = W.i4_Wp_s + t * K;
                const uint8_t** wh_s = W.i4_Wh_s + t * K;
                const float** ws_s = W.i4_WS_s + t * K;
                k::route_token(W.rlog + (size_t) t * c.E, Ly.probs_b, c.E, K, c.route_scale, Ly.tab_blob,
                               nullptr, nullptr, nullptr, top_t, w_t, W.blob + t * K, W.tok + t * K,
                               W.inv + t * K, W.sg + t * K, W.su + t * K, W.wt + t * K, g_s,
                               W.xq + (size_t) t * C, C, exp_tc_x ? (uint16_t*) W.gu32 : nullptr,
                               Ly.i4_tab_Wp[0], Ly.i4_tab_Wh[0], Ly.i4_tab_WS[0], wp_s, wh_s, ws_s);
                int8_t* xq_t = W.i4_xq + (size_t) t * C;
                float2* xd_t = W.i4_xd + (size_t) t * (C / 32 + 2);
                k::int4_rot_q8x(W.xq + (size_t) t * C, 1, C, xq_t, xd_t, g_s);
                float* gu_t = W.gu32;
                k::int4_matvec_grouped(wp_s, wh_s, ws_s, 2 * F, C, xq_t, xd_t, K, gu_t, 2 * F, inv4, g_s);
                if (g_i4_mid_fuse) {
                    k::int4_mid(gu_t, K, F, Kp, c.gate_cap, c.up_cap,
                                W.i4_hq + (size_t) t * K * Kp,
                                W.i4_hd + (size_t) t * K * (Kp / 32 + 2), g_s);
                } else {
                    k::swiglu_cap_f32(gu_t, K, F, W.sg + t * K, W.su + t * K, c.gate_cap, c.up_cap,
                                      W.h16 + (size_t) t * K * F, g_s);
                    float* hf_t = W.i4_hf + (size_t) t * K * Kp;
                    int8_t* hq_t = W.i4_hq + (size_t) t * K * Kp;
                    float2* hd_t = W.i4_hd + (size_t) t * K * (Kp / 32 + 2);
                    k::int4_prep_down(W.h16 + (size_t) t * K * F, hf_t, K, F, Kp, g_s);
                    k::int4_rot_q8x(hf_t, K, Kp, hq_t, hd_t, g_s);
                }
                k::int4_gather_slots(top_t, Ly.i4_tab_Wp[2], Ly.i4_tab_Wh[2], Ly.i4_tab_WS[2], K,
                                     W.i4_Wp_dn + t * K, W.i4_Wh_dn + t * K, W.i4_WS_dn + t * K, g_s);
                k::int4_matvec_grouped(W.i4_Wp_dn + t * K, W.i4_Wh_dn + t * K, W.i4_WS_dn + t * K,
                                       C, Kp, W.i4_hq + (size_t) t * K * Kp,
                                       W.i4_hd + (size_t) t * K * (Kp / 32 + 2), K,
                                       W.y32 + (size_t) t * K * C, C, inv4, g_s, Kp, Kp / 32);
            }
            for (int t = 0; t < T; ++t)
                k::moe_combine_f32(W.y32 + (size_t) t * K * C, W.inv + t * K, W.wt + t * K, 1, K, C,
                                   W.tmp + (size_t) t * C, false, g_s);
            conv_rec_w(Ly, W, 3, W.tmp, T, C);
            conv4_mlp_w(Ly, W, W.tmp, W.r, W.h, T, C);
            gxe_hash("moe", W.h, vis);
            continue;
        }
        const bool batch_moe_ok = g_batch_moe_device && W.batch && T > 1 && T <= 8 && S <= 64 &&
            g_int4_experts && Ly.i4_loaded && l < g_int4_layers && !g_int4_pf_off && !dual() &&
            !g_ep_fast && !g_prof && dump.empty() && !g_trace && g_pop.empty() && C % 256 == 0 &&
            (2 * c.F) % 32 == 0 && W.i4_Kpad_f % 256 == 0 && W.i4_pf_cap >= S &&
            W.i4_pf_Kmax >= C && W.i4_pf_x4 && W.i4_pf_xm && W.i4_pf_xr && Ly.tab_blob &&
            Ly.i4_tab_Wp[0] && Ly.i4_tab_Wh[0] && Ly.i4_tab_WS[0] && Ly.i4_tab_Wp[2] &&
            Ly.i4_tab_Wh[2] && Ly.i4_tab_WS[2] && T <= W.Tm;
        const bool fast_moe_ok = g_verify_fast && g_rec && !batch_moe_ok;
        if (fast_moe_ok && T > 1 && T <= 8 && S <= 64 && g_int4_experts && Ly.i4_loaded && l < g_int4_layers &&
            !g_int4_pf_off && !dual() && !g_ep_fast && !g_prof && dump.empty() && !g_trace && g_pop.empty() &&
            C % 256 == 0 && (2 * c.F) % 16 == 0 && W.i4_Kpad_f % 256 == 0 && Ly.tab_blob && T <= W.Tm &&
            S <= W.i4_pf_cap) {
            const double te0 = now();
            const int F = c.F, Kp = W.i4_Kpad_f;
            const float inv4 = 1.0f / (float) (1 << 12);   // k::kInt4Sh8
            // route_gather: inv[i]=i, tok[i]=i/K, sg=su=1, wt = the route weight only (the INT4 scale is
            // folded into WS). tab_blob is a dummy E-entry table on INT4 layers (never read by this path).
            k::route_gather(rtop, rw, Ly.tab_blob, nullptr, nullptr, nullptr, S, c.K, W.blob, W.tok,
                            W.inv, W.sg, W.su, W.wt, g_s);
            // IK2_MOE_GROUP_EXPERT (default OFF): group this visit's slots by routed expert ONCE on the
            // device (deterministic build, shared by gate_up and down); the grouped matvec reads each
            // expert's weights once for ALL its slots. Fixed grids throughout -> IK2_VERIFY_GRAPH-safe.
            // OFF: the identical per-slot gather + k_mv_grouped sequence as before, bit-for-bit.
            const bool grp = g_moe_group_exp && W.i4_g_cnt;
            if (grp) k::int4_moe_group_build(rtop, S, W.i4_g_cnt, W.i4_g_slots, W.i4_g_exp, W.i4_g_n, g_s);
            if (!grp)
                k::int4_gather_slots(rtop, Ly.i4_tab_Wp[0], Ly.i4_tab_Wh[0], Ly.i4_tab_WS[0], S,
                                     W.i4_Wp_s, W.i4_Wh_s, W.i4_WS_s, g_s);
            // per-slot input rows (slot s -> token s/K): the prefill scratch holds [cap][Kmax] f32.
            gather_f32_16_kernel<<<S, 256, 0, g_s>>>(W.x16, W.tok, S, C, W.i4_pf_x);
            k::int4_rot_q8x(W.i4_pf_x, S, C, W.i4_xq, W.i4_xd, g_s);
            if (grp)
                k::int4_matvec_grouped_exp(Ly.i4_tab_Wp[0], Ly.i4_tab_Wh[0], Ly.i4_tab_WS[0], 2 * F, C,
                                           W.i4_xq, W.i4_xd, S, W.gu32, 2 * F, inv4, g_s, C, C / 32,
                                           W.i4_g_cnt, W.i4_g_slots, W.i4_g_exp, W.i4_g_n);
            else
                k::int4_matvec_grouped(W.i4_Wp_s, W.i4_Wh_s, W.i4_WS_s, 2 * F, C, W.i4_xq, W.i4_xd,
                                       S, W.gu32, 2 * F, inv4, g_s, C, C / 32);
            k::swiglu_cap_f32(W.gu32, S, F, W.sg, W.su, c.gate_cap, c.up_cap, W.h16, g_s);
            k::int4_prep_down(W.h16, W.i4_hf, S, F, Kp, g_s);
            k::int4_rot_q8x(W.i4_hf, S, Kp, W.i4_hq, W.i4_hd, g_s);
            if (grp)
                k::int4_matvec_grouped_exp(Ly.i4_tab_Wp[2], Ly.i4_tab_Wh[2], Ly.i4_tab_WS[2], C, Kp,
                                           W.i4_hq, W.i4_hd, S, W.y32, C, inv4, g_s, Kp, Kp / 32,
                                           W.i4_g_cnt, W.i4_g_slots, W.i4_g_exp, W.i4_g_n);
            else {
                k::int4_gather_slots(rtop, Ly.i4_tab_Wp[2], Ly.i4_tab_Wh[2], Ly.i4_tab_WS[2], S,
                                     W.i4_Wp_dn, W.i4_Wh_dn, W.i4_WS_dn, g_s);
                k::int4_matvec_grouped(W.i4_Wp_dn, W.i4_Wh_dn, W.i4_WS_dn, C, Kp, W.i4_hq, W.i4_hd,
                                       S, W.y32, C, inv4, g_s, Kp, Kp / 32);
            }
            k::moe_combine_f32(W.y32, W.inv, W.wt, T, c.K, C, W.tmp, false, g_s);
            t_exp += now() - te0;
            conv_rec_w(Ly, W, 3, W.tmp, T, C);
            conv4_mlp_w(Ly, W, W.tmp, W.r, W.h, T, C);
            dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);  // no-op: dump.empty() in the guard
            continue;
        }
        // IK2_BATCH_MOE_DEVICE: device-only but BIT-EXACT vs the general MoE path. build the exact
        // expert-major row permutation on device, then run the SAME rot_pack8 + gemm8(kInt4Sh8) +
        // scatter_add_rows sequence as run_prefill_moe, instead of q8x/matvec/combine (which diverges).
        if (batch_moe_ok) {
            const double te0 = now();
            const int F = c.F, Kp = W.i4_Kpad_f;
            const bool moe_compact = g_batch_moe_compact && W.moe_active_map && W.moe_active_count;
            const int* active_map = moe_compact ? W.moe_active_map : nullptr;
            const int* active_count = moe_compact ? W.moe_active_count : nullptr;
            moe_batch_perm_kernel<<<1, 256, (size_t) c.E * 2 * sizeof(int), g_s>>>(
                rtop, rw, Ly.i4_tab_Wp[0], Ly.i4_tab_Wh[0], Ly.i4_tab_WS[0],
                Ly.i4_tab_Wp[2], Ly.i4_tab_Wh[2], Ly.i4_tab_WS[2], c.E, c.K, S,
                W.gidx, W.tok, W.wtr, W.sg, W.su,
                W.i4_Wp_s, W.i4_Wh_s, W.i4_WS_s, W.i4_Wp_dn, W.i4_Wh_dn, W.i4_WS_dn, W.inv,
                g_batch_moe_compact ? W.moe_active_map : nullptr,
                g_batch_moe_compact ? W.moe_active_count : nullptr);
            // W.tok[expert-major row] = its slot's token; this is the same row order run_prefill_moe
            // builds with fill[e]++, so each token receives the expert contributions in expert order.
            gather_f32_16_kernel<<<S, 256, 0, g_s>>>(W.x16, W.tok, S, C, W.i4_pf_x);
            k::int4_rot_pack8(W.i4_pf_x, S, C, W.i4_pf_x4, W.i4_pf_xm, W.i4_pf_xr, g_s);
            k::int4_gemm8_moe_masked(Ly.i4_Wp[0], Ly.i4_Wh[0], Ly.i4_WS[0],
                                     Ly.i4_wp_stride[0], Ly.i4_wh_stride[0], 2 * F, c.E, W.inv,
                                     2 * F, C, W.i4_pf_x4, W.i4_pf_xm, W.i4_pf_xr, S, 4, 8,
                                     W.gu32, nullptr, k::kInt4Sh8, g_s, 2 * F, active_map, active_count);
            k::swiglu_cap_f32(W.gu32, S, F, W.sg, W.su, c.gate_cap, c.up_cap, W.h16, g_s);
            i4_pad_h_kernel<<<S, 256, 0, g_s>>>(W.h16, S, F, Kp, W.i4_pf_x);
            k::int4_rot_pack8(W.i4_pf_x, S, Kp, W.i4_pf_x4, W.i4_pf_xm, W.i4_pf_xr, g_s);
            CU(cudaMemsetAsync(W.tmp, 0, (size_t) T * C * sizeof(float), g_s));
            k::int4_gemm8_moe_masked(Ly.i4_Wp[2], Ly.i4_Wh[2], Ly.i4_WS[2],
                                     Ly.i4_wp_stride[2], Ly.i4_wh_stride[2], C, c.E, W.inv,
                                     C, Kp, W.i4_pf_x4, W.i4_pf_xm, W.i4_pf_xr, S, 4, 8,
                                     W.y32, nullptr, k::kInt4Sh8, g_s, C, active_map, active_count);
            moe_batch_scatter_kernel<<<T, 256, 0, g_s>>>(W.y32, W.tok, W.wtr, S, C, W.tmp, C);
            g_pf.lap(g_pf.experts);
            t_exp += now() - te0;
            conv_rec_w(Ly, W, 3, W.tmp, T, C);
            conv4_mlp_w(Ly, W, W.tmp, W.r, W.h, T, C);
            continue;
        }
#endif
        if (g_vg_stage != VG_NORMAL && !(g_int4_experts && Ly.i4_loaded && l < g_int4_layers)) {
            // IK2_VERIFY_GRAPH: the layers the INT4 verify branch does not cover (l >= 20) must also be launch-only.
            // route_d already holds this visit's picks / weights; Ly.tab_blob and Ly.tab_sc are the resident expert
            // pointers and scales filled at load.  No D2H, no host grouping: the same launches on every replay.
            // dtype / offsets come from Ly.ty / Ly.eb, so this is the resident NVFP4/Q8 expert sequence.
            k::route_gather(W.route_d, (const float*) (W.route_d + S), Ly.tab_blob,
                            Ly.tab_sc[0], Ly.tab_sc[1], Ly.tab_sc[2], S, c.K,
                            W.blob, W.tok, W.inv, W.sg, W.su, W.wt, g_s);
            k::experts_gate_up_swiglu(W.blob, W.tok, S, Ly.ty[0], Ly.ty[1], Ly.eb[0], W.xq, C, c.F, W.sg, W.su,
                                      c.gate_cap, c.up_cap, W.h16, g_s);
            k::experts_down(W.blob, S, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32, g_s);
            k::moe_combine_f32(W.y32, W.inv, W.wt, T, c.K, C, W.tmp, false, g_s);
            conv_rec_w(Ly, W, 3, W.tmp, T, C);
            conv4_mlp_w(Ly, W, W.tmp, W.r, W.h, T, C);
            continue;
        }
        const bool cpu_lane = g_pool && g_overlap && !g_prof && T <= 8 && Ly.ty[0] == k::kNVFP4 &&
                              Ly.ty[1] == k::kNVFP4 && Ly.ty[2] == k::kNVFP4;
        const bool fast_ep_token = g_ep_fast && g_ep && T <= 8 && g_overlap && !g_prof &&
                                   dump.empty() && !g_trace && g_pop.empty();
        const int* top = nullptr;
        const float* w = nullptr;
        if (!fast_ep_token) {
            CU(cudaMemcpyAsync(W.route_h, W.route_d, (size_t) S * 8, cudaMemcpyDeviceToHost, g_s));
            if (cpu_lane) CU(cudaMemcpyAsync(W.x_h, W.xq, TC * 4, cudaMemcpyDeviceToHost, g_s));
            CU(cudaStreamSynchronize(g_s));
            top = W.route_h;
            w = (const float*) (W.route_h + S);
            pop_count(l, top, S);
            g_pf.lap(g_pf.route);
            if (!dump.empty()) {   // the experts picked, [T][K] int32 (compare with the reference's blk.L.picks)
                FILE* f = std::fopen((dump + "/blk." + std::to_string(vis) + ".picks.i32").c_str(), "wb");
                if (f) std::fwrite(top, 4, (size_t) S, f), std::fclose(f);
            }
            if (g_trace) {
                const int hdr[3] = {l, T, c.K};
                std::fwrite(hdr, 4, 3, g_trace);
                std::fwrite(top, 4, S, g_trace);
            }
        }
        if (g_tp && T == 1 && g_overlap && !g_prof) {   // tensor-parallel experts, one token
            const double te0 = now();
            const int F = c.F, Fd = F / 2, Cd = C / 2;
            CU(cudaEventRecord(g_ep_x, g_dev[0].s));
            const uint8_t* canon[2][16];
            // (1) each GPU: its halves in (hits run first), its half of h for every pick into tp_h (row = the pick)
            for (int d = 1; d >= 0; --d) {
                use_dev(d);
                Work& Wd = dev_work(d);
                ExpertCache& Ed = dev_cache(d);
                const uint64_t tick = ++Ed.tick;
                int ord[16], nhit = 0, nmis = 0, mis[16];
                for (int i = 0; i < S; ++i) {
                    bool missed = false;
                    canon[d][i] = Ed.get(Ly, l, top[i], tick, g_cs, &missed, true, false);
                    if (missed) mis[nmis++] = i;
                    else ord[nhit++] = i;
                }
                for (int j = 0; j < nmis; ++j) ord[nhit + j] = mis[j];
                CU(cudaEventRecord(g_ep_cp[d], g_cs));
                const int ps = Wd.pack_slot;
                Wd.pack_slot = (Wd.pack_slot + 1) % Work::kPackRing;
                uint8_t* pk = Wd.pack_h[ps];
                uint8_t* pk_d = Wd.pack_d[ps];
                const uint8_t* blob[16];
                int zeros[16] = {}, orow[16];
                float sgv[16], suv[16];
                for (int j = 0; j < S; ++j) {
                    const int i = ord[j], e = top[i];
                    blob[j] = canon[d][i], orow[j] = i;
                    sgv[j] = Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e];
                    suv[j] = Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e];
                }
                std::memcpy(pk, blob, S * sizeof(void*));
                std::memcpy(pk + 512, zeros, S * 4);
                std::memcpy(pk + 768, orow, S * 4);
                std::memcpy(pk + 1024, sgv, S * 4);
                std::memcpy(pk + 1280, suv, S * 4);
                std::memcpy(pk + 1536, canon[d], S * sizeof(void*));
                CU(cudaMemcpyAsync(pk_d, pk, 1600, cudaMemcpyHostToDevice, g_s));
                const auto* pb = (const uint8_t* const*) pk_d;
                const int* ptk = (const int*) (pk_d + 512);
                const int* por = (const int*) (pk_d + 768);
                const float* psg = (const float*) (pk_d + 1024);
                const float* psu = (const float*) (pk_d + 1280);
                if (d == 1) {
                    CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
                    CU(cudaMemcpyPeerAsync(Wd.xq, 1, dev_work(0).xq, 0, (size_t) C * 4, g_s));
                }
                size_t o, b0g;
                tp_part(Ly, d, 0, o, b0g);
                for (int part = 0; part < 2; ++part) {
                    const int p0 = part ? nhit : 0, m = part ? nmis : nhit;
                    if (part) CU(cudaStreamWaitEvent(g_s, g_ep_cp[d], 0));
                    if (!m) continue;
                    k::experts_gate_up_swiglu_rows(pb + p0, ptk + p0, m, Ly.ty[0], Ly.ty[1], b0g, Wd.xq, C, Fd,
                                                   psg + p0, psu + p0, c.gate_cap, c.up_cap, Wd.tp_h + (size_t) d * Fd,
                                                   por + p0, F, g_s);
                }
                CU(cudaEventRecord(g_tp_h[d], g_s));
                Wd.tp_pk = pk_d;
            }
            // (2) each GPU: the other's half of h in, its rows of down; GPU 1's rows to GPU 0, combined there
            for (int d = 1; d >= 0; --d) {
                use_dev(d);
                Work& Wd = dev_work(d);
                const Work& O = dev_work(1 - d);
                CU(cudaStreamWaitEvent(g_s, g_tp_h[1 - d], 0));
                CU(cudaMemcpy2DAsync(Wd.tp_h + (size_t) (1 - d) * Fd, (size_t) F * 2, O.tp_h + (size_t) (1 - d) * Fd,
                                     (size_t) F * 2, (size_t) Fd * 2, S, cudaMemcpyDefault, g_s));
                size_t o, b0g, b1u;
                tp_part(Ly, d, 0, o, b0g);
                tp_part(Ly, d, 1, o, b1u);
                k::experts_down((const uint8_t* const*) (Wd.tp_pk + 1536), S, Ly.ty[2], b0g + b1u, Wd.tp_h, Cd, F,
                                Wd.y32, g_s);
                if (d == 1) {
                    CU(cudaMemcpyPeerAsync(dev_work(0).y32 + (size_t) 8 * C, 0, Wd.y32, 1, (size_t) S * Cd * 4, g_s));
                    CU(cudaEventRecord(g_ep_done, g_s));
                }
            }
            CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
            {
                const int ps = W.pack_slot;
                W.pack_slot = (W.pack_slot + 1) % Work::kPackRing;
                uint8_t* pk = W.pack_h[ps];
                uint8_t* pk_d = W.pack_d[ps];
                int id[16];
                float wt[16];
                for (int i = 0; i < S; ++i) {
                    id[i] = i;
                    wt[i] = w[i] * (Ly.scale[2].empty() ? 1.0f : Ly.scale[2][top[i]]);
                }
                std::memcpy(pk + 768, id, S * 4);
                std::memcpy(pk + 1536, wt, S * 4);
                CU(cudaMemcpyAsync(pk_d + 768, pk + 768, 1024, cudaMemcpyHostToDevice, g_s));
                k::moe_combine_f32(W.y32, (const int*) (pk_d + 768), (const float*) (pk_d + 1536), 1, c.K, Cd, W.tmp,
                                   false, g_s);
                k::moe_combine_f32(W.y32 + (size_t) 8 * C, (const int*) (pk_d + 768), (const float*) (pk_d + 1536), 1,
                                   c.K, Cd, W.tmp + Cd, false, g_s);
            }
            t_exp += now() - te0;
            conv_rec_w(Ly, W, 3, W.tmp, T, C);
            conv4_mlp_w(Ly, W, W.tmp, W.r, W.h, T, C);
            dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
            gxe_hash("moe", W.h, vis);
            continue;
        }
        if (fast_ep_token) {
            const int F = c.F;
#if defined(__HIPCC__)
            static const int exp_mma = getenv("IK2_EXPERT_MMA") ? atoi(getenv("IK2_EXPERT_MMA")) : 0;
#else
            static const int exp_mma = getenv("IK2_EXPERT_MMA") ? atoi(getenv("IK2_EXPERT_MMA")) : 1;
#endif
            const bool exp_tc = exp_mma && k::experts_mma_fits(Ly.ty[0], Ly.ty[1], Ly.ty[2], C, F);

            // 1. Send xq (TC*4 B) and route_d (S*8 B) from GPU 0 to GPU 1
            if (g_ep_stage.d_xq[0]) {
                CU(cudaMemcpyAsync(g_ep_stage.d_xq[0], dev_work(0).xq, (size_t) T * C * 4, cudaMemcpyDeviceToDevice, g_s));
                CU(cudaMemcpyAsync(g_ep_stage.d_route[0], dev_work(0).route_d, (size_t) S * 8, cudaMemcpyDeviceToDevice, g_s));
            } else {
                CU(cudaMemcpyPeerAsync(dev_work(1).xq, 1, dev_work(0).xq, 0, (size_t) T * C * 4, g_s));
                CU(cudaMemcpyPeerAsync(dev_work(1).route_d, 1, dev_work(0).route_d, 0, (size_t) S * 8, g_s));
            }
            CU(cudaEventRecord(g_ep_x, g_s));

            // Helper to run experts on a device
            auto run_dev_exp = [&](int /*d*/, Work& Wd, cudaStream_t strm) {
                if (exp_tc) {
                    k::to_f16_limbs_tokens(Wd.xq, T, C, (uint16_t*) Wd.gu32, strm);
                    k::experts_gate_up_swiglu_mma(Wd.blob, S, Ly.eb[0], (const uint16_t*) Wd.gu32, C, F, Wd.sg, Wd.su,
                                                  c.gate_cap, c.up_cap, Wd.h16, strm, Wd.tok);
                    k::experts_down_mma(Wd.blob, S, Ly.eb[0] + Ly.eb[1], Wd.h16, C, F, Wd.y32, strm);
                } else {
                    k::experts_gate_up_swiglu(Wd.blob, Wd.tok, S, Ly.ty[0], Ly.ty[1], Ly.eb[0], Wd.xq, C, F, Wd.sg, Wd.su,
                                              c.gate_cap, c.up_cap, Wd.h16, strm);
                    k::experts_down(Wd.blob, S, Ly.ty[2], Ly.eb[0] + Ly.eb[1], Wd.h16, C, F, Wd.y32, strm);
                }
            };

            // 2. GPU 1: wait for xq & route_d, select resident experts, compute experts, copy result to GPU 0
            use_dev(1);
            CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
            const int* r_top1 = g_ep_stage.d_route[1] ? g_ep_stage.d_route[1] : dev_work(1).route_d;
            const float* r_w1 = g_ep_stage.d_route[1] ? (const float*)(g_ep_stage.d_route[1] + S) : (const float*)(dev_work(1).route_d + S);
            if (g_ep_stage.d_xq[1]) {
                CU(cudaMemcpyAsync(dev_work(1).xq, g_ep_stage.d_xq[1], (size_t) T * C * 4, cudaMemcpyDeviceToDevice, g_s));
            }
            k::ep_select_token(r_top1, r_w1, Ly.ep_tab_blob[1],
                               Ly.ep_tab_sc[1][0], Ly.ep_tab_sc[1][1], Ly.ep_tab_sc[1][2],
                               S, dev_work(1).blob, dev_work(1).tok, dev_work(1).inv, dev_work(1).sg, dev_work(1).su, dev_work(1).wt, g_s, c.K);
            run_dev_exp(1, dev_work(1), g_s);
            if (g_ep_stage.d_y[1]) {
                CU(cudaMemcpyAsync(g_ep_stage.d_y[1], dev_work(1).y32, (size_t) S * C * 4, cudaMemcpyDeviceToDevice, g_s));
            } else {
                CU(cudaMemcpyPeerAsync(dev_work(0).y32 + (size_t) S * C, 0, dev_work(1).y32, 1, (size_t) S * C * 4, g_s));
            }
            CU(cudaEventRecord(g_ep_done, g_s));

            // 3. GPU 0: select resident experts, compute experts, wait for GPU 1, combine & conv
            use_dev(0);
            k::ep_select_token(dev_work(0).route_d, (const float*)(dev_work(0).route_d + S), Ly.ep_tab_blob[0],
                               Ly.ep_tab_sc[0][0], Ly.ep_tab_sc[0][1], Ly.ep_tab_sc[0][2],
                               S, dev_work(0).blob, dev_work(0).tok, dev_work(0).inv, dev_work(0).sg, dev_work(0).su, dev_work(0).wt, g_s, c.K);
            run_dev_exp(0, dev_work(0), g_s);
            CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
            const float* y_peer = g_ep_stage.d_y[0] ? g_ep_stage.d_y[0] : (dev_work(0).y32 + (size_t) S * C);
            if (T == 1 && !g_rec) {
                k::ep_combine_conv(dev_work(0).y32, y_peer, dev_work(0).blob, dev_work(0).wt,
                                   S, C, Ly.h_mlp, Ly.sc_mlp, dev_work(0).r, dev_work(0).h, g_s);
            } else {
                if (T == 1) {
                    k::ep_combine_token(dev_work(0).y32, y_peer, dev_work(0).blob, dev_work(0).wt,
                                        S, C, dev_work(0).tmp, g_s);
                } else {
                    k::ep_combine_block(dev_work(0).y32, y_peer, dev_work(0).blob, dev_work(0).wt,
                                        T, c.K, C, dev_work(0).tmp, g_s);
                }
                Work& W0 = dev_work(0);
                conv_rec_w(Ly, W0, 3, W0.tmp, T, C);
                conv4_mlp_w(Ly, W0, W0.tmp, W0.r, W0.h, T, C);
            }
            dump_f("blk." + std::to_string(vis) + ".out", dev_work(0).h, TC);
            continue;
        }
        if (g_ep && T == 1 && g_overlap && !g_prof) {
            const double te0 = now();
            const int F = c.F;
            int row_of[16], n_d[2] = {0, 0};
            const uint8_t* blob_d[2][16];
            float sg_d[2][16], su_d[2][16];
            int dev_of[16];
            for (int i = 0; i < S; ++i) dev_of[i] = ep_owner(l, top[i]);
            // GPU 1 first (it has the farther to go): x over, its picks (hits first, then the copied ones), y back
            CU(cudaEventRecord(g_ep_x, g_dev[0].s));
            for (int d = 1; d >= 0; --d) {
                use_dev(d);
                Work& Wd = dev_work(d);
                ExpertCache& Ed = dev_cache(d);
                const uint64_t tick = ++Ed.tick;
                int hit[16], mis[16], nhit = 0, nmis = 0;
                const uint8_t* where[16];
                for (int i = 0; i < S; ++i) {
                    if (dev_of[i] != d) continue;
                    bool missed = false;
                    where[i] = Ed.get(Ly, l, top[i], tick, g_cs, &missed, true, false);
                    (missed ? mis[nmis++] : hit[nhit++]) = i;
                }
                CU(cudaEventRecord(g_ep_cp[d], g_cs));
                int n = 0;
                for (int pass = 0; pass < 2; ++pass)
                    for (int j = 0; j < (pass ? nmis : nhit); ++j) {
                        const int i = pass ? mis[j] : hit[j], e = top[i];
                        blob_d[d][n] = where[i];
                        sg_d[d][n] = Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e];
                        su_d[d][n] = Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e];
                        row_of[i] = n++;
                    }
                n_d[d] = n;
                if (d == 1) CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
                if (!n) continue;
                const int ps = Wd.pack_slot;
                Wd.pack_slot = (Wd.pack_slot + 1) % Work::kPackRing;
                uint8_t* pk = Wd.pack_h[ps];
                uint8_t* pk_d = Wd.pack_d[ps];
                const int zeros[16] = {};
                std::memcpy(pk, blob_d[d], n * sizeof(void*));
                std::memcpy(pk + 512, zeros, n * 4);
                std::memcpy(pk + 1024, sg_d[d], n * 4);
                std::memcpy(pk + 1280, su_d[d], n * 4);
                CU(cudaMemcpyAsync(pk_d, pk, 1536, cudaMemcpyHostToDevice, g_s));
                const auto* pb = (const uint8_t* const*) pk_d;
                const int* ptk = (const int*) (pk_d + 512);
                const float* psg = (const float*) (pk_d + 1024);
                const float* psu = (const float*) (pk_d + 1280);
                if (d == 1) CU(cudaMemcpyPeerAsync(Wd.xq, 1, dev_work(0).xq, 0, (size_t) C * 4, g_s));
                for (int part = 0; part < 2; ++part) {
                    const int p0 = part ? nhit : 0, m = part ? nmis : nhit;
                    if (part) CU(cudaStreamWaitEvent(g_s, g_ep_cp[d], 0));
                    if (!m) continue;
                    k::experts_gate_up_swiglu(pb + p0, ptk + p0, m, Ly.ty[0], Ly.ty[1], Ly.eb[0], Wd.xq, C, F, psg + p0,
                                              psu + p0, c.gate_cap, c.up_cap, Wd.h16 + (size_t) p0 * F, g_s);
                    k::experts_down(pb + p0, m, Ly.ty[2], Ly.eb[0] + Ly.eb[1], Wd.h16 + (size_t) p0 * F, C, F,
                                    Wd.y32 + (size_t) p0 * C, g_s);
                }
                if (d == 1) {
                    CU(cudaMemcpyPeerAsync(dev_work(0).y32 + (size_t) 8 * C, 0, Wd.y32, 1, (size_t) n * C * 4, g_s));
                    CU(cudaEventRecord(g_ep_done, g_s));
                }
            }
            // on GPU 0 (current): the combine over both GPUs' rows (GPU 1's from row 8)
            if (n_d[1]) CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
            {
                const int ps = W.pack_slot;
                W.pack_slot = (W.pack_slot + 1) % Work::kPackRing;
                uint8_t* pk = W.pack_h[ps];
                uint8_t* pk_d = W.pack_d[ps];
                int id[16];
                float wt[16];
                for (int i = 0; i < S; ++i) {
                    id[i] = (dev_of[i] ? 8 : 0) + row_of[i];
                    wt[i] = w[i] * (Ly.scale[2].empty() ? 1.0f : Ly.scale[2][top[i]]);
                }
                std::memcpy(pk + 768, id, S * 4);
                std::memcpy(pk + 1536, wt, S * 4);
                CU(cudaMemcpyAsync(pk_d + 768, pk + 768, 1024, cudaMemcpyHostToDevice, g_s));
                k::moe_combine_f32(W.y32, (const int*) (pk_d + 768), (const float*) (pk_d + 1536), T, c.K, C, W.tmp,
                                   false, g_s);
            }
            t_exp += now() - te0;
            conv_rec_w(Ly, W, 3, W.tmp, T, C);
            conv4_mlp_w(Ly, W, W.tmp, W.r, W.h, T, C);
            dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
            gxe_hash("moe", W.h, vis);
            continue;
        }
        const bool i4_layer = g_int4_experts && Ly.i4_loaded && l < g_int4_layers;
        if (g_fast && T > 1 && S <= W.rg_slots && !Ly.tab_h.empty() && !i4_layer) {
            ++g_pf.br_rg;
            // Every expert in VRAM and a prompt of a few tokens (a chat turn): the experts straight over their
            // blocks, an expert's blocks read once for up to kGroupMax of the tokens that picked it.  (The GEMM path
            // below decodes a whole expert to FP16 for the two or three tokens it has: a 47-token prompt took 0.34 s.)
            const double te0 = now();
            const int GM = W.rg_groups;
            const size_t o_n = (size_t) GM * 8, o_tok = o_n + (size_t) GM * 4, o_row = o_tok + (size_t) GM * k::kGroupMax * 4,
                         o_sg = o_row + (size_t) GM * k::kGroupMax * 4, o_su = o_sg + (size_t) W.rg_slots * 4,
                         o_wt = o_su + (size_t) W.rg_slots * 4, o_id = o_wt + (size_t) W.rg_slots * 4;
            auto** gb = (const uint8_t**) W.rg_h;
            int *gn = (int*) (W.rg_h + o_n), *gtok = (int*) (W.rg_h + o_tok), *grow = (int*) (W.rg_h + o_row);
            float *sg = (float*) (W.rg_h + o_sg), *su = (float*) (W.rg_h + o_su), *wt = (float*) (W.rg_h + o_wt);
            int* id = (int*) (W.rg_h + o_id);
            if (++W.stamp == 0) {
                std::fill(W.seen_stamp.begin(), W.seen_stamp.end(), 0);
                std::fill(W.gi_stamp.begin(), W.gi_stamp.end(), 0);
                W.stamp = 1;
            }
            const uint32_t stp = W.stamp;
            int G = 0;
            for (int i = 0; i < S; ++i) {   // slot i (token i / K, one of its picks): row i of h16 and y32
                const int e = top[i];
                int g = W.gi_stamp[e] == stp ? W.gi_idx[e] : -1;
                if (g < 0 || gn[g] == k::kGroupMax) {   // the expert's first group, or its next
                    g = G++;
                    W.gi_stamp[e] = stp, W.gi_idx[e] = g;
                    gb[g] = Ly.tab_h[e], gn[g] = 0;
                }
                gtok[(size_t) g * k::kGroupMax + gn[g]] = i / c.K;
                grow[(size_t) g * k::kGroupMax + gn[g]] = i;
                ++gn[g];
                sg[i] = Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e];
                su[i] = Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e];
                wt[i] = w[i] * (Ly.scale[2].empty() ? 1.0f : Ly.scale[2][e]);
                id[i] = i;
            }
            // (the stream was synchronised after the routing: the copy of the last layer's tables is done)
            CU(cudaMemcpyAsync(W.rg_d, W.rg_h, W.rg_bytes, cudaMemcpyHostToDevice, g_s));
            k::experts_gate_up_swiglu_g((const uint8_t* const*) W.rg_d, (const int*) (W.rg_d + o_n),
                                        (const int*) (W.rg_d + o_tok), (const int*) (W.rg_d + o_row), G, Ly.ty[0],
                                        Ly.ty[1], Ly.eb[0], W.xq, C, c.F, (const float*) (W.rg_d + o_sg),
                                        (const float*) (W.rg_d + o_su), c.gate_cap, c.up_cap, W.h16, g_s);
            k::experts_down_g((const uint8_t* const*) W.rg_d, (const int*) (W.rg_d + o_n), (const int*) (W.rg_d + o_row),
                              G, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32, g_s);
            k::moe_combine_f32(W.y32, (const int*) (W.rg_d + o_id), (const float*) (W.rg_d + o_wt), T, c.K, C, W.tmp,
                               false, g_s);
            g_pf.lap(g_pf.experts);
            t_exp += now() - te0;
            conv_rec_w(Ly, W, 3, W.tmp, T, C);
            conv4_mlp_w(Ly, W, W.tmp, W.r, W.h, T, C);
            g_pf.lap(g_pf.rest);
            dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
            continue;
        }
        if (g_mv && S <= 64 && !i4_layer) {
            ++g_pf.br_mv;
            if (g_prof && g_pf.br_mv <= 3) std::printf("MVDBG S=%d T=%d K=%d\n", S, T, c.K);
            const double te0 = now();
            // the token path: each slot (token t's pick i) its expert's blob straight from the cache
            // The slots already in VRAM first, then the ones copied in now: the first group's experts run while the
            // second's blocks are on their way (the copies on a stream of their own).
            const uint8_t* blob[64];
            int tk[64], id[64];
            float sg[64], su[64], wt[64];
            const uint64_t tick = ++EC.tick;
            const cudaStream_t cst = g_overlap && !g_prof ? g_cs : g_s;
            const uint8_t* where[64];
            char late[64] = {};
            int slot_of[64];
            if (++W.stamp == 0) {
                std::fill(W.seen_stamp.begin(), W.seen_stamp.end(), 0);
                std::fill(W.gi_stamp.begin(), W.gi_stamp.end(), 0);
                W.stamp = 1;
            }
            const uint32_t stp = W.stamp;
            lazy_poll();
            double acc_up = 0, acc_cpu = 0;
            int urgent[64], lazy[64];
            int n_urgent = 0, n_lazy = 0;
            for (int i = 0; i < S; ++i) {
                const int e = top[i];
                if (W.seen_stamp[e] == stp) {   // a second token of this call on an expert missing in it
                    late[i] = W.seen_late[e];
                    where[i] = EC.has(l, e) ? EC.get(Ly, l, e, tick, cst, nullptr, false, false) : nullptr;
                    slot_of[i] = where[i] ? EC.last_handle : -1;
                    continue;
                }
                if (EC.has(l, e)) {
                    where[i] = EC.get(Ly, l, e, tick, cst, nullptr, false, false);
                    slot_of[i] = EC.last_handle;
                    if (EC.slot_sseq(EC.last_handle) > g_seq_done) late[i] = 1;   // still arriving from a lazy copy
                    continue;
                }
                // a miss: the CPU's or the link's, whichever is free first
                if (cpu_lane && acc_cpu + g_t_cpu <= acc_up + g_t_up) {
                    late[i] = 2, acc_cpu += g_t_cpu;
                    if (g_cpu_lazy) {   // a slot for its blocks, copied in afterwards
                        where[i] = EC.get(Ly, l, e, tick, cst, nullptr, false, false);
                        slot_of[i] = EC.last_handle;
                        lazy[n_lazy++] = i;
                    } else {            // no slot taken: nothing in VRAM is evicted for it
                        ++EC.misses;
                        where[i] = nullptr, slot_of[i] = -1;
                    }
                } else {
                    late[i] = 1, acc_up += g_t_up;
                    where[i] = EC.get(Ly, l, e, tick, cst, nullptr, false, false);
                    slot_of[i] = EC.last_handle;
                    urgent[n_urgent++] = i;
                }
                W.seen_stamp[e] = stp;
                W.seen_late[e] = late[i];
            }
            int nh = 0, nw = 0;
            for (int i = 0; i < S; ++i) nh += late[i] == 0, nw += late[i] == 1;
            for (int ui = 0; ui < n_urgent; ++ui) {
                const int i = urgent[ui];
                EC.copy_in(Ly, top[i], (uint8_t*) where[i], cst);
            }
            if (nw && cst != g_s) CU(cudaEventRecord(g_ev, g_cs));
            if (n_lazy > 0) {
                const uint64_t seq = g_seq_next++;
                if (seq > g_seq_done + kRing - 1) {   // the ring's entry is still an unfinished batch's
                    CU(cudaStreamSynchronize(g_cs));
                    g_seq_done = seq - 1;
                }
                for (int li = 0; li < n_lazy; ++li) {
                    const int i = lazy[li];
                    EC.copy_in(Ly, top[i], (uint8_t*) where[i], g_cs);
                    EC.slot_sseq(slot_of[i]) = seq;
                }
                CU(cudaEventRecord(g_ring[seq % kRing], g_cs));
            }
            const int ps = W.pack_slot;
            W.pack_slot = (W.pack_slot + 1) % Work::kPackRing;
            uint8_t* pk = W.pack_h[ps];
            uint8_t* pk_d = W.pack_d[ps];
            float* yh = W.y_h[ps];
            int ph = 0, pm = nh, pc = nh + nw;
            ik2::cpu::ExpertJob jobs[64];
            const int n_jobs = S - nh - nw;
            const double tc0 = now();
            for (int i = 0; i < S; ++i) {
                const int e = top[i], p = late[i] == 2 ? pc++ : late[i] ? pm++ : ph++;
                blob[p] = where[i];
                tk[p] = i / c.K;
                id[i] = p;
                sg[p] = Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e];
                su[p] = Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e];
                wt[i] = w[i] * (Ly.scale[2].empty() ? 1.0f : Ly.scale[2][e]);
                if (late[i] == 2) {
                    auto& J = jobs[p - nh - nw];
                    J.gate = Ly.host[0] + (size_t) e * Ly.stride[0];
                    J.up = Ly.host[1] + (size_t) e * Ly.stride[1];
                    J.down = Ly.host[2] + (size_t) e * Ly.stride[2];
                    J.x = W.x_h + (size_t) (i / c.K) * C;
                    J.sg = sg[p], J.su = su[p];
                    J.y = yh + (size_t) (p - nh - nw) * C;
                    ik2::cpu::post_expert(*g_pool, &J, C, c.F, c.gate_cap, c.up_cap);
                }
            }
            g_n_cpu += n_jobs, g_n_wait += nw;
            // the groups: a part's (0: in VRAM, 1: waited for) slots by expert, each expert's blocks read once
            int ng[2] = {0, 0};
            const uint8_t* gb[64];
            int gn[64];
            int gtok[64 * k::kGroupMax];
            int grow[64 * k::kGroupMax];
            int G = 0;
            for (int part = 0; part < 2; ++part) {
                if (++W.stamp == 0) {
                    std::fill(W.seen_stamp.begin(), W.seen_stamp.end(), 0);
                    std::fill(W.gi_stamp.begin(), W.gi_stamp.end(), 0);
                    W.stamp = 1;
                }
                const uint32_t gstp = W.stamp;
                for (int i = 0; i < S; ++i) {
                    if (late[i] != part) continue;
                    const int e = top[i];
                    int g;
                    if (W.gi_stamp[e] != gstp) {
                        W.gi_stamp[e] = gstp;
                        g = G++;
                        W.gi_idx[e] = g;
                        gb[g] = where[i];
                        gn[g] = 0;
                        ++ng[part];
                    } else {
                        g = W.gi_idx[e];
                    }
                    gtok[(size_t) g * k::kGroupMax + gn[g]] = i / c.K;
                    grow[(size_t) g * k::kGroupMax + gn[g]] = id[i];
                    ++gn[g];
                }
            }
            g_pf.lap(g_pf.upload);
            // one copy of the tables from pinned memory: [64] pointers, then tok, inv, sg, su, wt [64] each
            std::memcpy(pk, blob, S * sizeof(void*));
            std::memcpy(pk + 512, tk, S * 4);
            std::memcpy(pk + 768, id, S * 4);
            std::memcpy(pk + 1024, sg, S * 4);
            std::memcpy(pk + 1280, su, S * 4);
            std::memcpy(pk + 1536, wt, S * 4);
            // ... and the groups: [64] pointers, n [64], tok [64][8], row [64][8]
            std::memcpy(pk + 1792, gb, G * sizeof(void*));
            std::memcpy(pk + 2304, gn, G * 4);
            std::memcpy(pk + 2560, gtok, (size_t) G * k::kGroupMax * 4);
            std::memcpy(pk + 4608, grow, (size_t) G * k::kGroupMax * 4);
            CU(cudaMemcpyAsync(pk_d, pk, 6656, cudaMemcpyHostToDevice, g_s));
            const auto* pgb = (const uint8_t* const*) (pk_d + 1792);
            const int* pgn = (const int*) (pk_d + 2304);
            const int* pgt = (const int*) (pk_d + 2560);
            const int* pgr = (const int*) (pk_d + 4608);
            const auto* pb = (const uint8_t* const*) pk_d;
            const int* ptk = (const int*) (pk_d + 512);
            const int* pid = (const int*) (pk_d + 768);
            const float* psg = (const float*) (pk_d + 1024);
            const float* psu = (const float*) (pk_d + 1280);
            const float* pwt = (const float*) (pk_d + 1536);
            const int F = c.F;
            for (int part = 0; part < 2; ++part) {
                const int p0 = part ? nh : 0, n = part ? nw : nh;
                if (part && cst != g_s) CU(cudaStreamWaitEvent(g_s, g_ev, 0));
                if (!n) continue;
                if (g_group && g_group_exp && T > 1) {
                    const int g0 = part ? ng[0] : 0, gc = ng[part];
                    k::experts_gate_up_swiglu_g(pgb + g0, pgn + g0, pgt + (size_t) g0 * k::kGroupMax,
                                                pgr + (size_t) g0 * k::kGroupMax, gc, Ly.ty[0], Ly.ty[1], Ly.eb[0],
                                                W.xq, C, F, psg, psu, c.gate_cap, c.up_cap, W.h16, g_s);
                    k::experts_down_g(pgb + g0, pgn + g0, pgr + (size_t) g0 * k::kGroupMax, gc, Ly.ty[2],
                                      Ly.eb[0] + Ly.eb[1], W.h16, C, F, W.y32, g_s);
                    continue;
                }
                k::experts_gate_up_swiglu(pb + p0, ptk + p0, n, Ly.ty[0], Ly.ty[1], Ly.eb[0], W.xq, C, F, psg + p0,
                                          psu + p0, c.gate_cap, c.up_cap, W.h16 + (size_t) p0 * F, g_s);
                k::experts_down(pb + p0, n, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16 + (size_t) p0 * F, C, F,
                                W.y32 + (size_t) p0 * C, g_s);
            }
            if (n_jobs > 0) {
                g_pool->wait();
                g_t_cpu = 0.9 * g_t_cpu + 0.1 * (now() - tc0) / n_jobs;
                CU(cudaMemcpyAsync(W.y32 + (size_t) (nh + nw) * C, yh, (size_t) n_jobs * C * 4, cudaMemcpyHostToDevice,
                                   g_s));
            }
            k::moe_combine_f32(W.y32, pid, pwt, T, c.K, C, W.tmp, false, g_s);
            g_pf.lap(g_pf.experts);
            t_exp += now() - te0;
            conv_rec_w(Ly, W, 3, W.tmp, T, C);
            conv4_mlp_w(Ly, W, W.tmp, W.r, W.h, T, C);
            g_pf.lap(g_pf.rest);
            dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
            continue;
        }
        ++g_pf.br_pf;
        if (g_prof && g_pf.br_pf <= 3) std::printf("PFDBG S=%d T=%d K=%d rg_slots=%d\n", S, T, c.K, W.rg_slots);
        run_prefill_moe(M, Ly, l, T, W.x16, top, w, W.r, W.h, W, EC, t_exp);
        g_pf.lap(g_pf.rest);
        dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
    }
    run_head(M, T, all_logits, out_mode, nb, W.h, W, logits_out, argmax_out);
    if (vg_capture) {
        // EndCapture / Instantiate do not execute the graph.  On success replay immediately so this
        // window's eight argmaxes are produced; on failure fall back to a direct run of this same window.
        g_vg_stage = VG_NORMAL;
        cudaGraph_t gr = nullptr;
        cudaError_t ce = cudaStreamEndCapture(g_s, &gr);
        if (ce == cudaSuccess) ce = cudaGraphInstantiate(&g_vgraph.exec, gr, 0);
        if (ce == cudaSuccess && g_vgraph.exec) {
            g_vgraph.graph = gr;
            g_vgraph.built = true;
            std::printf("verify: CUDA graph captured and instantiated!\n");
            std::fflush(stdout);
            CU(cudaGraphLaunch(g_vgraph.exec, g_s));
            CU(cudaMemcpyAsync(W.argmax_h, W.argmax_d, (size_t) T * 4, cudaMemcpyDeviceToHost, g_s));
            CU(cudaStreamSynchronize(g_s));
            argmax_out->resize(T);
            for (int t = 0; t < T; ++t) (*argmax_out)[t] = W.argmax_h[t];
        } else {
            std::printf("verify: graph %s failed (%s), fallback to direct launches\n",
                        gr ? "instantiate" : "end capture", cudaGetErrorString(ce));
            std::fflush(stdout);
            if (gr) cudaGraphDestroy(gr);
            g_vgraph.graph = nullptr;
            g_vgraph.exec = nullptr;
            g_vgraph.disabled = true;
            (void) cudaGetLastError();
            forward(M, tok, T, pos0, all_logits, out_mode, dump, logits_out, argmax_out, t_exp);
        }
    }
    if (bg_direct_read) {
        // The warm-up pass (or a failed capture begin) ran with g_vg_stage != VG_NORMAL, so run_head left the
        // argmaxes on the device.  Read them here before returning; the next eligible call captures the graph.
        g_vg_stage = VG_NORMAL;
        use_dev(0);
        Work& Wb = dev_work(0);
        bg_read_argmax(Wb);
        g_bgraph.warmed = true;
        return;
    }
    if (bg_began) {
        g_batch_graph_capturing = false;
        g_vg_stage = VG_NORMAL;
        cudaGraph_t gr = nullptr;
        cudaError_t ce = cudaStreamEndCapture(g_s, &gr);
        if (ce == cudaSuccess) ce = cudaGraphInstantiate(&g_bgraph.exec, gr, 0);
        if (ce == cudaSuccess && g_bgraph.exec) {
            g_bgraph.graph = gr;
            g_bgraph.built = true;
            g_bgraph.rows = T;
            std::printf("batch graph: captured T=%d, %d visits, one launch per step\n", T, c.n_visit);
            std::fflush(stdout);
            use_dev(0);
            CU(cudaGraphLaunch(g_bgraph.exec, g_s));
            bg_read_argmax(dev_work(0));
        } else {
            std::printf("batch graph: %s failed (%s), fallback to direct launches\n",
                        gr ? "instantiate" : "end capture", cudaGetErrorString(ce));
            std::fflush(stdout);
            if (gr) cudaGraphDestroy(gr);
            g_bgraph.graph = nullptr;
            g_bgraph.exec = nullptr;
            g_bgraph.disabled = true;
            (void) cudaGetLastError();
            forward(M, tok, T, pos0, all_logits, out_mode, dump, logits_out, argmax_out, t_exp);
            return;
        }
    }
#undef W
#undef EC
}

// IK2_BATCH_ATTN: one forward of n_rows independent greedy slots.  Each row has its own token, absolute position
// and KV/conv state; the 48 visits, grouped INT4 MoE and head all run as the existing T=2..8 verify window, with
// rows -> slots instead of rows -> consecutive draft positions.  Non-graph first; a graph can be added later by
// capturing this exact call shape.  Eligibility failures fall back to serial forward() calls, so correctness never
// depends on the new path being enabled.
void forward_batch(Model& M, const int* tok, int n_rows, const int* pos, const int* slot_ids, OutMode out_mode,
                   std::vector<int>* argmax_out, double& t_exp) {
    if (n_rows <= 0 || !tok || !pos || !slot_ids) return;
    if (argmax_out) argmax_out->clear();
    auto serial = [&]() {
        std::vector<float> lo;
        for (int i = 0; i < n_rows; ++i) {
            std::vector<int> one;
            if (g_batch_decode) load_batch_slot(M, slot_ids[i]);
            forward(M, tok + i, 1, pos[i], false, out_mode, std::string(), lo, &one, t_exp);
            if (argmax_out && !one.empty()) argmax_out->push_back(one[0]);
        }
    };
    // IK2_BATCH_EXACT (default 1): the true-batched T=2..8 general path is not numerically identical to the
    // serial resident T=1 path (dense Q8/INT4 GEMM vs rotated-int4 matvec, separate f16 sink GEMM vs fused f32
    // sink dot, rows attention with P=128 vs at_gqa, row argmax vs sum argmax).  Route every slot through the
    // proven byte-identical serial path.  IK2_BATCH_EXACT=0 keeps the old batched kernels as an escape hatch.
    // The resident CUDA graph captures raw per-visit pointers; with per-slot batch state those pointers change
    // on every load_batch_slot(), so replay would read the first captured slot's KV/conv state.  Direct launches
    // are required here; restore g_graph afterwards so non-batched requests keep their graph path.
    if (g_batch_exact) {
        const bool saved_graph = g_graph;
        g_graph = false;
        serial();
        g_graph = saved_graph;
        return;
    }
    bool ok = g_batch_decode && g_batch_slots.n_slot > 0 && n_rows > 1 &&
              n_rows <= BatchSlots::kMaxRows && out_mode == OutMode::kArgmax && argmax_out &&
              kv_is_int() && !g_kv_host && g_dev.size() == 1 && !dual() && !g_ep && !g_tp &&
              g_group && g_group_attn && !g_prof &&
              !g_prof_fast && !g_trace && g_pop.empty() && g_img.n == 0 && !g_cancel &&
              (size_t) n_rows * M.c.K <= 64 && n_rows <= dev_work(0).Tm;
    for (int i = 0; i < n_rows && ok; ++i) ok &= slot_ids[i] >= 0 && slot_ids[i] < g_batch_slots.n_slot;
    if (!ok) { serial(); return; }
    Work& W = dev_work(0);
    use_dev(0);
    bool got = false;
    {
        W.batch = true; W.batch_rows = n_rows;
        g_batch_active = true; g_batch_pos = pos; g_batch_slot_ids = slot_ids; g_batch_rows = n_rows;
        struct BatchGuard {
            Work& W;
            ~BatchGuard() { g_batch_active = false; g_batch_pos = nullptr; g_batch_slot_ids = nullptr; W.batch = false; W.batch_rows = 0; }
        } guard{W};
        std::vector<float> lo;
        std::vector<int> ag;
        forward(M, tok, n_rows, 0, true, OutMode::kArgmax, std::string(), lo, &ag, t_exp);
        got = (int) ag.size() == n_rows;
        if (got) *argmax_out = std::move(ag);
    }
    if (!got) serial();
}

// Layer-major super-chunk prefill (single GPU, no dump, no drafter):
// A long prompt in fewer passes: the prefill scratch for a pass of T tokens (~256 KB a token) is lent by the expert
// cache (ExpertCache::lend), which a long prefill streams through anyway, and given back afterwards - so the passes
// can be as long as the cache allows without the decode's cache being smaller (IK2_BORROW=0: off).
struct BigScratch {
    int pool = -1, T_sup = 0;
    size_t blk_ld = 0;
    float *h = nullptr, *r = nullptr, *tmp = nullptr, *blocks = nullptr, *gu32 = nullptr, *qkv = nullptr,
          *y32 = nullptr, *sg = nullptr, *su = nullptr, *wt = nullptr, *wtr = nullptr, *rlog = nullptr;
    uint16_t *xq16_sup = nullptr, *ex16 = nullptr, *h16 = nullptr;
    int *pos = nullptr, *route_d = nullptr, *gidx = nullptr, *inv = nullptr;
    float* lg = nullptr;
    uint16_t* lg16 = nullptr;
    size_t lg_cap = 0;
};
// the scratch of a T-token pass, carved from base (or only its size, base null)
size_t big_carve(const Config& c, int T, size_t gu32_have, size_t lg_want, size_t lg_have, Work* W, uint8_t* base) {
    const size_t C = c.C, F = c.F, S = (size_t) T * c.K, nblk = c.n_layer / c.block + 1;
    const size_t Tr = std::max<size_t>(64, T);
    size_t off = 0;
    auto take = [&](size_t bytes) {
        uint8_t* p = base ? base + off : nullptr;
        off += (bytes + 255) / 256 * 256;
        return p;
    };
    float* h = (float*) take(T * C * 4);
    float* r = (float*) take(T * C * 4);
    float* tmp = (float*) take(T * C * 4);
    float* blocks = (float*) take(nblk * T * C * 4);
    uint16_t* xq = (uint16_t*) take(T * C * 2);
    int* pos = (int*) take((size_t) T * 4);
    float* rlog = (float*) take((size_t) T * c.E * 4);
    int* route = (int*) take(S * 8);
    uint16_t* ex16 = (uint16_t*) take(T * C * 2);
    uint16_t* h16 = (uint16_t*) take(Tr * F * 2);
    const bool new_gu = (size_t) S * 2 * F * 4 > gu32_have;
    float* gu32 = new_gu ? (float*) take((size_t) S * 2 * F * 4) : nullptr;
    float* y32 = (float*) take(Tr * C * 4);
    float* sg = (float*) take(S * 4);
    float* su = (float*) take(S * 4);
    float* wt = (float*) take(S * 4);
    float* wtr = (float*) take(S * 4);
    int* gidx = (int*) take(S * 4);
    int* inv = (int*) take(S * 4);
    // the attention's logits and probabilities, room for kAttnRows queries at the prompt's end (lg_want of each)
    const bool new_lg = lg_want > lg_have;
    float* lg = new_lg ? (float*) take(lg_want * 4) : nullptr;
    uint16_t* lg16 = new_lg ? (uint16_t*) take(lg_want * 2) : nullptr;
    if (W && base) {
        W->h = h, W->r = r, W->tmp = tmp, W->blocks = blocks, W->xq16_sup = xq, W->pos = pos, W->rlog = rlog;
        W->route_d = route, W->ex16 = ex16, W->h16 = h16, W->y32 = y32;
        W->sg = sg, W->su = su, W->wt = wt, W->wtr = wtr, W->gidx = gidx, W->inv = inv;
        if (new_gu) W->gu32 = gu32, W->qkv = gu32, W->lg16 = (uint16_t*) gu32;
        if (new_lg) W->lg = lg, W->lg16 = lg16, W->lg_cap = lg_want;
        W->blk_ld = (size_t) T;
        W->T_sup = T;
    }
    return off;
}
// Tokens a pass for a prompt of n new tokens: W.T_sup, or more with a lent scratch (then big_end gives it back).
int big_begin(const Model& M, Work& W, ExpertCache& EC, int n, int ctx_end, BigScratch& bs) {
    bs.pool = -1;
    if (W.T_sup <= 0 || dual() || (getenv("IK2_BORROW") && !std::atoi(getenv("IK2_BORROW")))) return W.T_sup;
    const Config& c = M.c;
    // the attention's scratch for 256 queries against the whole context: few, large GEMMs (the default ~400 MB left
    // room for only ~20 queries a step at 100k tokens) - also lent for a short prompt at the end of a long context
    const size_t lg_want = (g_flash && !g_kv_fp8 && !kv_is_int() && c.D == 128) ? 0 : (size_t) c.H * 256 * (size_t) ctx_end;
    const bool long_ctx = lg_want > W.lg_cap;
    if (n <= W.T_sup && !long_ctx) return W.T_sup;
    int p = -1;
    size_t most = 0;
    for (size_t i = 0; i < EC.pools.size(); ++i) {
        const size_t b = (size_t) EC.pools[i].n_active * EC.pools[i].slot_bytes;
        if (b > most) most = b, p = (int) i;
    }
    if (p < 0) return W.T_sup;
    const auto& P = EC.pools[p];
    const int k_max = P.n_active - (2 * c.E + 80);   // (prefetch_layer still needs two layers' slots)
    if (k_max <= 0) return W.T_sup;
    const size_t avail = (size_t) k_max * P.slot_bytes;
    const size_t lg_bytes = long_ctx ? (lg_want * 4 + 255) / 256 * 256 + lg_want * 2 + 256 : 0;
    if (lg_bytes >= avail) return W.T_sup;
    int T = W.T_sup;
    if (n > W.T_sup) {
        const size_t per_tok = big_carve(c, 1 << 14, W.gu32_bytes, 0, 0, nullptr, nullptr) >> 14;
        const int T_cap = (int) std::min<size_t>((size_t) W.T_host, (avail - lg_bytes) / (per_tok + 1));
        if (T_cap > W.T_sup) {
            const int passes = (n + T_cap - 1) / T_cap;
            T = (n + passes - 1) / passes;
            while (T > W.T_sup && big_carve(c, T, W.gu32_bytes, lg_want, W.lg_cap, nullptr, nullptr) > avail) T -= 256;
            if (T < W.T_sup) T = W.T_sup;
        }
    }
    if (T == W.T_sup && !long_ctx) return W.T_sup;
    const size_t bytes = T > W.T_sup ? big_carve(c, T, W.gu32_bytes, lg_want, W.lg_cap, nullptr, nullptr) : lg_bytes;
    const int k = (int) ((bytes + P.slot_bytes - 1) / P.slot_bytes);
    CU(cudaStreamSynchronize(g_cs));
    CU(cudaStreamSynchronize(g_s));
    g_seq_done = g_seq_next - 1;
    uint8_t* base = EC.lend(p, k);
    if (!base) return W.T_sup;
    bs.h = W.h, bs.r = W.r, bs.tmp = W.tmp, bs.blocks = W.blocks, bs.gu32 = W.gu32, bs.qkv = W.qkv, bs.y32 = W.y32;
    bs.sg = W.sg, bs.su = W.su, bs.wt = W.wt, bs.wtr = W.wtr, bs.rlog = W.rlog, bs.xq16_sup = W.xq16_sup;
    bs.ex16 = W.ex16, bs.h16 = W.h16, bs.pos = W.pos, bs.route_d = W.route_d, bs.gidx = W.gidx, bs.inv = W.inv;
    bs.blk_ld = W.blk_ld, bs.T_sup = W.T_sup;
    bs.lg = W.lg, bs.lg16 = W.lg16, bs.lg_cap = W.lg_cap;
    if (T > W.T_sup) {
        big_carve(c, T, W.gu32_bytes, lg_want, W.lg_cap, &W, base);
    } else {   // only the attention's scratch
        W.lg = (float*) base;
        W.lg16 = (uint16_t*) (base + (lg_want * 4 + 255) / 256 * 256);
        W.lg_cap = lg_want;
    }
    bs.pool = p;
    return T;
}
void big_end(Work& W, ExpertCache& EC, BigScratch& bs) {
    if (bs.pool < 0) return;
    CU(cudaStreamSynchronize(g_cs));
    CU(cudaStreamSynchronize(g_s));
    W.h = bs.h, W.r = bs.r, W.tmp = bs.tmp, W.blocks = bs.blocks, W.gu32 = bs.gu32, W.qkv = bs.qkv, W.y32 = bs.y32;
    W.sg = bs.sg, W.su = bs.su, W.wt = bs.wt, W.wtr = bs.wtr, W.rlog = bs.rlog, W.xq16_sup = bs.xq16_sup;
    W.ex16 = bs.ex16, W.h16 = bs.h16, W.pos = bs.pos, W.route_d = bs.route_d, W.gidx = bs.gidx, W.inv = bs.inv;
    W.blk_ld = bs.blk_ld, W.T_sup = bs.T_sup;
    W.lg = bs.lg, W.lg16 = bs.lg16, W.lg_cap = bs.lg_cap;
    EC.give_back(bs.pool);
    bs.pool = -1;
}

// Sweeps each layer once across up to W.T_sup tokens (sub-chunked attention for KV/attention scratch, then one combined MoE pass).
void forward_prefill_sup(Model& M, const int* tok, int T, int pos0, int sub_chunk, OutMode out_mode,
                         std::vector<float>& logits_out, std::vector<int>* argmax_out, double& t_exp) {
    const Config& c = M.c;
    const int C = c.C;
    use_dev(M.L[0].dev);
    Work& W = dev_work(g_cur);
    ExpertCache& EC = dev_cache(g_cur);

    // Embed and norm all T tokens in sub-chunks of <= W.Tm
    {
        const size_t rb = (size_t) C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
        for (int off = 0; off < T; off += W.Tm) {
            const int ts = std::min(W.Tm, T - off);
            for (int t = 0; t < ts; ++t)
                k::decode_host(M.embd_type, M.embd + (size_t) tok[off + t] * rb, C, W.emb_h + (size_t) t * C);
            CU(cudaMemcpyAsync(W.tmp, W.emb_h, (size_t) ts * C * 4, cudaMemcpyHostToDevice, g_s));
            k::rms(W.tmp, M.embd_norm, ts, C, c.eps, W.h + (size_t) off * C, g_s);
            place_pictures(W.h + (size_t) off * C, ts, pos0 + off, C, g_s);
            if (off + ts < T) CU(cudaStreamSynchronize(g_s));   // (W.emb_h reused on next iteration)
        }
        for (int t = 0; t < T; ++t) W.pos_h[t] = pos0 + t;
        CU(cudaMemcpyAsync(W.pos, W.pos_h, (size_t) T * 4, cudaMemcpyHostToDevice, g_s));
    }

    // (enough tokens that a layer uses most of its experts; below, the misses copied after the routing cost less than
    // prefetching experts that turn out unused - a 119-token turn took 3.4 s with 2E, 1.9 s without)
    const bool pf = (size_t) T * c.K >= (size_t) 8 * c.E;
    EC.scan = pf && !(getenv("IK2_SCAN") && !std::atoi(getenv("IK2_SCAN")));
    if (dual()) dev_cache(1).scan = EC.scan;
    // (under EP each GPU prefetches the experts it owns, by GPU 0's picks so far)
    // One layer ahead: two or three layers in flight cost more than they hide - their slots squeeze the cache (2x RTX
    // PRO 6000, a 21.8k prompt: copies 244 -> 328 GiB, 8.05 -> 9.73 s); the next layer's prefetch issued before this
    // layer's routing round trip instead of after its experts are queued was slower too
    const int depth = 1;
    auto prefetch = [&](int ll, int ahead) {
        if (!dual()) {
            prefetch_layer(M, ll, W, EC, T, nullptr, -1, ahead);
            return;
        }
        use_dev(1);
        prefetch_layer(M, ll, dev_work(1), dev_cache(1), T, &W, g_tp ? -1 : 1, ahead);
        use_dev(0);
        prefetch_layer(M, ll, W, EC, T, nullptr, g_tp ? -1 : 0, ahead);
    };
    if (pf)
        for (int ll = 0; ll < depth; ++ll) prefetch(ll, ll + 1);
    int nb = 0;
    for (int vis = 0; vis < c.n_visit; ++vis) {   // (no loop here: a looped model's prompts go through forward())
        const int l = M.V[vis].l;
        Layer& Ly = bind_visit(M, vis);
        g_pf.start();
        for (int off = 0; off < T; off += sub_chunk) {
            const int ts = std::min(sub_chunk, T - off);
            run_layer_attn(M, Ly, l, ts, pos0 + off, W.h + (size_t) off * C, W.pos + off, nb, off,
                           W.r + (size_t) off * C, W);
        }
        if (l % c.block == 0) ++nb;
        {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = W.blocks + (size_t) i * W.blk_ld * C;
            st.st[nb] = W.r;
            k::RmsOuts outs{};
            outs.w[0] = Ly.ffn_norm;
            outs.y16[0] = W.xq16_sup;
            k::attn_res_rms_f16(st, nb + 1, Ly.ffn_res_score, 1, T, C, 1e-6f, c.eps, outs, g_s);
        }
        g_pf.lap(g_pf.attn);
        gemm16(Ly.router16, c.E, C, W.xq16_sup, T, W.rlog, c.E);
        const int S = T * c.K;
        int* rtop = W.route_d;
        float* rw = (float*) (W.route_d + S);
        k::route(W.rlog, Ly.probs_b, T, c.E, c.K, c.route_scale, rtop, rw, g_s);
        CU(cudaMemcpyAsync(W.route_h, W.route_d, (size_t) S * 8, cudaMemcpyDeviceToHost, g_s));
        CU(cudaStreamSynchronize(g_s));
        const int* top_h = W.route_h;
        const float* w_h = (const float*) (W.route_h + S);
        pop_count(l, top_h, S);
        g_pf.lap(g_pf.route);
        if (g_trace) {
            const int hdr[3] = {l, T, c.K};
            std::fwrite(hdr, 4, 3, g_trace);
            std::fwrite(top_h, 4, S, g_trace);
        }
        run_prefill_moe(M, Ly, l, T, W.xq16_sup, top_h, w_h, W.r, W.h, W, EC, t_exp);
        if (pf) prefetch(l + depth, depth);
        g_pf.lap(g_pf.rest);
    }
    EC.scan = false;
    if (dual()) dev_cache(1).scan = false;

    if (out_mode != OutMode::kNone) {
        g_pf.start();
        const int last = T - 1;
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = W.blocks + ((size_t) i * W.blk_ld + last) * C;
        st.st[nb] = W.h + (size_t) last * C;
        k::RmsOuts outs{};
        outs.w[0] = M.out_norm;
        outs.y16[0] = W.x16;
        outs.y32[0] = g_dense_q8 ? W.xq : nullptr;
        k::attn_res_rms_f16(st, nb + 1, M.out_res_score, 1, 1, C, 1e-6f, c.eps, outs, g_s);
        matmul(M.out_w, W.xq, W.x16, 1, W.logits, c.n_rows);
        if (out_mode == OutMode::kArgmax) {
            argmax_out->resize(1);
            k::argmax_sum(W.logits, nullptr, nullptr, c.n_vocab, W.argmax_d, g_s);
            CU(cudaMemcpyAsync(W.argmax_h, W.argmax_d, 4, cudaMemcpyDeviceToHost, g_s));
            CU(cudaStreamSynchronize(g_s));
            (*argmax_out)[0] = W.argmax_h[0];
        } else {
            k::softcap(W.logits, (size_t) c.n_rows, c.logit_cap, g_s);
            const size_t at = logits_out.size();
            logits_out.resize(at + (size_t) c.n_vocab);
            CU(cudaMemcpyAsync(logits_out.data() + at, W.logits, (size_t) c.n_vocab * 4, cudaMemcpyDeviceToHost, g_s));
            CU(cudaStreamSynchronize(g_s));
        }
        g_pf.lap(g_pf.rest);
    } else {
        CU(cudaStreamSynchronize(g_s));
    }
}

// every short conv's history as after the first n of the verify window's tokens
void rollback(Model& M, int n) {
    const int C = M.c.C, KD = M.c.KV * M.c.D;
    for (int vis = 0; vis < (int) M.V.size(); ++vis) {
        Layer& Ly = bind_visit(M, vis);
        float* hist[4] = {Ly.h_k, Ly.h_v, Ly.h_attn, Ly.h_mlp};
        const int cc[4] = {KD, KD, C, C};
        for (int i = 0; i < 4; ++i) k::conv_hist_after(Ly.rec_old[i], Ly.rec_x[i], n, cc[i], hist[i], g_s);
    }
}

// the embedding row of token t, normed as the model's input, into out (device [C])
void embed_row(Model& M, int t, float* raw, float* out) {
    const int C = M.c.C;
    float* e = dev_work(M.L[0].dev).emb_h;
    const size_t rb = (size_t) C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
    k::decode_host(M.embd_type, M.embd + (size_t) t * rb, C, e);
    CU(cudaMemcpyAsync(raw, e, (size_t) C * 4, cudaMemcpyHostToDevice, g_s));
    k::rms(raw, M.embd_norm, 1, C, M.c.eps, out, g_s);
}

// DSpark speculative window loop, shared by --tokens and --serve (greedy). Emit returns true when
// generation must stop (stop token / cancel / max_new); it is called for the initial anchor too.
// Commit records one token whose state the verify left committed in the KV cache. Both are called
// in output order; the last emitted token is NEVER committed (it is the pending next anchor), so
// the caller's committed history stays equal to the visible tokens minus that last one -- the same
// invariant the AR path has. Everything is byte-identical to the former inline CLI loop when
// cli_diag=true and no stop/cancel interrupts it.
struct SpecStats {
    int steps = 0;
    double t_draft = 0, t_verify = 0;
    std::vector<int> acc;   // [kBlock]
};

template <class Emit, class Commit>
static SpecStats generate_speculative(Model& M, Config& c, ik2::draft::Drafter* drafter,
        int anchor, int P, int ctx_n, int B,
        Emit emit, Commit commit, bool cli_diag, int verify_check, bool margin_check, double& t_exp) {
    static float* raw = nullptr;
    static float* anchor_h = nullptr;
    if (!raw) { raw = dalloc<float>(c.C); anchor_h = dalloc<float>(c.C); }
    SpecStats st;
    st.acc.assign(ik2::draft::kBlock, 0);
    use_dev(0);
    const auto argmax_row = [&](const float* r) { return (int) (std::max_element(r, r + c.n_vocab) - r); };
    if (emit(anchor)) return st;   // e.g. max_new=1, or the anchor itself is a stop token
    while (P + B <= ctx_n) {
        use_dev(0);
        const double t0 = now();
        embed_row(M, anchor, raw, anchor_h);
        int dr[ik2::draft::kBlock - 1];
        ik2::draft::propose(drafter, anchor, anchor_h, P, M.out_w.f16, c.n_rows, c.n_vocab, dr, B - 1);
        const double t1 = now();
        int tk[ik2::draft::kBlock];
        tk[0] = anchor;
        for (int j = 1; j < B; ++j) tk[j] = dr[j - 1];
        g_rec = true;
        std::vector<float> logits;
        std::vector<int> ag;
        const bool need_logits = cli_diag && (st.steps < verify_check || margin_check);
        const double t_exp_before = t_exp;
        forward(M, tk, B, P, true, need_logits ? OutMode::kLogits : OutMode::kArgmax, std::string(), logits, &ag, t_exp);
        const double vexp_ms = (t_exp - t_exp_before) * 1e3;
        if (cli_diag && st.steps < verify_check) {
            const bool ge = g_group_exp, ga = g_group_attn;
            const int modes[3][2] = {{0, 0}, {1, 0}, {0, 1}};
            const char* names[3] = {"neither grouped", "experts grouped only", "attention grouped only"};
            for (int m = 0; m < 3; ++m) {
                rollback(M, 0);
                g_group_exp = modes[m][0], g_group_attn = modes[m][1];
                std::vector<float> lb;
                forward(M, tk, B, P, true, OutMode::kLogits, std::string(), lb, nullptr, t_exp);
                double mx = 0;
                int dis = 0;
                for (int j = 0; j < B; ++j) {
                    const float* a = logits.data() + (size_t) j * c.n_vocab;
                    const float* b = lb.data() + (size_t) j * c.n_vocab;
                    for (int v = 0; v < c.n_vocab; ++v) mx = std::max(mx, (double) std::fabs(a[v] - b[v]));
                    dis += argmax_row(a) != argmax_row(b);
                }
                std::printf("verify check, step %d, grouped vs %s: max |dlogit| %.4f, argmax differs in %d of %d rows\n",
                            st.steps, names[m], mx, dis, B);
            }
            g_group_exp = ge, g_group_attn = ga;
            rollback(M, 0);
            logits.clear();
            forward(M, tk, B, P, true, OutMode::kLogits, std::string(), logits, nullptr, t_exp);
        }
        g_rec = false;
        int g[ik2::draft::kBlock];
        if (need_logits) {
            for (int j = 0; j < B; ++j) g[j] = argmax_row(logits.data() + (size_t) j * c.n_vocab);
        } else {
            for (int j = 0; j < B; ++j) g[j] = ag[j];
        }
        if (cli_diag && margin_check) {
            for (int j = 0; j < B; ++j) {
                const float* r = logits.data() + (size_t) j * c.n_vocab;
                float b1 = -1e30f, b2 = -1e30f;
                int i1 = -1, i2 = -1;
                for (int v = 0; v < c.n_vocab; ++v) {
                    if (r[v] > b1) b2 = b1, i2 = i1, b1 = r[v], i1 = v;
                    else if (r[v] > b2) b2 = r[v], i2 = v;
                }
                if (b1 - b2 < 0.05f)
                    std::printf("near tie at position %d: %d %.4f, %d %.4f\n", P + j, i1, b1, i2, b2);
            }
        }
        int n = 0;
        while (n < B - 1 && dr[n] == g[n]) ++n;
        if (cli_diag && getenv("IK2_DEBUG_DRAFT")) {
            std::printf("step %d pos %d anchor %d: dr=[", st.steps, P, anchor);
            for (int j = 0; j < B - 1; ++j) std::printf("%d%s", dr[j], j + 2 < B ? " " : "");
            std::printf("] g=[");
            for (int j = 0; j < B; ++j) std::printf("%d%s", g[j], j + 1 < B ? " " : "");
            std::printf("] match=%d\n", n);
        }
        const double tv2 = now();
        // Emit the accepted drafts then the target's token at the first disagreement. `e` counts how
        // many were emitted this step; a stop/cancel/max_new mid-window stops early, and the keep_local
        // partial commit below commits only the e tokens whose state the window actually verified.
        int e = 0;
        bool stop = false;
        for (int j = 0; j < n && !stop; ++j) { stop = emit(dr[j]); ++e; }
        if (!stop) { stop = emit(g[n]); ++e; }
        if (e < B) rollback(M, e);
        use_dev(0);
        commit(anchor);   // the pending token from the previous step is now committed
        for (int j = 0; j + 1 < e; ++j) commit(j < n ? dr[j] : g[n]);
        use_dev(0);
        ik2::draft::write_context(drafter, g_taps, g_taps_ld, 0, e, P);
        ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, e - 1);
        const double tv4 = now();
        if (cli_diag && getenv("IK2_VPROF")) {
            static double sv_fwd = 0, sv_rb = 0, sv_wc = 0, sv_exp = 0; static int sv_n = 0;
            sv_fwd += (t1 - t0) * 1e3; sv_rb += (tv2 - t1) * 1e3; sv_wc += (tv4 - tv2) * 1e3; sv_exp += vexp_ms; ++sv_n;
            if (sv_n % 8 == 0) std::printf("vprof: fwd %.1f (exp %.1f) rb %.1f wc %.1f ms/step (n=%d)\n",
                                           sv_fwd / sv_n, sv_exp / sv_n, sv_rb / sv_n, sv_wc / sv_n, sv_n);
        }
        if (cli_diag && getenv("IK2_VPROF2")) {
            static double sv_attn = 0, sv_moe = 0; static int sv_n2 = 0;
            for (int q = 0; q + 1 < vp_n; ++q) { float a = 0, b = 0;
                cudaEventElapsedTime(&a, vpA[q], vpB[q]); cudaEventElapsedTime(&b, vpB[q], vpA[q + 1]);
                sv_attn += a; sv_moe += b; }
            ++sv_n2;
            if (sv_n2 % 8 == 0) std::printf("vprof2: attn %.1f moe+rest %.1f ms/step over %d visits (n=%d)\n",
                                            sv_attn / sv_n2, sv_moe / sv_n2, vp_n - 1, sv_n2);
        }
        if (cli_diag && (getenv("IK2_VPROF3") || getenv("IK2_VPROF4"))) {
            static double s_res = 0, s_rms = 0, s_mm = 0, s_prep = 0, s_attn = 0, s_wo = 0; static int sv_n3 = 0;
            for (int q = 0; q + 1 < vp_n; ++q) { float a = 0;
                cudaEventElapsedTime(&a, vpA[q], vpR[q]); s_res += a;
                cudaEventElapsedTime(&a, vpR[q], vpS[q]); s_rms += a;
                cudaEventElapsedTime(&a, vpS[q], vpP[q]); s_mm += a;
                cudaEventElapsedTime(&a, vpP[q], vpK[q]); s_prep += a;
                cudaEventElapsedTime(&a, vpK[q], vpF[q]); s_attn += a;
                cudaEventElapsedTime(&a, vpF[q], vpB[q]); s_wo += a; }
            ++sv_n3;
            if (sv_n3 % 8 == 0) std::printf("vprof3: res %.1f rms %.1f mm %.1f prep %.1f attn %.1f wo %.1f ms/step (n=%d)\n",
                                            s_res / sv_n3, s_rms / sv_n3, s_mm / sv_n3, s_prep / sv_n3, s_attn / sv_n3, s_wo / sv_n3, sv_n3);
        }
        anchor = g[n];
        P += e;
        ++st.steps;
        ++st.acc[n];
        st.t_draft += t1 - t0;
        st.t_verify += now() - t1;
        if (stop) break;
    }
    return st;
}

}  // namespace

namespace ik2 {

// Pinned host memory for the conversation cache's snapshots.  A pageable buffer copies at 3-6 GB/s: measured
// 5.93 GB/s restoring the 452 MB of a 48k-token conversation and 3.17 GB/s parking it, which is most of what a
// cached turn still costs.  Pinned pages reach 20-25 GB/s on PCIe 4.0 x16.  If the lock fails the allocation
// falls back to the heap, so a big cache can never kill the engine - it only gets slower.
template <class T> struct PinAlloc {
    using value_type = T;
    PinAlloc() = default;
    template <class U> PinAlloc(const PinAlloc<U>&) noexcept {}
    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        const std::size_t bytes = n * sizeof(T);
        void* p = nullptr;
        if (cudaHostAlloc(&p, bytes + 16, cudaHostAllocDefault) == cudaSuccess) {
            *(std::uint64_t*) p = 0x50494E4E4544ULL;   // "PINNED"
            return (T*) ((char*) p + 16);
        }
        char* q = (char*) std::malloc(bytes + 16);
        if (!q) throw std::bad_alloc();
        *(std::uint64_t*) q = 0;
        return (T*) (q + 16);
    }
    void deallocate(T* p, std::size_t) noexcept {
        if (!p) return;
        char* q = (char*) p - 16;
        if (*(std::uint64_t*) q == 0x50494E4E4544ULL) cudaFreeHost(q);
        else std::free(q);
    }
    template <class U> bool operator==(const PinAlloc<U>&) const noexcept { return true; }
    template <class U> bool operator!=(const PinAlloc<U>&) const noexcept { return false; }
};
using PinBytes = std::vector<uint8_t, PinAlloc<uint8_t>>;

int run(int argc, char** argv) {
    std::string model, toks, dump, draft_path;
    int gen = 16, chunk = 2048, ctx = 262144, threads = 8, n_gpu = 1;
    bool serve = false, serve_strata = true;   // (--serve: the engine protocol; --serve-ik2: the @-line one)
    std::vector<int> eos_ids;                 // --eos a,b: the tokens a request ends at (the engine protocol)
    double cache_gb = 60;
    double vram_reserve_mib = 0;
    int image_token = 200054;   // --image-token: Inkling 2's <|unused_200054|>, the template's placeholder of a picture
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value of " + a);
            return argv[++i];
        };
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--tokens") {   // (--tokens @FILE: the ids from a file - a long prompt passes a shell's limit)
            toks = next();
            if (!toks.empty() && toks[0] == '@') {
                std::ifstream in(toks.substr(1));
                toks.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
                while (!toks.empty() && std::isspace((unsigned char) toks.back())) toks.pop_back();
            }
        }
        else if (a == "--gen" || a == "--max-new") gen = std::atoi(next().c_str());
        else if (a == "--chunk" || a == "--prefill") {
            const std::string v = next();
            if (v == "auto") chunk = 2048;
            else chunk = std::max(1, std::atoi(v.c_str()));
        }
        else if (a == "--vram-reserve-mib") vram_reserve_mib = std::atof(next().c_str());
        else if (a == "--ctx" || a == "--max-context") ctx = std::atoi(next().c_str());   // (--max-context: serve/server.py's name)
        else if (a == "--cache-gb" || a == "--expert-vram") cache_gb = std::atof(next().c_str());
        else if (a == "--threads" || a == "--cpu-threads" || a == "--pool-workers") threads = std::atoi(next().c_str());
        else if (a == "--gpus") n_gpu = std::atoi(next().c_str());
        else if (a == "--draft") {
            // Strata's serve/inkling_kind.py auto-adds "--draft off" for Inkling
            // engines that did not specify drafting. Treat "off"/"none"/"" as
            // no-op (plain AR) instead of trying to open a file literally named
            // "off" (which died with "drafter: cannot open off").
            const std::string v = next();
            if (v != "off" && v != "none" && v != "false" && !v.empty()) draft_path = v;
        }
        else if (a == "--serve") serve = true;
        else if (a == "--serve-ik2") serve = true, serve_strata = false;
        else if (a == "--eos") {
            const std::string v = next();
            for (const char* p = v.c_str(); *p;) {
                eos_ids.push_back(std::atoi(p));
                while (*p && *p != ',') ++p;
                if (*p == ',') ++p;
            }
        }
        else if (a == "--image-token") image_token = std::atoi(next().c_str());
        else if (a == "--dump") dump = next();
        else if (a.rfind("--", 0) == 0) {
            // Strata server passes generic flags (e.g. --layer-split, --vram-reserve-mib, etc.): skip unrecognized flag and its value if any
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) ++i;
        }
        else die("unknown argument " + a);
    }
    auto getenv_str = [](const char* name) -> std::string {
        const char* val = getenv(name);
        if (!val) return "";
        std::string s(val);
        while (!s.empty() && std::isspace((unsigned char) s.back())) s.pop_back();
        while (!s.empty() && std::isspace((unsigned char) s.front())) s.erase(0, 1);
        return s;
    };
    g_prof = getenv("IK2_PROF") && std::atoi(getenv("IK2_PROF"));
    // IK2_PF_PROFILE: print the prefill breakdown at the end of each served request. The profile block used to
    // print only on the CLI tail, which cannot read a chunk longer than 4,096 tokens (the prompt is an argument
    // and Windows caps that at 32,767 characters), so the 8,192-token chunk cliff had no measurement. The phase
    // timers are gated on g_prof, so this turns it on as well; it costs the fast decode path, not the prefill.
    const bool g_pf_profile = getenv("IK2_PF_PROFILE") != nullptr;
    if (g_pf_profile) g_prof = true;
    g_prof_fast = getenv("IK2_PROF_FAST") && std::atoi(getenv("IK2_PROF_FAST"));
    g_flash = !(getenv("IK2_FLASH") && !std::atoi(getenv("IK2_FLASH")));
    g_flash_wmma = !(getenv("IK2_FLASH_WMMA") && !std::atoi(getenv("IK2_FLASH_WMMA")));
    g_gate_fuse = !(getenv("IK2_GATE_FUSE") && !std::atoi(getenv("IK2_GATE_FUSE")));
    g_rms_fuse = !(getenv("IK2_RMS_FUSE") && !std::atoi(getenv("IK2_RMS_FUSE")));
    g_onepass = (getenv("IK2_ATTN_ONEPASS") && std::atoi(getenv("IK2_ATTN_ONEPASS")) != 0);
    g_loopbank_fuse = getenv("IK2_LOOPBANK_FUSE") && std::atoi(getenv("IK2_LOOPBANK_FUSE"));
    g_exp_prefill_wmma = getenv("IK2_EXPERTS_PREFILL_WMMA") && std::atoi(getenv("IK2_EXPERTS_PREFILL_WMMA"));
    if (getenv("IK2_INT4_EXPERTS")) g_int4_experts = std::atoi(getenv("IK2_INT4_EXPERTS"));
    if (getenv("IK2_INT4_LAYERS")) g_int4_layers = std::atoi(getenv("IK2_INT4_LAYERS"));
    if (getenv("IK2_I4_MID_FUSE")) g_i4_mid_fuse = std::atoi(getenv("IK2_I4_MID_FUSE"));
    if (getenv("IK2_I4_ROT_IN")) g_i4_rot_in = std::atoi(getenv("IK2_I4_ROT_IN"));
    g_int4_pf_off = getenv("IK2_INT4_PF") && !std::atoi(getenv("IK2_INT4_PF"));
    g_verify_fast = getenv("IK2_VERIFY_FAST") && std::atoi(getenv("IK2_VERIFY_FAST")) != 0;
    g_verify_graph = getenv("IK2_VERIFY_GRAPH") && std::atoi(getenv("IK2_VERIFY_GRAPH")) != 0;
    g_batch_decode = getenv("IK2_BATCH_ATTN") && std::atoi(getenv("IK2_BATCH_ATTN")) != 0;
    // IK2_BATCH_EXACT defaults to 1: batched greedy decoding must stay byte-identical to solo.
    g_batch_exact = !getenv("IK2_BATCH_EXACT") || std::atoi(getenv("IK2_BATCH_EXACT")) != 0;
    g_batch_moe_device = getenv("IK2_BATCH_MOE_DEVICE") && std::atoi(getenv("IK2_BATCH_MOE_DEVICE")) != 0;
    g_batch_moe_compact = getenv("IK2_BATCH_MOE_COMPACT") && std::atoi(getenv("IK2_BATCH_MOE_COMPACT")) != 0;
    g_batch_graph = getenv("IK2_BATCH_GRAPH") && std::atoi(getenv("IK2_BATCH_GRAPH")) != 0;
    g_batch_graph_exact = getenv("IK2_BATCH_GRAPH_EXACT") && std::atoi(getenv("IK2_BATCH_GRAPH_EXACT")) != 0;
    // The general MoE path does a route D2H + stream sync per visit, which CUDA graph capture cannot contain.
    // IK2_BATCH_GRAPH therefore selects the device-only batch MoE even if IK2_BATCH_MOE_DEVICE was not set.
    if (g_batch_graph) g_batch_moe_device = true;
    g_serve_dspec = getenv("IK2_SERVE_DSPEC") && std::atoi(getenv("IK2_SERVE_DSPEC")) != 0;
    g_batch_dense_mv = getenv("IK2_BATCH_DENSE_MV") && std::atoi(getenv("IK2_BATCH_DENSE_MV")) != 0;
    if (getenv("IK2_BATCH_DENSE_MV_MATS")) g_batch_dense_mv_mask = std::atoi(getenv("IK2_BATCH_DENSE_MV_MATS"));
    g_batch_rotmv = getenv("IK2_BATCH_ROTMV") && std::atoi(getenv("IK2_BATCH_ROTMV")) != 0;
    g_batch_solo_fast = getenv("IK2_BATCH_SOLO_FAST") && std::atoi(getenv("IK2_BATCH_SOLO_FAST")) != 0;
    if (const char* pp = getenv("IK2_PARALLEL")) { int v = std::atoi(pp); if (v >= 1 && v <= 8) g_parallel = v; }
    g_ar_dual = getenv("IK2_AR_DUAL") && std::atoi(getenv("IK2_AR_DUAL")) != 0;
    if (g_ar_dual) {
        if (g_parallel != 2) {
            std::fprintf(stderr, "ik2: IK2_AR_DUAL=1 needs IK2_PARALLEL=2; ignoring\n");
            g_ar_dual = false;
        } else {
            // Per-slot VRAM KV/conv state is mandatory; without it there is nothing to bake into slot graphs.
            g_batch_decode = true;
        }
    }
    g_ar_stream2 = getenv("IK2_AR_STREAM2") && std::atoi(getenv("IK2_AR_STREAM2")) != 0;
    if (g_ar_stream2) {
        if (!g_ar_dual || g_parallel != 2) {
            std::fprintf(stderr, "ik2: IK2_AR_STREAM2=1 needs IK2_AR_DUAL=1 and IK2_PARALLEL=2; ignoring\n");
            g_ar_stream2 = false;
        } else {
            g_ar2_stats = getenv("IK2_AR_STREAM2_STATS") && std::atoi(getenv("IK2_AR_STREAM2_STATS")) != 0;
        }
    }
    g_embd_gpu = getenv("IK2_EMBD_GPU") && std::atoi(getenv("IK2_EMBD_GPU")) != 0;
    g_ar_kstep = getenv("IK2_AR_KSTEP") && std::atoi(getenv("IK2_AR_KSTEP")) != 0;
    if (g_ar_kstep) {
        if (g_parallel != 2) {
            std::fprintf(stderr, "ik2: IK2_AR_KSTEP=1 needs IK2_PARALLEL=2; ignoring\n");
            g_ar_kstep = false;
        } else {
            // KSTEP is a self-contained opt-in: it must have the dual stream and the device embedding table.
            if (!g_ar_dual) { g_ar_dual = true; g_batch_decode = true; }
            g_ar_stream2 = true;
            g_ar2_stats = getenv("IK2_AR_STREAM2_STATS") && std::atoi(getenv("IK2_AR_STREAM2_STATS")) != 0;
            g_embd_gpu = true;
            g_ar_K = 4;
            // The A/B harness sets IK2_AR_K; the spec's canonical name is IK2_AR_KSTEP_K.  Accept both.
            const char* ks = getenv("IK2_AR_KSTEP_K");
            if (!ks) ks = getenv("IK2_AR_K");
            if (ks) {
                int v = std::atoi(ks);
                if (v < 2) v = 2;
                if (v > 8) v = 8;
                g_ar_K = v;
            }
        }
    }
    g_moe_group_exp = getenv("IK2_MOE_GROUP_EXPERT") && std::atoi(getenv("IK2_MOE_GROUP_EXPERT")) != 0;
    if (getenv("IK2_INT4_DENSE")) g_int4_dense = std::atoi(getenv("IK2_INT4_DENSE"));
    if (const char* pf = getenv("IK2_DECODE_PFULL")) { int v = std::atoi(pf); if (v >= 1 && v <= 512) g_decode_pfull = v; }
    if (const char* fw = getenv("IK2_FULL_WINDOW")) { int v = std::atoi(fw); if (v >= 0) g_full_window = v; }
    g_attn_wideload = getenv("IK2_ATTN_WIDELOAD") && std::atoi(getenv("IK2_ATTN_WIDELOAD")) != 0;
    g_attn_dpp = getenv("IK2_ATTN_DPP") && std::atoi(getenv("IK2_ATTN_DPP")) != 0;
    g_attn_gqa = getenv("IK2_ATTN_GQA") && std::atoi(getenv("IK2_ATTN_GQA")) != 0;
    g_int4_dense_multi = getenv("IK2_INT4_DENSE_MULTI") && std::atoi(getenv("IK2_INT4_DENSE_MULTI")) != 0;
    g_int4_prefill = getenv("IK2_INT4_PREFILL") && std::atoi(getenv("IK2_INT4_PREFILL")) != 0;
    g_int4_head = getenv("IK2_INT4_HEAD") && std::atoi(getenv("IK2_INT4_HEAD")) != 0;
    g_int4_head_decode = getenv("IK2_INT4_HEAD_DECODE") && std::atoi(getenv("IK2_INT4_HEAD_DECODE")) != 0;
    if (getenv("IK2_DRAFT_HEAD_INT4")) g_draft_head_int4 = std::atoi(getenv("IK2_DRAFT_HEAD_INT4")) != 0;
    g_int4_msg = getenv("IK2_INT4_MSG") && std::atoi(getenv("IK2_INT4_MSG")) != 0;
    if (getenv("IK2_INT4_DENSE_MV")) g_int4_dense_mv = std::atoi(getenv("IK2_INT4_DENSE_MV")) != 0;
    g_int4_rot_dedup = getenv("IK2_INT4_ROT_DEDUP") && std::atoi(getenv("IK2_INT4_ROT_DEDUP")) != 0;
    g_int4_mv_alt = getenv("IK2_INT4_MV_ALT") && std::atoi(getenv("IK2_INT4_MV_ALT")) != 0;
    g_mv_lds = getenv("IK2_MV_LDS") && std::atoi(getenv("IK2_MV_LDS")) != 0;
    g_mv = !getenv("IK2_MV") || std::atoi(getenv("IK2_MV"));
    g_group = !getenv("IK2_GROUP") || std::atoi(getenv("IK2_GROUP"));
    if (getenv("IK2_GROUP")) {
        const int v = std::atoi(getenv("IK2_GROUP"));
        g_group_exp = v == 1 || v >= 3, g_group_attn = v >= 2;
    }
    g_dense_q8 = getenv_str("IK2_DENSE") == "q8";
    if (getenv("IK2_KV")) {
        const std::string kv = getenv_str("IK2_KV");
        if (kv == "f16" || kv == "fp16") g_kv_mode = KV_F16;
        else if (kv == "fp8" || kv == "e4m3") g_kv_mode = KV_FP8;
        else if (kv == "int8") g_kv_mode = KV_INT8;
        else if (kv == "int4") g_kv_mode = KV_INT4;
        else die("unknown IK2_KV=" + kv + " (expected fp8, f16, int8 or int4)");
        g_kv_fp8 = g_kv_mode == KV_FP8;
        g_kv_int8 = g_kv_mode == KV_INT8;
        g_kv_int4 = g_kv_mode == KV_INT4;
    }
    if (getenv("IK2_KV_HOST")) g_kv_host = std::atoi(getenv("IK2_KV_HOST"));
    if (getenv("IK2_DENSE_HOST")) g_dense_host = std::atoi(getenv("IK2_DENSE_HOST"));
    if (getenv("IK2_TRACE") && *getenv("IK2_TRACE")) {
        g_trace = std::fopen(getenv("IK2_TRACE"), "wb");
        if (!g_trace) die(std::string("cannot write ") + getenv("IK2_TRACE"));
    }
    if (model.empty() || (toks.empty() && !serve))
        die("usage: strata-ik2 --model FIRST.gguf (--tokens 1,2,3 [--gen N] | --serve) [--chunk 2048] [--ctx N] ...");
    std::vector<int> prompt;
    for (const char* p = toks.c_str(); *p;) {
        prompt.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    const double t_start = now();
    Shards S(model);
    Model M;
    Config& c = M.c;
    const std::string a = "inkling2.";
    // a key under PR #32187's name, else under llama.cpp master's; a value master writes a visit (an array of equal
    // values) read as one
    auto key = [&](const char* k1, const char* k2 = nullptr) -> double {
        const auto* v = S.get(a + k1);
        if (!v && k2) v = S.get(a + k2);
        if (!v) die(std::string("no key ") + a + k1);
        return v->items.empty() ? v->num() : v->items[0].num();
    };
    c.n_visit = (int) key("block_count");
    c.C = (int) key("embedding_length");
    c.H = (int) key("attention.head_count");
    c.KV = (int) key("attention.head_count_kv");
    c.D = (int) key("attention.key_length");
    c.E = (int) key("expert_count");
    c.K = (int) key("expert_used_count");
    c.F = (int) key("expert_feed_forward_length");
    c.window = (int) key("attention.sliding_window");
    c.d_rel = (int) key("d_rel", "attention.relative_length");
    c.extent = (int) key("rel_extent", "attention.relative_extent");
    c.extent_swa = S.get(a + "rel_extent_swa") ? (int) key("rel_extent_swa") : c.window;   // (master: the window)
    c.block = (int) key("attn_res.block_size");
    c.n_vocab = (int) key("unpadded_vocab_size", "vocab_size_unpadded");
    c.eps = (float) key("attention.layer_norm_rms_epsilon");
    c.route_scale = (float) key("expert_weights_scale");
    c.qk_clip = (float) key("qk_clip_threshold", "attention.qk_clip");
    c.rel_clip = (float) key("rel_clip_threshold", "attention.relative_clip");
    c.alpha = (float) key("log_scaling_alpha", "attention.log_scale_alpha");
    c.n_floor = (float) key("log_scaling_n_floor", "attention.log_scale_floor");
    c.gate_cap = (float) key("mlp_gate_softcap", "activation.situ_beta");
    c.up_cap = (float) key("mlp_up_softcap", "activation.situ_linear_beta");
    c.logit_cap = (float) key("final_logit_softcapping");
    // the loop (Inkling-2-Mini): block_count counts the layer visits, the core's layers once a pass
    if (S.get(a + "num_loops")) c.n_pass = (int) key("num_loops");
    if (c.n_pass > 1) {
        c.core0 = (int) key("loop.core_start"), c.core1 = (int) key("loop.core_end"), c.rank = (int) key("loop.gate_rank");
        if (c.core0 <= 0 || c.core1 <= c.core0 || c.core0 % c.block || (c.core1 - c.core0) % c.block || c.rank <= 0)
            die("the looped core must start after the first block and span whole blocks");
    }
    const int n_core = c.core1 - c.core0;
    c.n_layer = c.n_visit - (c.n_pass - 1) * n_core;
    {   // the pattern is a visit's; a layer's is its first visit's (the same in every pass)
        const auto* sp = S.get(a + "attention.sliding_window_pattern");
        if (!sp || (int) sp->items.size() != c.n_visit) die("sliding_window_pattern");
        for (int l = 0; l < c.n_layer; ++l)
            c.sliding.push_back(sp->items[l < c.core1 ? l : l + (c.n_pass - 1) * n_core].u != 0);
    }
    // (a row of an expert is whole blocks of its type: checked per tensor below; 64 covers NVFP4 and Q8_0)
    if (c.D != 128 || c.H % c.KV || c.C % 64 || c.F % 64) die("unexpected shapes");
    const int n_sliding = (int) std::count(c.sliding.begin(), c.sliding.end(), true);
    std::printf("strata-ik2: %d layers (%d sliding), %d experts (%d a token), width %d, vocab %d; %zu shards\n",
                c.n_layer, n_sliding, c.E, c.K, c.C, c.n_vocab, S.files.size());
    if (c.n_pass > 1) {
        std::printf("looped: layers %d .. %d run %d times - %d layer visits a token; message gates of rank %d\n", c.core0,
                    c.core1 - 1, c.n_pass, c.n_visit, c.rank);
    }
    if (getenv("IK2_POP_OUT")) g_pop.assign((size_t) c.n_layer * c.E, 0), g_pop_E = c.E;
    g_tp_F = c.F, g_tp_C = c.C;
    if (n_gpu == 2 && getenv("IK2_EP") && std::atoi(getenv("IK2_EP"))) {
        const double share = getenv("IK2_EP_SPLIT") ? std::atof(getenv("IK2_EP_SPLIT")) : 0.5;
        g_pop_E_ep = c.E;
        g_ep_own.assign((size_t) c.n_layer * c.E, 0);
        std::vector<uint64_t> cnt;
        if (const char* wf = getenv("IK2_WARM"); wf && *wf) {
            FILE* in = std::fopen(wf, "rb");
            int32_t hdr[2] = {0, 0};
            if (in && std::fread(hdr, 4, 2, in) == 2 && hdr[0] == c.n_layer && hdr[1] == c.E) {
                cnt.resize((size_t) c.n_layer * c.E);
                if (std::fread(cnt.data(), 8, cnt.size(), in) != cnt.size()) cnt.clear();
            }
            if (in) std::fclose(in);
        }
        for (int l = 0; l < c.n_layer; ++l) {
            if (cnt.empty()) {
                for (int e = 0; e < c.E; ++e)
                    g_ep_own[(size_t) l * c.E + e] = ((uint32_t) (e * 2654435761u + l * 40503u) >> 8) % 10000 < share * 10000;
                continue;
            }
            std::vector<int> ord(c.E);
            for (int e = 0; e < c.E; ++e) ord[e] = e;
            std::sort(ord.begin(), ord.end(), [&](int a, int b) { return cnt[(size_t) l * c.E + a] > cnt[(size_t) l * c.E + b]; });
            double got[2] = {0, 0};
            const double want[2] = {1.0 - share, share};
            for (int e : ord) {   // (the GPU furthest below its share takes the next most picked)
                const double v = (double) cnt[(size_t) l * c.E + e] + 1.0;
                const int d = got[0] / want[0] <= got[1] / want[1] ? 0 : 1;
                g_ep_own[(size_t) l * c.E + e] = (uint8_t) d;
                got[d] += v;
            }
        }
    }

    {
        int nd = 0;
        CU(cudaGetDeviceCount(&nd));
        if (n_gpu < 1 || n_gpu > nd) die("--gpus " + std::to_string(n_gpu) + ": " + std::to_string(nd) + " visible");
    }
    g_dev.resize(n_gpu);
    for (int d = 0; d < n_gpu; ++d) {
        CU(cudaSetDevice(d));
        CU(cudaStreamCreate(&g_dev[d].s));
        CU(cudaStreamCreate(&g_dev[d].cs));
        CU(cudaEventCreateWithFlags(&g_dev[d].ev, cudaEventDisableTiming));
        CB(cublasCreate(&g_dev[d].blas));
        CB(cublasSetStream(g_dev[d].blas, g_dev[d].s));
        CB(cublasSetMathMode(g_dev[d].blas, CUBLAS_DEFAULT_MATH));
    }
    g_ep = n_gpu == 2 && getenv("IK2_EP") && std::atoi(getenv("IK2_EP"));
    g_tp = n_gpu == 2 && !g_ep && getenv("IK2_TP") && std::atoi(getenv("IK2_TP"));
    if (dual()) {
        int p01 = 0, p10 = 0;
        CU(cudaDeviceCanAccessPeer(&p01, 0, 1));
        CU(cudaDeviceCanAccessPeer(&p10, 1, 0));
        if (p01 && p10) {
            CU(cudaSetDevice(0));
            CU(cudaDeviceEnablePeerAccess(1, 0));
            CU(cudaSetDevice(1));
            CU(cudaDeviceEnablePeerAccess(0, 0));
        }
        CU(cudaSetDevice(0));
        CU(cudaEventCreateWithFlags(&g_ep_x, cudaEventDisableTiming));
        CU(cudaEventCreateWithFlags(&g_ep_cp[0], cudaEventDisableTiming));
        CU(cudaSetDevice(1));
        CU(cudaEventCreateWithFlags(&g_ep_cp[1], cudaEventDisableTiming));
        CU(cudaEventCreateWithFlags(&g_ep_done, cudaEventDisableTiming));
        CU(cudaSetDevice(0));
        CU(cudaEventCreateWithFlags(&g_tp_h[0], cudaEventDisableTiming));
        CU(cudaSetDevice(1));
        CU(cudaEventCreateWithFlags(&g_tp_h[1], cudaEventDisableTiming));
        std::printf("%s over 2 GPUs (peer access %s)\n", g_tp ? "experts tensor-parallel" : "expert parallel",
                    p01 && p10 ? "on" : "off: copies through the host");
        // Pinned mapped staging for inter-GPU zero-copy activation transfer
        auto alloc_mapped = [&](size_t bytes, void** h_ptr, void** d0_ptr, void** d1_ptr) {
            cudaError_t e = cudaHostAlloc(h_ptr, bytes, cudaHostAllocPortable | cudaHostAllocMapped);
            if (e != cudaSuccess) { (void) cudaGetLastError(); *h_ptr = nullptr; return false; }
            CU(cudaSetDevice(0));
            e = cudaHostGetDevicePointer(d0_ptr, *h_ptr, 0);
            if (e != cudaSuccess) { (void) cudaGetLastError(); return false; }
            CU(cudaSetDevice(1));
            e = cudaHostGetDevicePointer(d1_ptr, *h_ptr, 0);
            if (e != cudaSuccess) { (void) cudaGetLastError(); return false; }
            return true;
        };
        const int max_S = 64;
        const int max_T = 8;
        bool stage_ok = alloc_mapped((size_t) max_T * c.C * sizeof(float), &g_ep_stage.h_xq, (void**) &g_ep_stage.d_xq[0], (void**) &g_ep_stage.d_xq[1]) &&
                        alloc_mapped((size_t) max_S * 8, &g_ep_stage.h_route, (void**) &g_ep_stage.d_route[0], (void**) &g_ep_stage.d_route[1]) &&
                        alloc_mapped((size_t) max_S * c.C * sizeof(float), &g_ep_stage.h_y, (void**) &g_ep_stage.d_y[0], (void**) &g_ep_stage.d_y[1]);
        if (stage_ok) {
            std::printf("zero-copy mapped EP staging enabled (%zu KB)\n",
                        ((size_t) max_T * c.C * sizeof(float) + (size_t) max_S * 8 + (size_t) max_S * c.C * sizeof(float)) / 1024);
        } else {
            std::printf("zero-copy mapped EP staging disabled (fallback to standard transfer)\n");
        }
    }
    use_dev(0);
    g_overlap = !getenv("IK2_OVERLAP") || std::atoi(getenv("IK2_OVERLAP"));
    for (auto& e : g_ring) CU(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    // (off by default: on an RTX PRO 6000 node's Xeon 6737P an NVFP4 expert takes 0.73 ms on 7 threads against 0.33 ms
    //  over PCIe, and the lane measured 16.4 against 16.8 tok/s - IK2_CPU=1 to try it)
    if (getenv("IK2_CPU") && std::atoi(getenv("IK2_CPU")) && ik2::cpu::available() && threads > 1 && n_gpu == 1)
        g_pool = new ik2::cpu::Pool(threads - 1);
    if (getenv("IK2_UP_MS")) g_t_up = std::atof(getenv("IK2_UP_MS")) * 1e-3;
    g_cpu_lazy = !getenv("IK2_CPU_LAZY") || std::atoi(getenv("IK2_CPU_LAZY"));
    if (n_gpu > 2 && getenv("IK2_DENSE")) die("IK2_DENSE with --gpus > 2");

    // Do the experts all fit in VRAM beside the dense weights and the K / V cache (Inkling-2-Mini's 14.7 GiB on a 24 GB
    // card)?  Then they all stay there: the dense weights are kept as the GGUF's blocks unless IK2_DENSE says otherwise
    // (half an FP16 copy's bytes, a token's products straight over them) and tokens take the resident path
    // (IK2_FAST=0: off; IK2_GRAPH=0: its launches one by one instead of a CUDA graph a token).
    g_graph = !(getenv("IK2_GRAPH") && !std::atoi(getenv("IK2_GRAPH")));
    // gfx1100 has no CUDA mma16816: the decode tensor-core path is a HIP trap stub and would hang. Default it off on HIP.
#if defined(__HIPCC__)
    const int attn_mma_default = 0;
#else
    const int attn_mma_default = 1;
#endif
    g_attn_mma = c.D == 128 && c.H / c.KV <= 8
                     ? (getenv("IK2_ATTN_MMA") ? std::atoi(getenv("IK2_ATTN_MMA")) : attn_mma_default)
                     : 0;
    g_mma_split = k::attn_decode_mma_blocks(c.KV);
    const bool can_resident = (n_gpu == 1 || (dual() && g_ep)) && !g_pool;
    if (can_resident) {
        size_t exp_b = 0, dense_b = 0, kv_b = 0;
        for (const auto& kv : S.by_name) {
            const auto& t = *kv.second.second;
            if (!k::block_values((int) t.type)) continue;
            if (t.name.find("_exps.weight") != std::string::npos) exp_b += tensor_bytes(t);
            else if (t.name != "token_embd.weight") dense_b += tensor_bytes(t);
        }
        // IK2_FIT_ACTUAL=1: account the REAL resident bytes, not the GGUF's. The INT4 expert sidecars (l<20)
        // are smaller than the NVFP4 blocks they replace, so the GGUF-based estimate above over-counts and
        // wrongly drops residency at long context. This knob makes the fit check match what is actually loaded.
        const bool fit_actual = getenv("IK2_FIT_ACTUAL") && std::atoi(getenv("IK2_FIT_ACTUAL"));
        if (fit_actual && g_int4_experts && c.F % 64 == 0 && c.C % 256 == 0) {
            const size_t Kpad_gu = (size_t) c.C, Kpad_dn = (size_t) ((c.F + 255) / 256) * 256;
            const size_t i4_per_layer =
                (size_t) c.E * ((size_t) (2 * c.F / 2) * Kpad_gu + (size_t) (2 * c.F / 16) * (Kpad_gu / 256) * 64 + (size_t) (2 * c.F) * 4) +
                (size_t) c.E * ((size_t) (c.C / 2) * Kpad_dn + (size_t) (c.C / 16) * (Kpad_dn / 256) * 64 + (size_t) c.C * 4);
            size_t gguf_l20 = 0;
            for (const auto& kv : S.by_name) {
                const auto& t = *kv.second.second;
                if (!k::block_values((int) t.type)) continue;
                if (t.name.find("_exps.weight") == std::string::npos) continue;
                int tl = -1;
                if (sscanf(t.name.c_str(), "blk.%d.", &tl) == 1 && tl >= g_int4_layers) gguf_l20 += tensor_bytes(t);
            }
            exp_b = gguf_l20 + (size_t) g_int4_layers * i4_per_layer;   // l>=cut Q8_0 + l<cut INT4 sidecars
        }
        // bytes per KV token-position (K AND V are two buffers): int8 = 2*KD codes + 4*KV fp16 scales;
        // int4 = KD codes (2 x KD/2) + 4*KV scales; fp8 = 2*KD; f16 = 4*KD.
        const size_t kv_elems = (size_t) c.KV * c.D;
        const size_t kv_per_tok = kv_is_int()
            ? ((kv_int_bits() == 4 ? kv_elems : kv_elems * 2) + (size_t) 4 * c.KV)
            : kv_elems * (g_kv_fp8 ? 1 : 2) * 2;
        for (int l = 0; l < c.n_layer; ++l) {
            const size_t cap = c.sliding[l] ? (size_t) std::min(ctx, c.window + std::max(chunk, ik2::draft::kBlock)) : (size_t) ctx;
            const size_t visits = (l >= c.core0 && l < c.core1 ? c.n_pass : 1);
            // IK2_KV_HOST: the full-attention visits' KV is host-mapped (0 VRAM); only sliding KV counts toward VRAM.
            if (g_kv_host && kv_is_int() && !c.sliding[l]) continue;
            kv_b += visits * cap * kv_per_tok;
        }
        // IK2_AR_DUAL keeps TWO independent KV/conv states in VRAM (one resident graph per slot).  Account for
        // the second slot here so residency/fast-path decisions are made with the real footprint.  If it does
        // not fit, fall back to the legacy single stream instead of risking an OOM hang.
        const size_t kv_b_one = kv_b;
        if (g_ar_dual) kv_b *= 2;
        size_t fr0 = 0, tot0 = 0;
        CU(cudaMemGetInfo(&fr0, &tot0));
        const size_t reserve_default = (size_t) 3 << 29;   // (the work buffers and a quarter of a GiB of room: 1.5 GiB)
        const size_t reserve = getenv("IK2_RESERVE_MIB") ? (size_t) (std::atof(getenv("IK2_RESERVE_MIB")) * 1048576.0) : reserve_default;
        const size_t exp_b_dev0 = dual() ? (exp_b / 2) : exp_b;
        bool fits = exp_b_dev0 + dense_b + kv_b + reserve <= fr0 && exp_b_dev0 <= (size_t) (cache_gb * 1073741824.0);
        if (g_ar_dual && !fits) {
            std::fprintf(stderr, "ik2: IK2_AR_DUAL=1 needs 2x KV in VRAM (%.2f GiB) beside experts/dense/work; "
                                 "falling back to IK2_PARALLEL=1\n", kv_b / 1073741824.0);
            g_ar_dual = false;
            g_ar_stream2 = false;
            g_ar_kstep = false;
            g_ar_kstep_ready = false;
            if (!getenv("IK2_EMBD_GPU")) g_embd_gpu = false;
            g_batch_decode = false;
            g_parallel = 1;
            kv_b = kv_b_one;
            fits = exp_b_dev0 + dense_b + kv_b + reserve <= fr0 && exp_b_dev0 <= (size_t) (cache_gb * 1073741824.0);
        }
        if (fit_actual)
            std::printf("fit(actual): experts %.2f GiB + dense %.2f + KV %.2f + reserve %.2f = %.2f vs free %.2f GiB -> %s\n",
                        exp_b_dev0 / 1073741824.0, dense_b / 1073741824.0, kv_b / 1073741824.0, reserve / 1073741824.0,
                        (exp_b_dev0 + dense_b + kv_b + reserve) / 1073741824.0, fr0 / 1073741824.0, fits ? "FITS" : "no");
        if (fits && !getenv("IK2_DENSE")) g_dense_q8 = true;
        g_fast_want = fits && g_dense_q8 && !(getenv("IK2_FAST") && !std::atoi(getenv("IK2_FAST")));
        std::printf("experts %.1f GiB (%s), dense weights %.1f GiB as blocks, K / V cache %.1f GiB, %.1f GiB free: %s\n",
                    exp_b / 1073741824.0, dual() ? "split across 2 GPUs" : "1 GPU",
                    dense_b / 1073741824.0, kv_b / 1073741824.0, fr0 / 1073741824.0,
                    fits ? "every expert stays in VRAM" : "the experts in host memory, a cache of them in VRAM");
    }

    if (g_ar_dual && g_ar_stream2) {
        CU(cudaSetDevice(0));
        CU(cudaStreamCreateWithFlags(&g_s1, cudaStreamNonBlocking));
        if (g_ar2_stats) for (cudaEvent_t& e : g_ar2_ev) CU(cudaEventCreate(&e));
        std::printf("ik2: AR dual stream2 enabled (slot0 g_s, slot1 g_s1, isolated Work)\n");
        std::fflush(stdout);
    }

    // the dense weights
    const double t_dense = now();
    const size_t C = c.C, HD = (size_t) c.H * c.D, KD = (size_t) c.KV * c.D;
    M.L.resize(c.n_layer);
    vram_mark("before-dense-layers");
    // two layouts: llama.cpp PR #32187's (ref/inkling2.py's: the short convs folded, gate and up apart) and unsloth's
    // "GGUF_shard_rewrite" one that llama.cpp's master reads (names of its own, the convs' residual and scale outside
    // their weights, gate and up fused - gate's rows first - with one scale): the second's folded here as it loads
    const bool rw = S.info("blk.0.ffn_gate_up_exps.weight") != nullptr;
    auto nm = [&](const std::string& b, const char* old_name) -> std::string {
        static const std::map<std::string, std::string> m = {
            {"attn_r.weight", "attn_rel.weight"}, {"attn_r_clip.weight", "attn_rel_clip.weight"},
            {"attn_rel_proj.weight", "attn_rel_bias.weight"}, {"attn_res_score.weight", "attn_res_query.weight"},
            {"ffn_res_score.weight", "ffn_res_query.weight"}, {"output_res_score.weight", "output_res_query.weight"},
            {"attn_q_in_norm.weight", "attn_q_inp_norm.weight"}, {"attn_k_in_norm.weight", "attn_k_inp_norm.weight"},
            {"attn_v_in_norm.weight", "attn_v_inp_norm.weight"}, {"shortconv_k.weight", "attn_k_conv.weight"},
            {"shortconv_v.weight", "attn_v_conv.weight"}, {"shortconv_attn.weight", "attn_post_conv.weight"},
            {"shortconv_mlp.weight", "ffn_post_conv.weight"}, {"shortconv_k_dyn.weight", "attn_k_conv_proj.weight"},
            {"shortconv_v_dyn.weight", "attn_v_conv_proj.weight"}};
        auto it = m.find(old_name);
        return b + (rw && it != m.end() ? it->second : std::string(old_name));
    };
    // a conv's weights [C'][4] with the residual (tap 3: the current token) and the output scale folded in
    auto conv_w = [&](const std::string& b, const char* n, size_t Cc, bool dynamic) -> float* {
        auto w = host_f32(S, nm(b, n), Cc * 4);
        if (rw) {
            const float sc = 1.0f / std::sqrt((float) ((dynamic ? 2 : 1) * 4 + 1));
            for (size_t i = 0; i < Cc; ++i) {
                for (int j = 0; j < 4; ++j) w[i * 4 + j] *= sc;
                w[i * 4 + 3] += sc;
            }
        }
        return dev_f32(w);
    };
    auto dyn_host = [&](const std::string& b, const char* n) -> std::vector<float> {
        auto w = host_f32(S, nm(b, n), (size_t) c.KV * 4 * C);
        if (rw)
            for (auto& x : w) x *= 1.0f / std::sqrt(9.0f);
        return w;
    };
    std::printf("layout: %s\n", rw ? "unsloth's GGUF_shard_rewrite (llama.cpp master)" : "llama.cpp PR #32187");
    struct ExpJob {
        int l;
        std::string name;   // a tensor read whole into pinned memory
        std::vector<int> roles;
    };
    std::vector<ExpJob> exp_jobs;
    size_t host_total = 0;
    M.embd_norm = load_f32(S, "token_embd_norm.weight", C);   // (on GPU 0, where the embedding rows go)
    std::vector<float> lp_q, lp_kv;   // a looped core's message queries, [2 n_core][C] each
    std::vector<float> lp_all;
    if (c.n_pass > 1) {
        lp_q = host_f32(S, "loop_res_query.weight", (size_t) 2 * n_core * C);
        lp_kv = host_f32(S, "loop_kv_res_query.weight", (size_t) 2 * n_core * C);
    }
    for (int l = 0; l < c.n_layer; ++l) {
        Layer& Ly = M.L[l];
        const std::string b = "blk." + std::to_string(l) + ".";
        Ly.dev = dual() ? 0 : l * n_gpu / c.n_layer;
        use_dev(Ly.dev);
        Ly.sliding = c.sliding[l];
        auto sink_v = host_f32(S, b + "attn_sink_proj.weight", (size_t) c.H * C);
        if (Ly.sliding) {
            std::vector<MatPart> qkv_parts = {
                {b + "attn_q.weight", (int) HD, {}},
                {b + "attn_k.weight", (int) KD, {}},
                {b + "attn_v.weight", (int) KD, {}},
                {nm(b, "attn_r.weight"), c.H * c.d_rel, {}},
                {b + "attn_gate.weight", (int) HD, {}},
            };
            if (!g_dense_q8) {
                qkv_parts.push_back({"", c.H, std::move(sink_v)});
                Ly.wqkvrg = load_fused_f16(S, qkv_parts, (int) C);
            } else {
                if (!try_load_fused_raw(S, qkv_parts, (int) C, Ly.wqkvrg)) {
                    Ly.wq = load_mat(S, b + "attn_q.weight", (int) HD, (int) C);
                    Ly.wk = load_mat(S, b + "attn_k.weight", (int) KD, (int) C);
                    Ly.wv = load_mat(S, b + "attn_v.weight", (int) KD, (int) C);
                    Ly.wr = load_mat(S, nm(b, "attn_r.weight"), c.H * c.d_rel, (int) C);
                    Ly.wgate = load_mat(S, b + "attn_gate.weight", (int) HD, (int) C);
                }
                Ly.sink16 = dev_f16(sink_v);
            }
        } else {
            std::vector<MatPart> qrg_parts = {
                {b + "attn_q.weight", (int) HD, {}},
                {nm(b, "attn_r.weight"), c.H * c.d_rel, {}},
                {b + "attn_gate.weight", (int) HD, {}},
            };
            auto kd_v = dyn_host(b, "shortconv_k_dyn.weight");
            auto vd_v = dyn_host(b, "shortconv_v_dyn.weight");
            if (!g_dense_q8) {
                qrg_parts.push_back({"", c.H, std::move(sink_v)});
                Ly.wqrg = load_fused_f16(S, qrg_parts, (int) C);
                Ly.wk_dyn = load_fused_f16(S, {{b + "attn_k.weight", (int) KD, {}}, {"", c.KV * 4, std::move(kd_v)}},
                                           (int) C);
                Ly.wv_dyn = load_fused_f16(S, {{b + "attn_v.weight", (int) KD, {}}, {"", c.KV * 4, std::move(vd_v)}},
                                           (int) C);
            } else {
                if (!try_load_fused_raw(S, qrg_parts, (int) C, Ly.wqrg)) {
                    Ly.wq = load_mat(S, b + "attn_q.weight", (int) HD, (int) C);
                    Ly.wr = load_mat(S, nm(b, "attn_r.weight"), c.H * c.d_rel, (int) C);
                    Ly.wgate = load_mat(S, b + "attn_gate.weight", (int) HD, (int) C);
                }
                Ly.wk = load_mat(S, b + "attn_k.weight", (int) KD, (int) C);
                Ly.wv = load_mat(S, b + "attn_v.weight", (int) KD, (int) C);
                Ly.sink16 = dev_f16(sink_v);
                Ly.sc_k_dyn16 = dev_f16(kd_v);
                Ly.sc_v_dyn16 = dev_f16(vd_v);
            }
        }
        Ly.wo = load_mat(S, b + "attn_output.weight", (int) C, (int) HD);
        // IK2_BATCH_DENSE_MV_MATS: tag each dense projection for per-mat A/B gating (unset/null mats are inert).
        Ly.wqkvrg.batch_mv_cat = 1; Ly.wqrg.batch_mv_cat = 2; Ly.wq.batch_mv_cat = 2;
        Ly.wk.batch_mv_cat = 3; Ly.wv.batch_mv_cat = 3; Ly.wk_dyn.batch_mv_cat = 3; Ly.wv_dyn.batch_mv_cat = 3;
        Ly.wo.batch_mv_cat = 4; Ly.wr.batch_mv_cat = 5; Ly.wgate.batch_mv_cat = 5;
        Ly.router16 = load_f16(S, b + "ffn_gate_inp.weight", (size_t) c.E * C);
        Ly.probs_b = load_f32(S, b + "exp_probs_b.bias", c.E);
        Ly.ffn_norm = load_f32(S, b + "ffn_norm.weight", C);
        if (Ly.sliding) {
            Ly.attn_norm = load_f32(S, b + "attn_norm.weight", C);
        } else {
            Ly.q_in_norm = load_f32(S, nm(b, "attn_q_in_norm.weight"), C);
            Ly.k_in_norm = load_f32(S, nm(b, "attn_k_in_norm.weight"), C);
            Ly.v_in_norm = load_f32(S, nm(b, "attn_v_in_norm.weight"), C);
        }
        Ly.q_clip = load_f32(S, b + "attn_q_clip.weight", HD);
        Ly.r_clip = load_f32(S, nm(b, "attn_r_clip.weight"), (size_t) c.H * c.d_rel);
        {   // [extent][d_rel]: PR #32187's attn_rel_proj is [d_rel][extent], the rewrite's attn_rel_bias already this
            const int ext = Ly.sliding ? c.extent_swa : c.extent;
            const auto v = host_f32(S, nm(b, "attn_rel_proj.weight"), (size_t) ext * c.d_rel);
            std::vector<float> t(v.size());
            if (rw) {
                t = v;
            } else {
                for (int r = 0; r < c.d_rel; ++r)
                    for (int e = 0; e < ext; ++e) t[(size_t) e * c.d_rel + r] = v[(size_t) r * ext + e];
            }
            Ly.rel_proj = dev_f32(t);
        }
        if (l > 0) {
            const std::string rn = nm(b, "attn_res_score.weight");
            const auto* ti = S.info(rn);
            if (!ti) die(rn);
            Ly.n_res_q = (int) (ti->elements() / C);
            Ly.res_score = load_f32(S, rn, (size_t) Ly.n_res_q * C);
        }
        Ly.ffn_res_score = load_f32(S, nm(b, "ffn_res_score.weight"), C);
        Ly.sc_k = conv_w(b, "shortconv_k.weight", KD, !Ly.sliding);
        Ly.sc_v = conv_w(b, "shortconv_v.weight", KD, !Ly.sliding);
        Ly.sc_attn = conv_w(b, "shortconv_attn.weight", C, false);
        Ly.sc_mlp = conv_w(b, "shortconv_mlp.weight", C, false);
        // the experts: (role tensor, its roles); a fused gate_up is gate's F rows then up's, an expert's 2 F rows
        std::vector<std::pair<std::string, std::vector<int>>> parts;
        if (rw) parts = {{"gate_up", {0, 1}}, {"down", {2}}};
        else parts = {{"gate", {0}}, {"up", {1}}, {"down", {2}}};
        for (const auto& pr : parts) {
            const std::string n = b + "ffn_" + pr.first + "_exps.weight";
            const auto* ti = S.info(n);
            if (!ti) die("no " + n);
            const int nr = (int) pr.second.size();
            if (ti->elements() != (uint64_t) c.E * c.F * C * nr) die(n + ": shape");
            if (!k::block_values((int) ti->type)) die(n + ": type " + ti->type_name() + " not decoded here");
            if (C % k::block_values((int) ti->type) || c.F % k::block_values((int) ti->type))
                die(n + ": a row is not whole " + ti->type_name() + " blocks");
            const size_t per = tensor_bytes(*ti) / c.E;
            std::vector<float> sc;
            const std::string sn = b + "ffn_" + pr.first + "_exps.scale";
            if (S.info(sn)) sc = host_f32(S, sn, c.E);
            else if (ti->type == k::kNVFP4) die(n + ": NVFP4 without its .scale");
            for (int i = 0; i < nr; ++i) {
                const int r = pr.second[i];
                Ly.ty[r] = (int) ti->type;
                Ly.eb[r] = per / nr;
                Ly.stride[r] = per;
                Ly.scale[r] = sc;
            }
            host_total += tensor_bytes(*ti);
            exp_jobs.push_back({l, n, pr.second});
        }
        // IK2_INT4_EXPERTS: load rotated-int4 expert weights from int4-bin (replace NVFP4 residency)
        if (g_int4_experts && l < g_int4_layers) {
            const char* dir_c = getenv("IK2_INT4_DIR");
            const std::string idir = dir_c ? dir_c : "int4-bin";
            const char* roles[2] = {"gate_up", "down"};
            const int rr[2][3] = {{0, 1, -1}, {-1, -1, 2}};
            bool ok = true;
            for (int q = 0; q < 2; ++q) {
                const std::string pre = idir + "/L" + (l < 10 ? "0" : "") + std::to_string(l) + "_" + roles[q] + "_";
                const int role = rr[q][0] >= 0 ? rr[q][0] : rr[q][2];
                const int M = (q == 0) ? 2 * c.F : C;
                const int Kpad = (q == 0) ? C : ((c.F + 255) / 256) * 256;
                const size_t wpsz = (size_t) c.E * (M / 2) * Kpad;   // per-row Wp bytes = M*Kpad/2
                const size_t whsz = (size_t) c.E * (size_t) (M / 16) * (Kpad / 256) * 64;
                const size_t wssz = (size_t) c.E * M * sizeof(float);
                uint8_t* wp = dalloc<uint8_t>(wpsz);
                uint8_t* wh = dalloc<uint8_t>(whsz);
                float* ws = dalloc<float>(c.E * M);
                auto rd = [&](const std::string& fn, void* dst, size_t n) -> bool {
                    int fd = ::open(fn.c_str(), O_RDONLY);
                    if (fd < 0) return false;
                    size_t got = 0;
                    while (got < n) {
                        ssize_t r = ::pread(fd, (uint8_t*) dst + got, n - got, (int64_t) got);
                        if (r <= 0) break;
                        got += (size_t) r;
                    }
                    ::close(fd);
                    return got == n;
                };
                // host stage then upload (file sizes: Wp from manifest, Ch MSVC-safe)
                std::vector<uint8_t> hp(wpsz), hh(whsz); std::vector<float> hs(c.E * M);
                ok = ok && rd(pre + "Wp.bin", hp.data(), wpsz) && rd(pre + "Wh.bin", hh.data(), whsz) && rd(pre + "WS.bin", hs.data(), wssz);
                if (!ok) { std::printf("INT4: missing %s\n", pre.c_str()); break; }
                CU(cudaMemcpy(wp, hp.data(), wpsz, cudaMemcpyHostToDevice));
                CU(cudaMemcpy(wh, hh.data(), whsz, cudaMemcpyHostToDevice));
                CU(cudaMemcpy(ws, hs.data(), wssz, cudaMemcpyHostToDevice));
                Ly.i4_Wp[role] = wp; Ly.i4_Wh[role] = wh; Ly.i4_WS[role] = ws;
                Ly.i4_wp_stride[role] = wpsz / c.E; Ly.i4_wh_stride[role] = whsz / c.E;
                Ly.i4_Kpad[role] = Kpad;
                std::vector<const uint8_t*> tp(c.E), th(c.E); std::vector<const float*> ts(c.E);
                for (int e = 0; e < c.E; ++e) { tp[e] = wp + (size_t) e * Ly.i4_wp_stride[role]; th[e] = wh + (size_t) e * Ly.i4_wh_stride[role]; ts[e] = ws + (size_t) e * M; }
                Ly.i4_tab_Wp[role] = dalloc<const uint8_t*>(c.E); Ly.i4_tab_Wh[role] = dalloc<const uint8_t*>(c.E); Ly.i4_tab_WS[role] = dalloc<const float*>(c.E);
                CU(cudaMemcpy(Ly.i4_tab_Wp[role], tp.data(), c.E * sizeof(void*), cudaMemcpyHostToDevice));
                CU(cudaMemcpy(Ly.i4_tab_Wh[role], th.data(), c.E * sizeof(void*), cudaMemcpyHostToDevice));
                CU(cudaMemcpy(Ly.i4_tab_WS[role], ts.data(), c.E * sizeof(void*), cudaMemcpyHostToDevice));
            }
            Ly.i4_loaded = ok;
        }
        // IK2_INT4_DENSE: rotated-int4 sidecars for the dense projections' DECODE matvecs (default OFF). The Q8_0
        // planar copies stay resident for prefill; decode's mv1/mv1c take the sidecar when present. Same k_repack
        // layout as the experts (Wp nibble tiles / Wh 4-bit sub-scales / WS per-row), one matrix per role.
        if (g_int4_dense && g_dense_q8) {
            const char* dir_c = getenv("IK2_INT4_DENSE_DIR");
            const std::string ddir = dir_c ? dir_c : "int4-dense-bin";
            const std::string pre = ddir + "/L" + (l < 10 ? "0" : "") + std::to_string(l) + "_";
            auto rd = [&](const std::string& fn, void* dst, size_t n) -> bool {
                int fd = ::open(fn.c_str(), O_RDONLY);
                if (fd < 0) return false;
                size_t got = 0;
                while (got < n) {
                    ssize_t r = ::pread(fd, (uint8_t*) dst + got, n - got, (int64_t) got);
                    if (r <= 0) break;
                    got += (size_t) r;
                }
                ::close(fd);
                return got == n;
            };
            auto attach = [&](Mat& m, const char* role, int N, int K) {
                if (m.N != N || m.K != K || N % 16 || K % 256 || (K / 256 != 9 && K / 256 != 12)) {
                    std::printf("INT4-DENSE: L%d %s shape %dx%d unsupported, Q8 fallback\n", l, role, m.N, m.K);
                    return;
                }
                const size_t wpsz = (size_t) N * K / 2, whsz = (size_t) (N / 16) * (K / 256) * 64, wssz = (size_t) N * sizeof(float);
                std::vector<uint8_t> hp(wpsz), hh(whsz); std::vector<float> hs(N);
                const std::string p = pre + role + "_";
                if (!rd(p + "Wp.bin", hp.data(), wpsz) || !rd(p + "Wh.bin", hh.data(), whsz) || !rd(p + "WS.bin", hs.data(), wssz)) {
                    std::printf("INT4-DENSE: missing %s*, Q8 fallback\n", p.c_str());
                    return;
                }
                uint8_t* wp = dalloc<uint8_t>(wpsz); uint8_t* wh = dalloc<uint8_t>(whsz); float* ws = dalloc<float>(N);
                CU(cudaMemcpy(wp, hp.data(), wpsz, cudaMemcpyHostToDevice));
                CU(cudaMemcpy(wh, hh.data(), whsz, cudaMemcpyHostToDevice));
                CU(cudaMemcpy(ws, hs.data(), wssz, cudaMemcpyHostToDevice));
                m.i4Wp = wp; m.i4Wh = wh; m.i4WS = ws;
            };
            if (Ly.sliding) attach(Ly.wqkvrg, "wqkvrg", (int) (2 * HD + 2 * KD + c.H * c.d_rel), (int) C);
            else {
                attach(Ly.wqrg, "wqrg", (int) (2 * HD + c.H * c.d_rel), (int) C);
                attach(Ly.wk, "wk", (int) KD, (int) C);
                attach(Ly.wv, "wv", (int) KD, (int) C);
            }
            attach(Ly.wo, "wo", (int) C, (int) HD);
        }
        if (c.n_pass > 1 && l >= c.core0 && l < c.core1) {   // the core: its later passes' message gates and queries
            static const char* mod[4] = {"attn", "k", "v", "mlp"};
            for (int m = 0; m < 4; ++m) {
                const std::string p = b + "loop_" + mod[m] + "_";
                Ly.lp_down[m] = load_f32(S, p + "down.weight", (size_t) c.rank * 2 * C);
                for (int fg = 0; fg < 2; ++fg) {   // (the file's [C][rank] by rank, [rank][C]: k::loop_gate's layout)
                    const auto w = host_f32(S, p + (fg ? "gain.weight" : "forget.weight"), C * c.rank);
                    std::vector<float> t(w.size());
                    for (size_t ch = 0; ch < C; ++ch)
                        for (int r = 0; r < c.rank; ++r) t[(size_t) r * C + ch] = w[ch * c.rank + r];
                    (fg ? Ly.lp_gain[m] : Ly.lp_forget[m]) = dev_f32(t);
                }
            }
            // loop_res_query: rows 2 i (q's) and 2 i + 1 (the MoE's) of core layer i; loop_kv_res_query: k's and v's
            const size_t i = (size_t) (l - c.core0);
            std::vector<float> q(4 * C);
            std::memcpy(q.data(), lp_q.data() + 2 * i * C, C * 4);
            std::memcpy(q.data() + C, lp_kv.data() + 2 * i * C, C * 4);
            std::memcpy(q.data() + 2 * C, lp_kv.data() + (2 * i + 1) * C, C * 4);
            std::memcpy(q.data() + 3 * C, lp_q.data() + (2 * i + 1) * C, C * 4);
            Ly.lp_query = dev_f32(q);
            lp_all.insert(lp_all.end(), q.begin(), q.end());
            if (l == c.core1 - 1) M.lp_query_all = dev_f32(lp_all);
        }
    }
    // the visits of a token and their state (FP8 E4M3 or FP16 KV; sliding layers ring-buffered in both CLI and --serve)
    vram_mark("before-KV-visits");
    M.V.resize(c.n_visit);
    for (int v = 0; v < c.n_visit; ++v) {
        Visit& Vi = M.V[v];
        if (c.n_pass == 1 || v < c.core0) Vi.l = v;
        else if (v < c.core0 + c.n_pass * n_core) Vi.l = c.core0 + (v - c.core0) % n_core, Vi.pass = (v - c.core0) / n_core;
        else Vi.l = v - (c.n_pass - 1) * n_core;
        const Layer& Ly = M.L[Vi.l];
        use_dev(Ly.dev);
        Vi.kv_cap = Ly.sliding ? std::min(ctx, c.window + std::max(chunk, ik2::draft::kBlock)) : ctx;
        if (kv_is_int()) {
            const size_t code_b = (size_t) Vi.kv_cap * KD * kv_int_bits() / 8;
            const size_t scale_n = (size_t) Vi.kv_cap * c.KV;
            // IK2_KV_HOST=1: the full-attention visits' KV (kv_cap == ctx, the ctx-scaled hog) lives in host-mapped
            // RAM so 1M fits beside the experts; sliding visits stay VRAM-resident (small, read every token).
            if (g_kv_host && !Ly.sliding) {
                void* hk = nullptr; void* hv = nullptr; void* hks = nullptr; void* hvs = nullptr;
                Vi.kci = halloc_mapped<uint8_t>(code_b, &hk);
                Vi.vci = halloc_mapped<uint8_t>(code_b, &hv);
                Vi.ks = halloc_mapped<uint16_t>(scale_n, &hks);
                Vi.vs = halloc_mapped<uint16_t>(scale_n, &hvs);
                Vi.hkci = (uint8_t*) hk; Vi.hvci = (uint8_t*) hv; Vi.hks = (uint16_t*) hks; Vi.hvs = (uint16_t*) hvs;
                Vi.kv_host_cap = Vi.kv_cap;
            } else {
                Vi.kci = dalloc<uint8_t>(code_b);
                Vi.vci = dalloc<uint8_t>(code_b);
                Vi.ks = dalloc<uint16_t>(scale_n);
                Vi.vs = dalloc<uint16_t>(scale_n);
            }
        } else if (g_kv_fp8) {
            Vi.kc8 = dalloc<uint8_t>((size_t) Vi.kv_cap * KD);
            Vi.vc8 = dalloc<uint8_t>((size_t) Vi.kv_cap * KD);
        } else {
            Vi.kc = dalloc<uint16_t>((size_t) Vi.kv_cap * KD);
            Vi.vc = dalloc<uint16_t>((size_t) Vi.kv_cap * KD);
        }
        Vi.h_k = dalloc<float>(3 * KD);
        Vi.h_v = dalloc<float>(3 * KD);
        Vi.h_attn = dalloc<float>(3 * C);
        Vi.h_mlp = dalloc<float>(3 * C);
    }
    if (g_batch_decode) {
        // IK2_BATCH_ATTN: real extra VRAM -- one complete KV+conv state per (slot, visit).  Host-mapped KV,
        // FP8/FP16 KV, CPU-offload and multi-GPU are explicitly unsupported; falling back keeps those paths exact.
        if (g_parallel < 2 || g_dev.size() != 1 || g_ep || g_tp || !kv_is_int() || g_kv_host) {
            fprintf(stderr, "ik2: IK2_BATCH_ATTN=1 ignored (requires IK2_PARALLEL>=2, one GPU, INT KV in VRAM)\n");
            g_batch_decode = false;
        } else {
            use_dev(0);
            BatchSlots& B = g_batch_slots;
            B.n_slot = std::min(g_parallel, BatchSlots::kMaxRows);
            B.n_visit = c.n_visit;
            const size_t nv = (size_t) B.n_visit, ns = (size_t) B.n_slot;
            const int bits = kv_int_bits();
            B.kci.resize(nv * ns); B.vci.resize(nv * ns);
            B.ks.resize(nv * ns);  B.vs.resize(nv * ns);
            B.h_k.resize(nv * ns); B.h_v.resize(nv * ns);
            B.h_attn.resize(nv * ns); B.h_mlp.resize(nv * ns);
            B.bytes = 0;
            for (int v = 0; v < c.n_visit; ++v) {
                const Visit& Vi = M.V[v];
                const size_t code_b = (size_t) Vi.kv_cap * KD * bits / 8;
                const size_t scale_b = (size_t) Vi.kv_cap * c.KV * sizeof(uint16_t);
                const size_t conv_b = 3 * KD * sizeof(float), conv_c = 3 * C * sizeof(float);
                B.bytes += ns * (2 * code_b + 2 * scale_b + 2 * conv_b + 2 * conv_c);
                for (int s = 0; s < B.n_slot; ++s) {
                    const size_t i = B.idx(s, v);
                    B.kci[i] = dalloc<uint8_t>(code_b); B.vci[i] = dalloc<uint8_t>(code_b);
                    B.ks[i]  = dalloc<uint16_t>((size_t) Vi.kv_cap * c.KV);
                    B.vs[i]  = dalloc<uint16_t>((size_t) Vi.kv_cap * c.KV);
                    B.h_k[i] = dalloc<float>(3 * KD); B.h_v[i] = dalloc<float>(3 * KD);
                    B.h_attn[i] = dalloc<float>(3 * C); B.h_mlp[i] = dalloc<float>(3 * C);
                    CU(cudaMemset(B.h_k[i], 0, conv_b)); CU(cudaMemset(B.h_v[i], 0, conv_b));
                    CU(cudaMemset(B.h_attn[i], 0, conv_c)); CU(cudaMemset(B.h_mlp[i], 0, conv_c));
                }
            }
            printf("ik2: IK2_BATCH_ATTN: %d slots x %d visits of extra KV/conv VRAM = %.2f GiB\n",
                   B.n_slot, B.n_visit, B.bytes / 1073741824.0);
        }
    }
    vram_mark("after-KV-visits");
    c.n_rows = (int) (S.info("output.weight")->elements() / C);
    // IK2_INT4_HEAD_DECODE: skip the head's Q8/raw copy entirely; the INT4 sidecar (loaded below) is the only
    // weight copy. Saves ~490 MiB (host-mapped pinned RAM or VRAM) so the 1M-ctx budget keeps the expert cache.
    M.out_w = load_mat(S, "output.weight", c.n_rows, (int) C, false, false);
    vram_mark("after-out_w-load");
    // IK2_INT4_HEAD: rotated-int4 sidecar for the verify HEAD GEMM (out_w, T=2..8), same k_repack layout as the
    // dense sidecars (gw4 sh12). i4_head_only keeps it out of mv1/mv1c and the DENSE_MULTI gate: T=1 decode and the
    // prompt path keep their existing route. The drafter's fold/propose read M.out_w.f16 (the f16 weight copy built
    // below when --draft), which stays resident and untouched; the INT4 GEMM only writes W.logits (f32).
    if (g_int4_head && g_dense_q8) {
        const char* dir_c = getenv("IK2_INT4_DENSE_DIR");
        const std::string ddir = dir_c ? dir_c : "int4-dense-bin";
        const int N = M.out_w.N, K = M.out_w.K;
        auto rd = [&](const std::string& fn, void* dst, size_t n) -> bool {
            int fd = ::open(fn.c_str(), O_RDONLY);
            if (fd < 0) return false;
            size_t got = 0;
            while (got < n) {
                ssize_t r = ::pread(fd, (uint8_t*) dst + got, n - got, (int64_t) got);
                if (r <= 0) break;
                got += (size_t) r;
            }
            ::close(fd);
            return got == n;
        };
        if (N % 16 || K % 256 || (K / 256 != 9 && K / 256 != 12)) {
            std::printf("INT4-HEAD: out_w shape %dx%d unsupported, f16/Q8 fallback\n", N, K);
        } else {
            const size_t wpsz = (size_t) N * K / 2, whsz = (size_t) (N / 16) * (K / 256) * 64, wssz = (size_t) N * sizeof(float);
            std::vector<uint8_t> hp(wpsz), hh(whsz); std::vector<float> hs(N);
            const std::string p = ddir + "/out_w_";
            if (!rd(p + "Wp.bin", hp.data(), wpsz) || !rd(p + "Wh.bin", hh.data(), whsz) || !rd(p + "WS.bin", hs.data(), wssz)) {
                std::printf("INT4-HEAD: missing %s*, f16/Q8 fallback\n", p.c_str());
            } else {
                // IK2_INT4_HEAD_DECODE: the decode head reads this VRAM sidecar, never the Q8 blocks. Free the Q8
                // head FIRST (host-mapped ~462 MiB pinned RAM, or VRAM) so the sidecar fits the 1M-ctx budget.
                if (g_int4_head_decode && M.out_w.q8) {
                    cudaFree(M.out_w.q8); cudaFree(M.out_w.q8d);
                    M.out_w.q8 = nullptr; M.out_w.q8d = nullptr;
                    std::printf("INT4-HEAD-DECODE: freed Q8 head before sidecar (decode head over the int4 sidecar)\n");
                }
                uint8_t* wp = dalloc<uint8_t>(wpsz); uint8_t* wh = dalloc<uint8_t>(whsz); float* ws = dalloc<float>(N);
                CU(cudaMemcpy(wp, hp.data(), wpsz, cudaMemcpyHostToDevice));
                CU(cudaMemcpy(wh, hh.data(), whsz, cudaMemcpyHostToDevice));
                CU(cudaMemcpy(ws, hs.data(), wssz, cudaMemcpyHostToDevice));
                M.out_w.i4Wp = wp; M.out_w.i4Wh = wh; M.out_w.i4WS = ws;
                M.out_w.i4_head_only = true;
                std::printf("INT4-HEAD: out_w sidecar %dx%d (%.1f MiB)\n", N, K, (double) (wpsz + whsz + wssz) / (1024.0 * 1024.0));
            }
        }
    }
    if (g_dense_q8) {   // (a layer's largest; the head's logits go a token at a time over its blocks)
        g_w16_n = (2 * HD + 2 * KD + (size_t) c.H * c.d_rel) * C;
        g_w16 = dalloc<uint16_t>(g_w16_n);
    }
    M.out_norm = load_f32(S, "output_norm.weight", C);
    M.out_res_score = load_f32(S, rw ? "output_res_query.weight" : "output_res_score.weight", C);
    M.embd = S.data("token_embd.weight");
    M.embd_type = (int) S.info("token_embd.weight")->type;
    if (!k::block_values(M.embd_type)) die("token_embd's type");
    if (g_embd_gpu) {
        // IK2_EMBD_GPU / IK2_AR_KSTEP: one device copy of the host Q8_0 embedding table.  Use raw cudaMalloc so an
        // allocation failure can gracefully disable KSTEP instead of exiting from dalloc's CU check.  The EC budget
        // below is computed from free VRAM afterwards, so this reservation is accounted for automatically.
        use_dev(0);
        const strata::TensorInfo* ti = S.info("token_embd.weight");
        const size_t emb_bytes = tensor_bytes(*ti);
        const int emb_rows = (int) (ti->elements() / (size_t) C);
        if (emb_rows < c.n_vocab || emb_bytes < (size_t) emb_rows) {
            std::fprintf(stderr, "ik2: embedding table too small (%d rows, vocab %d); disabling KSTEP\n",
                         emb_rows, c.n_vocab);
            g_ar_kstep = false; g_ar_kstep_ready = false; g_embd_gpu = false;
        } else {
            M.embd_rows = emb_rows;
            M.embd_row_bytes = (size_t) emb_bytes / (size_t) emb_rows;
            uint8_t* emb_d = nullptr;
            cudaError_t ce = cudaMalloc((void**) &emb_d, emb_bytes);
            if (ce == cudaSuccess) ce = cudaMemcpy(emb_d, M.embd, emb_bytes, cudaMemcpyHostToDevice);
            if (ce == cudaSuccess) {
                M.embd_d = emb_d;
                std::printf("ik2: embedding table on device (%.2f GiB, %d rows, %zu bytes/row)\n",
                            (double) emb_bytes / 1073741824.0, emb_rows, M.embd_row_bytes);
            } else {
                std::fprintf(stderr, "ik2: embedding upload failed (%s); disabling KSTEP\n", cudaGetErrorString(ce));
                if (emb_d) (void) cudaFree(emb_d);
                (void) cudaGetLastError();
                g_ar_kstep = false; g_ar_kstep_ready = false; g_embd_gpu = false;
            }
        }
    }
    if (g_stage) cudaFree(g_stage), g_stage = nullptr, g_stage_bytes = 0;
    vram_mark("after-dense-out_w");
    {
        size_t kv_bytes = 0;
        for (const auto& Vi : M.V) {
            if (kv_is_int())
                kv_bytes += (size_t) Vi.kv_cap * ((KD * kv_int_bits()) / 4 + 4 * c.KV);  // K+V codes + K+V fp16 scales
            else
                kv_bytes += (size_t) Vi.kv_cap * KD * (g_kv_fp8 ? 1 : 2) * 2;
        }
        const char* kv_name = kv_is_int() ? (kv_int_bits() == 4 ? "INT4 (per-head fp16 scale)" : "INT8 (per-head fp16 scale)") : (g_kv_fp8 ? "FP8 E4M3" : "FP16");
        std::printf("kv cache: %s (%.2f GiB across %d layer%s)\n", kv_name,
                    kv_bytes / 1073741824.0, c.n_visit, c.n_pass > 1 ? " visits" : "s");
        std::fflush(stdout);
    }
    size_t fr = 0, tot = 0;
    for (int d = 0; d < n_gpu; ++d) {
        use_dev(d);
        CU(cudaMemGetInfo(&fr, &tot));
        std::printf("dense weights in VRAM: %.1f s; GPU %d: %.1f GiB of %.1f free\n", now() - t_dense, d,
                    fr / 1073741824.0, tot / 1073741824.0);
        std::fflush(stdout);
    }

    // the experts into pinned host memory: the allocations, then the reads, several files at once.  IK2_HOST=mmap:
    // read where the files are mapped instead (pageable: slower copies, no 323 GiB of host memory - for several
    // GPUs whose caches hold most of the experts)
    const double t_host = now();
    Pinned pin;
    double t_alloc = 0;
    // (the resident path keeps every expert in VRAM, so no pinned host copy is needed: the experts go from where the
    //  files are mapped straight to their slots, and stay in the page cache for the next load - IK2_HOST=pinned: as
    //  for the other models)
#ifdef _WIN32
    // Windows restricts physical memory pinning (cudaHostAlloc fails with OOM for multi-gigabyte models).
    // Default to direct mmap reading unless IK2_HOST=pinned is explicitly requested.
    const bool host_mmap = !getenv_str("IK2_HOST").empty() ? getenv_str("IK2_HOST") != "pinned" : true;
#else
    const bool host_mmap = !getenv_str("IK2_HOST").empty() ? getenv_str("IK2_HOST") == "mmap" : g_fast_want;
#endif
    if (host_mmap) {
        if (g_pool) die("IK2_HOST=mmap with the CPU lane");
        for (const auto& J : exp_jobs) {
            Layer& Ly = M.L[J.l];
            const uint8_t* p = S.data(J.name);
            for (size_t i = 0; i < J.roles.size(); ++i) Ly.host[J.roles[i]] = p + i * Ly.eb[J.roles[i]];
            {   // (read ahead: a cold file otherwise comes in a page fault at a time)
                const size_t pg = 4096, a0 = (size_t) p & ~(pg - 1);
                ::madvise((void*) a0, (size_t) p - a0 + tensor_bytes(*S.info(J.name)), MADV_WILLNEED);
            }
        }
        std::printf("experts: read where the files are mapped (IK2_HOST=mmap)\n");
        std::fflush(stdout);
        vram_mark("after-expert-host-mmap");
    } else {
        // Pinned host memory: anonymous mappings read into and then registered with CUDA, a layer as soon as its
        // reads are done (323 GiB in ~60 s from VAST); IK2_ALLOC=hostalloc: cudaHostAlloc instead, which pins at
        // ~3 GiB/s (the driver serializes it; 110-190 s)
#ifndef _WIN32
        bool use_mmap_reg = !(getenv("IK2_ALLOC") && std::string(getenv("IK2_ALLOC")) == "hostalloc");
        if (use_mmap_reg) {
            const size_t probe_bytes = 2ULL << 20;
            void* mp = ::mmap(nullptr, probe_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (mp == MAP_FAILED) {
                use_mmap_reg = false;
            } else {
                *((volatile uint8_t*) mp) = 0;
                if (cudaHostRegister(mp, probe_bytes, cudaHostRegisterPortable) != cudaSuccess) {
                    cudaGetLastError();
                    use_mmap_reg = false;
                } else {
                    CU(cudaHostUnregister(mp));
                }
                ::munmap(mp, probe_bytes);
            }
        }
        pin.blocks.resize(c.n_layer);
        if (use_mmap_reg) {
            for (int l = 0; l < c.n_layer; ++l) {
                const Layer& Ly = M.L[l];
                const size_t slot_bytes = Ly.eb[0] + Ly.eb[1] + Ly.eb[2];
                const size_t layer_bytes = (size_t) c.E * slot_bytes;
                void* p = ::mmap(nullptr, layer_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (p == MAP_FAILED) {
                    for (int k = 0; k < l; ++k) ::munmap(pin.blocks[k].ptr, pin.blocks[k].bytes);
                    pin.blocks.assign(c.n_layer, {});
                    use_mmap_reg = false;
                    break;
                }
                pin.blocks[l] = {p, layer_bytes, true, false};
            }
        }
#else
        bool use_mmap_reg = false;
        pin.blocks.resize(c.n_layer);
#endif
        // (pinned: the allocations and the reads overlap - a layer's reads start as soon as its buffer exists;
        // IK2_ALLOC_THREADS threads allocate, default 1)
        std::vector<std::atomic<int>> ready(c.n_layer);
        for (auto& r : ready) r.store(use_mmap_reg ? 1 : 0);
        for (int l = 0; l < c.n_layer; ++l) {
            Layer& Ly = M.L[l];
            const size_t slot_bytes = Ly.eb[0] + Ly.eb[1] + Ly.eb[2];
            Ly.host_contig = true;
            Ly.slot_bytes = slot_bytes;
            Ly.stride[0] = Ly.stride[1] = Ly.stride[2] = slot_bytes;
        }
        std::vector<std::thread> alloc_th;
        std::atomic<int> next_alloc{0};
        double t_alloc_end = 0;
        std::atomic<int> alloc_left{use_mmap_reg ? 0 : c.n_layer};
        if (!use_mmap_reg) {
            const int na = std::max(1, getenv("IK2_ALLOC_THREADS") ? std::atoi(getenv("IK2_ALLOC_THREADS")) : 1);
            for (int t = 0; t < na; ++t)
                alloc_th.emplace_back([&]() {
                    for (;;) {
                        const int l = next_alloc++;
                        if (l >= c.n_layer) return;
                        const size_t layer_bytes = (size_t) c.E * M.L[l].slot_bytes;
                        void* p = nullptr;
                        CU(cudaHostAlloc(&p, layer_bytes, cudaHostAllocPortable));
                        pin.blocks[l] = {p, layer_bytes, false, false};
                        ready[l].store(1, std::memory_order_release);
                        if (--alloc_left == 0) t_alloc_end = now();
                    }
                });
        }
        auto wait_layer = [&](int l) {
            while (!ready[l].load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            Layer& Ly = M.L[l];
            uint8_t* buf = (uint8_t*) pin.blocks[l].ptr;
            Ly.host[0] = buf;
            Ly.host[1] = buf + Ly.eb[0];
            Ly.host[2] = buf + Ly.eb[0] + Ly.eb[1];
        };
        std::vector<std::atomic<int>> jobs_left(c.n_layer);
        for (auto& v : jobs_left) v.store(0);
        for (const auto& J : exp_jobs) ++jobs_left[J.l];
        std::mutex reg_m;
        std::condition_variable reg_cv;
        std::deque<int> reg_q;
        int reg_fail = 0;
        std::thread reg_th;
        if (use_mmap_reg)
            reg_th = std::thread([&]() {   // registers each layer once its reads are done, while the others are read
                for (int done = 0; done < c.n_layer; ++done) {
                    int l;
                    {
                        std::unique_lock<std::mutex> lk(reg_m);
                        reg_cv.wait(lk, [&] { return !reg_q.empty(); });
                        l = reg_q.front();
                        reg_q.pop_front();
                    }
                    if (cudaHostRegister(pin.blocks[l].ptr, pin.blocks[l].bytes, cudaHostRegisterPortable) == cudaSuccess)
                        pin.blocks[l].registered = true;
                    else
                        cudaGetLastError(), ++reg_fail;
                }
            });
        auto layer_read = [&](int l) {
            if (!use_mmap_reg) return;
            {
                std::lock_guard<std::mutex> lk(reg_m);
                reg_q.push_back(l);
            }
            reg_cv.notify_one();
        };
        for (int l = 0; l < c.n_layer; ++l)
            if (jobs_left[l].load() == 0) layer_read(l);
        std::vector<std::thread> th;
        std::atomic<int> next{0};
        for (int t = 0; t < threads; ++t)
            th.emplace_back([&]() {
                for (;;) {
                    const int j = next++;
                    if (j >= (int) exp_jobs.size()) return;
                    const auto& J = exp_jobs[j];
                    wait_layer(J.l);
                    const Layer& Ly = M.L[J.l];
                    const size_t role_off = J.roles[0] == 0 ? 0 : (J.roles[0] == 1 ? Ly.eb[0] : Ly.eb[0] + Ly.eb[1]);
                    const size_t exp_bytes = tensor_bytes(*S.info(J.name)) / c.E;
                    uint8_t* dst = (uint8_t*) pin.blocks[J.l].ptr + role_off;
                    read_into_strided(S.paths[S.file_of(J.name)], S.file_offset(J.name), c.E, exp_bytes,
                                      Ly.slot_bytes, dst);
                    if (--jobs_left[J.l] == 0) layer_read(J.l);
                }
            });
        for (auto& t : alloc_th) t.join();
        for (auto& t : th) t.join();
        for (int l = 0; l < c.n_layer; ++l) wait_layer(l);   // (host[] of layers no job read)
        t_alloc = (use_mmap_reg ? 0.0 : t_alloc_end - t_host);
        const double t_read = now() - t_host - t_alloc;   // (the reads left after the last allocation)
        double t_pin = 0;
        if (use_mmap_reg) {
            const double tp0 = now();
            reg_th.join();   // (the layers read last are still registering)
            t_pin = now() - tp0;
            if (reg_fail > 0)
                std::fprintf(stderr, "strata-ik2: warning: cudaHostRegister failed on %d layers (running pageable)\n",
                             reg_fail);
        }
        if (use_mmap_reg) {
            std::printf("experts in pinned (mmap+register) memory: %.1f GiB; allocating %.1f s, reading %.1f s "
                        "(%.2f GiB/s), pinning %.1f s\n",
                        host_total / 1073741824.0, t_alloc, t_read, host_total / 1073741824.0 / t_read, t_pin);
        } else {
            std::printf("experts in pinned memory: %.1f GiB; allocating %.1f s, reading %.1f s (%.2f GiB/s)\n",
                        host_total / 1073741824.0, t_alloc, t_read, host_total / 1073741824.0 / t_read);
        }
        std::fflush(stdout);
    }

    // the drafter (before the expert cache takes what VRAM is left)
    ik2::draft::Drafter* drafter = nullptr;
    if (!draft_path.empty()) {
        if (n_gpu > 2 || (n_gpu == 2 && !g_ep)) die("--draft: single GPU or dual-GPU EP only so far");
        use_dev(0);
        if (!M.out_w.f16) {
            const auto* t = S.info("output.weight");
            const size_t nb = tensor_bytes(*t);
            if (nb > g_stage_bytes) {
                if (g_stage) cudaFree(g_stage);
                g_stage = dalloc<uint8_t>(g_stage_bytes = nb);
            }
            CU(cudaMemcpy(g_stage, S.data("output.weight"), nb, cudaMemcpyHostToDevice));
            M.out_w.f16 = dalloc<uint16_t>((size_t) c.n_rows * c.C);
            k::decode_f16(M.out_w.type, g_stage, (size_t) c.n_rows * c.C, M.out_w.f16, g_s);
            CU(cudaStreamSynchronize(g_s));
        }
        const double t0 = now();
        try {
            drafter = ik2::draft::load(draft_path, c.C, ctx, g_blas, g_s);
            ik2::draft::fold(drafter, M.out_w.f16, c.n_rows);
            // IK2_DRAFT_HEAD_INT4: arm propose()'s base logits over the head's INT4 sidecar when it is
            // loaded (IK2_INT4_HEAD=1). Documented choice: if the sidecar is absent (IK2_INT4_HEAD off
            // or bins missing) fall back to gemm_h -- no second sidecar load here. fold keeps its f16 copy.
            if (g_draft_head_int4) {
                if (ik2::draft::set_head_int4(drafter, M.out_w.i4Wp, M.out_w.i4Wh, M.out_w.i4WS, c.n_rows)) {
                    std::printf("DRAFT-HEAD-INT4: propose base logits over the out_w int4 sidecar (NC=%d)\n",
                                ik2::draft::kBlock - 1);
                    // fold() already consumed the f16 head copy (one-time load cost); propose now reads the
                    // INT4 sidecar and the verify head uses it too, so nothing reads M.out_w.f16 anymore.
                    // Free it (~926 MiB) to leave room for the expert cache / long-context KV residency.
                    cudaFree(M.out_w.f16);
                    M.out_w.f16 = nullptr;
                } else {
                    std::printf("DRAFT-HEAD-INT4: requested but the out_w sidecar is unavailable/unsupported, "
                                "gemm_h fallback\n");
                }
            }
        } catch (const std::exception& e) {
            die(e.what());
        }
        g_taps_n = ik2::draft::taps(drafter);
        g_taps_ld = std::max(chunk, ik2::draft::kBlock);
        g_taps = dalloc<float>((size_t) g_taps_n * g_taps_ld * c.C);
        for (auto& Vi : M.V) {
            const size_t cc[4] = {KD, KD, C, C};
            for (int i = 0; i < 4; ++i) {
                Vi.rec_old[i] = dalloc<float>(3 * cc[i]);
                Vi.rec_x[i] = dalloc<float>((size_t) ik2::draft::kBlock * cc[i]);
            }
        }
        std::printf("drafter: %s, %.2f GiB of VRAM, %.1f s\n", draft_path.c_str(),
                    ik2::draft::bytes(drafter) / 1073741824.0, now() - t0);
    }

    // the work buffers and the expert cache, a set a GPU
    if (getenv("IK2_VRAM_TRACE")) { size_t f0 = 0, t0 = 0; CU(cudaMemGetInfo(&f0, &t0));
        std::printf("VRAM-TRACE before work-buffer loop: free %.2f GiB (ctx %d)\n", f0 / 1073741824.0, ctx); }
    for (int d = 0; d < n_gpu; ++d) {
    use_dev(d);
    CU(cudaMemGetInfo(&fr, &tot));
    Work& W = g_dev[d].W;
    W.Tm = std::max({chunk, 1, ik2::draft::kBlock});
    W.ctx = ctx;
    W.nk_max = std::min(ctx, ctx);
    // Tokens a layer-major prefill pass (every expert not in VRAM streams in once a pass): 16384 when there is room
    // (a pass's scratch is ~280 KB a token: 4.6 GB, taken from the expert cache), else 4096.
    // (a looped model's prompts go through forward(): forward_prefill_sup has no loop yet)
    // ... nor the resident path's models: a pass borrows cache slots, which would evict experts
    W.T_sup = ((n_gpu == 1 || dual()) && draft_path.empty() && c.n_pass == 1 && !g_fast_want && fr >= (20ULL << 30)) ? (fr >= (60ULL << 30) ? 16384 : 4096) : 0;
    // IK2_SUP=N: tokens a layer-major prefill pass (each pass streams every expert not in the cache once)
    if (W.T_sup > 0 && getenv("IK2_SUP")) W.T_sup = std::max(W.Tm, std::atoi(getenv("IK2_SUP")));
    const size_t Tm = W.Tm, S8 = Tm * c.K;
    const size_t T_buf = std::max(Tm, (size_t) W.T_sup), S_buf = T_buf * c.K;
    W.blk_ld = T_buf;
    for (float** p : {&W.h, &W.r, &W.tmp}) *p = dalloc<float>(T_buf * C);
    for (float** p : {&W.res, &W.a, &W.xq, &W.xk, &W.xv}) *p = dalloc<float>(Tm * C);
    W.q = dalloc<float>(Tm * HD), W.gate = dalloc<float>(Tm * HD), W.o = dalloc<float>(Tm * HD);
    W.kk = dalloc<float>(Tm * KD), W.vv = dalloc<float>(Tm * KD), W.kconv = dalloc<float>(Tm * KD);
    W.vconv = dalloc<float>(Tm * KD);
    W.rr = dalloc<float>(Tm * c.H * c.d_rel), W.sink = dalloc<float>(Tm * c.H);
    W.dyn = dalloc<float>(2 * Tm * c.KV * 4);   // (a full layer's dynamic taps: k's, and on the resident path v's beside)
    // IK2_LG_CTX: cap the linear-in-ctx logits scratch (W.lg / W.gu32 alias) at this many tokens instead of the
    // full --ctx. The flash decode path never touches W.lg, and prefill chunks (<= --prefill) only need scratch
    // for the chunk, not the whole context. At 1M ctx the default lg/gu32 pair is ~24 GiB; capping to the chunk
    // size frees it so the Q8 expert pool stays fully resident. Default 0 = unchanged (full ctx).
    const size_t lg_ctx = getenv("IK2_LG_CTX") ? (size_t) std::max(1, std::atoi(getenv("IK2_LG_CTX"))) : (size_t) ctx;
    W.lg_cap = std::max((size_t) 8 * c.H * lg_ctx + (size_t) 8 * c.H * 128 * 130, (size_t) 64 << 20);
    W.lg = dalloc<float>(W.lg_cap);
    W.blocks = dalloc<float>((size_t) (c.n_layer / c.block + 1) * T_buf * C);
    W.x16 = dalloc<uint16_t>(Tm * std::max(C, HD)), W.xk16 = dalloc<uint16_t>(Tm * C), W.xv16 = dalloc<uint16_t>(Tm * C);
    if (W.T_sup > 0 || dual()) W.xq16_sup = dalloc<uint16_t>(T_buf * C);
    if (!getenv("IK2_RELB") || std::atoi(getenv("IK2_RELB"))) {
        W.relb_ext = std::max(c.extent, c.extent_swa);
        W.relb = dalloc<float>((size_t) Work::kRelbRows * c.H * W.relb_ext);
    }
    if (g_kv_fp8 || kv_is_int()) {
        W.kv_tmp_cap = std::min(ctx, 32768);
        W.kv_k16 = dalloc<uint16_t>((size_t) W.kv_tmp_cap * KD);
        W.kv_v16 = dalloc<uint16_t>((size_t) W.kv_tmp_cap * KD);
    }
    W.pos = dalloc<int>(T_buf);
    if (g_batch_decode) {
        const int R = BatchSlots::kMaxRows;
        const int NS = g_batch_moe_device ? Work::kBatchStageSlots : 1;
        W.bkci_d = dalloc<uint8_t*>(R); W.bvci_d = dalloc<uint8_t*>(R);
        W.bks_d = dalloc<uint16_t*>(R); W.bvs_d = dalloc<uint16_t*>(R);
        W.bh_k_d = dalloc<float*>(R); W.bh_v_d = dalloc<float*>(R);
        W.bh_attn_d = dalloc<float*>(R); W.bh_mlp_d = dalloc<float*>(R);
        CU(cudaHostAlloc((void**) &W.bkci_h, (size_t) R * NS * sizeof(uint8_t*), cudaHostAllocDefault));
        CU(cudaHostAlloc((void**) &W.bvci_h, (size_t) R * NS * sizeof(uint8_t*), cudaHostAllocDefault));
        CU(cudaHostAlloc((void**) &W.bks_h, (size_t) R * NS * sizeof(uint16_t*), cudaHostAllocDefault));
        CU(cudaHostAlloc((void**) &W.bvs_h, (size_t) R * NS * sizeof(uint16_t*), cudaHostAllocDefault));
        CU(cudaHostAlloc((void**) &W.bh_k_h, (size_t) R * NS * sizeof(float*), cudaHostAllocDefault));
        CU(cudaHostAlloc((void**) &W.bh_v_h, (size_t) R * NS * sizeof(float*), cudaHostAllocDefault));
        CU(cudaHostAlloc((void**) &W.bh_attn_h, (size_t) R * NS * sizeof(float*), cudaHostAllocDefault));
        CU(cudaHostAlloc((void**) &W.bh_mlp_h, (size_t) R * NS * sizeof(float*), cudaHostAllocDefault));
        if (g_batch_moe_device)
            for (int i = 0; i < Work::kBatchStageSlots; ++i)
                CU(cudaEventCreateWithFlags(&W.batch_stage_ev[i], cudaEventDisableTiming));
    }
    W.rlog = dalloc<float>(T_buf * c.E), W.route_d = dalloc<int>(S_buf * 2);
    W.ex16 = dalloc<uint16_t>(T_buf * C), W.w16 = dalloc<uint16_t>((size_t) 2 * c.F * C);   // (one expert's tokens)
    W.h16 = dalloc<uint16_t>(std::max<size_t>(64, T_buf) * c.F);
    W.gu32_bytes = 4 * std::max({S8 * 2 * (size_t) c.F,
                  Tm * std::max<size_t>(2 * HD + 2 * KD + (size_t) c.H * c.d_rel + c.H,
                                        (2 * HD + (size_t) c.H * c.d_rel + c.H) + 2 * (KD + (size_t) c.KV * 4)),
                  T_buf * 2 * (size_t) c.F, (W.lg_cap + 1) / 2});
    W.gu32 = dalloc<float>(W.gu32_bytes / 4);
    if (getenv("IK2_VRAM_TRACE")) { size_t f2 = 0, t2 = 0; CU(cudaMemGetInfo(&f2, &t2));
        std::printf("VRAM-TRACE after work buffers: free %.2f GiB (lg %.2f gu32 %.2f GiB)\n", f2 / 1073741824.0,
                    W.lg_cap * 4 / 1073741824.0, W.gu32_bytes / 1073741824.0); }
    W.qkv = W.gu32;
    W.lg16 = (uint16_t*) W.gu32;
    W.y32 = dalloc<float>(std::max<size_t>(64, T_buf) * C);
    if (g_tp) W.tp_h = dalloc<uint16_t>(std::max<size_t>(64, S_buf) * c.F);   // (one expert's tokens; the token path's <= 64 rows)
    W.wtr = dalloc<float>(S_buf);
    W.sg = dalloc<float>(S_buf), W.su = dalloc<float>(S_buf), W.wt = dalloc<float>(S_buf);
    W.gidx = dalloc<int>(S_buf), W.inv = dalloc<int>(S_buf);
    if (g_batch_moe_compact) {
        W.moe_active_map = dalloc<int>(S_buf);
        W.moe_active_count = dalloc<int>(1);
    }
    W.blob = dalloc<const uint8_t*>(S8), W.tok = dalloc<int>(S8);
    if (g_int4_experts) {
        const int Kpad_f = ((c.F + 255) / 256) * 256;
        W.i4_Kpad_f = Kpad_f;
        // verify-fast rotates S=T*K slot rows into i4_xq/i4_xd; batched dense MV rotates T rows per projection.
        // IK2_BATCH_GRAPH_EXACT needs one q8x row per decode slot for the solo-equivalent MoE matvec.
        // Without any of these these hold the T=1 token only (unchanged).
        const int i4_act_rows = (g_verify_fast || g_batch_dense_mv || g_batch_graph_exact)
                                    ? (int) std::max<size_t>(Tm, 8) : 1;
        W.i4_xq_rows = i4_act_rows;
        W.i4_xq = dalloc<int8_t>((size_t) i4_act_rows * C); W.i4_xd = dalloc<float2>((size_t) i4_act_rows * (C / 32 + 2));
        W.i4_hf = dalloc<float>(S8 * Kpad_f); W.i4_hq = dalloc<int8_t>(S8 * Kpad_f); W.i4_hd = dalloc<float2>(S8 * (Kpad_f / 32 + 2));
        W.i4_Wp_s = dalloc<const uint8_t*>(S8); W.i4_Wh_s = dalloc<const uint8_t*>(S8); W.i4_WS_s = dalloc<const float*>(S8);
        W.i4_Wp_dn = dalloc<const uint8_t*>(S8); W.i4_Wh_dn = dalloc<const uint8_t*>(S8); W.i4_WS_dn = dalloc<const float*>(S8);
        if (g_moe_group_exp) {   // IK2_MOE_GROUP_EXPERT scratch: fixed 64-slot capacity (verify-fast guard: S<=64)
            W.i4_g_cnt = dalloc<int>(64); W.i4_g_slots = dalloc<int>(64 * 8);
            W.i4_g_exp = dalloc<int>(64); W.i4_g_n = dalloc<int>(1);
        }
#if defined(STRATA_USE_HIP) && defined(STRATA_WMMA_GFX11)
        // prefill scratch: one expert's rows at a time, capped at T_buf (rot_pack8 pads rows to 256)
        const int cap = (int) ((T_buf + 255) / 256 * 256);
        const size_t Kmax_sz = std::max<size_t>(C, (size_t) Kpad_f);
        const int Kmax = (int) ((g_int4_dense_multi || g_int4_prefill || g_batch_dense_mv) ? std::max<size_t>(Kmax_sz, HD) : Kmax_sz);
        W.i4_pf_cap = cap;
        W.i4_pf_Kmax = Kmax;
        W.i4_pf_x = dalloc<float>((size_t) cap * Kmax);
        W.i4_pf_x4 = dalloc<uint8_t>((size_t) cap * Kmax);   // two planes of cap * Kmax / 2 bytes
        W.i4_pf_xm = dalloc<uint8_t>((size_t) (cap / 16) * (Kmax / 256) * 16);
        W.i4_pf_xr = dalloc<float>(cap);
        // IK2_INT4_DENSE_MV: decode-style matvec activation scratch (T rows, up to Kmax wide).
        if (g_int4_dense_mv) {
            W.i4_mv_xq = dalloc<int8_t>((size_t) i4_act_rows * Kmax);
            W.i4_mv_xd = dalloc<float2>((size_t) i4_act_rows * (Kmax / 32 + 2));
            if (g_int4_rot_dedup)   // per-input slots: in_q's codes survive the in_k/in_v rotations (msg branch)
                for (int i = 0; i < Work::kI4RotSlots; ++i) {
                    W.i4_rd_xq[i] = dalloc<int8_t>((size_t) i4_act_rows * Kmax);
                    W.i4_rd_xd[i] = dalloc<float2>((size_t) i4_act_rows * (Kmax / 32 + 2));
                }
            if (g_int4_mv_alt)   // IK2_INT4_MV_ALT: in_q/in_k/in_v codes live concurrently for the fused msg matvec
                for (int i = 0; i < 3; ++i) {
                    W.i4_ma_xq[i] = dalloc<int8_t>((size_t) i4_act_rows * Kmax);
                    W.i4_ma_xd[i] = dalloc<float2>((size_t) i4_act_rows * (Kmax / 32 + 2));
                }
        }
#endif
    }
    for (int i = 0; i < Work::kPackRing; ++i) {
        CU(cudaHostAlloc(&W.pack_h[i], 8192, cudaHostAllocPortable));
        W.pack_d[i] = dalloc<uint8_t>(8192);
        if (g_pool) CU(cudaHostAlloc(&W.y_h[i], (size_t) 64 * C * 4, cudaHostAllocPortable));
    }
    W.T_host = (int) std::max<size_t>(T_buf, W.T_sup > 0 ? (size_t) ctx : 0);   // (a lent scratch can take up to ctx)
    const size_t S_host = (size_t) W.T_host * c.K;
    for (int i = 0; i < 2; ++i) {
        CU(cudaHostAlloc(&W.gidx_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.inv_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.sg_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.su_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.wt_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.wtr_h[i], S_host * 4, cudaHostAllocPortable));
    }
    CU(cudaHostAlloc(&W.x_h, (size_t) 8 * C * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&W.emb_h, Tm * C * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&W.pos_h, (size_t) W.T_host * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&W.route_h, S_host * 8, cudaHostAllocPortable));
    W.argmax_d = dalloc<int>(Tm);
    W.argmax_part_v = dalloc<float>(256);
    W.argmax_part_i = dalloc<int>(256);
    CU(cudaHostAlloc(&W.argmax_h, Tm * 4, cudaHostAllocPortable));
    // IK2_AR_KSTEP: per-Work K graph staging.  The clone for slot 1 allocates these freshly.
    if (g_ar_kstep && M.embd_d) {
        W.kstep_K = g_ar_K;
        W.emb_dK = dalloc<float>((size_t) (g_ar_K - 1) * C);
        W.argmax_dK = dalloc<int>(g_ar_K);
        CU(cudaHostAlloc(&W.argmax_hK, (size_t) g_ar_K * 4, cudaHostAllocPortable));
    }
    CU(cudaHostAlloc(&W.logits_h, (size_t) c.n_rows * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&W.cand_h, (size_t) (2 + 2 * kCandCap) * 4, cudaHostAllocPortable));
    {   // (IK2_RG_SLOTS: the slots - tokens x picks - up to which a prompt's experts go by groups; 0: never)
        // (96 tokens by default: past ~128 the FP16 GEMMs are the faster way)
        const int want = getenv("IK2_RG_SLOTS") ? std::atoi(getenv("IK2_RG_SLOTS")) : 96 * c.K;
        W.rg_slots = std::max(0, std::min(want, (int) std::max<size_t>(64, T_buf)));
        W.rg_groups = c.E + W.rg_slots / k::kGroupMax;
        W.rg_bytes = (size_t) W.rg_groups * (8 + 4 + 8 * k::kGroupMax) + (size_t) W.rg_slots * 16;
        CU(cudaHostAlloc(&W.rg_h, W.rg_bytes, cudaHostAllocPortable));
        W.rg_d = dalloc<uint8_t>(W.rg_bytes);
    }
    W.cand_d = dalloc<int>(2 + 2 * kCandCap);
    W.cand_part = dalloc<int>(k::cand_scratch(c.n_vocab));
    const int n_ev = (c.E + Work::kMissBatch - 1) / Work::kMissBatch;
    W.miss_ev.resize(n_ev);
    for (int i = 0; i < n_ev; ++i) CU(cudaEventCreateWithFlags(&W.miss_ev[i], cudaEventDisableTiming));
    for (int r = 0; r < Work::kPfRing; ++r) {
        W.pf_evs[r].resize(n_ev);
        for (int i = 0; i < n_ev; ++i) CU(cudaEventCreateWithFlags(&W.pf_evs[r][i], cudaEventDisableTiming));
    }
    W.cnt.assign(c.E + 1, 0);
    W.start.assign(c.E + 1, 0);
    W.fill.assign(c.E + 1, 0);
    W.hit_exps.reserve(c.E);
    W.miss_exps.reserve(c.E);
    W.slot_ptrs.assign(c.E, nullptr);
    W.seen_stamp.assign(c.E, 0);
    W.gi_stamp.assign(c.E, 0);
    W.seen_late.assign(c.E, 0);
    W.gi_idx.assign(c.E, 0);
    W.logits = dalloc<float>((size_t) 16 * c.n_rows);
    if (c.n_pass > 1) {
        W.lp_pre = dalloc<float>(Tm * C);
        W.lp_bank = dalloc<float>((size_t) (n_core / c.block) * Tm * C);   // (the pass's blocks after its first, its final stream)
        W.lp_msg = dalloc<float>(4 * Tm * C);
        W.lp_msg_all = dalloc<float>((size_t) (c.core1 - c.core0) * 4 * C);
        W.lp_in = dalloc<float>(3 * Tm * C);
        W.lp_g = dalloc<float>(3 * Tm * (2 * (size_t) c.rank + 2));
        W.lp_a = dalloc<float>(3 * Tm * (size_t) c.rank);
        W.lp_f = dalloc<float>(2 * Tm * C);
    }
    if (d == 0 && g_ar_dual && g_ar_stream2) clone_ar_work(g_ar_work1, W, c);
    for (auto& Vi : M.V) {   // the conv histories start at zero
        if (M.L[Vi.l].dev != d) continue;
        CU(cudaMemset(Vi.h_k, 0, 3 * KD * 4));
        CU(cudaMemset(Vi.h_v, 0, 3 * KD * 4));
        CU(cudaMemset(Vi.h_attn, 0, 3 * C * 4));
        CU(cudaMemset(Vi.h_mlp, 0, 3 * C * 4));
    }
    ExpertCache& EC = g_dev[d].EC;
    CU(cudaMemGetInfo(&fr, &tot));
    if (getenv("IK2_VRAM_TRACE"))
        std::printf("VRAM-TRACE GPU%d before EC: free %.2f GiB (tot %.2f)\n", d, fr / 1073741824.0, tot / 1073741824.0);
    // (what is left free: 2 GiB beside a cache that streams experts in and out; a quarter of one when every expert
    //  stays in VRAM - nothing more is allocated but a token's graph). IK2_EC_ROOM_MIB overrides the resident 256 MiB
    //  margin for the tight 1M-ctx budget (the graph + small buffers need < 128 MiB).
    const size_t resident_room = getenv("IK2_EC_ROOM_MIB")
        ? (size_t) (std::atof(getenv("IK2_EC_ROOM_MIB")) * 1048576.0) : (size_t) 256 << 20;
    const size_t room = (g_fast_want ? resident_room : (size_t) 2 << 30) +
                        (vram_reserve_mib > 0 ? (size_t) (vram_reserve_mib * 1048576.0) : 0);
    const size_t budget = std::min((size_t) (cache_gb * 1073741824.0), fr > room ? fr - room : (size_t) 0);
    std::printf("EC budget: free %.2f GiB, room %.2f GiB, cache_gb %.1f -> budget %.2f GiB (fast_want %d)\n",
                fr / 1073741824.0, room / 1073741824.0, cache_gb, budget / 1073741824.0, (int) g_fast_want);
    EC.tp = g_tp ? d : -1;
    EC.init(M.L, dual() ? -1 : d, c.n_layer, c.E, budget, (g_int4_experts && !g_int4_pf_off) ? g_int4_layers : 0);
    std::printf("expert cache, GPU %d: %d slots across %zu pools (%.1f GiB); load %.1f s in all\n", d, EC.total_slots(),
                EC.pools.size(), EC.total_bytes() / 1073741824.0, now() - t_start);
    for (size_t p = 0; p < EC.pools.size(); ++p)
        std::printf("  pool %zu: %d slots of %.1f MiB (%.1f GiB, %d layers)\n", p, EC.pools[p].n_slots,
                    EC.pools[p].slot_bytes / 1048576.0,
                    (double) EC.pools[p].n_slots * EC.pools[p].slot_bytes / 1073741824.0, EC.pools[p].n_layers);
    const char* wf = getenv("IK2_WARM");
    if ((wf && *wf) || g_ep) {   // the cache filled with the most picked experts (IK2_POP_OUT's file or all owned for EP)
        FILE* in = (wf && *wf) ? std::fopen(wf, "rb") : nullptr;
        int32_t hdr[2] = {0, 0};
        std::vector<uint64_t> cnt;
        if (in && std::fread(hdr, 4, 2, in) == 2 && hdr[0] == c.n_layer && hdr[1] == c.E) {
            cnt.resize((size_t) c.n_layer * c.E);
            if (std::fread(cnt.data(), 8, cnt.size(), in) != cnt.size()) cnt.clear();
        }
        if (in) std::fclose(in);
        if (cnt.empty() && g_ep) cnt.assign((size_t) c.n_layer * c.E, 1);
        if (cnt.empty()) {
            std::printf("warm: the cache starts empty\n");
        } else {
            const double tw = now();
            const uint64_t m0 = EC.misses, b0 = EC.bytes_up;
            int n = 0;
            for (size_t p = 0; p < EC.pools.size(); ++p) {
                std::vector<int> idx;
                for (int l = 0; l < c.n_layer; ++l)
                    if (EC.layer_pool[l] == (int) p)
                        for (int e = 0; e < c.E; ++e)
                            if (!g_ep || ep_owner(l, e) == d) idx.push_back(l * c.E + e);
                // the most picked first; the chosen ones then go in least picked first, so that the evictions that
                // start from the oldest take the least picked
                std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
                const int k = std::min((int) idx.size(), EC.pools[p].n_active);
                for (int i = k - 1; i >= 0; --i) {
                    const int l = idx[i] / c.E, e = idx[i] % c.E;
                    EC.get(M.L[l], l, e, EC.tick, g_cs, nullptr, true, true);
                    ++n;
                }
            }
            CU(cudaStreamSynchronize(g_cs));
            ++EC.tick;   // (taken in the tick the first prefetch uses, they would all count as in use: none evictable)
            // the profile also seeds the prefetch's picture of which experts a pass uses (W.use, as 100k picks a
            // layer): without it the first long-enough request prefetches all 18k experts
            W.use.assign((size_t) c.n_layer * c.E, 0), W.use_tot.assign(c.n_layer, 0);
            for (int l = 0; l < c.n_layer; ++l) {
                long double sum = 0;
                for (int e = 0; e < c.E; ++e) sum += cnt[(size_t) l * c.E + e];
                if (sum <= 0) continue;
                for (int e = 0; e < c.E; ++e) {
                    W.use[(size_t) l * c.E + e] = (uint64_t) (100000.0L * cnt[(size_t) l * c.E + e] / sum + 0.5L);
                    W.use_tot[l] += W.use[(size_t) l * c.E + e];
                }
            }
            std::printf("warm: %d experts (%.1f GiB) copied in %.1f s\n", n, (EC.bytes_up - b0) / 1073741824.0,
                        now() - tw);
            EC.misses = m0, EC.bytes_up = b0;
        }
    }
    if (g_fast_want && d == 0) {   // the resident path: every expert in now, and where each is on the device
        auto blocks = [](const Mat& m) { return m.raw || m.q8; };   // (kept as the GGUF's blocks, whole or planar)
        bool all = !EC.pools.empty(), raw = blocks(M.out_w);
        if (g_int4_experts) {   // only the l>=20 Q8_0 layers need the cache; l<20 live in i4_* buffers
            std::vector<bool> need(EC.pools.size(), false);
            for (int l = g_int4_layers; l < c.n_layer; ++l) need[EC.layer_pool[l]] = true;
            for (size_t p = 0; p < EC.pools.size(); ++p)
                if (need[p]) all = all && EC.pools[p].n_active >= EC.pools[p].n_layers * c.E;
            // IK2_INT4_DENSE / IK2_INT4_HEAD load rotated-int4 sidecars for the dense decode matvecs IN ADDITION to
            // the Q8_0 planar copies (kept for prefill). When every projection that decode reads carries its sidecar,
            // the resident decode path never touches the raw Q8 blocks -> residency holds without them. (raw still
            // gates the no-sidecar fallback so a missing/unsupported sidecar cannot silently drop residency.)
            if (g_int4_dense) {
                bool dense_i4 = M.out_w.i4Wp != nullptr || !g_int4_head;
                for (const auto& Ly : M.L) {
                    dense_i4 = dense_i4 && Ly.wo.i4Wp != nullptr &&
                               (Ly.sliding ? Ly.wqkvrg.i4Wp != nullptr
                                           : (Ly.wqrg.i4Wp != nullptr && Ly.wk.i4Wp != nullptr && Ly.wv.i4Wp != nullptr));
                }
                if (dense_i4) raw = true;
            }
        } else {
            for (const auto& P : EC.pools) all = all && P.n_active >= P.n_layers * c.E;
        }
        for (const auto& Ly : M.L)
            raw = raw && blocks(Ly.wo) && (Ly.sliding ? blocks(Ly.wqkvrg) : (blocks(Ly.wqrg) && blocks(Ly.wk) && blocks(Ly.wv)));
        if (g_int4_experts && g_int4_dense) {
            // every decode projection carries an int4 sidecar -> residency holds without the Q8 planar dense copies
            bool dense_i4 = M.out_w.i4Wp != nullptr || !g_int4_head;
            for (const auto& Ly : M.L)
                dense_i4 = dense_i4 && Ly.wo.i4Wp != nullptr &&
                           (Ly.sliding ? Ly.wqkvrg.i4Wp != nullptr
                                       : (Ly.wqrg.i4Wp != nullptr && Ly.wk.i4Wp != nullptr && Ly.wv.i4Wp != nullptr));
            if (dense_i4) raw = true;
        }
        if (all && raw) {
            const double tw = now();
            const uint64_t m0 = EC.misses, b0 = EC.bytes_up;
            int n = 0;
            // INT4 layers (<20) keep their weights in i4_* buffers: no NVFP4 cache copy for them.
            for (int l = (g_int4_experts ? g_int4_layers : 0); l < c.n_layer; ++l)
                for (int e = 0; e < c.E; ++e)
                    if (!EC.has(l, e)) EC.get(M.L[l], l, e, EC.tick, g_cs, nullptr, true, true), ++n;
            CU(cudaStreamSynchronize(g_cs));
            ++EC.tick;
            for (int l = 0; l < c.n_layer; ++l) {
                Layer& Ly = M.L[l];
                if (g_int4_experts && l < g_int4_layers) {   // dummy blob: route_token writes it, the i4 path ignores it
                    std::vector<const uint8_t*> tab(c.E, Ly.i4_Wp[0]);
                    Ly.tab_blob = dalloc<const uint8_t*>(c.E);
                    CU(cudaMemcpy(Ly.tab_blob, tab.data(), (size_t) c.E * sizeof(void*), cudaMemcpyHostToDevice));
                    Ly.tab_h = tab;
                } else {   // layers 20-23 stay Q8_0: real expert pointers from the cache
                    const auto& P = EC.pools[EC.layer_pool[l]];
                    std::vector<const uint8_t*> tab(c.E);
                    for (int e = 0; e < c.E; ++e) tab[e] = P.data + (size_t) EC.loc[(size_t) l * c.E + e] * P.slot_bytes;
                    Ly.tab_blob = dalloc<const uint8_t*>(c.E);
                    CU(cudaMemcpy(Ly.tab_blob, tab.data(), (size_t) c.E * sizeof(void*), cudaMemcpyHostToDevice));
                    Ly.tab_h = tab;
                    for (int r = 0; r < 3; ++r)
                        if (!Ly.scale[r].empty()) Ly.tab_sc[r] = dev_f32(Ly.scale[r]);
                }
            }
            EC.misses = m0, EC.bytes_up = b0;
            g_fast = true;
            if (g_int4_experts)
                std::printf("resident: INT4 experts (l<20) + Q8_0 (l>=20) all in VRAM; tokens take the resident path\n");
            else
                std::printf("resident: every expert in VRAM (%d copied in %.1f s); tokens take the resident path%s\n", n,
                            now() - tw, g_graph ? ", a CUDA graph each" : "");
        } else {
            std::printf("resident path off: %s\n", all ? "a dense matrix is not kept as blocks" : "the cache does not hold every expert");
        }
    }
    std::fflush(stdout);
    }
    if (g_ep && n_gpu == 2) {
        bool ep_all = true;
        for (int d = 0; d < 2; ++d) {
            auto& EC = dev_cache(d);
            for (int l = 0; l < c.n_layer; ++l) {
                for (int e = 0; e < c.E; ++e) {
                    if (ep_owner(l, e) == d && !EC.has(l, e)) {
                        ep_all = false;
                        break;
                    }
                }
                if (!ep_all) break;
            }
        }
        if (ep_all) {
            for (int d = 0; d < 2; ++d) {
                use_dev(d);
                auto& EC = dev_cache(d);
                for (int l = 0; l < c.n_layer; ++l) {
                    Layer& Ly = M.L[l];
                    const auto& P = EC.pools[EC.layer_pool[l]];
                    std::vector<const uint8_t*> tab(c.E, nullptr);
                    for (int e = 0; e < c.E; ++e) {
                        if (ep_owner(l, e) == d) {
                            tab[e] = P.data + (size_t) EC.loc[(size_t) l * c.E + e] * P.slot_bytes;
                        }
                    }
                    Ly.ep_tab_blob[d] = dalloc<const uint8_t*>(c.E);
                    CU(cudaMemcpy((void*) Ly.ep_tab_blob[d], tab.data(), (size_t) c.E * sizeof(void*), cudaMemcpyHostToDevice));
                    for (int r = 0; r < 3; ++r) {
                        if (!Ly.scale[r].empty()) Ly.ep_tab_sc[d][r] = dev_f32(Ly.scale[r]);
                    }
                }
            }
            use_dev(0);
            g_ep_fast = true;
            std::printf("resident: every expert in VRAM across 2 GPUs; zero-wait GPU EP dispatch enabled\n");
            std::fflush(stdout);
        } else {
            std::printf("resident EP path off: not all owned experts resident in VRAM\n");
            std::fflush(stdout);
        }
    }
    // the caches' counters, summed over the GPUs
    auto ec_sum = [&](auto f) {
        double v = 0;
        for (auto& D : g_dev) v += (double) f(D.EC);
        return v;
    };

    // IK2_AR_KSTEP: the lazy one-step capture cannot be used here.  A K graph computes K tokens while one-step
    // bookkeeping would consume 1, so both slot graphs MUST exist before the serve loop starts.  A direct warm-up
    // first runs the exact launch sequence (cuBLAS/hipBLAS shape setup may allocate outside capture), then the
    // capture records it.  Anything that fails disables K globally and leaves the proven one-step path intact.
    if (g_ar_kstep && !M.embd_d) {
        std::fprintf(stderr, "ik2: IK2_AR_KSTEP needs the device embedding table; disabling\n");
        g_ar_kstep = false;
    }
    if (g_ar_kstep) {
        use_dev(0);
        bool ok = true;
        for (int slot = 0; slot < 2 && ok; ++slot) {
            Work& W = slot == 1 ? g_ar_work1 : dev_work(0);
            cudaStream_t stream = slot == 1 ? g_s1 : g_s;
            cudaStream_t old_s = g_s;
            if (slot == 1) g_s = stream;   // all launch code, capture and replay use the same global stream
            load_batch_slot(M, slot);
            cudaError_t ce = cudaStreamSynchronize(stream);
            const int C = M.c.C;
            std::memset(W.emb_h, 0, (size_t) C * 4);
            W.pos_h[0] = 0;
            const double t0 = now();
            if (ce == cudaSuccess) fast_token_launches_k(M, W, g_ar_K);
            const double t1 = now();
            if (ce == cudaSuccess) ce = cudaStreamSynchronize(stream);
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t exec = nullptr;
            const double t2 = now();
            if (ce == cudaSuccess) ce = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            if (ce == cudaSuccess) {
                g_fast_graph_capturing = true;
                fast_token_launches_k(M, W, g_ar_K);
                ce = cudaStreamEndCapture(stream, &graph);
                g_fast_graph_capturing = false;
            }
            const double t3 = now();
            double inst_s = 0.0;
            if (ce == cudaSuccess) {
                const double ti0 = now();
                ce = cudaGraphInstantiate(&exec, graph, 0);
                inst_s = now() - ti0;
            }
            const double t4 = now();
            if (ce == cudaSuccess) {
                g_ar_kgraph[slot].graph = graph;
                g_ar_kgraph[slot].exec = exec;
                size_t nn = 0;
                if (cudaGraphGetNodes(graph, nullptr, &nn) != cudaSuccess) nn = (size_t) -1;
                std::printf("ik2: AR K-step graph slot %d K=%d nodes=%zu warm=%.1fms capture=%.1fms inst=%.1fms\n",
                            slot, g_ar_K, nn, (t1 - t0) * 1e3, (t3 - t2) * 1e3, inst_s * 1e3);
                (void) t4;
            } else {
                std::fprintf(stderr, "ik2: AR K-step graph slot %d failed (%s); disabling KSTEP\n",
                             slot, cudaGetErrorString(ce));
                if (graph) (void) cudaGraphDestroy(graph);
                if (exec) (void) cudaGraphExecDestroy(exec);
                ok = false;
                (void) cudaGetLastError();
            }
            if (slot == 1) g_s = old_s;
        }
        g_ar_kstep_ready = ok;
        if (!ok) {
            g_ar_kstep = false;
            for (int slot = 0; slot < 2; ++slot) {
                if (g_ar_kgraph[slot].exec) cudaGraphExecDestroy(g_ar_kgraph[slot].exec);
                if (g_ar_kgraph[slot].graph) cudaGraphDestroy(g_ar_kgraph[slot].graph);
                g_ar_kgraph[slot].exec = nullptr; g_ar_kgraph[slot].graph = nullptr;
            }
            std::fflush(stdout);
        } else {
            std::printf("ik2: AR K-step ready (K=%d, both slots pre-built)\n", g_ar_K);
            std::fflush(stdout);
        }
    }

    if (serve) {
        // --serve: requests on stdin, a line each - "GEN max_new temperature top_p seed stop,stop ids,ids,..." - and
        // the answer on stdout as "@TOK id" lines then "@END reason prompt_tokens cached_tokens prompt_s gen_s".  The
        // tokens whose state the model holds are kept: a request that starts with them runs only what follows.
        std::vector<int> hist;
        std::vector<uint64_t> hist_sig;   // a picture row's hash at its position (shorter than hist: the rest are tokens)
        std::vector<float> last;
        double t_exp = 0;
        // Snapshots of every short conv's history (plus sliding-layer KV windows in host RAM) at the end of a
        // request's prompt (before its generation): an agent's next turn usually agrees with the last one up to
        // there and no further (the model's own reply comes back re-rendered, its thinking dropped), so the state
        // goes back to that point instead of the start. Full-layer KV needs nothing: positions rewrite cleanly.

        struct Snap {
            int pos = -1;
            bool full = false;   // batch slot snapshots must park full-attention KV too (different prompts share cells)
            std::vector<float*> h;   // [layer * 4 + conv]
            std::vector<PinBytes> kv_k, kv_v;
            std::vector<PinBytes> kv_ks, kv_vs;   // int8/int4: per-head fp16 scales
            // The tokens this state covers, so it can be matched against a prompt on its own (see the search before
            // a prompt is read): a snapshot is a whole conversation, not a position in whichever one is live now.
            std::vector<int> toks;
            std::vector<uint64_t> sigs;
        };
        const size_t kv_elem_b = g_kv_fp8 ? 1 : 2;
        // Per-token KV row bytes.  For INT4 the old `KD * bits / 8` was computed as `KD * (bits/8)` = 0,
        // so every int4 KV snapshot silently copied zero bytes.  Batch slots switch between different
        // prompts, so that bug shows up as cross-slot state corruption; compute the row width after the
        // multiply, matching the allocation at :5224 (`cap * KD * kv_int_bits() / 8`).
        const size_t kv_row_b = kv_is_int() ? ((size_t) KD * kv_int_bits() / 8) : ((size_t) KD * kv_elem_b);
        // IK2_CONV_CACHE_MIB: conversation-cache budget in host RAM (default 8192 = 8 GiB). The cache parks each
        // session's snapshot (conv histories + sliding KV windows +, with IK2_CONV_CACHE_FULL, the full-attention
        // KV) so an agent's next turn reuses the shared prefix instead of re-running it. The number of snapshots is
        // sized to fit the budget; more snapshots = more concurrent conversations parked.
        const size_t conv_cache_mib = getenv("IK2_CONV_CACHE_MIB") ? (size_t) std::atof(getenv("IK2_CONV_CACHE_MIB")) : 8192;
        // The full-attention KV (8 layers of 8 KV heads x 128 x 2 bytes = 8 KiB a token on this model) is what makes
        // a parked conversation worth going back to: without it the state only answers for the tokens the device
        // happens to hold, so any request that shares just the opening tokens - an agent's title or summary call
        // between two turns - left every snapshot unusable and the whole prompt was read again.  It is sized per
        // conversation when that conversation is parked (park_state below), not for the whole context up front, so
        // this is on by default; IK2_CONV_CACHE_FULL=0 turns it off.
        const bool conv_cache_full = !getenv("IK2_CONV_CACHE_FULL") || std::atoi(getenv("IK2_CONV_CACHE_FULL")) != 0;
    // IK2_CONV_CACHE_PARK_MIB: the most host bytes one park may copy (default 1024). A park is synchronous, so an
    // unbounded one turns a 393K-token prompt into a 42 s stall; past the cap the conversation is simply not cached.
    const size_t conv_park_cap_mib = getenv("IK2_CONV_CACHE_PARK_MIB") ? (size_t) std::atof(getenv("IK2_CONV_CACHE_PARK_MIB")) : 1024;
        size_t snap_bytes = 0;   // host bytes per snapshot (sliding KV + scales; conv histories are VRAM)
        size_t batch_snap_bytes = 0;   // a batch slot must restore full-attention KV of a different prompt
        size_t full_row_b = 0;   // host bytes a token takes in every full-attention layer (K and V)
        for (size_t l = 0; l < M.V.size(); ++l) {
            const Layer& Ly = bind_visit(M, (int) l);
            if (Ly.sliding && Ly.kv_cap < ctx) {
                const size_t cap = (size_t) c.window;
                snap_bytes += 2 * cap * kv_row_b;
                if (kv_is_int()) snap_bytes += 2 * cap * c.KV * 2;
            }
            if (!Ly.sliding) {
                full_row_b += 2 * kv_row_b;
                if (kv_is_int()) full_row_b += 2 * (size_t) c.KV * 2;
            }
            const bool batch_full = !Ly.sliding;
            if ((Ly.sliding && Ly.kv_cap < ctx) || batch_full) {
                const size_t cap = batch_full ? (size_t) ctx : (size_t) c.window;
                batch_snap_bytes += 2 * cap * kv_row_b;
                if (kv_is_int()) batch_snap_bytes += 2 * cap * c.KV * 2;
            }
        }
        int n_snaps = 8;
        // The slot count follows the sliding part alone (small and fixed); the full-attention KV of a parked
        // conversation grows with that conversation and is charged to the same budget as it is written.
        if (snap_bytes > 0) n_snaps = (int) std::max<size_t>(1, std::min<size_t>(64, (conv_cache_mib << 20) / snap_bytes));
        size_t conv_full_budget = conv_cache_mib << 20;
        const size_t conv_fixed_bytes = snap_bytes * (size_t) n_snaps;
        conv_full_budget = conv_fixed_bytes < conv_full_budget ? conv_full_budget - conv_fixed_bytes : 0;
        std::printf("conversation cache: %zu MiB budget, %d snapshots of %.1f MiB host RAM (%s, %.1f KiB a token)\n",
                    conv_cache_mib, n_snaps, snap_bytes / 1048576.0,
                    conv_cache_full ? "sliding + full-attention KV" : "sliding KV", full_row_b / 1024.0);
        std::fflush(stdout);
        auto alloc_snap = [&](Snap& sn, bool full, bool eager) {
            sn.full = full;
            sn.kv_k.resize(M.V.size());
            sn.kv_v.resize(M.V.size());
            sn.kv_ks.resize(M.V.size());
            sn.kv_vs.resize(M.V.size());
            for (size_t l = 0; l < M.V.size(); ++l) {   // (l: a visit)
                auto& Ly = bind_visit(M, (int) l);
                use_dev(Ly.dev);
                const size_t cc[4] = {KD, KD, C, C};
                for (int i = 0; i < 4; ++i) sn.h.push_back(dalloc<float>(3 * cc[i]));
                const bool park_full = full && !Ly.sliding;
                if ((Ly.sliding && Ly.kv_cap < ctx) || park_full) {
                    // Sized when the conversation is parked (snap_copy grows it): a full-attention window is as
                    // long as the conversation, and a sliding one is c.window rows - pinning those up front for
                    // every slot would lock gigabytes nobody uses yet.  Only a batch slot allocates eagerly.
                    const size_t cap = eager ? (park_full ? (size_t) ctx : (size_t) c.window) : 0;
                    if (cap > 0) {
                        const size_t bytes = cap * kv_row_b;
                        sn.kv_k[l].resize(bytes);
                        sn.kv_v[l].resize(bytes);
                        if (kv_is_int()) {
                            sn.kv_ks[l].resize(cap * c.KV * 2);
                            sn.kv_vs[l].resize(cap * c.KV * 2);
                        }
                    }
                }
            }
        };
        std::vector<Snap> snaps(n_snaps);
        for (auto& sn : snaps) alloc_snap(sn, conv_cache_full, false);
        int snap_next = 0;
        auto snap_copy = [&](Snap& sn, bool save) {
            const size_t cc[4] = {KD, KD, C, C};
            const cudaMemcpyKind kv_dir = save ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
            for (size_t l = 0; l < M.V.size(); ++l) {   // (l: a visit)
                Layer& Ly = bind_visit(M, (int) l);
                use_dev(Ly.dev);
                float* hs[4] = {Ly.h_k, Ly.h_v, Ly.h_attn, Ly.h_mlp};
                for (int i = 0; i < 4; ++i) {
                    float* sv = sn.h[l * 4 + i];
                    CU(cudaMemcpyAsync(save ? sv : hs[i], save ? hs[i] : sv, 3 * cc[i] * 4, cudaMemcpyDeviceToDevice, g_s));
                }
                const bool park_full = sn.full && !Ly.sliding;
                if (sn.pos > 0 && ((Ly.sliding && Ly.kv_cap < ctx) || park_full)) {
                    const int w = park_full ? sn.pos : std::min(sn.pos, c.window);
                    if (save) {   // (grown when the conversation is parked: see alloc_snap)
                        const size_t need = (size_t) w * kv_row_b;
                        if (sn.kv_k[l].size() < need) sn.kv_k[l].resize(need);
                        if (sn.kv_v[l].size() < need) sn.kv_v[l].resize(need);
                        if (kv_is_int()) {
                            const size_t sneed = (size_t) w * c.KV * 2;
                            if (sn.kv_ks[l].size() < sneed) sn.kv_ks[l].resize(sneed);
                            if (sn.kv_vs[l].size() < sneed) sn.kv_vs[l].resize(sneed);
                        }
                    }
                    const int k_start = sn.pos - w;
                    const int s0 = k_start % Ly.kv_cap;
                    const int n1 = std::min(w, Ly.kv_cap - s0);
                    const int n2 = w - n1;
                    const size_t row_b = kv_row_b;
                    uint8_t* dk = kv_is_int() ? Ly.kci : (g_kv_fp8 ? Ly.kc8 : (uint8_t*) Ly.kc);
                    uint8_t* dv = kv_is_int() ? Ly.vci : (g_kv_fp8 ? Ly.vc8 : (uint8_t*) Ly.vc);
                    uint8_t* hk = sn.kv_k[l].data();
                    uint8_t* hv = sn.kv_v[l].data();
                    auto ring_copy = [&](uint8_t* d, uint8_t* h, size_t rb) {
                        CU(cudaMemcpyAsync(save ? h : d + (size_t) s0 * rb, save ? d + (size_t) s0 * rb : h,
                                           (size_t) n1 * rb, kv_dir, g_s));
                        if (n2 > 0)
                            CU(cudaMemcpyAsync(save ? h + (size_t) n1 * rb : d,
                                               save ? d : h + (size_t) n1 * rb, (size_t) n2 * rb, kv_dir, g_s));
                    };
                    ring_copy(dk, hk, row_b);
                    ring_copy(dv, hv, row_b);
                    if (kv_is_int()) {
                        const size_t srow = (size_t) c.KV * 2;
                        ring_copy((uint8_t*) Ly.ks, sn.kv_ks[l].data(), srow);
                        ring_copy((uint8_t*) Ly.vs, sn.kv_vs[l].data(), srow);
                    }
                }
            }
            for (auto& D : g_dev) CU(cudaStreamSynchronize(D.s));
        };
        // What the parked conversations hold in host RAM right now (a conversation's full-attention KV grows with it).
        size_t conv_host_bytes = 0;
        auto snap_host = [&](const Snap& sn) {
            size_t b = 0;
            for (const auto& v : sn.kv_k) b += v.size();
            for (const auto& v : sn.kv_v) b += v.size();
            for (const auto& v : sn.kv_ks) b += v.size();
            for (const auto& v : sn.kv_vs) b += v.size();
            return b;
        };
        auto count_parked = [&]() {
            int n = 0;
            for (const auto& sn : snaps) if (sn.pos > 0) ++n;
            return n;
        };
        // Park the live state - the tokens it covers, its conv histories, its sliding windows and, with the
        // conversation cache on, its full-attention KV - so a later prompt that begins with those tokens goes back to
        // it instead of reading them again.  The device holds one state at a time, so this copy is the only one; when
        // the budget is full the least useful parked conversation (the one with the fewest tokens) makes room.
        // Returns the tokens parked, 0 when this one did not fit.
        auto park_state = [&](const std::vector<int>& toks, const std::vector<uint64_t>& sigs) -> int {
            if (snaps.empty() || toks.empty()) return 0;
            const size_t need = conv_cache_full ? toks.size() * full_row_b : 0;
            // A park copies the conversation's full-attention KV to host RAM, and that copy is synchronous: at
            // 393K tokens it cost ~42 s, which is more than the whole decode it was meant to speed up. Never
            // spend more than this on one park; past it the conversation simply is not cached (a re-read of the
            // prompt is the cheaper mistake). IK2_CONV_CACHE_PARK_MIB overrides the cap, 0 = no cap.
            const size_t park_cap = conv_park_cap_mib << 20;
            if (park_cap > 0 && need > park_cap) return 0;
            while (conv_host_bytes + need > conv_full_budget) {
                int victim = -1;
                for (int i = 0; i < (int) snaps.size(); ++i)
                    if (snaps[i].pos > 0 && (victim < 0 || snaps[i].pos < snaps[victim].pos)) victim = i;
                if (victim < 0 || snaps[victim].pos > (int) toks.size()) break;   // (nothing worth giving up)
                conv_host_bytes -= snap_host(snaps[victim]);
                for (size_t l = 0; l < snaps[victim].kv_k.size(); ++l) {
                    if (bind_visit(M, (int) l).sliding) continue;   // (the sliding windows are preallocated: keep them)
                    snaps[victim].kv_k[l] = {};
                    snaps[victim].kv_v[l] = {};
                    snaps[victim].kv_ks[l] = {};
                    snaps[victim].kv_vs[l] = {};
                }
                snaps[victim].toks.clear();
                snaps[victim].sigs.clear();
                snaps[victim].pos = -1;
            }
            if (conv_host_bytes + need > conv_full_budget) return 0;
            Snap& sn = snaps[snap_next];
            snap_next = (snap_next + 1) % (int) snaps.size();
            conv_host_bytes -= snap_host(sn);
            sn.toks = toks;
            sn.sigs = sigs;
            sn.pos = (int) toks.size();
            snap_copy(sn, true);
            conv_host_bytes += snap_host(sn);
            return sn.pos;
        };
        auto reset = [&]() {
            // (the parked conversations stay: a snapshot carries its own tokens and KV, so it no longer depends on
            // what the device holds - which is why one prompt that matches nothing must not throw the cache away)
            for (int vis = 0; vis < (int) M.V.size(); ++vis) {
                Layer& Ly = bind_visit(M, vis);
                use_dev(Ly.dev);
                CU(cudaMemset(Ly.h_k, 0, 3 * KD * 4));
                CU(cudaMemset(Ly.h_v, 0, 3 * KD * 4));
                CU(cudaMemset(Ly.h_attn, 0, 3 * C * 4));
                CU(cudaMemset(Ly.h_mlp, 0, 3 * C * 4));
            }
            hist.clear();
            hist_sig.clear();
            last.clear();
        };
        auto parse_ints = [](const std::string& t) {
            std::vector<int> v;
            for (const char* p = t.c_str(); *p;) {
                if (*p == '-' && p[1] == 0) break;
                v.push_back(std::atoi(p));
                while (*p && *p != ',') ++p;
                if (*p == ',') ++p;
            }
            return v;
        };
        // ---- P1 continuous batching (IK2_PARALLEL>1). The solo path below is untouched when it is off. ----
        struct Ik2Req {
            bool active = false, cancel = false, admitted = false;
            int slot = -1, max_new = 0, generated = 0, prompt_len = 0, prefill_pos = 0, next_tok = -1;
            double temp = 0, top_p = 1, top_k = 0;
            long long seed = -1;
            std::mt19937_64 rng;
            std::vector<int> prompt_ids, hist, stops;
            std::vector<uint64_t> sig;
            std::vector<float> last;
            bool cand = false;
            bool have_state = false;
            Snap state;
            double t_start = 0, prompt_ms = 0, decode_t0 = 0;
            int cached_prefix = 0;
        };
        // A batch slot restores full-attention KV of a different prompt, so its snapshot is much bigger than
        // the sliding-only conversation snapshot. Cap the total host footprint and the free VRAM; never let
        // a misconfigured IK2_PARALLEL turn into an OOM hang (q36 does the same kind of early capacity check).
        if (g_parallel > 1 && batch_snap_bytes > 0) {
            const size_t budget = (size_t) (getenv("IK2_BATCH_HOST_MIB")
                                            ? std::atof(getenv("IK2_BATCH_HOST_MIB")) : 8192) << 20;
            const int fit = (int) std::max<size_t>(1, budget / batch_snap_bytes);
            if (fit < g_parallel) {
                std::fprintf(stderr, "ERR batch host KV needs %.1f MiB/slot (ctx=%d), IK2_BATCH_HOST_MIB=%zu: "
                                     "lowering IK2_PARALLEL %d -> %d\n",
                             batch_snap_bytes / 1048576.0, ctx, budget >> 20, g_parallel, fit);
                g_parallel = fit;
            }
        }
        // AR dual pre-allocates both slots' KV, isolated Work and graphs at load time; replay needs
        // no fresh device allocation, so a post-load free<256MiB (EC deliberately fills VRAM for
        // expert residency) must NOT disable it. The guard is for P1's dynamic host-snapshot path.
        if (g_parallel > 1 && !g_ar_dual) {
            size_t free_b = 0, total_b = 0;
            if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && free_b < (256u << 20)) {
                std::fprintf(stderr, "ERR only %.1f MiB VRAM free; lowering IK2_PARALLEL %d -> 1\n",
                             free_b / 1048576.0, g_parallel);
                g_parallel = 1;
            }
        }
        if (g_ar_dual && g_parallel != 2) {
            std::fprintf(stderr, "ik2: AR dual disabled (IK2_PARALLEL lowered to %d)\n", g_parallel);
            g_ar_dual = false;
            g_ar_stream2 = false;
            g_ar_kstep = false;
            g_ar_kstep_ready = false;
            if (!getenv("IK2_EMBD_GPU")) g_embd_gpu = false;
            g_batch_decode = false;
        }
        std::vector<Ik2Req> reqs;
        int loaded_slot = -1;
        bool batch_dirty = false;
        long long batch_steps = 0, batch_tokens = 0;
        long long ar2_steps = 0;
        double ar2_sum_slot0 = 0.0, ar2_sum_slot1 = 0.0, ar2_sum_ms = 0.0, ar2_wall_ms = 0.0;
        const bool batch_stats = getenv("IK2_BATCH_STATS") && std::atoi(getenv("IK2_BATCH_STATS"));
        if (g_parallel > 1) {
            if (batch_stats)
                std::printf("batch host KV: %.1f MiB/slot, %d slots\n", batch_snap_bytes / 1048576.0, g_parallel);
            reqs.resize(g_parallel);
            for (int b = 0; b < g_parallel; ++b) {
                reqs[b].slot = b;
                alloc_snap(reqs[b].state, true, true);
            }
        }
        // Sampling, shared by the solo path and the batch scheduler. Parameterized so the solo path keeps its
        // original RNG stream byte for byte (same code, same order of draws).
        auto pick_top = [&](std::vector<std::pair<float, int>>& cd, double temp, double top_p, int top_k,
                            std::mt19937_64& rg) -> int {
            const int kk = std::min((int) cd.size(), top_k > 0 ? std::min(top_k, kCandK) : kCandK);
            std::partial_sort(cd.begin(), cd.begin() + kk, cd.end(),
                              [](const std::pair<float, int>& a2, const std::pair<float, int>& b2) {
                                  return a2.first > b2.first || (a2.first == b2.first && a2.second < b2.second);
                              });
            std::vector<double> pr(kk);
            double z = 0;
            for (int j = 0; j < kk; ++j) z += (pr[j] = std::exp((cd[j].first - cd[0].first) / temp));
            double acc = 0;
            int keep = kk;
            for (int j = 0; j < kk; ++j) {
                acc += pr[j] / z;
                if (acc >= top_p) { keep = j + 1; break; }
            }
            double zk = 0;
            for (int j = 0; j < keep; ++j) zk += pr[j];
            double u = std::uniform_real_distribution<double>(0, zk)(rg);
            for (int j = 0; j < keep; ++j) {
                u -= pr[j];
                if (u <= 0) return cd[j].second;
            }
            return cd[keep - 1].second;
        };
        auto pick = [&](const float* r, double temp, double top_p, int top_k, std::mt19937_64& rg) -> int {
            if (temp <= 0) return (int) (std::max_element(r, r + c.n_vocab) - r);
            std::vector<std::pair<float, int>> cd(c.n_vocab);
            for (int v = 0; v < c.n_vocab; ++v) cd[v] = {r[v], v};
            return pick_top(cd, temp, top_p, top_k, rg);
        };
        auto pick_cand = [&](double temp, double top_p, int top_k, std::mt19937_64& rg) -> int {
            std::vector<std::pair<float, int>> cd(g_cand.n);
            for (int i = 0; i < g_cand.n; ++i) cd[i] = {g_cand.val[i], g_cand.idx[i]};
            return pick_top(cd, temp, top_p, top_k, rg);
        };
        const bool use_cand = !(getenv("IK2_CAND") && !std::atoi(getenv("IK2_CAND")));
        // IK2_MONITOR_INFO (default 0): opt-in Monitor fields. The default printf calls below stay byte-identical.
        const char* monitor_env = getenv("IK2_MONITOR_INFO");
        const bool monitor_info = monitor_env && std::atoi(monitor_env) != 0;
        auto monitor_counts = [&](bool want_hits) -> uint64_t {
            uint64_t n = 0;
            for (auto& D : g_dev) n += want_hits ? (uint64_t) D.EC.hits : (uint64_t) D.EC.misses;
            return n;
        };
        std::signal(SIGUSR1, on_usr1);
        // The engine protocol (serve/server.py's StrataEngine): "GEN <max_new> [key=value ...] <ids>" in; "PP ..." per
        // prompt pass, "T <id>" per token and "DONE <generated> <prompt> <prompt_ms> <decode_ms> <stop|length|cancel>
        // 0 0 <reused>" out; "STOP" mid-request ends it at the next token (or between the prompt's passes), "QUIT"
        // ends the engine.  The tokens a request ends at: --eos, else the GGUF's eos and <|endoftext|>.
        if (eos_ids.empty()) {
            if (const auto* e = S.get("tokenizer.ggml.eos_token_id")) eos_ids.push_back((int) e->u);
            eos_ids.push_back(199999);   // (Inkling 2's <|endoftext|>)
        }
        if (serve_strata) {
            // Single-stream (g_parallel==1) must report batch_slots=0: serve.py:955 sets
            // self.batch=int(info["batch_slots"]); any nonzero makes it take generate_batched,
            // skipping the solo FIFO -> two concurrent GENs deadlock the one-stream engine.
            // IK2_REPORT_BATCH=1 restores the old value for real batched deployments.
            static const bool report_batch = getenv("IK2_REPORT_BATCH") && std::atoi(getenv("IK2_REPORT_BATCH"));
            const int reported_slots = report_batch ? g_parallel : 0;
            if (monitor_info) {
                int expert_slots = 0;
                size_t expert_bytes = 0;
                for (auto& D : g_dev) {
                    expert_slots += D.EC.total_slots();
                    expert_bytes += D.EC.total_bytes();
                }
                size_t vram_free_b = 0, vram_total_b = 0;
                use_dev(0);
                if (cudaMemGetInfo(&vram_free_b, &vram_total_b) != cudaSuccess) vram_free_b = 0;
#if defined(STRATA_USE_HIP) && defined(_WIN32)
                // The Windows cudaMemGetInfo shim subtracts the WDDM budget withheld from this process; a 640k-ctx
                // load can leave that figure at 0 even while the card has free memory. Fall back to the driver's
                // unadjusted figure only for this opt-in Monitor field (the default path is untouched).
                if (vram_free_b == 0) {
                    size_t raw_free_b = 0, raw_total_b = 0;
                    if (hipMemGetInfo(&raw_free_b, &raw_total_b) == hipSuccess) vram_free_b = raw_free_b;
                }
#endif
                std::printf("INFO engine=ik2 engine_kind=ik2 model=Inkling-2 kv=%s ctx=%d gpus=%d batch_slots=%d%s "
                            "expert_slots=%d expert_cache_mib=%d vram_free_mib=%d conversation_cache_mib=%d "
                            "conversation_cache_slots=%d\n",
                            kv_is_int() ? (kv_int_bits() == 4 ? "int4" : "int8") : (g_kv_fp8 ? "fp8" : "f16"), ctx, n_gpu,
                            reported_slots, reported_slots ? " slot_read=1" : "", expert_slots,
                            (int) (expert_bytes >> 20), (int) (vram_free_b >> 20), (int) conv_cache_mib, n_snaps);
            } else {
                std::printf("INFO engine=ik2 engine_kind=ik2 model=Inkling-2 kv=%s ctx=%d gpus=%d batch_slots=%d%s\n",
                            kv_is_int() ? (kv_int_bits() == 4 ? "int4" : "int8") : (g_kv_fp8 ? "fp8" : "f16"), ctx, n_gpu,
                            reported_slots, reported_slots ? " slot_read=1" : "");
            }
            std::printf("READY %d stop\n", ctx);
        } else {
            std::printf("@READY ctx %d\n", ctx);
        }
        std::fflush(stdout);
        // stdin on a thread: a STOP while a request runs is seen at once (the request loop looks at g_cancel)
        // (static: the reader thread is detached and may outlive this scope at QUIT)
        static std::mutex in_m;
        static std::condition_variable in_cv;
        static std::deque<std::string> in_q;
        static bool in_eof = false;
        static std::atomic<bool> busy{false};
        std::thread([]() {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (l == "STOP" || l.rfind("STOP ", 0) == 0) {
                    if (busy) g_cancel = 1;
                    continue;
                }
                std::lock_guard<std::mutex> lk(in_m);
                in_q.push_back(l);
                in_cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(in_m);
            in_eof = true;
            in_cv.notify_one();
        }).detach();
        auto next_line = [&](std::string& l) {
            std::unique_lock<std::mutex> lk(in_m);
            in_cv.wait(lk, [&] { return !in_q.empty() || in_eof; });
            if (in_q.empty()) return false;
            l = std::move(in_q.front());
            in_q.pop_front();
            return true;
        };
        // Non-blocking companion to next_line: while batch slots run we must keep serving new BGEN/BSTOP lines.
        auto poll_line = [&](std::string& l) {
            std::lock_guard<std::mutex> lk(in_m);
            if (in_q.empty()) return false;
            l = std::move(in_q.front());
            in_q.pop_front();
            return true;
        };
        auto rq_pos = [](const Ik2Req& r) {
            return r.prefill_pos < r.prompt_len ? r.prefill_pos : (int) r.hist.size();
        };
        auto any_active = [&]() {
            for (const auto& r : reqs) if (r.active) return true;
            return false;
        };
        auto finish_req = [&](int b, const char* reason) {
            Ik2Req& r = reqs[b];
            const double d0 = r.decode_t0 > 0 ? r.decode_t0 : r.t_start;
            std::printf("BDONE %d %d %s %.1f read=%.1f\n", b, r.generated, reason, (now() - d0) * 1e3, r.prompt_ms);
            r.active = false;
            r.cancel = false;
        };
        // The device holds one state at a time. Park the outgoing slot's conv state + full/sliding KV in its own
        // snapshot, then restore the incoming one. Solo requests never call this (g_parallel==1 path unchanged).
        auto ensure_loaded = [&](int b) {
            if (g_batch_decode) {
                // Every request has its own resident KV/conv state; no snapshot swap.  Preflills use M.V, so point
                // the layer state at this slot's storage, and zero conv histories on first use.
                if (loaded_slot == b) return;
                Ik2Req& r = reqs[b];
                if (!r.have_state) { reset_batch_slot(b, M.c.KV * M.c.D, M.c.C); r.have_state = true; }
                load_batch_slot(M, b);
                loaded_slot = b;
                return;
            }
            if (loaded_slot == b) return;
            if (loaded_slot >= 0 && reqs[loaded_slot].active) {
                Ik2Req& p = reqs[loaded_slot];
                p.state.pos = rq_pos(p);
                snap_copy(p.state, true);
            }
            Ik2Req& r = reqs[b];
            r.state.pos = rq_pos(r);
            if (r.have_state) snap_copy(r.state, false);
            else { reset(); r.have_state = true; }
            loaded_slot = b;
        };
        auto admit_bgen = [&](const std::string& ln) {
            std::istringstream s(ln);
            std::string cmd, ids_s, w;
            int b = -1, mx = 0;
            double temp = 0, top_p = 1, top_k = 0;
            long long seed = -1;
            s >> cmd >> b >> mx;
            while (s >> w) {
                const size_t eq = w.find('=');
                if (eq == std::string::npos) { ids_s = w; continue; }
                const std::string k = w.substr(0, eq), v = w.substr(eq + 1);
                if (k == "temperature") temp = std::atof(v.c_str());
                else if (k == "top_p") top_p = std::atof(v.c_str());
                else if (k == "top_k") top_k = std::atof(v.c_str());
                else if (k == "seed") seed = std::atoll(v.c_str());
            }
            const std::vector<int> ids = parse_ints(ids_s);
            const char* err = nullptr;
            if (cmd != "BGEN") err = "BGENI is not supported by the ik2 batch scheduler (no per-slot pictures)";
            else if (b < 0 || b >= g_parallel) err = "no such batch slot";
            else if (reqs[b].active) err = "batch slot is busy";
            else if (mx <= 0) err = "bad max_new";
            else if (ids.empty()) err = "an empty prompt";
            else if ((int) ids.size() + mx > ctx) err = "the prompt and max_tokens pass the context";
            if (err) {
                std::printf("ERR BGEN: %s (--batch %d)\n", err, g_parallel);
                std::fflush(stdout);
                return;
            }
            Ik2Req& r = reqs[b];
            r.active = true; r.cancel = false; r.admitted = true;
            r.slot = b; r.max_new = mx; r.generated = 0;
            r.prompt_len = (int) ids.size(); r.prefill_pos = 0; r.next_tok = -1;
            r.temp = temp; r.top_p = top_p; r.top_k = top_k; r.seed = seed;
            r.rng.seed(seed >= 0 ? (uint64_t) seed : (uint64_t) (1234 + b));
            r.prompt_ids = ids; r.hist.clear(); r.sig.assign(ids.size(), 0); r.stops = eos_ids;
            r.last.clear(); r.cand = false; r.next_tok = -1;
            r.have_state = false; r.state.pos = 0;
            r.t_start = now(); r.prompt_ms = 0; r.decode_t0 = 0; r.cached_prefix = 0;
            // slot_read=1: zero-HoL admission goes on the control stream before the prompt is read.
            std::printf("DONE 0 %zu 0.0 0.0 admit 0 0 0\n", r.prompt_ids.size());
            std::printf("BADM %d 1\n", b);
            std::fflush(stdout);
        };
        auto batch_control = [&](const std::string& ln, bool& quit) {
            if (ln == "QUIT") {
                quit = true;
                for (auto& r : reqs) r.cancel = true;
                return;
            }
            if (ln.rfind("BSTOP ", 0) == 0 || ln.rfind("CANCEL ", 0) == 0) {
                const int b = std::atoi(ln.c_str() + (ln[0] == 'B' ? 6 : 7));
                if (b >= 0 && b < g_parallel && reqs[b].active) reqs[b].cancel = true;
                return;
            }
            if (ln == "BYIELD" || ln.rfind("BYIELD ", 0) == 0 || ln.rfind("BSTAT", 0) == 0) return;
            if (ln.rfind("BGEN ", 0) == 0 || ln.rfind("BGENI ", 0) == 0) { admit_bgen(ln); return; }
            if (ln == "STOP" || ln.rfind("STOP ", 0) == 0) return;   // solo-only cancel
            if (ln == "GEN" || ln.rfind("GEN ", 0) == 0)
                std::printf("ERR solo GEN while batch slots are active\nDONE 0 0 0.0 0.0 cancel 0 0 0\n");
        };
        // One continuous-batching step: decode one token for every decoding slot first (no head-of-line
        // blocking), then fill the remaining token budget with chunked prefill, round-robin over slots.
        // P1 calls the existing single-token / single-chunk forward for each step; the kernels are unchanged.
        auto batch_step = [&]() {
            if (!any_active()) { busy = false; return; }
            busy = true;
            ++batch_steps;
            // IK2_BATCH_SOLO_FAST: exactly one slot decoding, none prefilling -> the existing solo T=1
            // fast path (forward -> fast_token resident graph), bit-exact and ~107 t/s/slot. The batched
            // branch below needs >=2 ready rows, so a lone slot would otherwise take the general forward.
            if (g_batch_solo_fast) {
                int solo = -1, n_dec = 0, n_pf_chk = 0;
                for (int b = 0; b < g_parallel; ++b) {
                    const Ik2Req& r = reqs[b];
                    if (!r.active) continue;
                    if (r.prefill_pos < r.prompt_len) { ++n_pf_chk; continue; }
                    if (r.next_tok < 0 || r.cancel || r.generated >= r.max_new || r.temp > 0) continue;
                    ++n_dec; solo = b;
                }
                if (solo >= 0 && n_dec == 1 && n_pf_chk == 0) {
                    Ik2Req& r = reqs[solo];
                    ensure_loaded(solo);
                    const int tok = r.next_tok;
                    std::vector<float> lo;
                    std::vector<int> ag;
                    forward(M, &tok, 1, rq_pos(r), false, OutMode::kArgmax, std::string(), lo, &ag, t_exp);
                    const int ntok = ag.empty() ? 0 : ag[0];
                    r.hist.push_back(tok);
                    ++r.generated;
                    r.next_tok = ntok;
                    r.cand = false;
                    r.last.clear();
                    std::printf("BT %d %d\n", solo, ntok);
                    ++batch_tokens;
                    if (r.cancel || std::find(r.stops.begin(), r.stops.end(), ntok) != r.stops.end())
                        finish_req(solo, r.cancel ? "cancel" : "stop");
                    else if (r.generated >= r.max_new) finish_req(solo, "length");
                    if (batch_stats && batch_steps % 64 == 0) {
                        int n_act = 0;
                        for (const auto& r : reqs) if (r.active) ++n_act;
                        std::printf("BSTAT active=%d steps=%lld tokens=%lld\n", n_act, batch_steps, batch_tokens);
                    }
                    if (!any_active()) busy = false;
                    batch_dirty = true;
                    std::fflush(stdout);
                    return;
                }
            }
            // IK2_BATCH_ATTN: one greedy decode forward for all ready slots.  Sampling slots (and single ready
            // slots) keep the original serial path, so their RNG/order and output are unchanged.
            std::vector<char> batched((size_t) g_parallel, 0);
            if (g_batch_decode) {
                std::vector<int> ids, toks, poss;
                for (int b = 0; b < g_parallel; ++b) {
                    Ik2Req& r = reqs[b];
                    if (!r.active || r.prefill_pos < r.prompt_len || r.next_tok < 0 || r.cancel ||
                        r.generated >= r.max_new || r.temp > 0) continue;
                    ensure_loaded(b);
                    ids.push_back(b); toks.push_back(r.next_tok); poss.push_back(rq_pos(r));
                }
                if (g_ar_dual && !ids.empty()) {
                    // One or two greedy slots ready: one replay of EACH ready slot's own resident T=1 graph.
                    // Same kernels/order as a solo token, so the emitted ids stay bit-exact.  With IK2_AR_STREAM2
                    // the two replays are submitted to g_s and g_s1 without a host sync between them, then both
                    // streams are waited before either argmax is read.  One ready slot keeps the serial path.
                    // IK2_AR_KSTEP branch: K tokens per graph replay, both slots submitted back-to-back.
                    // This MUST precede the one-step ar2_ready path: a K graph writes K ids into argmax_hK,
                    // which the one-step bookkeeping (W.argmax_h[0]) would misread as a single token.
                    const bool kstep_ready = g_ar_kstep && g_ar_kstep_ready && ids.size() == 2 &&
                        g_ar_kgraph[0].exec && g_ar_kgraph[1].exec;
                    if (kstep_ready) {
                        fast_token_k(M, ids[0], toks[0], poss[0], true);
                        fast_token_k(M, ids[1], toks[1], poss[1], true);
                        CU(cudaStreamSynchronize(g_s));
                        CU(cudaStreamSynchronize(g_s1));
                        if (g_ar2_stats) {
                            float ms0 = 0.0f, ms1 = 0.0f;
                            CU(cudaEventElapsedTime(&ms0, g_ar2_ev[0], g_ar2_ev[1]));
                            CU(cudaEventElapsedTime(&ms1, g_ar2_ev[2], g_ar2_ev[3]));
                            ar2_sum_slot0 += ms0; ar2_sum_slot1 += ms1;
                            ar2_sum_ms += (double) ms0 + ms1;
                            ar2_wall_ms += std::max((double) ms0, (double) ms1);
                            ++ar2_steps;
                            if (batch_stats && batch_steps % 64 == 0 && ar2_wall_ms > 0.0)
                                std::printf("AR2KSTAT steps=%lld tokens/sample=%d slot0=%.3fms slot1=%.3fms wall=%.3fms speedup=%.2fx\n",
                                            ar2_steps, g_ar_K, ar2_sum_slot0 / ar2_steps, ar2_sum_slot1 / ar2_steps,
                                            ar2_wall_ms / ar2_steps, ar2_sum_ms / ar2_wall_ms);
                        }
                        for (int i = 0; i < 2; ++i) {
                            const int b = ids[i];
                            Ik2Req& r = reqs[b];
                            Work& W = (b == 1) ? g_ar_work1 : dev_work(0);
                            const int nvalid = std::min(W.kstep_K, std::max(0, r.max_new - r.generated));
                            int cur = toks[i];
                            for (int s = 0; s < nvalid; ++s) {
                                const int ntok = W.argmax_hK[s];
                                r.hist.push_back(cur);
                                ++r.generated;
                                r.next_tok = ntok;
                                r.cand = false;
                                r.last.clear();
                                std::printf("BT %d %d\n", b, ntok);
                                ++batch_tokens;
                                if (r.cancel || std::find(r.stops.begin(), r.stops.end(), ntok) != r.stops.end()) {
                                    finish_req(b, r.cancel ? "cancel" : "stop");
                                    break;
                                }
                                if (r.generated >= r.max_new) { finish_req(b, "length"); break; }
                                cur = ntok;
                            }
                            batched[b] = 1;
                        }
                        loaded_slot = -1;   // prefill must re-bind M.V after two graph replays
                        batch_dirty = true;
                        if (!any_active()) busy = false;
                        std::fflush(stdout);
                        return;
                    }
                    const bool ar2_ready = g_ar_stream2 && ids.size() == 2 &&
                        g_ar_fgraph[0][(int) OutMode::kArgmax].exec && g_ar_fgraph[1][(int) OutMode::kArgmax].exec;
                    if (ar2_ready) {
                        std::vector<float> lo;
                        std::vector<int> ag[2];
                        fast_token(M, toks[0], poss[0], OutMode::kArgmax, lo, &ag[0], ids[0], true);
                        fast_token(M, toks[1], poss[1], OutMode::kArgmax, lo, &ag[1], ids[1], true);
                        CU(cudaStreamSynchronize(g_s));
                        CU(cudaStreamSynchronize(g_s1));
                        if (g_ar2_stats) {
                            float ms0 = 0.0f, ms1 = 0.0f;
                            CU(cudaEventElapsedTime(&ms0, g_ar2_ev[0], g_ar2_ev[1]));
                            CU(cudaEventElapsedTime(&ms1, g_ar2_ev[2], g_ar2_ev[3]));
                            ar2_sum_slot0 += ms0; ar2_sum_slot1 += ms1;
                            ar2_sum_ms += (double) ms0 + ms1;
                            ar2_wall_ms += std::max((double) ms0, (double) ms1);
                            ++ar2_steps;
                            if (batch_stats && batch_steps % 64 == 0 && ar2_wall_ms > 0.0)
                                std::printf("AR2STAT steps=%lld slot0=%.3fms slot1=%.3fms wall=%.3fms speedup=%.2fx\n",
                                            ar2_steps, ar2_sum_slot0 / ar2_steps, ar2_sum_slot1 / ar2_steps,
                                            ar2_wall_ms / ar2_steps, ar2_sum_ms / ar2_wall_ms);
                        }
                        for (int i = 0; i < 2; ++i) {
                            const int b = ids[i];
                            Ik2Req& r = reqs[b];
                            Work& W = (b == 1) ? g_ar_work1 : dev_work(0);
                            const int ntok = W.argmax_h[0];
                            r.hist.push_back(toks[i]);
                            ++r.generated;
                            r.next_tok = ntok;
                            r.cand = false;
                            r.last.clear();
                            std::printf("BT %d %d\n", b, ntok);
                            ++batch_tokens;
                            batched[b] = 1;
                            if (r.cancel || std::find(r.stops.begin(), r.stops.end(), ntok) != r.stops.end())
                                finish_req(b, r.cancel ? "cancel" : "stop");
                            else if (r.generated >= r.max_new) finish_req(b, "length");
                        }
                    } else {
                        for (size_t i = 0; i < ids.size(); ++i) {
                            const int b = ids[i];
                            Ik2Req& r = reqs[b];
                            std::vector<float> lo;
                            std::vector<int> ag;
                            fast_token(M, toks[i], poss[i], OutMode::kArgmax, lo, &ag, b);
                            const int ntok = ag.empty() ? 0 : ag[0];
                            r.hist.push_back(toks[i]);
                            ++r.generated;
                            r.next_tok = ntok;
                            r.cand = false;
                            r.last.clear();
                            std::printf("BT %d %d\n", b, ntok);
                            ++batch_tokens;
                            batched[b] = 1;
                            if (r.cancel || std::find(r.stops.begin(), r.stops.end(), ntok) != r.stops.end())
                                finish_req(b, r.cancel ? "cancel" : "stop");
                            else if (r.generated >= r.max_new) finish_req(b, "length");
                        }
                    }
                    loaded_slot = -1;   // prefill must re-bind M.V to the right slot after two graph replays
                } else if (ids.size() >= 2 && (int) ids.size() <= BatchSlots::kMaxRows) {
                    std::vector<int> ag;
                    forward_batch(M, toks.data(), (int) ids.size(), poss.data(), ids.data(), OutMode::kArgmax,
                                  &ag, t_exp);
                    if ((int) ag.size() == (int) ids.size()) {
                        for (size_t i = 0; i < ids.size(); ++i) {
                            const int b = ids[i], ntok = ag[i];
                            Ik2Req& r = reqs[b];
                            r.hist.push_back(toks[i]);
                            ++r.generated;
                            r.next_tok = ntok;
                            r.cand = false;
                            r.last.clear();
                            std::printf("BT %d %d\n", b, ntok);
                            ++batch_tokens;
                            batched[b] = 1;
                            if (r.cancel || std::find(r.stops.begin(), r.stops.end(), ntok) != r.stops.end())
                                finish_req(b, r.cancel ? "cancel" : "stop");
                            else if (r.generated >= r.max_new) finish_req(b, "length");
                        }
                        loaded_slot = -1;
                    }
                }
            }
            for (int b = 0; b < g_parallel; ++b) {
                if (batched[b]) continue;
                Ik2Req& r = reqs[b];
                if (!r.active || r.prefill_pos < r.prompt_len || r.next_tok < 0) continue;
                if (r.cancel) { finish_req(b, "cancel"); continue; }
                if (r.generated >= r.max_new) { finish_req(b, "length"); continue; }
                ensure_loaded(b);
                const int tok = r.next_tok;
                int ntok = -1;
                if (r.temp <= 0) {
                    std::vector<float> lo;
                    std::vector<int> ag;
                    forward(M, &tok, 1, rq_pos(r), false, OutMode::kArgmax, std::string(), lo, &ag, t_exp);
                    ntok = ag.empty() ? 0 : ag[0];
                    r.cand = false;
                    r.last.clear();
                } else {
                    std::vector<float> lo;
                    forward(M, &tok, 1, rq_pos(r), false, use_cand ? OutMode::kCand : OutMode::kLogits,
                            std::string(), lo, nullptr, t_exp);
                    r.cand = use_cand && g_cand.n >= 0;
                    r.last = std::move(lo);
                    ntok = r.cand ? pick_cand(r.temp, r.top_p, r.top_k, r.rng)
                                  : pick(r.last.data(), r.temp, r.top_p, r.top_k, r.rng);
                }
                r.hist.push_back(tok);
                ++r.generated;
                r.next_tok = ntok;
                std::printf("BT %d %d\n", b, ntok);
                ++batch_tokens;
                if (r.cancel || std::find(r.stops.begin(), r.stops.end(), ntok) != r.stops.end()) {
                    finish_req(b, r.cancel ? "cancel" : "stop");
                    continue;
                }
                if (r.generated >= r.max_new) finish_req(b, "length");
            }
            int n_pf = 0;
            for (const auto& r : reqs) if (r.active && r.prefill_pos < r.prompt_len) ++n_pf;
            if (n_pf > 0) {
                const int per = std::max(1, std::min(chunk, 512) / n_pf);
                for (int b = 0; b < g_parallel; ++b) {
                    Ik2Req& r = reqs[b];
                    if (!r.active || r.prefill_pos >= r.prompt_len) continue;
                    if (r.cancel) { finish_req(b, "cancel"); continue; }
                    ensure_loaded(b);
                    const int p0 = r.prefill_pos;
                    const int n = std::min(per, r.prompt_len - p0);
                    const bool last_c = p0 + n == r.prompt_len;
                    std::vector<float> lo;
                    forward(M, r.prompt_ids.data() + p0, n, p0, false,
                            last_c ? OutMode::kLogits : OutMode::kNone, std::string(), lo, nullptr, t_exp);
                    r.prefill_pos += n;
                    if (last_c) {
                        r.hist = r.prompt_ids;
                        r.decode_t0 = now();
                        r.prompt_ms = (r.decode_t0 - r.t_start) * 1e3;
                        const int first = pick(lo.data(), r.temp, r.top_p, r.top_k, r.rng);
                        r.last = std::move(lo);
                        r.next_tok = first;
                        r.generated = 1;
                        r.cand = false;
                        std::printf("BT %d %d\n", b, first);
                        ++batch_tokens;
                        if (r.cancel || std::find(r.stops.begin(), r.stops.end(), first) != r.stops.end())
                            finish_req(b, r.cancel ? "cancel" : "stop");
                        else if (r.generated >= r.max_new) finish_req(b, "length");
                    }
                }
            }
            if (batch_stats && batch_steps % 64 == 0) {
                int n_act = 0;
                for (const auto& r : reqs) if (r.active) ++n_act;
                std::printf("BSTAT active=%d steps=%lld tokens=%lld\n", n_act, batch_steps, batch_tokens);
            }
            if (!any_active()) busy = false;
            batch_dirty = true;
            std::fflush(stdout);
        };
        std::string line;
        std::mt19937_64 rng(1234);
        bool quit = false;
        while (!quit) {
            // With batch slots active we poll control lines and run one scheduler step per iteration: a
            // blocking next_line here would stall already-running requests behind an idle stdin.
            if (g_parallel > 1 && any_active()) {
                while (poll_line(line)) {
                    batch_control(line, quit);
                    if (quit) break;
                }
                if (quit) break;
                batch_step();
                continue;
            }
            if (!next_line(line)) break;
            if (g_parallel > 1) {
                if (line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0) {
                    admit_bgen(line);
                    batch_step();
                    continue;
                }
                if (line == "QUIT" || line.rfind("BSTOP ", 0) == 0 || line.rfind("CANCEL ", 0) == 0 ||
                    line == "BYIELD" || line.rfind("BYIELD ", 0) == 0 || line.rfind("BSTAT", 0) == 0) {
                    batch_control(line, quit);
                    if (quit) break;
                    continue;
                }
            }
            std::istringstream in(line);
            std::string cmd, stops_s, ids_s, emb_path;
            int max_new = 0, top_k = 0;
            double temp = 0, top_p = 1;
            long long seed = -1;
            in >> cmd;
            if (cmd == "QUIT") {
                pop_dump();
                break;
            }
            // GENI: GEN with the request's pictures - "GENI <max_new> [key=value ...] <pictures file> <ids>", the file
            // the image encoder's rows (SVE1 records back to back), one row an image token of the ids in order
            const bool geni = cmd == "GENI";
            if (cmd != "GEN" && !geni) {   // (PAUSE / WRAP and the like: not offered in READY, so not sent)
                if (!serve_strata) {
                    std::printf("@END error unknown_command 0 0 0 0\n");
                    std::fflush(stdout);
                }
                continue;
            }
            std::vector<int> stops;
            if (serve_strata) {
                in >> max_new;
                std::string w;
                while (in >> w) {
                    const size_t eq = w.find('=');
                    if (eq == std::string::npos) {
                        if (geni && emb_path.empty() && ids_s.empty() && w.find_first_not_of("0123456789,-") != std::string::npos)
                            emb_path = w;   // (the pictures' file comes before the ids)
                        else
                            ids_s = w;
                        continue;
                    }
                    const std::string k = w.substr(0, eq), v = w.substr(eq + 1);
                    if (k == "temperature") temp = std::atof(v.c_str());
                    else if (k == "top_p") top_p = std::atof(v.c_str());
                    else if (k == "top_k") top_k = std::atoi(v.c_str());
                    else if (k == "seed") seed = std::atoll(v.c_str());
                }
                stops = eos_ids;
            } else {
                in >> max_new >> temp >> top_p >> seed >> stops_s >> ids_s;
                stops = parse_ints(stops_s);
            }
            const std::vector<int> ids = parse_ints(ids_s);
            if (ids.empty() || (int) ids.size() + max_new > ctx) {
                if (serve_strata)
                    std::printf("ERR %s\nDONE 0 %zu 0 0 cancel 0 0 0\n",
                                ids.empty() ? "an empty prompt" :
                                    ("the prompt and max_tokens pass the context (" + std::to_string(ctx) + ")").c_str(),
                                ids.size());
                else
                    std::printf("@END error %s %zu 0 0 0\n", ids.empty() ? "empty_prompt" : "context_length", ids.size());
                std::fflush(stdout);
                continue;
            }
            g_img.clear();
            std::vector<uint64_t> sig(ids.size(), 0);
            if (!emb_path.empty()) {
                const std::string err = load_pictures(emb_path, ids, image_token, c.C, g_img, sig);
                if (!err.empty()) {
                    g_img.clear();
                    std::printf("ERR %s\nDONE 0 %zu 0 0 cancel 0 0 0\n", err.c_str(), ids.size());
                    std::fflush(stdout);
                    continue;
                }
            }
            busy = true;
            if (g_parallel > 1 && batch_dirty) {
                // a solo GEN after batch slots: batch left a different prompt's state in the device
                reset();
                batch_dirty = false;
                loaded_slot = -1;
            }
            if (seed >= 0) rng.seed((uint64_t) seed);
            g_cancel = 0;
            // IK2_SERVE_DSPEC (default OFF): greedy (temp<=0) spec decoding needs a drafter and ctx room
            // for one verify window. temp>0 or a too-short ctx keep the AR path byte-identical. When a
            // prefix is reused the drafter's ring context is not in Snap, but the target verifies every
            // draft so the emitted stream stays bit-exact regardless of the drafter's cached state.
            int spec_B = ik2::draft::kBlock;
            if (getenv("IK2_SPEC_N")) spec_B = std::atoi(getenv("IK2_SPEC_N"));
            if (spec_B < 2) spec_B = 2;
            if (spec_B > ik2::draft::kBlock) spec_B = ik2::draft::kBlock;
            const bool spec_active = serve_strata && g_serve_dspec && drafter && temp <= 0 &&
                                     (int) ids.size() + spec_B <= ctx;
            size_t L = 0;   // (a picture's position is the same only with the same rows: its hash beside its token)
            auto hsig = [&](size_t i) { return i < hist_sig.size() ? hist_sig[i] : (uint64_t) 0; };
            while (L < hist.size() && L < ids.size() && hist[L] == ids[L] && hsig(L) == sig[L]) ++L;
            // Which parked conversation to go back to: the one covering the longest prefix of THIS prompt.  A snapshot
            // carries the tokens it was taken at, so it is matched against the prompt itself; matching on position
            // alone (L, i.e. what the device happens to hold) meant that one request sharing only the opening tokens -
            // an agent's title or summary call between two turns - left every snapshot unusable, the whole prompt was
            // read again, and reset() threw the cache away.
            int best = -1;
            for (int i = 0; i < (int) snaps.size(); ++i) {
                const Snap& sn = snaps[i];
                if (sn.pos <= 0 || sn.toks.size() >= ids.size()) continue;   // (one token stays to run: its logits)
                if (best >= 0 && sn.pos <= snaps[best].pos) continue;
                if (!std::equal(sn.toks.begin(), sn.toks.end(), ids.begin())) continue;
                bool same = true;
                for (size_t k = 0; k < sn.sigs.size() && same; ++k) same = sn.sigs[k] == (k < sig.size() ? sig[k] : 0);
                if (same) best = i;
            }
            // The live state answers this prompt only when it ends where the prompt ends (L is where they part) and
            // its logits are still there; a parked conversation reaching further in wins when one does.
            const bool live_ok = L == hist.size() && (L < ids.size() || !last.empty());
            if (best >= 0 && ((int) L < snaps[best].pos || !live_ok)) {
                const int sp = snaps[best].pos;
                const double tr0 = now();
                snap_copy(snaps[best], false);
                const double tr1 = now();
                hist = snaps[best].toks;   // (the state and the tokens it covers, together)
                hist_sig = snaps[best].sigs;
                last.clear();
                L = (size_t) sp;
                if (serve_strata) {
                    // (stderr: the server's log is the engine's stderr, and the Monitor's cache card reads it)
                    const size_t moved = snap_host(snaps[best]);
                    std::fprintf(stderr,
                                 "strata serve: conversation cache: restored %d tokens; parked=%d bytes=%zu; "
                                 "%.1f ms (%.2f GB/s)\n",
                                 sp, count_parked(), conv_host_bytes, (tr1 - tr0) * 1e3,
                                 (tr1 - tr0) > 1e-9 ? moved / (tr1 - tr0) / 1e9 : 0.0);
                    std::fflush(stderr);
                }
            } else if (!live_ok) {
                reset(), L = 0;
            }
            const int cached = (int) L;
            uint64_t monitor_h0 = 0, monitor_m0 = 0;
            if (monitor_info) {
                monitor_h0 = monitor_counts(true);
                monitor_m0 = monitor_counts(false);
            }
            if (serve_strata) {   // (the server reports how much of the prompt was already there: cached tokens)
                std::printf("RESUME %d\n", cached);
                std::fflush(stdout);
            }
            const double t0 = now();
            const uint64_t miss0 = dev_cache(0).misses, bytes0 = dev_cache(0).bytes_up;
            for (auto& v : g_cnt_hist) v = 0;
            for (auto& v : g_src_hist) v = 0;
            bool cancelled = false;
            if (L < ids.size()) {
                const int T_sup = g_dev[0].W.T_sup;
                const int sub_chunk = std::min(g_dev[0].W.Tm, 512);
                if (T_sup > 0 && (n_gpu == 1 || dual()) && (int) (ids.size() - L) > 64) {
                    BigScratch bs;
                    const int T_pass = big_begin(M, g_dev[0].W, dev_cache(0), (int) (ids.size() - L), (int) ids.size(), bs);
                    for (size_t p0 = L; p0 < ids.size(); p0 += T_pass) {
                        if (g_cancel) {   // (a prompt half run: nothing of it is kept)
                            cancelled = true;
                            break;
                        }
                        const int n = (int) std::min<size_t>(T_pass, ids.size() - p0);
                        const bool last_c = p0 + n == ids.size();
                        std::vector<float> lo;
                        forward_prefill_sup(M, ids.data() + p0, n, (int) p0, sub_chunk,
                                            last_c ? OutMode::kLogits : OutMode::kNone, lo, nullptr, t_exp);
                        if (last_c) last = std::move(lo);
                        if (serve_strata) {   // (read, total, ms, tok/s, reused)
                            const double ms = (now() - t0) * 1e3;
                            std::printf("PP %zu %zu %.0f %.1f %d\n", p0 + n, ids.size(), ms,
                                        (p0 + n - L) / std::max(1e-3, ms / 1e3), (int) L);
                            std::fflush(stdout);
                        }
                    }
                    big_end(g_dev[0].W, dev_cache(0), bs);
                } else {
                    for (size_t p0 = L; p0 < ids.size(); p0 += chunk) {
                        if (g_cancel) {   // (a prompt half run: nothing of it is kept)
                            cancelled = true;
                            break;
                        }
                        const int n = (int) std::min<size_t>(chunk, ids.size() - p0);
                        const bool last_c = p0 + n == ids.size();
                        std::vector<float> lo;
                        forward(M, ids.data() + p0, n, (int) p0, false, last_c ? OutMode::kLogits : OutMode::kNone,
                                std::string(), lo, nullptr, t_exp);
                        if (last_c) last = std::move(lo);
                        if (serve_strata) {   // (read, total, ms, tok/s, reused) - a short read still shows progress
                            const double ms = (now() - t0) * 1e3;
                            std::printf("PP %zu %zu %.0f %.1f %d\n", p0 + n, ids.size(), ms,
                                        (p0 + n - L) / std::max(1e-3, ms / 1e3), (int) L);
                            std::fflush(stdout);
                        }
                        if (spec_active && drafter && g_taps) {
                            // Seed the drafter from the chunk this forward() just ran: g_taps holds exactly those rows.
                            // Doing it per chunk is what the CLI path does; the old code seeded once with the whole
                            // prompt length, which read past the taps buffer for any prompt longer than one chunk.
                            ik2::draft::write_context(drafter, g_taps, g_taps_ld, 0, n, (int) p0);
                            if (last_c) ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, n - 1);
                        }
                    }
                }
            }
            if (cancelled) {
                reset();
                if (serve_strata) {
                    if (monitor_info) {
                        const uint64_t h = monitor_counts(true), m = monitor_counts(false);
                        const uint64_t dh = h > monitor_h0 ? h - monitor_h0 : 0;
                        const uint64_t dm = m > monitor_m0 ? m - monitor_m0 : 0;
                        std::printf("DONE 0 %zu %.1f 0 cancel 0 0 %d %llu %llu\n", ids.size(), (now() - t0) * 1e3,
                                    cached, (unsigned long long) dh, (unsigned long long) (dh + dm));
                    } else {
                        std::printf("DONE 0 %zu %.1f 0 cancel 0 0 %d\n", ids.size(), (now() - t0) * 1e3, cached);
                    }
                } else std::printf("@END cancelled %zu %d %.3f %.3f\n", ids.size(), cached, now() - t0, 0.0);
                std::fflush(stdout);
                busy = false;
                continue;
            }
            if (L < ids.size()) {
                hist = ids;
                hist_sig = sig;
                // The state at the end of the prompt: an agent's next turn begins with exactly these tokens, so it
                // reads only what follows them.
                const int parked = park_state(hist, hist_sig);
                if (serve_strata && parked > 0) {
                    std::fprintf(stderr, "strata serve: conversation cache: parked %d tokens; parked=%d bytes=%zu\n",
                                 parked, count_parked(), conv_host_bytes);
                    std::fflush(stderr);
                }
            }
            const double t1 = now();
            if (monitor_info) {
                monitor_h0 = monitor_counts(true);
                monitor_m0 = monitor_counts(false);
            }
            if (getenv("IK2_STATS"))
                std::printf("prefill stats: %zu tokens run, %llu expert copies (%.1f GiB) in %.2f s; experts that ran 1 / 2 / 3-7 / 8+ "
                            "tokens: %llu / %llu / %llu / %llu\n",
                            ids.size() - (size_t) cached, (unsigned long long) (dev_cache(0).misses - miss0),
                            (dev_cache(0).bytes_up - bytes0) / 1073741824.0, t1 - t0,
                            (unsigned long long) g_cnt_hist[0], (unsigned long long) g_cnt_hist[1],
                            (unsigned long long) g_cnt_hist[2], (unsigned long long) g_cnt_hist[3]);
            if (getenv("IK2_STATS"))
                std::printf("prefill sources: used and cached %llu, used and prefetched %llu, used and copied after the "
                            "routing %llu, prefetched and unused %llu\n",
                            (unsigned long long) g_src_hist[0], (unsigned long long) g_src_hist[1],
                            (unsigned long long) g_src_hist[2], (unsigned long long) g_src_hist[3]);
            // Sampling helpers (pick/pick_cand/pick_top) and `use_cand` live above the serve loop so the batch
            // scheduler can share them; the solo path passes its own rng/options and keeps the same draw order.
            bool cand = false;   // the last token's logits came as candidates (g_cand), not in `last`
            const char* reason = "length";
            int n_out = 0;
            int greedy_tok = temp <= 0 && !last.empty() ? pick(last.data(), temp, top_p, top_k, rng) : -1;
            // IK2_SERVE_DSPEC (default OFF): greedy requests take the shared DSpark speculative loop,
            // which emits only the target's own argmax (accepted drafts equal it), so the stream is
            // bit-identical to the AR greedy path. The commit callback keeps `hist` aligned with the
            // tokens actually emitted (the last emitted token stays pending as the next anchor), so
            // prefix-cache reuse and the snapshot history never run ahead of the visible stream.
            if (spec_active && greedy_tok >= 0) {
                // The drafter was seeded per chunk by the prefill loop above (that is the only place g_taps holds
                // rows that belong to this request). A prompt that was entirely reused runs no chunk, so the
                // drafter keeps the cache it built for the same conversation on the previous request - the target
                // verifies every draft, so a stale drafter can only cost accept rate, never correctness.
                auto emit = [&](int t) -> bool {
                    std::printf("T %d\n", t);
                    std::fflush(stdout);
                    ++n_out;
                    if (g_cancel) { reason = "cancelled"; return true; }
                    if (std::find(stops.begin(), stops.end(), t) != stops.end()) { reason = "stop"; return true; }
                    return n_out >= max_new;   // reason stays "length"
                };
                auto commit = [&](int t) { hist.push_back(t); };
                const SpecStats st = generate_speculative(M, c, drafter, greedy_tok, (int) hist.size(), ctx, spec_B,
                                                          emit, commit, /*cli_diag=*/false, /*verify_check=*/0,
                                                          /*margin_check=*/false, t_exp);
                if (serve_strata && st.steps > 0) {
                    // Served requests used to throw this away, so nobody could tell whether speculation paid. The
                    // accept rate is the number that decides it, and it is a property of the text, not the engine.
                    std::fprintf(stderr,
                                 "strata serve: speculative: %d steps, %.2f tokens a step; drafting %.1f ms a step, "
                                 "verifying %.1f ms\n",
                                 st.steps, (double) n_out / st.steps, st.t_draft * 1e3 / st.steps,
                                 st.t_verify * 1e3 / st.steps);
                    std::fflush(stderr);
                }
            } else for (int i = 0; i < max_new; ++i) {
                if (g_cancel) {
                    reason = "cancelled";
                    break;
                }
                const int tok = temp <= 0 ? greedy_tok : cand ? pick_cand(temp, top_p, top_k, rng)
                                                              : pick(last.data(), temp, top_p, top_k, rng);
                ++n_out;
                std::printf(serve_strata ? "T %d\n" : "@TOK %d\n", tok);
                std::fflush(stdout);
                if (std::find(stops.begin(), stops.end(), tok) != stops.end()) {
                    reason = "stop";
                    break;
                }
                if (i + 1 == max_new) break;
                if (temp <= 0) {
                    std::vector<float> lo;
                    std::vector<int> ag;
                    forward(M, &tok, 1, (int) hist.size(), false, OutMode::kArgmax, std::string(), lo, &ag, t_exp);
                    greedy_tok = ag[0];
                    last.clear();
                } else {
                    std::vector<float> lo;
                    forward(M, &tok, 1, (int) hist.size(), false, use_cand ? OutMode::kCand : OutMode::kLogits,
                            std::string(), lo, nullptr, t_exp);
                    cand = use_cand && g_cand.n >= 0;
                    last = std::move(lo);
                }
                hist.push_back(tok);
            }
            // The state at the end of the reply is parked as well: a client that sends the assistant's message back
            // verbatim, or a tool call echoed into the next turn, then finds the reply's own tokens already read.
            // It happens AFTER the DONE line below, not before it: at a 48K conversation the copy is ~405 MB and it
            // used to sit inside the decode timer, which read 64 tok/s where the decode itself ran at 89.
            const double t_done = now();
            // IK2_PROF_FAST: the resident decode path's phase split for this request. On stderr, because stdout is
            // the protocol pipe; report() clears the accumulators so each request shows its own split.
            if (g_prof_fast) pf_fast::report(stderr);
            // IK2_PF_PROFILE: the prefill breakdown for this request, so a chunk longer than 4,096 tokens can be
            // measured at all (the CLI prompt is an argument and Windows caps it there). Same lines as the CLI
            // tail, on stderr because stdout is the protocol pipe, then cleared so each request stands alone.
            if (g_pf_profile) {
                const Prof& p = g_pf;
                std::fprintf(stderr, "strata serve: PREFILL profile: attention %.2f s, router %.2f s, uploads %.2f s,"
                                     " experts %.2f s, rest %.2f s\n", p.attn, p.route, p.upload, p.experts, p.rest);
                std::fprintf(stderr, "strata serve: PREFILL moe: host %.2f s, gpu %.2f s (sync %.2f s), calls %d;"
                                     " branches rg %ld mv %ld pf %ld\n", p.pf_host, p.pf_gpu, p.pf_sync, p.pf_calls,
                             p.br_rg, p.br_mv, p.br_pf);
                std::fprintf(stderr, "strata serve: PREFILL run_exp: gather %.3f s, decode_f16 %.3f s, gemm16 %.3f s,"
                                     " scatter %.3f s\n", p.pf_gather, p.pf_dec, p.pf_gemm, p.pf_scatter);
                std::fprintf(stderr, "strata serve: PREFILL attn: proj %.3f s, conv %.3f s, QK %.3f s, probs %.3f s,"
                                     " PV %.3f s, out %.3f s\n", p.a_proj, p.a_conv, p.a_qk, p.a_probs, p.a_pv, p.a_out);
                std::fprintf(stderr, "strata serve: PREFILL loop_gates %.3f s\n", p.pf_loopgate);
                std::fflush(stderr);
                g_pf = Prof();
            }
            if (serve_strata) {
                if (monitor_info) {
                    const uint64_t h = monitor_counts(true), m = monitor_counts(false);
                    const uint64_t dh = h > monitor_h0 ? h - monitor_h0 : 0;
                    const uint64_t dm = m > monitor_m0 ? m - monitor_m0 : 0;
                    std::printf("DONE %d %zu %.1f %.1f %s 0 0 %d %llu %llu\n", n_out, ids.size(), (t1 - t0) * 1e3,
                                (t_done - t1) * 1e3, std::strcmp(reason, "cancelled") == 0 ? "cancel" : reason, cached,
                                (unsigned long long) dh, (unsigned long long) (dh + dm));
                } else {
                    std::printf("DONE %d %zu %.1f %.1f %s 0 0 %d\n", n_out, ids.size(), (t1 - t0) * 1e3,
                                (t_done - t1) * 1e3, std::strcmp(reason, "cancelled") == 0 ? "cancel" : reason, cached);
                }
            } else
                std::printf("@END %s %zu %d %.3f %.3f\n", reason, ids.size(), cached, t1 - t0, t_done - t1);
            std::fflush(stdout);
            if (hist.size() > ids.size()) {
                const double tp0 = now();
                const int parked = park_state(hist, hist_sig);
                const double tp1 = now();
                if (serve_strata && parked > 0) {
                    const size_t moved = snap_host(snaps[(snap_next + (int) snaps.size() - 1) % (int) snaps.size()]);
                    std::fprintf(stderr,
                                 "strata serve: conversation cache: parked %d tokens; parked=%d bytes=%zu; "
                                 "%.1f ms (%.2f GB/s)\n",
                                 parked, count_parked(), conv_host_bytes, (tp1 - tp0) * 1e3,
                                 (tp1 - tp0) > 1e-9 ? moved / (tp1 - tp0) / 1e9 : 0.0);
                    std::fflush(stderr);
                }
            }
            busy = false;
        }
        return 0;
    }
    if ((int) prompt.size() + gen > ctx) die("prompt + gen past --ctx");
    // the prompt
    std::vector<float> logits;
    double t_exp = 0;
    const double tp = now();
    const bool all = !dump.empty();
    int first_best = -1;
    const int T_sup = g_dev[0].W.T_sup;
    const int sub_chunk = std::min(g_dev[0].W.Tm, 512);
    if (T_sup > 0 && (n_gpu == 1 || dual()) && !drafter && !all && (int) prompt.size() > 64) {
        BigScratch bs;
        const int T_pass = big_begin(M, g_dev[0].W, dev_cache(0), (int) prompt.size(), (int) prompt.size(), bs);
        if (T_pass != T_sup) std::printf("prefill passes of %d tokens (scratch lent by the expert cache)\n", T_pass);
        for (int p0 = 0; p0 < (int) prompt.size(); p0 += T_pass) {
            const int n = std::min(T_pass, (int) prompt.size() - p0);
            const bool last = p0 + n == (int) prompt.size();
            const OutMode om = (last && gen > 0) ? OutMode::kArgmax : OutMode::kNone;
            std::vector<float> lo;
            std::vector<int> ag;
            forward_prefill_sup(M, prompt.data() + p0, n, p0, sub_chunk, om, lo, &ag, t_exp);
            if (last && !ag.empty()) first_best = ag[0];
        }
        big_end(g_dev[0].W, dev_cache(0), bs);
    } else {
        for (int p0 = 0; p0 < (int) prompt.size(); p0 += chunk) {
            const int n = std::min(chunk, (int) prompt.size() - p0);
            const bool last = p0 + n == (int) prompt.size();
            const OutMode om = all ? OutMode::kLogits : (!last ? OutMode::kNone : ((gen > 0 || drafter) ? OutMode::kArgmax : OutMode::kNone));
            std::vector<float> lo;
            std::vector<int> ag;
            forward(M, prompt.data() + p0, n, p0, all, om, all ? dump : std::string(), lo, &ag, t_exp);
            if (all) logits.insert(logits.end(), lo.begin(), lo.end());
            else if (last && !ag.empty()) first_best = ag[0];
            if (drafter) {
                ik2::draft::write_context(drafter, g_taps, g_taps_ld, 0, n, p0);
                if (last) ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, n - 1);
            }
        }
    }
    const double prompt_s = now() - tp;
    std::printf("prompt: %zu tokens in %.2f s (%.1f tok/s); experts %.2f s of it, uploads %.2f s; cache %llu hits, "
                "%llu misses, %.1f GiB copied\n",
                prompt.size(), prompt_s, prompt.size() / prompt_s, t_exp, ec_sum([](ExpertCache& e) { return e.t_up; }),
                (unsigned long long) ec_sum([](ExpertCache& e) { return e.hits; }),
                (unsigned long long) ec_sum([](ExpertCache& e) { return e.misses; }),
                ec_sum([](ExpertCache& e) { return e.bytes_up; }) / 1073741824.0);
    std::fflush(stdout);
    if (all) {
        // log-probabilities of every prompt position
        std::vector<float> lp(logits.size());
        for (size_t t = 0; t < prompt.size(); ++t) {
            const float* r = logits.data() + t * c.n_vocab;
            double m = -1e30, s = 0;
            for (int v = 0; v < c.n_vocab; ++v) m = std::max(m, (double) r[v]);
            for (int v = 0; v < c.n_vocab; ++v) s += std::exp(r[v] - m);
            const double ls = m + std::log(s);
            for (int v = 0; v < c.n_vocab; ++v) lp[t * c.n_vocab + v] = (float) (r[v] - ls);
        }
        FILE* f = std::fopen((dump + "/logprobs.f32").c_str(), "wb");
        std::fwrite(lp.data(), 4, lp.size(), f);
        std::fclose(f);
        const float* r_last = logits.data() + (prompt.size() - 1) * c.n_vocab;
        first_best = (int) (std::max_element(r_last, r_last + c.n_vocab) - r_last);
        logits.clear();
    }
    // greedy generation
    const Prof pf_prefill = g_pf;   // snapshot prefill profiling before the reset
    g_pf = Prof{};
    std::vector<int> out;
    const double tg = now();
    const uint64_t h0 = (uint64_t) ec_sum([](ExpertCache& e) { return e.hits; }),
                   m0 = (uint64_t) ec_sum([](ExpertCache& e) { return e.misses; });
    int n_steps = 0;
    const int verify_check = getenv("IK2_VERIFY_CHECK") ? std::atoi(getenv("IK2_VERIFY_CHECK")) : 0;
    double t_draft = 0, t_verify = 0;
    std::vector<int> acc_hist(ik2::draft::kBlock, 0);
    // IK2_FORCE=FILE (comma-separated ids): the tokens fed are these instead of the model's own picks, and how often
    // its pick agrees with the next one is counted - two builds then run the same tokens (a fair speed comparison)
    // and the agreement measures their numerics on real text
    std::vector<int> force;
    int agree = 0;
    if (getenv("IK2_FORCE")) {
        FILE* ff = std::fopen(getenv("IK2_FORCE"), "r");
        if (!ff) die("IK2_FORCE: cannot read the file");
        int v;
        while (std::fscanf(ff, "%d,", &v) == 1) force.push_back(v);
        std::fclose(ff);
    }
    // IK2_LOGITS_OUT=FILE: every generated token's logits [n_vocab] F32 appended there (with IK2_FORCE: two builds'
    // logits over the same tokens, to compare)
    FILE* lo = getenv("IK2_LOGITS_OUT") ? std::fopen(getenv("IK2_LOGITS_OUT"), "wb") : nullptr;
    for (int i = 0, best = first_best; i < gen && !drafter; ++i) {
        const int tok = i < (int) force.size() ? force[i] : best;
        if (i < (int) force.size()) agree += best == force[i];
        out.push_back(tok);
        std::printf("%d%s", tok, i + 1 < gen ? "," : "\n");
        std::fflush(stdout);
        if (i + 1 == gen) break;
        std::vector<int> ag;
        if (lo) logits.clear();
        forward(M, &tok, 1, (int) prompt.size() + i, false, lo ? OutMode::kLogits : OutMode::kArgmax, std::string(),
                logits, &ag, t_exp);
        if (lo) {
            std::fwrite(logits.data(), sizeof(float), (size_t) c.n_vocab, lo);
            best = (int) (std::max_element(logits.begin(), logits.begin() + c.n_vocab) - logits.begin());
        } else {
            best = ag[0];
        }
    }
    if (lo) std::fclose(lo);
    if (!force.empty())
        std::printf("forced: %d tokens fed, the model's pick agreed with the next one %d times (%.2f%%)\n",
                    (int) std::min<size_t>(force.size(), out.size()), agree,
                    100.0 * agree / std::max<size_t>(1, std::min<size_t>(force.size(), out.size())));

    if (drafter) {
        use_dev(0);
        // draft 7 after the anchor, verify the 8 in one call, keep the agreeing prefix and the target's next token
        int anchor = first_best, P = (int) prompt.size();
        // IK2_SPEC_N (default 8): verify window B = N tokens (anchor + N-1 drafts), 2..8. Smaller B trades
        // accept-per-step against verify cost (fewer tokens, fewer experts touched); the drafter still computes
        // all kBlock-1 Markov steps, only the host copy / verify window is truncated. B=8 is the trained block.
        int B = ik2::draft::kBlock;
        if (getenv("IK2_SPEC_N")) B = std::atoi(getenv("IK2_SPEC_N"));
        if (B < 2) B = 2;
        if (B > ik2::draft::kBlock) B = ik2::draft::kBlock;
        const bool margin_check = getenv("IK2_MARGIN") != nullptr;
        SpecStats st = generate_speculative(M, c, drafter, anchor, P, ctx, B,
            [&](int t) { out.push_back(t); return (int) out.size() >= gen; },  // emit: append; stop at gen
            [](int) {},                                        // commit: CLI has no incremental history
            /*cli_diag=*/true, verify_check, margin_check, t_exp);
        n_steps = st.steps;
        t_draft = st.t_draft;
        t_verify = st.t_verify;
        acc_hist = std::move(st.acc);
        for (size_t j = 0; j < out.size(); ++j) std::printf("%d%s", out[j], j + 1 < out.size() ? "," : "\n");
    }
    const double gen_s = now() - tg;
    if (drafter && n_steps) {
        std::printf("speculative: %d steps, %.2f tokens a step (accepted drafts: ", n_steps,
                    (double) (out.size() - 1) / n_steps);
        for (int j = 0; j < ik2::draft::kBlock; ++j) std::printf("%d%s", acc_hist[j], j + 1 < ik2::draft::kBlock ? " " : "");
        std::printf("); drafting %.1f ms a step, verifying %.1f ms\n", t_draft * 1e3 / n_steps, t_verify * 1e3 / n_steps);
    }
    if (g_pool)
        std::printf("CPU lane: %d threads, %llu experts computed on the CPU (%.2f ms each at the end), %llu waited for\n",
                    g_pool->size(), (unsigned long long) g_n_cpu, g_t_cpu * 1e3, (unsigned long long) g_n_wait);
    if (g_prof) {
        std::printf("profile (all calls): attention %.2f s, router %.2f s, uploads %.2f s, experts %.2f s, rest %.2f s\n",
                    g_pf.attn, g_pf.route, g_pf.upload, g_pf.experts, g_pf.rest);
        std::printf("prefill-moe detail: host %.2f s, gpu %.2f s (sync %.2f s), calls %d\n", g_pf.pf_host, g_pf.pf_gpu,
                    g_pf.pf_sync, g_pf.pf_calls);
        std::printf("MoE branch counts: resident-grouped %ld, token %ld, run_prefill_moe %ld\n", g_pf.br_rg, g_pf.br_mv,
                    g_pf.br_pf);
        std::printf("PREFILL profile: attention %.2f s, router %.2f s, uploads %.2f s, experts %.2f s, rest %.2f s\n",
                    pf_prefill.attn, pf_prefill.route, pf_prefill.upload, pf_prefill.experts, pf_prefill.rest);
        std::printf("PREFILL moe detail: host %.2f s, gpu %.2f s (sync %.2f s), calls %d; branches rg %ld mv %ld pf %ld\n",
                    pf_prefill.pf_host, pf_prefill.pf_gpu, pf_prefill.pf_sync, pf_prefill.pf_calls, pf_prefill.br_rg,
                    pf_prefill.br_mv, pf_prefill.br_pf);
        std::printf("PREFILL run_exp (cnt>=128 experts): gather %.3f s, decode_f16 %.3f s, gemm16 %.3f s, scatter %.3f s\n",
                    pf_prefill.pf_gather, pf_prefill.pf_dec, pf_prefill.pf_gemm, pf_prefill.pf_scatter);
        std::printf("PREFILL attn sub: proj %.3f s, conv/prep %.3f s, QK %.3f s, probs %.3f s, PV %.3f s, out %.3f s\n",
                    pf_prefill.a_proj, pf_prefill.a_conv, pf_prefill.a_qk, pf_prefill.a_probs, pf_prefill.a_pv,
                    pf_prefill.a_out);
        std::printf("PREFILL loop_gates (FP32 cuBLAS): %.3f s\n", pf_prefill.pf_loopgate);
    }
    if (out.size() > 1)
        std::printf("generation: %d tokens in %.2f s (%.2f tok/s); cache %llu hits, %llu misses in it\n",
                    (int) out.size() - 1, gen_s, (out.size() - 1) / gen_s, (unsigned long long) (ec_sum([](ExpertCache& e) { return e.hits; }) - h0),
                    (unsigned long long) (ec_sum([](ExpertCache& e) { return e.misses; }) - m0));
    std::printf("generated:");
    for (int t : out) std::printf(" %d", t);
    std::printf("\n");
    pf_fast::report();
    std::fflush(stdout);
    if (g_trace) std::fclose(g_trace);
    pop_dump();
    return 0;
}

}  // namespace ik2

namespace strata::ik2 {
int run(int argc, char** argv) {
    return ::ik2::run(argc, argv);
}
}  // namespace strata::ik2

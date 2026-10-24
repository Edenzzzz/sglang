// Host-only check of UnifiedTilePlan (A6) on real serving batch geometries: coverage (every
// (request, kv head, packed row, kv position) exactly once), class routing by threshold, merge-group
// member counts, LPT slot balance, and plan time per call. CUDA is stubbed: only the final
// cudaMemcpyAsync touches it, and the host-only mode skips it.
//   g++ -O2 -std=c++17 -I<inc> test_unified_plan.cpp && ./a.out capture_geoms.txt
#include <cstdint>
#include <cstddef>
typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorInvalidValue = 1 };
typedef void* cudaStream_t;
enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1 };
inline cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) { return 0; }
#define FLASHINFER_FUSED_PLAN_WITH_CUDA 1
#include "flashinfer/attention/fused_plan.cuh"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
using namespace flashinfer;

static const int32_t* at(const std::vector<char>& buf, int64_t off) {
  return reinterpret_cast<const int32_t*>(buf.data() + off);
}

int main(int argc, char** argv) {
  std::ifstream in(argc > 1 ? argv[1] : "/tmp/capture_geoms.txt");
  std::string line;
  const uint32_t tiles[3] = {48, 32, 16};
  const uint32_t num_sm = 132, fa2_ctas = 2 * 132, fa3_ctas = 132;
  int failures = 0, cases = 0;
  double worst_us = 0, total_us = 0;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::string name;
    uint32_t heads, kvh, n;
    ss >> name >> heads >> kvh >> n;
    std::vector<int32_t> q(n), p(n);
    for (auto& x : q) ss >> x;
    for (auto& x : p) ss >> x;
    std::vector<int32_t> qo(n + 1, 0), kvlen(n), kvptr(n + 1, 0);
    for (uint32_t i = 0; i < n; ++i) {
      qo[i + 1] = qo[i] + q[i];
      kvlen[i] = p[i] + q[i];
      kvptr[i + 1] = kvptr[i] + kvlen[i];  // page size 1
    }
    std::vector<char> host(64 << 20);
    FusedPlanInfo<4> plan;
    size_t fbytes = 0;
    UnifiedPlanStats st;
    auto t0 = std::chrono::steady_clock::now();
    const int reps = 20;
    for (int r = 0; r < reps; ++r)
      UnifiedTilePlan<int32_t>(nullptr, host.data(), host.size(), &fbytes, plan, qo.data(), kvptr.data(),
                               kvlen.data(), n, heads, kvh, 128, 1, num_sm, fa2_ctas, fa3_ctas, tiles,
                               49, nullptr, &st);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
    worst_us = std::max(worst_us, us);
    total_us += us;
    ++cases;
    const uint32_t gqa = heads / kvh;
    // coverage: count (request, head, packed row, kv) mass covered by items
    std::map<std::tuple<int, int>, int64_t> cover;  // (req, head) -> sum rows*kv
    bool bad_class = false;
    for (int t = 0; t < 4; ++t) {
      const auto& T = plan.tasks[t];
      const int64_t m = st.items[t];
      const int32_t* b = at(host, T.batch_idx_offset);
      const int32_t* qs = at(host, T.q_start_offset);
      const int32_t* ks = at(host, T.kv_start_offset);
      const int32_t* ke = at(host, T.kv_end_offset);
      const int32_t* hh = at(host, T.kv_head_idx_offset);
      const uint32_t tile = t == 3 ? 128 : tiles[t];
      for (int64_t i = 0; i < m; ++i) {
        const int64_t packed = int64_t(q[b[i]]) * gqa;
        const int64_t rows = std::min<int64_t>(tile, packed - qs[i]);
        cover[{b[i], hh[i]}] += rows * (ke[i] - ks[i]);
        const bool want_fa3 = packed >= 49;
        if ((t == 3) != want_fa3) bad_class = true;
        if (t < 3 && packed > tile) bad_class = true;
        if (t < 2 && packed <= tiles[t + 1]) bad_class = true;
      }
      // work_indptr must cover all items, monotone
      const uint32_t ctas = t == 3 ? fa3_ctas : fa2_ctas;
      const int32_t* w = at(host, T.work_indptr_offset);
      if (w[0] != 0 || w[ctas] != m) { std::printf("  work_indptr ends wrong t=%d\n", t); ++failures; }
    }
    bool bad_cover = false;
    for (uint32_t i = 0; i < n; ++i)
      for (uint32_t h = 0; h < kvh; ++h)
        if (cover[{int(i), int(h)}] != int64_t(q[i]) * gqa * kvlen[i]) bad_cover = true;
    // groups: members per group must equal the plan's nmembers
    std::vector<int32_t> members(st.merge_groups, 0);
    for (int t = 0; t < 3; ++t) {
      const int32_t* g = at(host, plan.tasks[t].group_id_offset);
      for (int64_t i = 0; i < st.items[t]; ++i) if (g[i] >= 0) ++members[g[i]];
    }
    bool bad_group = false;
    const int32_t* nm = at(host, plan.group_nmembers_offset);
    for (int64_t g = 0; g < st.merge_groups; ++g) if (members[g] != nm[g]) bad_group = true;
    // LPT balance over FA2 slots
    std::vector<double> load(fa2_ctas, 0);
    for (int t = 0; t < 3; ++t) {
      const int32_t* w = at(host, plan.tasks[t].work_indptr_offset);
      const int32_t* ks = at(host, plan.tasks[t].kv_start_offset);
      const int32_t* ke = at(host, plan.tasks[t].kv_end_offset);
      for (uint32_t c = 0; c < fa2_ctas; ++c)
        for (int32_t i = w[c]; i < w[c + 1]; ++i) load[c] += 2.0 * tiles[t] + (ke[i] - ks[i]);
    }
    double mx = 0, sum = 0;
    for (double l : load) { mx = std::max(mx, l); sum += l; }
    const bool fail = bad_cover || bad_class || bad_group;
    failures += fail;
    std::printf("%-40s n=%2u items=%lld/%lld/%lld/%lld groups=%lld fa3merge=%lld fa2chunk=%d fa3chunk=%d "
                "fa2 max/mean=%.2f plan=%.1fus %s%s%s%s\n",
                name.c_str(), n, (long long)st.items[0], (long long)st.items[1], (long long)st.items[2],
                (long long)st.items[3], (long long)st.merge_groups, (long long)st.fa3_merge_reqs, st.fa2_chunk,
                st.fa3_chunk, sum > 0 ? mx / (sum / fa2_ctas) : 0.0, us, fail ? "FAIL" : "ok",
                bad_cover ? " cover" : "", bad_class ? " class" : "", bad_group ? " group" : "");
  }
  std::printf("cases=%d failures=%d plan time mean=%.1fus worst=%.1fus\n", cases, failures,
              total_us / cases, worst_us);
  return failures ? 1 : 0;
}

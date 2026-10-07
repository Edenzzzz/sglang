// organize_draft_results + build_tree_kernel_efficient as ONE kernel, one CTA per request.
//
// Replaces topk -> sort -> gather -> cat(verified_id) -> tree_mask fill -> retrive fill ->
// build_tree_efficient (and its bit-packed twin). The outputs match those two functions
// element for element on the region the verify pass reads; the mask prefix and the -1
// retrive defaults that the reference gets from separate fill launches are written here.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

enum TreeMaskMode : int { FULL_MASK = 0, QLEN_ONLY = 1, QLEN_ONLY_BITPACKING = 2 };

constexpr int kThreads = 256;

// torch.topk orders NaN above every number; equal keys keep the lower index first.
__device__ __forceinline__ bool ranks_before(float a, int ia, float b, int ib) {
  const bool na = isnan(a), nb = isnan(b);
  if (na != nb) return na;
  if (!na && a != b) return a > b;
  return ia < ib;
}

__device__ __forceinline__ void fill_true(uint8_t* p, int64_t len, int tid, int nthreads) {
  const uintptr_t addr = reinterpret_cast<uintptr_t>(p);
  int64_t head = (16 - static_cast<int64_t>(addr & 15)) & 15;
  if (head > len) head = len;
  for (int64_t i = tid; i < head; i += nthreads) p[i] = 1;
  const int64_t nvec = (len - head) >> 4;
  uint4* v = reinterpret_cast<uint4*>(p + head);
  const uint4 ones = make_uint4(0x01010101u, 0x01010101u, 0x01010101u, 0x01010101u);
  for (int64_t i = tid; i < nvec; i += nthreads) v[i] = ones;
  for (int64_t i = head + (nvec << 4) + tid; i < len; i += nthreads) p[i] = 1;
}

__global__ void __launch_bounds__(kThreads) fused_draft_tree_build_kernel(
    const float* __restrict__ scores, const int64_t* __restrict__ tokens, const int L,
    const int64_t* __restrict__ parent_list, const int64_t* __restrict__ verified_id,
    const int64_t* __restrict__ seq_lens, int64_t* __restrict__ top_scores_index,
    int64_t* __restrict__ draft_tokens, uint8_t* __restrict__ tree_mask,
    int64_t* __restrict__ positions, int64_t* __restrict__ retrive_index,
    int64_t* __restrict__ retrive_next_token, int64_t* __restrict__ retrive_next_sibling,
    const int topk, const int depth, const int n, const int mode,
    const int num_bytes_per_item) {
  extern __shared__ unsigned char smem_raw[];
  float* s = reinterpret_cast<float*>(smem_raw);                   // [L]
  int* chosen = reinterpret_cast<int*>(s + L);                     // [L] 0/1
  int64_t* sel = reinterpret_cast<int64_t*>(                       // [n-1], 8B aligned
      smem_raw + ((static_cast<size_t>(2 * L) * 4 + 7) & ~size_t(7)));
  __shared__ int64_t s_seq_prefix;

  const int bid = blockIdx.x, tid = threadIdx.x;
  const int m = n - 1;
  const float* srow = scores + static_cast<int64_t>(bid) * L;
  const int64_t* trow = tokens + static_cast<int64_t>(bid) * L;
  const int64_t* prow = parent_list + static_cast<int64_t>(bid) * (topk * (depth - 1) + 1);

  for (int i = tid; i < L; i += kThreads) s[i] = srow[i];
  if (tid == 0) {
    int64_t acc = 0;
    if (mode == FULL_MASK)
      for (int i = 0; i < bid; ++i) acc += seq_lens[i];
    s_seq_prefix = acc;
  }
  __syncthreads();

  // top-m by rank, then ascending index order (torch.topk followed by torch.sort).
  for (int i = tid; i < L; i += kThreads) {
    int rank = 0;
    const float si = s[i];
    for (int j = 0; j < L && rank < m; ++j) rank += ranks_before(s[j], j, si, i) ? 1 : 0;
    chosen[i] = rank < m ? 1 : 0;
  }
  __syncthreads();
  for (int i = tid; i < L; i += kThreads) {
    if (chosen[i]) {
      int pos = 0;
      for (int j = 0; j < i; ++j) pos += chosen[j];
      sel[pos] = i;
    }
  }
  __syncthreads();

  const int64_t seq_len = seq_lens[bid];
  const int64_t rbase = static_cast<int64_t>(bid) * n;
  int64_t* sel_out = top_scores_index + static_cast<int64_t>(bid) * m;
  for (int i = tid; i < m; i += kThreads) {
    sel_out[i] = sel[i];
    draft_tokens[rbase + 1 + i] = trow[sel[i]];
  }
  if (tid == 0) draft_tokens[rbase] = verified_id[bid];

  // Mask rows' fixed parts. FULL_MASK rows are (seq_len + n) wide and the reference leaves
  // the first seq_len of each at the fill value True.
  const int64_t seq_tree_idx = static_cast<int64_t>(n) * n * bid + s_seq_prefix * n;
  if (mode == FULL_MASK) {
    for (int r = 0; r < n; ++r)
      fill_true(tree_mask + seq_tree_idx + (seq_len + n) * r, seq_len, tid, kThreads);
  }
  if (tid < n) {
    retrive_next_token[rbase + tid] = -1;
    retrive_next_sibling[rbase + tid] = -1;
  }
  __syncthreads();
  if (tid >= n) return;

  if (mode == QLEN_ONLY_BITPACKING) {
    uint8_t* row = tree_mask + (rbase + tid) * num_bytes_per_item;
    for (int b = 0; b < num_bytes_per_item; ++b) row[b] = 0;
    row[0] = 1;
  } else {
    const int64_t token_tree_idx =
        mode == FULL_MASK ? seq_tree_idx + (seq_len + n) * tid + seq_len + 1
                          : static_cast<int64_t>(n) * n * bid + static_cast<int64_t>(n) * tid + 1;
    tree_mask[token_tree_idx - 1] = 1;
    for (int i = 0; i < n - 1; ++i) tree_mask[token_tree_idx + i] = 0;
  }

  if (tid == 0) {
    positions[rbase] = seq_len;
    for (int i = n - 1; i > 0; --i) {
      retrive_index[rbase + i] = rbase + i;
      const int64_t parent_tb_idx = sel[i - 1] / topk;
      int parent_position = 0;
      if (parent_tb_idx > 0) {
        const int64_t parent_token_idx = prow[parent_tb_idx];
        for (; parent_position < n; ++parent_position) {
          // the reference scans n entries of an (n-1)-wide row; entry n-1 is the next
          // request's first index, never a match for an in-row parent, so stop at m
          if (parent_position < m && sel[parent_position] == parent_token_idx) {
            ++parent_position;
            break;
          }
        }
      }
      if (parent_position == n) {
        printf("WARNING: invalid eagle tree!!! Detected a token with no parent token selected. "
               "Please check if the logprob has nan. The token will be ignored to keep "
               "proceeding.\n");
        continue;
      }
      const int64_t slot = rbase + parent_position;
      if (retrive_next_token[slot] == -1) {
        retrive_next_token[slot] = i;
      } else {
        retrive_next_sibling[rbase + i] = retrive_next_token[slot];
        retrive_next_token[slot] = i;
      }
    }
    retrive_index[rbase] = rbase;
  } else {
    int cur = tid - 1;
    int position = 0;
    const int64_t token_tree_idx =
        mode == FULL_MASK ? seq_tree_idx + (seq_len + n) * tid + seq_len + 1
                          : static_cast<int64_t>(n) * n * bid + static_cast<int64_t>(n) * tid + 1;
    uint8_t* prow_mask = tree_mask + (rbase + tid) * num_bytes_per_item;
    while (true) {
      position += 1;
      if (mode == QLEN_ONLY_BITPACKING) {
        prow_mask[(cur + 1) / 8] |= static_cast<uint8_t>(1u << ((cur + 1) % 8));
      } else {
        tree_mask[token_tree_idx + cur] = 1;
      }
      const int64_t parent_tb_idx = sel[cur] / topk;
      if (parent_tb_idx == 0) break;
      const int64_t token_idx = prow[parent_tb_idx];
      for (cur = 0; cur < n; ++cur) {
        if (cur < m && sel[cur] == token_idx) break;
      }
      if (cur >= m) break;  // reference reads past the row here; see the note above
    }
    positions[rbase + tid] = position + seq_len;
  }
}

}  // namespace

// scores/tokens: the concatenated score_list / token_list, [bs, L]. parent_list: [bs, >=1].
void fused_draft_tree_build(at::Tensor scores, at::Tensor tokens, at::Tensor parent_list,
                            at::Tensor verified_id, at::Tensor seq_lens,
                            at::Tensor top_scores_index, at::Tensor draft_tokens,
                            at::Tensor tree_mask, at::Tensor positions, at::Tensor retrive_index,
                            at::Tensor retrive_next_token, at::Tensor retrive_next_sibling,
                            int64_t topk, int64_t depth, int64_t n, int64_t mode) {
  const int bs = static_cast<int>(scores.size(0));
  const int L = static_cast<int>(scores.size(1));
  TORCH_CHECK(scores.scalar_type() == at::kFloat && scores.is_contiguous());
  TORCH_CHECK(tokens.scalar_type() == at::kLong && tokens.is_contiguous());
  TORCH_CHECK(n >= 1 && n <= kThreads && n - 1 <= L, "num_verify_tokens out of range");
  TORCH_CHECK(mode != QLEN_ONLY_BITPACKING || n <= 32, "bit-packed tree mask holds at most 32 tokens");
  int nbytes = 1;
  if (mode == QLEN_ONLY_BITPACKING) nbytes = n > 16 ? 4 : (n > 8 ? 2 : 1);
  const size_t smem = ((static_cast<size_t>(2 * L) * 4 + 7) & ~size_t(7)) + 8 * (n > 1 ? n - 1 : 1);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (bs == 0) return;
  fused_draft_tree_build_kernel<<<bs, kThreads, smem, stream>>>(
      scores.data_ptr<float>(), tokens.data_ptr<int64_t>(), L,
      parent_list.data_ptr<int64_t>(), verified_id.data_ptr<int64_t>(),
      seq_lens.data_ptr<int64_t>(), top_scores_index.data_ptr<int64_t>(),
      draft_tokens.data_ptr<int64_t>(), static_cast<uint8_t*>(tree_mask.data_ptr()),
      positions.data_ptr<int64_t>(), retrive_index.data_ptr<int64_t>(),
      retrive_next_token.data_ptr<int64_t>(), retrive_next_sibling.data_ptr<int64_t>(),
      static_cast<int>(topk), static_cast<int>(depth), static_cast<int>(n),
      static_cast<int>(mode), nbytes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_draft_tree_build", &fused_draft_tree_build);
}

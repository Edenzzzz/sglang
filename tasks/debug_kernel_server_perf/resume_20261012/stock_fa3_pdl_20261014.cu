#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <flashinfer/attention/fused_plan.cuh>
#include <flashinfer/attention/hopper/fa3_upstream.cuh>
#include <cutlass/device_kernel.h>

namespace {

template <int Slot, bool Maximum = false>
__device__ __forceinline__ void record_time(unsigned long long* timeline) {
  if (timeline != nullptr && threadIdx.x == 0) {
    unsigned long long timestamp;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp));
    if constexpr (Maximum) atomicMax(timeline + Slot, timestamp);
    else atomicMin(timeline + Slot, timestamp);
  }
}

template <typename Kernel>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
void StockFA3PdlPrimary(CUTLASS_GRID_CONSTANT typename Kernel::Params const params,
                        bool release_dependent, unsigned long long* timeline) {
  extern __shared__ char shared_memory[];
  if (release_dependent) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  record_time<0>(timeline);
  Kernel kernel;
  kernel(params, shared_memory);
  cute::tma_store_wait<0>();
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  record_time<1, true>(timeline);
}

void integer_metadata(const at::Tensor& tensor, const at::Tensor& query) {
  TORCH_CHECK(tensor.device() == query.device() && tensor.scalar_type() == at::kInt &&
                  tensor.is_contiguous(),
              "metadata must be contiguous same-device int32 tensors");
}

void float_workspace(const at::Tensor& tensor, const at::Tensor& query, int64_t required) {
  TORCH_CHECK(tensor.device() == query.device() && tensor.scalar_type() == at::kFloat &&
                  tensor.is_contiguous() && tensor.numel() >= required,
              "insufficient same-device float32 workspace");
}

void run_stock_fa3(at::Tensor query, at::Tensor key_cache, at::Tensor value_cache,
                  at::Tensor output, at::Tensor lse, at::Tensor query_indptr,
                  at::Tensor kv_lengths, at::Tensor page_table, at::Tensor partial_output,
                  at::Tensor partial_lse, at::Tensor split_counts, at::Tensor tile_counter,
                  int64_t splits, int64_t maximum_query, int64_t sm_budget,
                  double scale, int64_t phase, at::Tensor timeline) {
  TORCH_CHECK(query.is_cuda() && query.scalar_type() == at::kBFloat16 && query.dim() == 3 &&
                  query.size(2) == 128 && query.stride(2) == 1,
              "query must be a CUDA bf16 [tokens, heads, 128] tensor");
  TORCH_CHECK(key_cache.device() == query.device() && value_cache.device() == query.device() &&
                  key_cache.scalar_type() == at::kBFloat16 &&
                  value_cache.scalar_type() == at::kBFloat16 && key_cache.dim() == 3 &&
                  key_cache.sizes() == value_cache.sizes() &&
                  key_cache.strides() == value_cache.strides() && key_cache.size(2) == 128 &&
                  key_cache.stride(2) == 1 && key_cache.size(1) > 0 &&
                  query.size(1) % key_cache.size(1) == 0,
              "stock PDL prototype requires matching page-1 NHD KV tensors");
  TORCH_CHECK(output.device() == query.device() && output.scalar_type() == at::kBFloat16 &&
                  output.sizes() == query.sizes() && output.stride(2) == 1,
              "output must match query shape, dtype and device");
  TORCH_CHECK(phase >= 0 && phase <= 3 && splits >= 1 && splits <= 32 && sm_budget > 0,
              "invalid phase, split count or SM budget");
  for (const auto& tensor : {query_indptr, kv_lengths, page_table, split_counts, tile_counter}) {
    integer_metadata(tensor, query);
  }
  const int64_t batch = kv_lengths.numel();
  TORCH_CHECK(query_indptr.dim() == 1 && query_indptr.numel() == batch + 1 &&
                  kv_lengths.dim() == 1 && page_table.dim() == 2 &&
                  page_table.size(0) == batch && split_counts.numel() == batch &&
                  tile_counter.numel() == 1,
              "inconsistent stock FA3 metadata shapes");
  TORCH_CHECK(query.size(0) <= INT32_MAX && key_cache.size(0) <= INT32_MAX &&
                  maximum_query >= 0 && maximum_query <= query.size(0),
              "invalid query bound or int32 extent");
  const int64_t lse_elements = query.size(0) * query.size(1);
  float_workspace(lse, query, lse_elements);
  float_workspace(partial_lse, query, splits * lse_elements);
  float_workspace(partial_output, query, splits * lse_elements * 128);
  TORCH_CHECK(timeline.device() == query.device() && timeline.scalar_type() == at::kLong &&
                  timeline.is_contiguous() && (timeline.numel() == 0 || timeline.numel() == 4),
              "timeline must be empty or four CUDA int64 values");
  if (batch == 0 || query.size(0) == 0) return;
  TORCH_CHECK(maximum_query > 0 && page_table.size(1) > 0, "empty launch bounds");

  const c10::cuda::OptionalCUDAGuard guard(query.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  flashinfer::UpstreamFA3Plan plan{};
  plan.q_ptr = static_cast<cutlass::bfloat16_t const*>(query.data_ptr());
  plan.k_ptr = static_cast<cutlass::bfloat16_t*>(key_cache.data_ptr());
  plan.v_ptr = static_cast<cutlass::bfloat16_t*>(value_cache.data_ptr());
  plan.page_table = page_table.data_ptr<int>();
  plan.cu_seqlens_q = query_indptr.data_ptr<int>();
  plan.seqused_k = kv_lengths.data_ptr<int>();
  plan.total_q = int(query.size(0));
  plan.fa3_total_q = int(query.size(0));
  plan.num_pages = int(key_cache.size(0));
  plan.page_size = 1;
  plan.num_qo_heads = int(query.size(1));
  plan.num_kv_heads = int(key_cache.size(1));
  plan.head_dim = 128;
  plan.q_stride_tok = query.stride(0);
  plan.q_stride_head = query.stride(1);
  plan.kv_stride_page = key_cache.stride(0);
  plan.kv_stride_tok = key_cache.stride(0);
  plan.kv_stride_head = key_cache.stride(1);
  plan.o_stride_tok = output.stride(0);
  plan.o_stride_head = output.stride(1);
  plan.num_requests_for_combine = int(batch);
  plan.max_seqlen_q = int(maximum_query);
  plan.num_splits_dynamic = split_counts.data_ptr<int>();
  plan.fa3_batch_list = nullptr;
  plan.num_fa3_requests = int(batch);
  plan.softmax_scale = float(scale);
  plan.num_splits = int(splits);

  if (phase == 3) {
    if (splits > 1) {
      const auto status = flashinfer::LaunchUpstreamCombine<128>(
          plan, static_cast<cutlass::bfloat16_t*>(output.data_ptr()), lse.data_ptr<float>(),
          partial_output.data_ptr<float>(), partial_lse.data_ptr<float>(), stream);
      TORCH_CHECK(status == cudaSuccess, "stock FA3 combine failed: ", cudaGetErrorString(status));
    }
    return;
  }
  int device = 0;
  int device_sms = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&device_sms, cudaDevAttrMultiProcessorCount, device));
  TORCH_CHECK(sm_budget <= device_sms, "SM budget exceeds device capacity");
  C10_CUDA_CHECK(cudaMemsetAsync(tile_counter.data_ptr<int>(), 0, sizeof(int), stream));
  auto scheduler = flashinfer::MakeUpstreamOwnSchedArgs(plan, tile_counter.data_ptr<int>());
  auto mainloop = flashinfer::MakeUpstreamMainloopArgs<128>(
      plan, page_table.data_ptr<int>(), int(page_table.size(1)), int(batch));
  auto epilogue = flashinfer::MakeUpstreamEpilogueArgs<128>(
      plan, static_cast<cutlass::bfloat16_t*>(output.data_ptr()),
      partial_output.data_ptr<float>(), lse.data_ptr<float>(), partial_lse.data_ptr<float>());
  if (phase == 0) {
    const auto status = flashinfer::LaunchUpstreamFA3OwnSched<128>(
        mainloop, epilogue, scheduler, int(sm_budget), stream);
    TORCH_CHECK(status == cudaSuccess, "stock FA3 launch failed: ", cudaGetErrorString(status));
    return;
  }
  using Kernel = flashinfer::UpstreamFA3KernelOwnSched<128>;
  cutlass::KernelHardwareInfo hardware{};
  hardware.sm_count = int(sm_budget);
  typename Kernel::Params params{
      flashinfer::UpstreamFA3Mainloop<128>::to_underlying_arguments(mainloop),
      flashinfer::UpstreamFA3Epilogue<128>::to_underlying_arguments(epilogue), hardware,
      Kernel::TileScheduler::to_underlying_arguments(scheduler)};
  auto kernel = StockFA3PdlPrimary<Kernel>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel::SharedStorageSize));
  auto* timestamps = timeline.numel() ? reinterpret_cast<unsigned long long*>(timeline.data_ptr()) : nullptr;
  kernel<<<Kernel::get_grid_shape(params), Kernel::get_block_shape(), Kernel::SharedStorageSize, stream>>>(
      params, phase == 2, timestamps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}

TORCH_LIBRARY(stock_fa3_pdl, library) {
  library.def(
      "run(Tensor query, Tensor key_cache, Tensor value_cache, Tensor(a!) output, "
      "Tensor(b!) lse, Tensor query_indptr, Tensor kv_lengths, Tensor page_table, "
      "Tensor(c!) partial_output, Tensor(d!) partial_lse, Tensor split_counts, "
      "Tensor(e!) tile_counter, int splits, int maximum_query, int sm_budget, "
      "float scale, int phase, Tensor(f!) timeline) -> ()",
      &run_stock_fa3);
}

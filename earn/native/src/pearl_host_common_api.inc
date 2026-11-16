// The share-proof readers both host files export (pearl_core.cc declares them). Included
// by pearl_host.cu and pearl_host_hip.cpp at file scope, after pearl_host_common.inc.

// Copy whole 1024-byte leaf chunks out of an operand.
extern "C" bool pearl_host_leaf_chunks(void *handle, int isA,
                                       const uint32_t *leaf_indices,
                                       uint32_t count, uint8_t *out) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx || !leaf_indices || !out) return false;
  DeviceScope scope(ctx->device);
  const int8_t *src = isA ? ctx->dA : ctx->dB;
  const uint64_t bytes = isA ? (uint64_t)ctx->profile.m * ctx->profile.k
                             : (uint64_t)ctx->profile.n * ctx->profile.k;
  for (uint32_t i = 0; i < count; i++) {
    const uint64_t off = (uint64_t)leaf_indices[i] * 1024;
    if (off + 1024 > bytes) return false;
    gpuMemcpy(out + (size_t)i * 1024, operand_chunk(ctx, src, leaf_indices[i]), 1024,
               gpuMemcpyDeviceToHost);
  }
  return true;
}

// Copy 32-byte nodes out of one level of a commitment tree.
extern "C" bool pearl_host_tree_nodes(void *handle, int isA, uint32_t level,
                                      const uint32_t *indices, uint32_t count,
                                      uint8_t *out) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx || !indices || !out) return false;
  DeviceScope scope(ctx->device);
  const std::vector<uint64_t> &offs = isA ? ctx->layerOffA : ctx->layerOffB;
  const uint32_t *tree = isA ? ctx->dTreeA : ctx->dTreeB;
  if (!tree || level >= offs.size()) return false;
  const uint64_t base = offs[level];
  for (uint32_t i = 0; i < count; i++) {
    gpuMemcpy(out + (size_t)i * PEARL_HASH_BYTES,
               tree + (base + indices[i]) * 8, PEARL_HASH_BYTES,
               gpuMemcpyDeviceToHost);
  }
  return true;
}

// How many levels the tree has, so the host knows where to stop walking.
extern "C" uint32_t pearl_host_tree_levels(void *handle, int isA) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return 0;
  return (uint32_t)(isA ? ctx->layerOffA.size() : ctx->layerOffB.size());
}

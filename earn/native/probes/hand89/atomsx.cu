// atomsx.cu (goal-r1-rtx50): does sm_120 have a native shared-memory XOR atomic? nv50box.sh's "sass" stage compiles
// this for the card and prints the SASS of k, so the hand loop can take the instruction's word as ptxas writes it.
extern "C" __global__ void k(unsigned v, unsigned s, unsigned *out) {
  extern __shared__ unsigned sm[];
  unsigned x = (v << (s & 31)) | (v >> ((32 - s) & 31));
  atomicXor(&sm[threadIdx.x + 256u * (s & 7u)], x);
  __syncthreads();
  out[threadIdx.x] = sm[threadIdx.x];
}

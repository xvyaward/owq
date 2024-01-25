#include "cuda_utils.h"
#include "owq_cuda.h"

__global__ void VecQuant3MatMulKernel(
    const    float* __restrict__ vec,
    const      int* __restrict__ mat,
             float* __restrict__ mul,
    const    float* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    int height,
    int width
) {
  int row = BLOCKHEIGHT * blockIdx.x;
  int col =  BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int bwidth = ((height - row) < BLOCKHEIGHT) ? ((height - row) * 32 / 3) : BLOCKWIDTH;

  __shared__ float blockvec[BLOCKWIDTH];
  if (threadIdx.x < bwidth)
    blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT) * BLOCKWIDTH + threadIdx.x];
  __syncthreads();

  if (col < width){
    float scale = scales[col];
    float zero = threadIdx.x % 2 ? \
                 float(zeros[col / 2] >> 4) * scale: \
                 float(zeros[col / 2] & 0xf) * scale;

    float res = 0;
    int i = width * row + col;
    int k = 0;

    unsigned int tmp1;
    unsigned int tmp2;
    unsigned int tmp;

    while (k < bwidth) {
      tmp1 = as_unsigned(mat[i]);
      res += (scale * float((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
      res += (scale * float((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
      res += (scale * float((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
      res += (scale * float((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
      res += (scale * float((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
      res += (scale * float((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
      res += (scale * float((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
      res += (scale * float((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
      res += (scale * float((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
      res += (scale * float((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];
      i += width;
      tmp2 = as_unsigned(mat[i]);
      tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
      tmp2 >>= 1;
      res += (scale * float(tmp) - zero) * blockvec[k + 10];
      k += 11;
      res += (scale * float((tmp2 >>  0) & 0x7) - zero) * blockvec[k + 0];
      res += (scale * float((tmp2 >>  3) & 0x7) - zero) * blockvec[k + 1];
      res += (scale * float((tmp2 >>  6) & 0x7) - zero) * blockvec[k + 2];
      res += (scale * float((tmp2 >>  9) & 0x7) - zero) * blockvec[k + 3];
      res += (scale * float((tmp2 >> 12) & 0x7) - zero) * blockvec[k + 4];
      res += (scale * float((tmp2 >> 15) & 0x7) - zero) * blockvec[k + 5];
      res += (scale * float((tmp2 >> 18) & 0x7) - zero) * blockvec[k + 6];
      res += (scale * float((tmp2 >> 21) & 0x7) - zero) * blockvec[k + 7];
      res += (scale * float((tmp2 >> 24) & 0x7) - zero) * blockvec[k + 8];
      res += (scale * float((tmp2 >> 27) & 0x7) - zero) * blockvec[k + 9];
      i += width;
      tmp1 = as_unsigned(mat[i]);
      tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
      tmp1 >>= 2;
      res += (scale * float(tmp) - zero) * blockvec[k + 10];
      k += 11;
      res += (scale * float((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
      res += (scale * float((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
      res += (scale * float((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
      res += (scale * float((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
      res += (scale * float((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
      res += (scale * float((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
      res += (scale * float((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
      res += (scale * float((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
      res += (scale * float((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
      res += (scale * float((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];
      i += width;
      k += 10;
    }
    atomicAdd(&mul[col], res);
  }
}

template <typename T1, typename T2>
__global__ void VecQuant3MatMulKernelFaster(
    const      T2* __restrict__ vec,
    const     int* __restrict__ mat,
               T2* __restrict__ mul,
    const      T1* __restrict__ scales,
    const uint8_t* __restrict__ zeros,
    int height,
    int width
) {
  const int blockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int bwidth = ((height - row) < BLOCKHEIGHT) ? ((height - row) * 16 / 3) : blockwidth2;

  __shared__ T2 blockvec[blockwidth2];
  if (threadIdx.x < bwidth)
    blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT) * blockwidth2 + threadIdx.x];

  __shared__ T2 deq2[64][32];
  int val = threadIdx.x / 32;
  int off = threadIdx.x % 32;
  for (; val < 64; val += BLOCKWIDTH / 32) {
    deq2[val][off] = pair2pack(
       int2T<T1>(val & 0x7), int2T<T1>(val >> 3)
    );
  }

  __syncthreads();

  if (col < width) {
    T2 scale = TtoT2(scales[col]);
    T2 zero = threadIdx.x % 2 ? \
              TtoT2(hmul(int2T<T1>(zeros[col / 2] >> 4), hneg(scale.x))) : \
              TtoT2(hmul(int2T<T1>(zeros[col / 2] & 0xf), hneg(scale.x)));

    int i = width * row + col;
    int k = 0;

    float res = 0;
    T2 res2;

    unsigned int tmp1;
    unsigned int tmp2;
    unsigned int tmp;

    while (k < bwidth) {
      res2 = {};
      tmp1 = as_unsigned(mat[i]);
      res2 = hfma2(hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
      res2 = hfma2(hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
      res2 = hfma2(hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
      res2 = hfma2(hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
      res2 = hfma2(hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero), blockvec[k + 4], res2);
      i += width;
      tmp2 = as_unsigned(mat[i]);
      tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x3c);
      res2 = hfma2(hfma2(deq2[tmp][off], scale, zero), blockvec[k + 5], res2);
      tmp2 >>= 4;
      k += 6;
      res2 = hfma2(hfma2(deq2[(tmp2 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
      res2 = hfma2(hfma2(deq2[(tmp2 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
      res2 = hfma2(hfma2(deq2[(tmp2 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
      res2 = hfma2(hfma2(deq2[(tmp2 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
      i += width;
      tmp1 = as_unsigned(mat[i]);
      tmp = (tmp2 >> 24) | ((tmp1 << 4) & 0x30);
      res2 = hfma2(hfma2(deq2[tmp][off], scale, zero), blockvec[k + 4], res2);
      tmp1 >>= 2;
      k += 5;
      res2 = hfma2(hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
      res2 = hfma2(hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
      res2 = hfma2(hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
      res2 = hfma2(hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
      res2 = hfma2(hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero), blockvec[k + 4], res2);
      i += width;
      k += 5;
      res += T2float(res2.x) + T2float(res2.y);
    }

    if (col % 2 == 0){
      res2 = pair2pack(float2T<T1>(res), getzero<T1>());
    }
    else{
      res2 = pair2pack(getzero<T1>(), float2T<T1>(res));
    }
    atomicAdd(&mul[col / 2], res2);
  }
}

__global__ void VecQuant3OutlierMatMulKernel(
    const    float* __restrict__ vec,
    const      int* __restrict__ mat,
             float* __restrict__ mul,
    const    float* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    const    float* __restrict__ outlierMat,
    const      int* __restrict__ outlieridx,
    const      int* __restrict__ outrow,
    const      int* __restrict__ cnt,
    int height,
    int width
) {
  int row = BLOCKHEIGHT * blockIdx.x;
  int col =  BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int bwidth = ((height - row) < BLOCKHEIGHT) ? ((height - row) * 32 / 3) : BLOCKWIDTH;

  int oidx = -1;
  int blockoutrow = outrow[blockIdx.x];
  int blockcnt = cnt[blockIdx.x];

  outlierMat += blockoutrow * width;
  outlieridx += blockoutrow;

  for (int i = 0; i < blockcnt; i++){
    if (threadIdx.x == outlieridx[i] % BLOCKWIDTH)
      oidx = i;
  }
  
  __shared__ float blockvec[BLOCKWIDTH];
  __shared__ float blockveco[MAXOUTLIER];

  if (threadIdx.x < bwidth){
    blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT) * BLOCKWIDTH + threadIdx.x];
    if (oidx > -1)
      blockveco[oidx] = blockvec[threadIdx.x];
  }

  __syncthreads();

  if (col < width){
    float scale = scales[col];
    float zero = threadIdx.x % 2 ? \
                 float(zeros[col / 2] >> 4) * scale: \
                 float(zeros[col / 2] & 0xf) * scale;

    float res = 0;
    int i = width * row + col;
    int k = 0;

    unsigned int tmp1;
    unsigned int tmp2;
    unsigned int tmp;

    while (k < bwidth) {
      tmp1 = as_unsigned(mat[i]);
      res += (scale * float((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
      res += (scale * float((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
      res += (scale * float((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
      res += (scale * float((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
      res += (scale * float((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
      res += (scale * float((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
      res += (scale * float((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
      res += (scale * float((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
      res += (scale * float((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
      res += (scale * float((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];
      i += width;
      tmp2 = as_unsigned(mat[i]);
      tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
      tmp2 >>= 1;
      res += (scale * float(tmp) - zero) * blockvec[k + 10];
      k += 11;
      res += (scale * float((tmp2 >>  0) & 0x7) - zero) * blockvec[k + 0];
      res += (scale * float((tmp2 >>  3) & 0x7) - zero) * blockvec[k + 1];
      res += (scale * float((tmp2 >>  6) & 0x7) - zero) * blockvec[k + 2];
      res += (scale * float((tmp2 >>  9) & 0x7) - zero) * blockvec[k + 3];
      res += (scale * float((tmp2 >> 12) & 0x7) - zero) * blockvec[k + 4];
      res += (scale * float((tmp2 >> 15) & 0x7) - zero) * blockvec[k + 5];
      res += (scale * float((tmp2 >> 18) & 0x7) - zero) * blockvec[k + 6];
      res += (scale * float((tmp2 >> 21) & 0x7) - zero) * blockvec[k + 7];
      res += (scale * float((tmp2 >> 24) & 0x7) - zero) * blockvec[k + 8];
      res += (scale * float((tmp2 >> 27) & 0x7) - zero) * blockvec[k + 9];
      i += width;
      tmp1 = as_unsigned(mat[i]);
      tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
      tmp1 >>= 2;
      res += (scale * float(tmp) - zero) * blockvec[k + 10];
      k += 11;
      res += (scale * float((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
      res += (scale * float((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
      res += (scale * float((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
      res += (scale * float((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
      res += (scale * float((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
      res += (scale * float((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
      res += (scale * float((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
      res += (scale * float((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
      res += (scale * float((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
      res += (scale * float((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];
      i += width;
      k += 10;
    }

    if (blockcnt > 0){
      for (int k = 0; k < blockcnt; k++) {
        res += outlierMat[col + k * width] * blockveco[k];
      }
    }
    atomicAdd(&mul[col], res);
  }
}

template <typename T1, typename T2>
__global__ void VecQuant3OutlierMatMulKernelFaster(
    const       T2* __restrict__ vec,
    const      int* __restrict__ mat,
                T2* __restrict__ mul,
    const       T1* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    const       T1* __restrict__ outlierMat,
    const      int* __restrict__ outlieridx,
    const      int* __restrict__ outrow,
    const      int* __restrict__ cnt,
    int height,
    int width
) {
  const int blockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int bwidth = ((height - row) < BLOCKHEIGHT) ? ((height - row) * 16 / 3) : blockwidth2;

  __shared__ T2 deq2[64][32];
  int val = threadIdx.x / 32;
  int off = threadIdx.x % 32;
  for (; val < 64; val += BLOCKWIDTH / 32) {
    deq2[val][off] = pair2pack(
       int2T<T1>(val & 0x7), int2T<T1>(val >> 3)
    );
  }
  
  int flag = 0, oidx = 0;
  int blockoutrow = outrow[blockIdx.x];
  int blockcnt = cnt[blockIdx.x];

  outlierMat += blockoutrow * width;
  outlieridx += blockoutrow;

  for (int i = 0, outidx = 0; i < blockcnt; i++){
    outidx = outlieridx[i];
    if (threadIdx.x == (outidx / 2) % blockwidth2){
      flag += (outidx % 2) + 1; // 1 (x) 2 (y) 3(x,y)
      oidx = i;
    }
  }
  
  __shared__ T2 blockvec[blockwidth2];
  __shared__ T1 blockveco[MAXOUTLIER];

  if (threadIdx.x < bwidth){
    blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT) * blockwidth2 + threadIdx.x];
    if (flag == 1)
      blockveco[oidx] = blockvec[threadIdx.x].x;
    else if (flag == 2)
      blockveco[oidx] = blockvec[threadIdx.x].y;
    else if (flag == 3){
      blockveco[oidx - 1] = blockvec[threadIdx.x].x;
      blockveco[oidx] = blockvec[threadIdx.x].y;
    }
  }

  __syncthreads();

  if (col < width) {
    T2 scale = TtoT2(scales[col]);
    T2 zero = threadIdx.x % 2 ? \
              TtoT2(hmul(int2T<T1>(zeros[col / 2] >> 4), hneg(scale.x))) : \
              TtoT2(hmul(int2T<T1>(zeros[col / 2] & 0xf), hneg(scale.x)));

    int i = width * row + col;
    int k = 0;

    float res = 0;
    T2 res2;

    unsigned int tmp1;
    unsigned int tmp2;
    unsigned int tmp;

    while (k < bwidth) {
      res2 = {};
      tmp1 = as_unsigned(mat[i]);
      res2 = hfma2(hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
      res2 = hfma2(hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
      res2 = hfma2(hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
      res2 = hfma2(hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
      res2 = hfma2(hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero), blockvec[k + 4], res2);
      i += width;
      tmp2 = as_unsigned(mat[i]);
      tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x3c);
      res2 = hfma2(hfma2(deq2[tmp][off], scale, zero), blockvec[k + 5], res2);
      tmp2 >>= 4;
      k += 6;
      res2 = hfma2(hfma2(deq2[(tmp2 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
      res2 = hfma2(hfma2(deq2[(tmp2 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
      res2 = hfma2(hfma2(deq2[(tmp2 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
      res2 = hfma2(hfma2(deq2[(tmp2 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
      i += width;
      tmp1 = as_unsigned(mat[i]);
      tmp = (tmp2 >> 24) | ((tmp1 << 4) & 0x30);
      res2 = hfma2(hfma2(deq2[tmp][off], scale, zero), blockvec[k + 4], res2);
      tmp1 >>= 2;
      k += 5;
      res2 = hfma2(hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
      res2 = hfma2(hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
      res2 = hfma2(hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
      res2 = hfma2(hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
      res2 = hfma2(hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero), blockvec[k + 4], res2);
      i += width;
      k += 5;
      res += T2float(res2.x) + T2float(res2.y);
    }

    if (blockcnt > 0){
      T1 res_o = {};
      for (int k = 0; k < blockcnt; k++) {
        res_o = hfma(outlierMat[col + k * width], blockveco[k], res_o);
      }
      res += T2float(res_o);
    }

    if (col % 2 == 0){
      res2 = pair2pack(float2T<T1>(res), getzero<T1>());
    }
    else{
      res2 = pair2pack(getzero<T1>(), float2T<T1>(res));
    }
    atomicAdd(&mul[col / 2], res2);
  }
}

__global__ void VecQuant4MatMulKernel(
    const    float* __restrict__ vec,
    const      int* __restrict__ mat,
             float* __restrict__ mul,
    const    float* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    int height,
    int width
) {
  int row = BLOCKHEIGHT4B * blockIdx.x;
  int col =  BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int bwidth = ((height - row) < BLOCKHEIGHT4B) ? ((height - row) * 8) : BLOCKWIDTH;

  __shared__ float blockvec[BLOCKWIDTH];
  if (threadIdx.x < bwidth)
    blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT4B) * BLOCKWIDTH + threadIdx.x];
  __syncthreads();

  if (col < width){
    float scale = scales[col];
    float zero = threadIdx.x % 2 ? \
                 float(zeros[col / 2] >> 4) * scale: \
                 float(zeros[col / 2] & 0xf) * scale;

    float res = 0;
    int i = width * row + col;
    int k = 0;

    unsigned int tmp;

    while (k < bwidth) {
      tmp = as_unsigned(mat[i]);
      for (int a = 0; a < 8; a++){
        res += (scale * float((tmp >> (a * 4)) & 0xf) - zero) * blockvec[k + a];
      }
      i += width;
      k += 8;
    }
    atomicAdd(&mul[col], res);
  }
}

template <typename T1, typename T2>
__global__ void VecQuant4MatMulKernelFaster(
    const       T2* __restrict__ vec,
    const      int* __restrict__ mat,
                T2* __restrict__ mul,
    const       T1* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    int height,
    int width
) {
  const int blockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT4B * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int bwidth = ((height - row) < BLOCKHEIGHT4B) ? ((height - row) * 4) : blockwidth2;

  __shared__ T2 blockvec[blockwidth2];
  if (threadIdx.x < bwidth)
    blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT4B) * blockwidth2 + threadIdx.x];

  __syncthreads();

  if (col < width) {
    T2 scale = TtoT2(scales[col]);
    T2 zero = threadIdx.x % 2 ? \
              TtoT2(hmul(int2T<T1>(zeros[col / 2] >> 4), hneg(scale.x))) : \
              TtoT2(hmul(int2T<T1>(zeros[col / 2] & 0xf), hneg(scale.x)));

    int i = width * row + col;
    int k = 0;

    float res = 0;
    T2 temp;
    T2 res2;

    unsigned int tmp;

    while (k < bwidth) {
      tmp = as_unsigned(mat[i]);
      res2 = {};
      for (int a = 0; a < 4; a++){
        temp = pair2pack(
          int2T<T1>((tmp >> (a * 8)) & 0x0f),
          int2T<T1>((tmp >> (a * 8 + 4)) & 0x0f)
        );
        res2 = hfma2(hfma2(temp, scale, zero), blockvec[k + a], res2);
      }
      i += width;
      k += 4;
      res += T2float(res2.x) + T2float(res2.y);
    }
    if (col % 2 == 0){
      res2 = pair2pack(float2T<T1>(res), getzero<T1>());
    }
    else{
      res2 = pair2pack(getzero<T1>(), float2T<T1>(res));
    }
    atomicAdd(&mul[col / 2], res2);
  }
}

__global__ void VecQuant4OutlierMatMulKernel(
    const    float* __restrict__ vec,
    const      int* __restrict__ mat,
             float* __restrict__ mul,
    const    float* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    const    float* __restrict__ outlierMat,
    const      int* __restrict__ outlieridx,
    const      int* __restrict__ outrow,
    const      int* __restrict__ cnt,
    int height,
    int width
) {
  int row = BLOCKHEIGHT4B * blockIdx.x;
  int col =  BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int bwidth = ((height - row) < BLOCKHEIGHT4B) ? ((height - row) * 8) : BLOCKWIDTH;

  int oidx = -1;
  int blockoutrow = outrow[blockIdx.x];
  int blockcnt = cnt[blockIdx.x];

  outlierMat += blockoutrow * width;
  outlieridx += blockoutrow;

  for (int i = 0; i < blockcnt; i++){
    if (threadIdx.x == outlieridx[i] % BLOCKWIDTH)
      oidx = i;
  }
  
  __shared__ float blockvec[BLOCKWIDTH];
  __shared__ float blockveco[MAXOUTLIER];

  if (threadIdx.x < bwidth){
    blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT4B) * BLOCKWIDTH + threadIdx.x];
    if (oidx > -1)
      blockveco[oidx] = blockvec[threadIdx.x];
  }

  __syncthreads();

  if (col < width){
    float scale = scales[col];
    float zero = threadIdx.x % 2 ? \
                 float(zeros[col / 2] >> 4) * scale: \
                 float(zeros[col / 2] & 0xf) * scale;

    float res = 0;
    int i = width * row + col;
    int k = 0;

    unsigned int tmp;

    while (k < bwidth) {
      tmp = as_unsigned(mat[i]);
      for (int a = 0; a < 8; a++){
        res += (scale * float((tmp >> (a * 4)) & 0xf) - zero) * blockvec[k + a];
      }
      i += width;
      k += 8;
    }
    
    if (blockcnt > 0){
      for (int k = 0; k < blockcnt; k++) {
        res += outlierMat[col + k * width] * blockveco[k];
      }
    }
    atomicAdd(&mul[col], res);
  }
}

template <typename T1, typename T2>
__global__ void VecQuant4OutlierMatMulKernelFaster(
    const       T2* __restrict__ vec,
    const      int* __restrict__ mat,
                T2* __restrict__ mul,
    const       T1* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    const       T1* __restrict__ outlierMat,
    const      int* __restrict__ outlieridx,
    const      int* __restrict__ outrow,
    const      int* __restrict__ cnt,
    int height,
    int width
) {
  const int blockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT4B * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int bwidth = ((height - row) < BLOCKHEIGHT4B) ? ((height - row) * 4) : blockwidth2;

  int flag = 0, oidx = 0;
  int blockoutrow = outrow[blockIdx.x];
  int blockcnt = cnt[blockIdx.x];

  outlierMat += blockoutrow * width;
  outlieridx += blockoutrow;

  for (int i = 0, outidx = 0; i < blockcnt; i++){
    outidx = outlieridx[i];
    if (threadIdx.x == (outidx / 2) % blockwidth2){
      flag += (outidx % 2) + 1; // 1 (x) 2 (y) 3(x,y)
      oidx = i;
    }
  }
  
  __shared__ T2 blockvec[blockwidth2];
  __shared__ T1 blockveco[MAXOUTLIER];

  if (threadIdx.x < bwidth){
    blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT4B) * blockwidth2 + threadIdx.x];
    if (flag == 1)
      blockveco[oidx] = blockvec[threadIdx.x].x;
    else if (flag == 2)
      blockveco[oidx] = blockvec[threadIdx.x].y;
    else if (flag == 3){
      blockveco[oidx - 1] = blockvec[threadIdx.x].x;
      blockveco[oidx] = blockvec[threadIdx.x].y;
    }
  }
  __syncthreads();

  if (col < width) {
    T2 scale = TtoT2(scales[col]);
    T2 zero = threadIdx.x % 2 ? \
              TtoT2(hmul(int2T<T1>(zeros[col / 2] >> 4), hneg(scale.x))) : \
              TtoT2(hmul(int2T<T1>(zeros[col / 2] & 0xf), hneg(scale.x)));

    int i = width * row + col;
    int k = 0;

    float res = 0;
    T2 temp;
    T2 res2;

    unsigned int tmp;

    while (k < bwidth) {
      tmp = as_unsigned(mat[i]);
      res2 = {};
      for (int a = 0; a < 4; a++){
        temp = pair2pack(
          int2T<T1>((tmp >> (a * 8)) & 0x0f),
          int2T<T1>((tmp >> (a * 8 + 4)) & 0x0f)
        );
        res2 = hfma2(hfma2(temp, scale, zero), blockvec[k + a], res2);
      }
      i += width;
      k += 4;
      res += T2float(res2.x) + T2float(res2.y);
    }

    if (blockcnt > 0){
      T1 res_o = {};
      for (int k = 0; k < blockcnt; k++) {
        res_o = hfma(outlierMat[col + k * width], blockveco[k], res_o);
      }
      res += T2float(res_o);
    }

    if (col % 2 == 0){
      res2 = pair2pack(float2T<T1>(res), getzero<T1>());
    }
    else{
      res2 = pair2pack(getzero<T1>(), float2T<T1>(res));
    }
    atomicAdd(&mul[col / 2], res2);
  }

}

void vecquant3matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant3matmul_cuda", ([&] {
      VecQuant3MatMulKernel<<<blocks, threads>>>(
        vec.data<float>(), mat.data<int>(), mul.data<float>(),
        scales.data<float>(), zeros.data<uint8_t>(),
        height, width
      );
    })
  );
}

void vecquant3matmul_faster_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);
  if (scales.dtype() == torch::kBFloat16){
    VecQuant3MatMulKernelFaster<<<blocks, threads>>>(
      (nv_bfloat162*) vec.data_ptr(),
      mat.data_ptr<int>(),
      (nv_bfloat162*) mul.data_ptr(),
      (nv_bfloat16*) scales.data_ptr(),
      zeros.data_ptr<uint8_t>(),
      height, width
    );
  }
  else {
    VecQuant3MatMulKernelFaster<<<blocks, threads>>>(
      (half2*) vec.data_ptr(),
      mat.data_ptr<int>(),
      (half2*) mul.data_ptr(),
      (half*) scales.data_ptr(),
      zeros.data_ptr<uint8_t>(),
      height, width
    );
  }
}

void vecquant3outliermatmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor outlierMat,
  torch::Tensor outlieridx,
  torch::Tensor outrow,
  torch::Tensor cnt
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3OutlierMatMulKernel<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    scales.data_ptr<float>(),
    zeros.data_ptr<uint8_t>(),
    outlierMat.data_ptr<float>(),
    outlieridx.data_ptr<int>(), 
    outrow.data_ptr<int>(), 
    cnt.data_ptr<int>(),
    height, width
  );
}

void vecquant3outliermatmul_faster_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor outlierMat,
  torch::Tensor outlieridx,
  torch::Tensor outrow,
  torch::Tensor cnt
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  if (scales.dtype() == torch::kBFloat16){
    VecQuant3OutlierMatMulKernelFaster<<<blocks, threads>>>(
      (nv_bfloat162*) vec.data_ptr(),
      mat.data_ptr<int>(),
      (nv_bfloat162*) mul.data_ptr(),
      (nv_bfloat16*) scales.data_ptr(),
      zeros.data_ptr<uint8_t>(),
      (nv_bfloat16*) outlierMat.data_ptr(), 
      outlieridx.data_ptr<int>(), 
      outrow.data_ptr<int>(), 
      cnt.data_ptr<int>(), 
      height, width
    );
  }
  else {
    VecQuant3OutlierMatMulKernelFaster<<<blocks, threads>>>(
      (half2*) vec.data_ptr(),
      mat.data_ptr<int>(),
      (half2*) mul.data_ptr(),
      (half*) scales.data_ptr(),
      zeros.data_ptr<uint8_t>(),
      (half*) outlierMat.data_ptr(), 
      outlieridx.data_ptr<int>(), 
      outrow.data_ptr<int>(), 
      cnt.data_ptr<int>(), 
      height, width
    );
  }
}

void vecquant4matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4B - 1) / BLOCKHEIGHT4B,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant4matmul_cuda", ([&] {
      VecQuant4MatMulKernel<<<blocks, threads>>>(
        vec.data<float>(), mat.data<int>(), mul.data<float>(),
        scales.data<float>(), zeros.data<uint8_t>(),
        height, width
      );
    })
  );
}

void vecquant4matmul_faster_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4B - 1) / BLOCKHEIGHT4B,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  if (scales.dtype() == torch::kBFloat16){
    VecQuant4MatMulKernelFaster<<<blocks, threads>>>(
      (nv_bfloat162*) vec.data_ptr(),
      mat.data_ptr<int>(),
      (nv_bfloat162*) mul.data_ptr(),
      (nv_bfloat16*) scales.data_ptr(),
      zeros.data_ptr<uint8_t>(),
      height, width
    );
  }
  else{
    VecQuant4MatMulKernelFaster<<<blocks, threads>>>(
      (half2*) vec.data_ptr(),
      mat.data_ptr<int>(),
      (half2*) mul.data_ptr(),
      (half*) scales.data_ptr(),
      zeros.data_ptr<uint8_t>(),
      height, width
    );
  }
}

void vecquant4outliermatmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor outlierMat,
  torch::Tensor outlieridx,
  torch::Tensor outrow,
  torch::Tensor cnt
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4B - 1) / BLOCKHEIGHT4B,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4OutlierMatMulKernel<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    scales.data_ptr<float>(),
    zeros.data_ptr<uint8_t>(),
    outlierMat.data_ptr<float>(),
    outlieridx.data_ptr<int>(), 
    outrow.data_ptr<int>(), 
    cnt.data_ptr<int>(),
    height, width
  );
}

void vecquant4outliermatmul_faster_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor outlierMat,
  torch::Tensor outlieridx,
  torch::Tensor outrow,
  torch::Tensor cnt
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4B - 1) / BLOCKHEIGHT4B,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);
  
  if (scales.dtype() == torch::kBFloat16){
    VecQuant4OutlierMatMulKernelFaster<<<blocks, threads>>>(
      (nv_bfloat162*) vec.data_ptr(),
      mat.data_ptr<int>(),
      (nv_bfloat162*) mul.data_ptr(),
      (nv_bfloat16*) scales.data_ptr(),
      zeros.data_ptr<uint8_t>(),
      (nv_bfloat16*) outlierMat.data_ptr(), 
      outlieridx.data_ptr<int>(), 
      outrow.data_ptr<int>(), 
      cnt.data_ptr<int>(), 
      height, width
    );
  }
  else { // Half
    VecQuant4OutlierMatMulKernelFaster<<<blocks, threads>>>(
      (half2*) vec.data_ptr(),
      mat.data_ptr<int>(),
      (half2*) mul.data_ptr(),
      (half*) scales.data_ptr(),
      zeros.data_ptr<uint8_t>(),
      (half*) outlierMat.data_ptr(), 
      outlieridx.data_ptr<int>(), 
      outrow.data_ptr<int>(), 
      cnt.data_ptr<int>(), 
      height, width
    );
  }
}

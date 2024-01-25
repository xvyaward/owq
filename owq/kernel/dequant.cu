#include "cuda_utils.h"
#include "owq_cuda.h"

__global__ void MatQuant3DequantKernel(
    const      int* __restrict__ mat,
             float* __restrict__ out,
    const    float* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    int height,
    int width
) {
  int row =  MMBLOCKHEIGHT * blockIdx.x;
  int col =  MMBLOCKWIDTH * blockIdx.y + threadIdx.x;
  int new_row = ((MMBLOCKHEIGHT * 32) / 3) * blockIdx.x;
  int bwidth = ((height - row) < MMBLOCKHEIGHT) ? ((height - row) * 32 / 3) : MMBLOCKWIDTH;

  __shared__ float out_temp[32][MMBLOCKWIDTH];

  if (col < width){
    float scale = scales[col];
    float zero = threadIdx.x % 2 ? \
                 float(zeros[col / 2] >> 4) * scale: \
                 float(zeros[col / 2] & 0xf) * scale;

    int i = width * row + col;
    int k = 0;

    unsigned int tmp1;
    unsigned int tmp2;
    unsigned int tmp;

    while (k < bwidth) {
      tmp1 = as_unsigned(mat[i]);
      out_temp[0][threadIdx.x] = (scale * float((tmp1 >>  0) & 0x7) - zero);
      out_temp[1][threadIdx.x] = (scale * float((tmp1 >>  3) & 0x7) - zero);
      out_temp[2][threadIdx.x] = (scale * float((tmp1 >>  6) & 0x7) - zero);
      out_temp[3][threadIdx.x] = (scale * float((tmp1 >>  9) & 0x7) - zero);
      out_temp[4][threadIdx.x] = (scale * float((tmp1 >> 12) & 0x7) - zero);
      out_temp[5][threadIdx.x] = (scale * float((tmp1 >> 15) & 0x7) - zero);
      out_temp[6][threadIdx.x] = (scale * float((tmp1 >> 18) & 0x7) - zero);
      out_temp[7][threadIdx.x] = (scale * float((tmp1 >> 21) & 0x7) - zero);
      out_temp[8][threadIdx.x] = (scale * float((tmp1 >> 24) & 0x7) - zero);
      out_temp[9][threadIdx.x] = (scale * float((tmp1 >> 27) & 0x7) - zero);
      i += width;
      tmp2 = as_unsigned(mat[i]);
      tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
      tmp2 >>= 1;
      out_temp[10][threadIdx.x] = (scale * float(tmp) - zero);
      out_temp[11][threadIdx.x] = (scale * float((tmp2 >>  0) & 0x7) - zero);
      out_temp[12][threadIdx.x] = (scale * float((tmp2 >>  3) & 0x7) - zero);
      out_temp[13][threadIdx.x] = (scale * float((tmp2 >>  6) & 0x7) - zero);
      out_temp[14][threadIdx.x] = (scale * float((tmp2 >>  9) & 0x7) - zero);
      out_temp[15][threadIdx.x] = (scale * float((tmp2 >> 12) & 0x7) - zero);
      out_temp[16][threadIdx.x] = (scale * float((tmp2 >> 15) & 0x7) - zero);
      out_temp[17][threadIdx.x] = (scale * float((tmp2 >> 18) & 0x7) - zero);
      out_temp[18][threadIdx.x] = (scale * float((tmp2 >> 21) & 0x7) - zero);
      out_temp[19][threadIdx.x] = (scale * float((tmp2 >> 24) & 0x7) - zero);
      out_temp[20][threadIdx.x] = (scale * float((tmp2 >> 27) & 0x7) - zero);
      i += width;
      tmp1 = as_unsigned(mat[i]);
      tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
      tmp1 >>= 2;
      out_temp[21][threadIdx.x] = (scale * float(tmp) - zero);
      out_temp[22][threadIdx.x] = (scale * float((tmp1 >>  0) & 0x7) - zero);
      out_temp[23][threadIdx.x] = (scale * float((tmp1 >>  3) & 0x7) - zero);
      out_temp[24][threadIdx.x] = (scale * float((tmp1 >>  6) & 0x7) - zero);
      out_temp[25][threadIdx.x] = (scale * float((tmp1 >>  9) & 0x7) - zero);
      out_temp[26][threadIdx.x] = (scale * float((tmp1 >> 12) & 0x7) - zero);
      out_temp[27][threadIdx.x] = (scale * float((tmp1 >> 15) & 0x7) - zero);
      out_temp[28][threadIdx.x] = (scale * float((tmp1 >> 18) & 0x7) - zero);
      out_temp[29][threadIdx.x] = (scale * float((tmp1 >> 21) & 0x7) - zero);
      out_temp[30][threadIdx.x] = (scale * float((tmp1 >> 24) & 0x7) - zero);
      out_temp[31][threadIdx.x] = (scale * float((tmp1 >> 27) & 0x7) - zero);
      i += width;
      k += 32;
      __syncthreads();

      for (int a = 0; a < 32; a++){
        out[(new_row + (k - 32) + a) * width + col] = out_temp[a][threadIdx.x];
      }
    }
    __syncthreads();
  }
}

template <typename T1, typename T2>
__global__ void MatQuant3DequantKernelFaster(
    const     int* __restrict__ mat,
               T1* __restrict__ out,
    const      T1* __restrict__ scales,
    const uint8_t* __restrict__ zeros,
    int height,
    int width
) {
  const int mmblockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int new_row = ((BLOCKHEIGHT * 32) / 3) * blockIdx.x;
  int bwidth = ((height - row) < BLOCKHEIGHT) ? ((height - row) * 16 / 3) : mmblockwidth2;

  __shared__ T1 out_temp[32][BLOCKWIDTH];

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

    T2 res;

    unsigned int tmp1;
    unsigned int tmp2;
    unsigned int tmp;

    while (k < bwidth) {
      tmp1 = as_unsigned(mat[i]);
      res = hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero);
      out_temp[0][threadIdx.x] = res.x;
      out_temp[1][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero);
      out_temp[2][threadIdx.x] = res.x;
      out_temp[3][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero);
      out_temp[4][threadIdx.x] = res.x;
      out_temp[5][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero);
      out_temp[6][threadIdx.x] = res.x;
      out_temp[7][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero);
      out_temp[8][threadIdx.x] = res.x;
      out_temp[9][threadIdx.x] = res.y;
      i += width;
      tmp2 = as_unsigned(mat[i]);
      tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x3c);
      res = hfma2(deq2[tmp][off], scale, zero);
      out_temp[10][threadIdx.x] = res.x;
      out_temp[11][threadIdx.x] = res.y;
      tmp2 >>= 4;
      res = hfma2(deq2[(tmp2 >>  0) & 0x3f][off], scale, zero);
      out_temp[12][threadIdx.x] = res.x;
      out_temp[13][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp2 >>  6) & 0x3f][off], scale, zero);
      out_temp[14][threadIdx.x] = res.x;
      out_temp[15][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp2 >> 12) & 0x3f][off], scale, zero);
      out_temp[16][threadIdx.x] = res.x;
      out_temp[17][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp2 >> 18) & 0x3f][off], scale, zero);
      out_temp[18][threadIdx.x] = res.x;
      out_temp[19][threadIdx.x] = res.y;
      i += width;
      tmp1 = as_unsigned(mat[i]);
      tmp = (tmp2 >> 24) | ((tmp1 << 4) & 0x30);
      res = hfma2(deq2[tmp][off], scale, zero);
      out_temp[20][threadIdx.x] = res.x;
      out_temp[21][threadIdx.x] = res.y;
      tmp1 >>= 2;
      res = hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero);
      out_temp[22][threadIdx.x] = res.x;
      out_temp[23][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero);
      out_temp[24][threadIdx.x] = res.x;
      out_temp[25][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero);
      out_temp[26][threadIdx.x] = res.x;
      out_temp[27][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero);
      out_temp[28][threadIdx.x] = res.x;
      out_temp[29][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero);
      out_temp[30][threadIdx.x] = res.x;
      out_temp[31][threadIdx.x] = res.y;
      i += width;
      k += 16;
      __syncthreads();
    
      for (int a = 0; a < 32; a++){
        out[(new_row + (k - 16) * 2 + a) * width + col] = out_temp[a][threadIdx.x];
      }
    }
    __syncthreads();
  }
}

template <typename T1, typename T2>
__global__ void MatQuant3DequantOutlierKernelFaster(
    const     int* __restrict__ mat,
               T1* __restrict__ out,
    const      T1* __restrict__ scales,
    const uint8_t* __restrict__ zeros,
    const      T1* __restrict__ outlierMat,
    const     int* __restrict__ outlieridx,
    const     int* __restrict__ outrow,
    const     int* __restrict__ cnt,
    int height,
    int width
) {
  const int mmblockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int new_row = ((BLOCKHEIGHT * 32) / 3) * blockIdx.x;
  int bwidth = ((height - row) < BLOCKHEIGHT) ? ((height - row) * 16 / 3) : mmblockwidth2;

  __shared__ T1 out_temp[32][BLOCKWIDTH];

  __shared__ T2 deq2[64][32];
  int val = threadIdx.x / 32;
  int off = threadIdx.x % 32;
  for (; val < 64; val += BLOCKWIDTH / 32) {
    deq2[val][off] = pair2pack(
       int2T<T1>(val & 0x7), int2T<T1>(val >> 3)
    );
  }

  int blockoutrow = outrow[blockIdx.x];
  int blockcnt = cnt[blockIdx.x];

  outlierMat += blockoutrow * width;
  outlieridx += blockoutrow;

  __syncthreads();

  if (col < width) {
    T2 scale = TtoT2(scales[col]);
    T2 zero = threadIdx.x % 2 ? \
              TtoT2(hmul(int2T<T1>(zeros[col / 2] >> 4), hneg(scale.x))) : \
              TtoT2(hmul(int2T<T1>(zeros[col / 2] & 0xf), hneg(scale.x)));

    int i = width * row + col;
    int k = 0;

    T2 res;

    unsigned int tmp1;
    unsigned int tmp2;
    unsigned int tmp;

    while (k < bwidth) {
      tmp1 = as_unsigned(mat[i]);
      res = hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero);
      out_temp[0][threadIdx.x] = res.x;
      out_temp[1][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero);
      out_temp[2][threadIdx.x] = res.x;
      out_temp[3][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero);
      out_temp[4][threadIdx.x] = res.x;
      out_temp[5][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero);
      out_temp[6][threadIdx.x] = res.x;
      out_temp[7][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero);
      out_temp[8][threadIdx.x] = res.x;
      out_temp[9][threadIdx.x] = res.y;
      i += width;
      tmp2 = as_unsigned(mat[i]);
      tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x3c);
      res = hfma2(deq2[tmp][off], scale, zero);
      out_temp[10][threadIdx.x] = res.x;
      out_temp[11][threadIdx.x] = res.y;
      tmp2 >>= 4;
      res = hfma2(deq2[(tmp2 >>  0) & 0x3f][off], scale, zero);
      out_temp[12][threadIdx.x] = res.x;
      out_temp[13][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp2 >>  6) & 0x3f][off], scale, zero);
      out_temp[14][threadIdx.x] = res.x;
      out_temp[15][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp2 >> 12) & 0x3f][off], scale, zero);
      out_temp[16][threadIdx.x] = res.x;
      out_temp[17][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp2 >> 18) & 0x3f][off], scale, zero);
      out_temp[18][threadIdx.x] = res.x;
      out_temp[19][threadIdx.x] = res.y;
      i += width;
      tmp1 = as_unsigned(mat[i]);
      tmp = (tmp2 >> 24) | ((tmp1 << 4) & 0x30);
      res = hfma2(deq2[tmp][off], scale, zero);
      out_temp[20][threadIdx.x] = res.x;
      out_temp[21][threadIdx.x] = res.y;
      tmp1 >>= 2;
      res = hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero);
      out_temp[22][threadIdx.x] = res.x;
      out_temp[23][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero);
      out_temp[24][threadIdx.x] = res.x;
      out_temp[25][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero);
      out_temp[26][threadIdx.x] = res.x;
      out_temp[27][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero);
      out_temp[28][threadIdx.x] = res.x;
      out_temp[29][threadIdx.x] = res.y;
      res = hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero);
      out_temp[30][threadIdx.x] = res.x;
      out_temp[31][threadIdx.x] = res.y;
      i += width;
      k += 16;
      __syncthreads();
    
      for (int a = 0; a < 32; a++){
        out[(new_row + (k - 16) * 2 + a) * width + col] = out_temp[a][threadIdx.x];
      }
    }
    if (blockcnt > 0){
      for (int k = 0; k < blockcnt; k++) {
        out[outlieridx[k] * width + col] = outlierMat[k * width + col];
      }
    }
    __syncthreads();
  }
}

__global__ void MatQuant4DequantKernel(
    const      int* __restrict__ mat,
             float* __restrict__ out,
    const    float* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    int height,
    int width
) {
  int row = MMBLOCKHEIGHT4B * blockIdx.x;
  int col =  MMBLOCKWIDTH * blockIdx.y + threadIdx.x;
  int new_row = (MMBLOCKHEIGHT4B * 8) * blockIdx.x;
  int bwidth = ((height - row) < MMBLOCKHEIGHT4B) ? ((height - row) * 8) : MMBLOCKWIDTH;

  __shared__ float out_temp[8][MMBLOCKWIDTH];

  if (col < width){
    float scale = scales[col];
    float zero = threadIdx.x % 2 ? \
                 float(zeros[col / 2] >> 4) * scale: \
                 float(zeros[col / 2] & 0xf) * scale;

    int i = width * row + col;
    int k = 0;

    unsigned int tmp;

    while (k < bwidth) {
      tmp = as_unsigned(mat[i]);
      for (int a = 0; a < 8; a++){
        out_temp[a][threadIdx.x] = (scale * float((tmp >> (a * 4)) & 0xf) - zero);
      }
      i += width;
      k += 8;

      for (int a = 0; a < 8; a++){
        out[(new_row + (k - 8) + a) * width + col] = out_temp[a][threadIdx.x];
      }
    }
    __syncthreads();
  }
}

template <typename T1, typename T2>
__global__ void MatQuant4DequantKernelFaster(
    const      int* __restrict__ mat,
                T1* __restrict__ out,
    const       T1* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    int height,
    int width
) {
  const int mmblockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT4B * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int new_row = (BLOCKHEIGHT4B * 8) * blockIdx.x;
  int bwidth = ((height - row) < BLOCKHEIGHT4B) ? ((height - row) * 4) : mmblockwidth2;

  __shared__ T1 out_temp[8][BLOCKWIDTH];

  if (col < width) {
    T2 scale = TtoT2(scales[col]);
    T2 zero = threadIdx.x % 2 ? \
              TtoT2(hmul(int2T<T1>(zeros[col / 2] >> 4), hneg(scale.x))) : \
              TtoT2(hmul(int2T<T1>(zeros[col / 2] & 0xf), hneg(scale.x)));

    int i = width * row + col;
    int k = 0;

    T2 res;
    T2 temp;

    unsigned int tmp;

    while (k < bwidth) {
      tmp = as_unsigned(mat[i]);
      for (int a = 0; a < 4; a++){
        temp = pair2pack(
          int2T<T1>(((tmp >> (a * 8))) & 0x0f),
          int2T<T1>(((tmp >> (a * 8 + 4))) & 0x0f)
        );
        res = hfma2(temp, scale, zero);
        out_temp[2*a][threadIdx.x] = res.x;
        out_temp[2*a+1][threadIdx.x] = res.y;
      }
      i += width;
      k += 4;
    
      for (int a = 0; a < 8; a++){
        out[(new_row + (k - 4) * 2 + a) * width + col] = out_temp[a][threadIdx.x];
      }
    }
    __syncthreads();
  }
}

void matquant3dequant_cuda(
  torch::Tensor mat,
  torch::Tensor out,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + MMBLOCKHEIGHT - 1) / MMBLOCKHEIGHT,
    (width + MMBLOCKWIDTH - 1) / MMBLOCKWIDTH
  );
  dim3 threads(MMBLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    scales.type(), "matquant3dequant_cuda", ([&] {
      MatQuant3DequantKernel<<<blocks, threads>>>(
        mat.data<int>(), out.data<float>(),
        scales.data<float>(), zeros.data<uint8_t>(),
        height, width
      );
    })
  );
}

void matquant3dequantoutlier_faster_cuda(
  torch::Tensor mat,
  torch::Tensor out,
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
    MatQuant3DequantOutlierKernelFaster<nv_bfloat16, nv_bfloat162><<<blocks, threads>>>(
      mat.data_ptr<int>(),
      (nv_bfloat16*) out.data_ptr(),
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
    MatQuant3DequantOutlierKernelFaster<half, half2><<<blocks, threads>>>(
      mat.data_ptr<int>(),
      (half*) out.data_ptr(),
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

void matquant3dequant_faster_cuda(
  torch::Tensor mat,
  torch::Tensor out,
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
    MatQuant3DequantKernelFaster<nv_bfloat16, nv_bfloat162><<<blocks, threads>>>(
      mat.data_ptr<int>(),
      (nv_bfloat16*) out.data_ptr(),
      (nv_bfloat16*) scales.data_ptr(),
      zeros.data_ptr<uint8_t>(),
      height, width
    );
  }
  else{
    MatQuant3DequantKernelFaster<half, half2><<<blocks, threads>>>(
      mat.data_ptr<int>(),
      (half*) out.data_ptr(),
      (half*) scales.data_ptr(),
      zeros.data_ptr<uint8_t>(),
      height, width
    );
  }
}

void matquant4dequant_cuda(
  torch::Tensor mat,
  torch::Tensor out,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + MMBLOCKHEIGHT4B - 1) / MMBLOCKHEIGHT4B,
    (width + MMBLOCKWIDTH - 1) / MMBLOCKWIDTH
  );
  dim3 threads(MMBLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    scales.type(), "matquant4dequant_cuda", ([&] {
      MatQuant4DequantKernel<<<blocks, threads>>>(
        mat.data<int>(), out.data<float>(),
        scales.data<float>(), zeros.data<uint8_t>(),
        height, width
      );
    })
  );
}

void matquant4dequant_faster_cuda(
  torch::Tensor mat,
  torch::Tensor out,
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
    MatQuant4DequantKernelFaster<nv_bfloat16, nv_bfloat162><<<blocks, threads>>>(
      mat.data_ptr<int>(),
      (nv_bfloat16*) out.data_ptr(),
      (nv_bfloat16*) scales.data_ptr(),
      zeros.data_ptr<uint8_t>(),
      height, width
    );
  }
  else{
    MatQuant4DequantKernelFaster<half, half2><<<blocks, threads>>>(
      mat.data_ptr<int>(),
      (half*) out.data_ptr(),
      (half*) scales.data_ptr(),
      zeros.data_ptr<uint8_t>(),
      height, width
    );
  }
}

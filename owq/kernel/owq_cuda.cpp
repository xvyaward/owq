#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include "owq_cuda.h"

int GetBLOCKWIDTH(){
  return BLOCKWIDTH;
};

void vecquant3matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant3matmul_faster_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
); 

void vecquant3outliermatmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor outlierMat, torch::Tensor outlieridx,
  torch::Tensor outrow, torch::Tensor cnt
); 

void vecquant3outliermatmul_faster_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor outlierVec, torch::Tensor outlierMat
); 

void matquant3dequant_cuda(
  torch::Tensor mat, torch::Tensor out,
  torch::Tensor scales, torch::Tensor zeros
);

void matquant3dequant_faster_cuda(
  torch::Tensor mat, torch::Tensor out,
  torch::Tensor scales, torch::Tensor zeros
);

void matquant3dequantoutlier_faster_cuda(
  torch::Tensor mat, torch::Tensor out,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor outlierMat, torch::Tensor outlieridx,
  torch::Tensor outrow, torch::Tensor cnt
);

void vecquant4matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant4matmul_faster_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
); 

void vecquant4outliermatmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor outlierMat, torch::Tensor outlieridx,
  torch::Tensor outrow, torch::Tensor cnt
);

void vecquant4outliermatmul_faster_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor outlierMat, torch::Tensor outlieridx,
  torch::Tensor outrow, torch::Tensor cnt
);

void matquant4dequant_cuda(
  torch::Tensor mat, torch::Tensor out,
  torch::Tensor scales, torch::Tensor zeros
);

void matquant4dequant_faster_cuda(
  torch::Tensor mat, torch::Tensor out,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant3matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_cuda(vec, mat, mul, scales, zeros);
}

void vecquant3matmul_faster(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_faster_cuda(vec, mat, mul, scales, zeros);
}

void vecquant3outliermatmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor outlierMat, torch::Tensor outlieridx,
  torch::Tensor outrow, torch::Tensor cnt
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3outliermatmul_cuda(vec, mat, mul, scales, zeros, outlierMat, outlieridx, outrow, cnt);
}

void vecquant3outliermatmul_faster(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor outlierMat, torch::Tensor outlieridx,
  torch::Tensor outrow, torch::Tensor cnt
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3outliermatmul_faster_cuda(vec, mat, mul, scales, zeros, outlierMat, outlieridx, outrow, cnt);
}

void matquant3dequant(
  torch::Tensor mat, torch::Tensor out,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(scales));
  matquant3dequant_cuda(mat, out, scales, zeros);
}

void matquant3dequant_faster(
  torch::Tensor mat, torch::Tensor out,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(scales));
  matquant3dequant_faster_cuda(mat, out, scales, zeros);
}

void matquant3dequantoutlier_faster(
  torch::Tensor mat, torch::Tensor out,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor outlierMat, torch::Tensor outlieridx,
  torch::Tensor outrow, torch::Tensor cnt
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(scales));
  matquant3dequantoutlier_faster_cuda(mat, out, scales, zeros, outlierMat, outlieridx, outrow, cnt);
}

void vecquant4matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_cuda(vec, mat, mul, scales, zeros);
}

void vecquant4matmul_faster(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_faster_cuda(vec, mat, mul, scales, zeros);
}

void vecquant4outliermatmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor outlierMat, torch::Tensor outlieridx,
  torch::Tensor outrow, torch::Tensor cnt
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4outliermatmul_cuda(vec, mat, mul, scales, zeros, outlierMat, outlieridx, outrow, cnt);
}

void vecquant4outliermatmul_faster(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor outlierMat, torch::Tensor outlieridx,
  torch::Tensor outrow, torch::Tensor cnt
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4outliermatmul_faster_cuda(vec, mat, mul, scales, zeros, outlierMat, outlieridx, outrow, cnt);
}

void matquant4dequant(
  torch::Tensor mat, torch::Tensor out,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(scales));
  matquant4dequant_cuda(mat, out, scales, zeros);
}

void matquant4dequant_faster(
  torch::Tensor mat, torch::Tensor out,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(scales));
  matquant4dequant_faster_cuda(mat, out, scales, zeros);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("GetBLOCKWIDTH", &GetBLOCKWIDTH);
  // 3bit
  m.def("vecquant3matmul", &vecquant3matmul, "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant3matmul_faster", &vecquant3matmul_faster, "Vector 3-bit Quantized Matrix Multiplication (CUDA), float16, bfloat16 faster version");
  m.def("vecquant3outliermatmul", &vecquant3outliermatmul, "Vector 3-bit Quantized Matrix Multiplication with outlier columns (CUDA)");
  m.def("vecquant3outliermatmul_faster", &vecquant3outliermatmul_faster, "Vector 3-bit Quantized Matrix Multiplication with outlier columns (CUDA), float16, bfloat16 faster version");
  m.def("matquant3dequant", &matquant3dequant, "Dequantize 3-bit weight matrix to fp16, bf16 weight matrix");
  m.def("matquant3dequant_faster", &matquant3dequant_faster, "Dequantize 3-bit weight matrix to fp16, bf16 weight matrix, float16, bfloat16 faster version");
  m.def("matquant3dequantoutlier_faster", &matquant3dequantoutlier_faster, "Dequantize 3-bit weight matrix to fp16, bf16 weight matrix, float16, bfloat16 faster version");

  // 4bit
  m.def("vecquant4matmul", &vecquant4matmul, "Vector 4-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant4matmul_faster", &vecquant4matmul_faster, "Vector 4-bit Quantized Matrix Multiplication (CUDA), float16, bfloat16 faster version");
  m.def("vecquant4outliermatmul", &vecquant4outliermatmul, "Vector 4-bit Quantized Matrix Multiplication with outlier columns (CUDA)");
  m.def("vecquant4outliermatmul_faster", &vecquant4outliermatmul_faster, "Vector 4-bit Quantized Matrix Multiplication with outlier columns (CUDA), float16, bfloat16 faster version");
  m.def("matquant4dequant", &matquant4dequant, "Dequantize 4-bit weight matrix to fp16, bf16 weight matrix");
  m.def("matquant4dequant_faster", &matquant4dequant_faster, "Dequantize 4-bit weight matrix to fp16, bf16 weight matrix, float16, bfloat16 faster version");
}

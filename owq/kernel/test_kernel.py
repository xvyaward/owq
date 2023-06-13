import torch
import torch.nn as nn

import quant_cuda
import time

import sys
sys.path.append('../../')
from owq.quant import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEV = torch.device('cuda:0')
error_tolerance = 1e-10

def outlier_quantize(x, scale, zero, maxq, outlieridx=None):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    qx = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    qx = scale * (qx - zero)

    if outlieridx is not None:
        for idx in outlieridx:
            qx[:, idx] = x[:, idx]
    return qx

class OutlierQuantizer(nn.Module):

    def __init__(self, shape=1):
        super(OutlierQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False,
        outlieridx=None
    ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        if trits:
            self.maxq = torch.tensor(-1)
        if outlieridx is None:
            self.outlieridx = []
        else:
            self.outlieridx = outlieridx


    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                # Currently obtaining params from whole weight matrix, not main
                # Scale and weight must be overwritten for actual implementation anyway
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
          self.scale = xmax
          self.zero = xmin
        else:
          self.scale = (xmax - xmin) / self.maxq
          if self.sym:
              self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
          else:
              self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


def latency_measure(RUN_COUNT, M=12288*4, N=12288, extra_bits=0.01):
    DTYPE = torch.half
    mat = torch.randn((M, N), device=DEV, dtype=DTYPE)
    vec = torch.randn((1, M), device=DEV, dtype=DTYPE)
    mul = torch.zeros((1, N), device=DEV, dtype=DTYPE)

    # tick = time.time()
    # for _ in range(RUN_COUNT):
    #     torch.matmul(vec, mat, out=mul) 
    #     torch.cuda.synchronize()
    # print('FP16:', (time.time() - tick) / RUN_COUNT)

    DTYPE = torch.float
    mat = mat.to(DTYPE)
    vec = vec.to(DTYPE)
    mul = mul.to(DTYPE)

    # GPTQ divides by 1024 and multiplies 32 * 3 to make sure the matrix size is divisible by 1024
    mat = torch.randint(-1000000000, 1000000000, (M // 1024 * 96, N), device=DEV, dtype=torch.int)
    scales = torch.randn(N, device=DEV, dtype=DTYPE)
    zeros = torch.randn(N, device=DEV, dtype=DTYPE)

    tick = time.time()
    for _ in range(RUN_COUNT):
        quant_cuda.vecquant3matmul(vec, mat, mul, scales, zeros)
        torch.cuda.synchronize()
    GPTQ = (time.time() - tick) / RUN_COUNT
    # print('3bit GPTQ:', (time.time() - tick) / RUN_COUNT)

    tick = time.time()
    for _ in range(RUN_COUNT):
        quant_cuda.vecquant3matmul_faster(vec, mat, mul, scales, zeros)
        torch.cuda.synchronize()
    GPTQ_FASTER = (time.time() - tick) / RUN_COUNT
    # print('3bit GPTQ:', (time.time() - tick) / RUN_COUNT, '(faster)', mat.shape)

    ### Outlier

    O = int(M*extra_bits * 12 / 13 / 6 / 4)
    # Faster outlier kernel also have 1024 size condition
    mat3 = torch.randint(-1000000000, 1000000000, (M // 1024 * 96, N), device=DEV, dtype=torch.int)
    vec3 = torch.randn((M,), device=DEV, dtype=DTYPE)
    matOutlier = torch.randn((N, O), device=DEV, dtype=DTYPE)
    vecOutlier = torch.randn((O,), device=DEV, dtype=DTYPE)

    vec3 = vec3.to(DTYPE)
    matOutlier = matOutlier.to(DTYPE)
    vecOutlier = vecOutlier.to(DTYPE)
    # vecOutlier = vecOutlier.to(torch.int)

    tick = time.time()
    for _ in range(RUN_COUNT):
        quant_cuda.vecquant3outliermatmul(vec3, mat3, mul, scales, zeros, vecOutlier, matOutlier)
        torch.cuda.synchronize()
    OUTL = (time.time() - tick) / RUN_COUNT
    # print('3bit OUTL:', (time.time() - tick) / RUN_COUNT)
    print(f'{3 + extra_bits} target_bit: {(OUTL-GPTQ)/GPTQ*100} % overhead, GPTQ {GPTQ}, OUTL {OUTL}')

    matOutlier = torch.randn((N, (int)((O+1)/2)), device=DEV, dtype=torch.half)
    vecOutlier = torch.randn(((int)((O+1)/2),), device=DEV, dtype=torch.half)
    matOutlier = matOutlier.to(torch.half)
    vecOutlier = vecOutlier.to(torch.half)

    tick = time.time()
    for _ in range(RUN_COUNT):
        quant_cuda.vecquant3outliermatmul_faster(vec3, mat3, mul, scales, zeros, vecOutlier, matOutlier)
        torch.cuda.synchronize()
    OUTL_FASTER = (time.time() - tick) / RUN_COUNT
    # print('3bit OUTL:', (time.time() - tick) / RUN_COUNT)
    print(f'{3 + extra_bits} target_bit: {(OUTL_FASTER-GPTQ_FASTER)/GPTQ_FASTER*100} % overhead, GPTQ_FASTER {GPTQ_FASTER}, OUTL_FASTER {OUTL_FASTER}')

def correctness(M=4*4096, N=4096, outlieridx=[0], faster=False):
    layer = nn.Linear(M, N)
    vec = torch.randn(M)

    # import pdb; pdb.set_trace()
    if faster:
        layer = layer.to(dtype=torch.float16)
        vec = vec.to(dtype=torch.float16)

    quantizer = OutlierQuantizer()
    quantizer.configure(3, perchannel=True, sym=False, mse=False, outlieridx=outlieridx)
    quantizer.find_params(layer.weight.data, weight=True)
    layer.weight.data = outlier_quantize(
        layer.weight.data, quantizer.scale, quantizer.zero, quantizer.maxq, quantizer.outlieridx
    )
    if faster:
        layer.weight.data = layer.weight.data.to(dtype=torch.float16)

    qlayer = Quant3Linear(layer.in_features, layer.out_features, len(quantizer.outlieridx), bias=True, faster=faster)
    outlieridx = torch.tensor(quantizer.outlieridx)
    qlayer.pack(layer, quantizer.scale, quantizer.zero, outlieridx)

    qlayer = qlayer.to(DEV)
    layer = layer.to(DEV)
    vec = vec.to(DEV)

    with torch.no_grad():
        if faster:
            print('Simu:', layer(vec))
            print('Kern:', qlayer(vec))
        if ((layer(vec)-qlayer(vec))**2).sum()/N < error_tolerance:
            print('Passed, MSE:', ((layer.to(DEV)(vec)-qlayer(vec))**2).sum()/N)
        else:
            print('Failed, MSE:', ((layer.to(DEV)(vec)-qlayer(vec))**2).sum()/N)

if __name__=="__main__":
    import random
    # torch.set_printoptions(precision=10)
    print('Benchmarking OPT-175B FC2 matvec with outlier ...')
    print(f'::: Latency Benchmark :::')
    for i in range(1, 6):
        latency_measure(1000, M=12288*4, N=12288, extra_bits=0.01*i)
    print(f'::: Correctness check :::')
    for i in range(1, 6):
        n_outcol = round(12288 * 0.01 * i * 12 / 13 / 6)
        print(f'# of weak columns: {n_outcol}')
        correctness(outlieridx=random.shuffle(list(range(0,n_outcol))))
    # TODO: faster-kernel test logic debugging
    # print(f'::: Correctness check for faster mode. CAUTION: may fail due to bad fp16 conversion. check output.:::')
    # for i in range(1, 5):
    #     print(f'Removing first {i*32+1}')
    #     correctness(outlieridx=range(0,i*32+1), faster=True)

import torch
import torch.nn as nn
import numpy as np
import time

import owq_cuda

import sys
sys.path.append('../../')
from owq.quant import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEV = torch.device('cuda:0')
error_tolerance = 1e-6

def outlier_quantize(x, scale, zero, maxq, outlieridx=None):
    qx = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    qx = scale * (qx - zero)

    if outlieridx is not None:
        for idx in outlieridx:
            qx[:, idx] = x[:, idx]
    return qx


def latency_measure(RUN_COUNT, M=12288*4, N=12288, numout=0, bits=3):
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=DEV)
    def flush_cache():
        cache.zero_()
    # GPTQ divides by 1024 and multiplies 32 * 3 to make sure the matrix size is divisible by 1024
    vec = torch.randn((1, M), device=DEV, dtype=torch.half)
    fmat = torch.randn((N, M), device=DEV, dtype=torch.half)
    mul = torch.zeros((1, N), device=DEV, dtype=torch.half)
    
    warm_up = 1000
    for _ in range(warm_up):
        nn.functional.linear(vec, fmat, mul)
    
    times = []
    for _ in range(RUN_COUNT):
        tick = time.perf_counter()
        flush_cache()
        nn.functional.linear(vec, fmat, mul)
        # torch.cuda.synchronize()
        times.append(time.perf_counter() - tick)
    FP16 = np.median(times)
    
    mul = mul.float()
    mat = torch.randint(-1000000000, 1000000000, (M // 32 * 3, N), device=DEV, dtype=torch.int)
    scales = torch.randn(N, device=DEV, dtype=torch.half)
    zeros = torch.randint(0, 2**bits, (N // 2,), device=DEV, dtype=torch.uint8)
    
    times = []
    for _ in range(RUN_COUNT):
        tick = time.perf_counter()
        flush_cache()
        owq_cuda.vecquant3matmul_faster(vec, mat, mul, scales, zeros)
        # torch.cuda.synchronize()
        times.append(time.perf_counter() - tick)
    NO_OUTLIER = np.median(times)

    ### Outlier
    O = numout
    outidx = torch.randint(0, M, [O], device=DEV, dtype=torch.int)
    matOutlier = torch.randn((O, N), device=DEV, dtype=torch.half)
    
    mul = mul.half()
    
    BLOCKWIDTH = owq_cuda.GetBLOCKWIDTH()
    NUMBLOCK = (M + BLOCKWIDTH - 1) // BLOCKWIDTH
    cnt = torch.bincount(
        (outidx).to(torch.long) // BLOCKWIDTH,
        minlength=NUMBLOCK).to(torch.int).to(vec.device)
    outrow = torch.zeros_like(cnt)
    for i in range(1, NUMBLOCK):
        outrow[i] = outrow[i-1] + cnt[i-1];
    
    times = []
    for _ in range(RUN_COUNT):
        tick = time.perf_counter()
        flush_cache()
        owq_cuda.vecquant3outliermatmul_faster(vec, mat, mul, scales, zeros, matOutlier, outidx, outrow, cnt)
        # torch.cuda.synchronize()
        times.append(time.perf_counter() - tick)
    OWQ = np.median(times)
    
    print(f'FP16 {FP16:.4e}, NO_OUTLIER {NO_OUTLIER:.4e}, OWQ {OWQ:.4e} (+{(OWQ - NO_OUTLIER)/NO_OUTLIER*100:.4f} % overhead over NO_OUTLIER)\n')

def correctness(M=4*12288, N=12288, bits=3, outlieridx=[], faster=False):
    layer = nn.Linear(M, N, dtype=torch.half)
    vec = torch.randn(M, dtype=torch.half)
    
    quantizer = Quantizer(bits, perchannel=True, sym=False, mse=False)
    quantizer.find_params(layer.weight.data, weight=True)
    layer.weight.data = outlier_quantize(
        layer.weight.data.float(), quantizer.scale, quantizer.zero, quantizer.maxq, outlieridx
    )
    layer.weight.data = layer.weight.data.to(dtype=torch.half)

    qlayer = QuantLinear(bits,
                         layer.in_features, 
                         layer.out_features, 
                         len(outlieridx), 
                         bias=layer.bias is not None, 
                         dtype=layer.weight.dtype,)
    outlieridx = torch.tensor(outlieridx)
    qlayer.pack(layer, quantizer.scale, quantizer.zero, outlieridx)
    if not faster:
        qlayer.faster = False
        qlayer.oweight = qlayer.oweight.float()
    qlayer.set_kernel()

    qlayer = qlayer.to(DEV)
    layer = layer.to(DEV)
    vec = vec.to(DEV)

    with torch.no_grad():
        res_sim = layer(vec).float()
        res_ker = qlayer(vec).float()
        print('Simu :', res_sim)
        print('Kern :', res_ker)
        if faster:
            non = res_ker.isnan().sum()
            if non > 0:
                print(f"# of nan : {non} / {res_ker.numel()}")
                print(f"nan idx : {torch.where(res_ker.isnan())}")
                print(f"outlier idx : {outlieridx}")
        err = ((res_sim-res_ker)**2).sum()/N
        print(f"{'Passed' if err < error_tolerance else 'Failed'}" + ', MSE :', err)

if __name__=="__main__":
    bits=3
    for d, model in zip([4096],['opt-6.7b']): # opt-6.7b
    # for d, model in zip([12288],['opt-175b']): # opt-175b
        n = 6
        print(f'Benchmarking {model.upper()} matvec with outlier ...')
        for M,N in [[d,d],[d,d*4],[d*4,d]]:
            print(f"M={M}, N={N}")
            r = (M * N) / (d * d)
            print(f'::: Latency Benchmark :::')
            for i in [1, 10]:
                n_outcol = round(M * 0.01 * i * 12 / (16 - bits) / n / r)
                n_outcol = n_outcol+1 if n_outcol % 2 else n_outcol
                print(f'target_bit : {bits + 0.01 * i} bits | # of weak columns: {n_outcol}')
                latency_measure(10000, M=M, N=N, numout=n_outcol, bits=bits)
                # break
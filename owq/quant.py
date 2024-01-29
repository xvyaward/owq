import numpy as np
import torch
import torch.nn as nn
from transformers.models.falcon.modeling_falcon import FalconLinear

try:
    import owq_cuda
except:
    print('OWQ CUDA kernel extension is not installed.')

def quantize(x, scale, zero, minq, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, minq, maxq)
    return scale * (q - zero)

def quantize_efficient(x_round, scale, zero, minq, maxq):
    q = torch.clamp(x_round + zero, minq, maxq)
    return scale * (q - zero)

class Quantizer(nn.Module):
    def __init__(
            self,
            bits, perchannel=False, sym=False, 
            mse=False, norm=2.4, 
        ):
        super(Quantizer, self).__init__()
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        self.register_buffer('out_ids', torch.zeros(1))
        
        self.bits = bits
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.perchannel = perchannel
        self.n_levels = 2 ** bits
        
        if self.sym:
            self.minq, self.maxq = -((self.n_levels - 1) // 2 + 1), (self.n_levels - 1) // 2
        else:
            self.minq, self.maxq = 0, self.n_levels - 1
        
        self.num = 100
        self.eps = torch.tensor(1e-8)
        
    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if not self.perchannel:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)
    
    def find_params(self, x, weight=False, num=100):
        self.num = num
        dev = x.device
        minq, maxq = self.minq, self.maxq
        
        shape = x.shape
        if self.perchannel: # row-wise
            if weight:
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
        
        if self.mse:
            if self.perchannel:
                new_shape = [-1] + [1] * (len(x.shape) -  1)
            
            best_score = torch.zeros_like(xmin) + (1e+10)
            best_min = xmin.clone()
            best_max = xmax.clone()
            
            if self.sym:
                xrange = torch.max(xmin.abs(), xmax)
                zero = torch.zeros_like(xmin)
                if self.perchannel:
                    zero = zero.reshape(new_shape)
                for i in range(1, self.num + 1):
                    tmp_max = xrange / self.num * i
                    scale = torch.max(tmp_max / -minq, self.eps)
                    if self.perchannel:
                        scale = scale.reshape(new_shape)
                    x_round = torch.round(x / scale)
                    x_q = quantize_efficient(x_round, scale, zero, minq, maxq)
                    score = self.lp_loss(x, x_q, 2.4)
                    best_max = torch.where(score < best_score, tmp_max, best_max)
                    best_score = torch.min(score, best_score)
                
                max_val = torch.max(best_max, torch.zeros_like(best_max))

                self.scale = torch.max(max_val / -minq, self.eps)
                self.zero = torch.zeros_like(self.scale)
            else:
                xrange = xmax - xmin
                tmp_min = torch.zeros_like(xmin)
                for i in range(1, self.num + 1):
                    tmp_max = xrange / self.num * i
                    scale = torch.max((tmp_max - tmp_min) / (maxq - minq), self.eps)
                    delta = scale.clone()
                    if self.perchannel:
                        scale = scale.reshape(new_shape)
                    x_round = torch.round(x / scale)
                    for zp in range(0, self.n_levels):
                        new_min = tmp_min - zp * delta
                        new_max = tmp_max - zp * delta
                        zero = torch.clamp(minq - torch.round(new_min / delta), minq, maxq)
                        if self.perchannel:
                            zero = zero.reshape(new_shape)
                        x_q = quantize_efficient(x_round, scale, zero, minq, maxq)
                        score = self.lp_loss(x, x_q, 2.4)
                        best_min = torch.where(score < best_score, new_min, best_min)
                        best_max = torch.where(score < best_score, new_max, best_max)
                        best_score = torch.min(best_score, score)
            
                min_val_neg = torch.min(best_min, torch.zeros_like(best_min))
                max_val_pos = torch.max(best_max, torch.zeros_like(best_max))

                self.scale = torch.max((max_val_pos - min_val_neg) / (maxq - minq), self.eps)
                self.zero = torch.clamp(minq - torch.round(min_val_neg / self.scale), minq, maxq)
        else:
            if self.sym:
                xmax = torch.maximum(torch.abs(xmin), xmax)
                tmp = xmin < 0
                if torch.any(tmp):
                    xmin[tmp] = -xmax[tmp]

            tmp = (xmin == 0) & (xmax == 0) 
            xmin[tmp] = -1
            xmax[tmp] = +1

            if self.sym:
                self.scale = xmax / -minq
                self.zero = torch.zeros_like(self.scale)
            else:
                self.scale = (xmax - xmin) / maxq
                self.zero = torch.round(-xmin / self.scale)
        
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
            return quantize(x, self.scale, self.zero, self.minq, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

def make_quant(module, n_out_infos, wbits, name=''):
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in n_out_infos:
            setattr(
                module, attr, 
                QuantLinear(wbits, 
                            tmp.in_features, 
                            tmp.out_features, 
                            n_out_infos[name1].n_out, 
                            tmp.bias is not None, 
                            tmp.weight.dtype,
                            name1).to(tmp.weight.device)
            )
    for name1, child in module.named_children():
        make_quant(child, n_out_infos, wbits, name + '.' + name1 if name != '' else name1)

def lm_pack(model, quantinfos, wbits, linears=[nn.Linear, FalconLinear]):
    from owq.utils.misc import find_layers
    layers = find_layers(model, linears)
    layers = {n: layers[n] for n in quantinfos}
    make_quant(model, quantinfos, wbits)
    qlayers = find_layers(model, [QuantLinear])
    print('Packing ...')
    for name in qlayers:
        quantinfos[name] = quantinfos[name].cpu()
        qlayers[name].pack(layers[name], 
                           quantinfos[name].scale, 
                           quantinfos[name].zero, 
                           quantinfos[name].out_ids
                           )
    print('Done.')
    return model

class QuantMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, oweight, fn_dequant, qweight, scales, zeros, shape, n_out, outids, bias):
        
        # 1. Dequantize
        out = torch.empty(shape, dtype=oweight.dtype, device=oweight.device)        
        fn_dequant(qweight, out, scales, zeros)
        out[outids, :] = oweight
        out = out.t()
        
        # 2. Matmul
        output = torch.nn.functional.linear(x.to(bias.dtype), out.to(bias.dtype), bias)
        
        ctx.dequant_params = [oweight, fn_dequant, qweight, scales, zeros, shape, n_out, outids]
        ctx.tensors = torch.index_select(x, -1, outids)
        ctx.n_out = n_out
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x_outlier = ctx.tensors
        oweight, fn_dequant, qweight, scales, zeros, shape, n_out, outids = ctx.dequant_params
        
        # Dequantize
        out = torch.empty(shape, dtype=oweight.dtype, device=oweight.device)
        fn_dequant(qweight, out, scales, zeros)
        out[outids, :] = oweight
        out = out.t()
            
        grad_input = None
        grad_oweight = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, out.to(grad_output.dtype))
        if ctx.needs_input_grad[1]:
            grad_oweight = torch.matmul(grad_output.transpose(-2,-1), x_outlier.to(grad_output.dtype))
        
        return grad_input, grad_oweight, None, None, None, None, None, None, None, None

class QuantLinear(nn.Module):

    def __init__(self, bits, infeatures, outfeatures, outlierfeatures, bias, dtype, name):
        super().__init__()
        assert bits in [3,4], "Only 3,4 bits are supported."
        
        self.bits = bits
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.outlierfeatures = outlierfeatures
        
        self.register_buffer(
            'qweight', torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32)
        )
        self.register_buffer('scales', torch.zeros((outfeatures, 1), dtype=dtype))
        self.register_buffer('zeros', torch.zeros((outfeatures // 2, 1), dtype=torch.uint8))
        self.register_buffer('bias', torch.zeros(outfeatures, dtype=dtype))
        
        self.register_buffer(
            'oweight', torch.zeros((outlierfeatures, outfeatures), dtype=dtype)
        )
        self.register_buffer(
            'outlieridx', torch.zeros((outlierfeatures), dtype=torch.int)
        )
        
        self.faster = True
        self.dtype = dtype
        self.name = name
        
    def pack(self, linear, scales, zeros, outlieridx:torch.Tensor, sym:bool=False):
        dtype = linear.weight.dtype
        
        if sym:
            zeros += 2**(self.bits - 1)
            
        if linear.bias is not None:
            self.bias = linear.bias.to(dtype)
            
        self.outlieridx = outlieridx

        if self.outlierfeatures > 0:
            self.oweight = torch.index_select(linear.weight.data, 1, self.outlieridx).t().contiguous()
        
        intweight = torch.round((linear.weight.data + zeros * scales) / scales).to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        if self.outlierfeatures > 0:
            for idx in outlieridx:
                intweight[idx,:] = zeros.numpy().astype(np.uint32).squeeze()
        qweight = np.zeros(
            (self.infeatures // 32 * self.bits, self.outfeatures), dtype=np.uint32
        )
        
        self.scales = scales.to(dtype)
        zeros = zeros.to(torch.uint8)
        zeros_int = torch.zeros((zeros.shape[0] // 2, zeros.shape[1]), dtype=torch.uint8)
        for i in range(zeros_int.shape[0]):
            zeros_int[i] = (zeros[2*i] | zeros[2*i + 1] << 4)
        self.zeros = zeros_int
        
        i = 0
        row = 0
        if self.bits == 3:
            while row < qweight.shape[0]:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):    
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):    
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
        elif self.bits == 4:
            while row < qweight.shape[0]:
                for j in range(i, i + 8):
                    qweight[row] |= intweight[j] << (4 * (j - i))
                i += 8
                row += 1
        else:
            raise NotImplementedError

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)
    
    def set_kernel(self, faster):
        if self.outlierfeatures % 2 > 0:
            print("Number of outlier is not even. manually set to faster=False.")
            faster = False
        self.faster = faster
        
        if faster == False:
            self.oweight = self.oweight.float()
            self.scales = self.scales.float()
        
        # for outliermatvec kernel
        if self.outlierfeatures > 0:
            BLOCKWIDTH = owq_cuda.GetBLOCKWIDTH()
            NUMBLOCK = (self.infeatures + BLOCKWIDTH - 1) // BLOCKWIDTH
            self.register_buffer('cnt', torch.zeros([NUMBLOCK], dtype=torch.int))
            self.register_buffer('outrow', torch.zeros([NUMBLOCK], dtype=torch.int))
            self.cnt = torch.bincount(
                (self.outlieridx).to(torch.long) // BLOCKWIDTH,
                minlength=NUMBLOCK).to(torch.int)
            self.outrow = torch.zeros_like(self.cnt)

            for i in range(1, NUMBLOCK):
                self.outrow[i] = self.outrow[i-1] + self.cnt[i-1]
        
        # set operation kernel
        if self.bits == 3:
            if faster:
                self.matvec = owq_cuda.vecquant3matmul_faster
                self.outmatvec = owq_cuda.vecquant3outliermatmul_faster
                self.dequant = owq_cuda.matquant3dequant_faster
            else:
                self.matvec = owq_cuda.vecquant3matmul
                self.outmatvec = owq_cuda.vecquant3outliermatmul
                self.dequant = owq_cuda.matquant3dequant
        elif self.bits == 4:
            if faster:
                self.matvec = owq_cuda.vecquant4matmul_faster
                self.outmatvec = owq_cuda.vecquant4outliermatmul_faster
                self.dequant = owq_cuda.matquant4dequant_faster
            else:
                self.matvec = owq_cuda.vecquant4matmul
                self.outmatvec = owq_cuda.vecquant4outliermatmul
                self.dequant = owq_cuda.matquant4dequant
        else: # support only 3, 4 bits
            raise NotImplementedError
        
        if self.outlierfeatures > 0:
            self.matmul = QuantMatMul.apply  
            if self.faster:
                self.forward = self.forward_faster_outlier
            else:
                self.forward = self.forward_normal_outlier
        else:
            if self.faster:
                self.forward = self.forward_faster
            else:
                self.forward = self.forward_normal

    def forward_faster_outlier(self, x):
        if x.shape[-1] == x.numel():
            y = self.bias.clone()
            self.outmatvec(
                x, self.qweight, y,
                self.scales, self.zeros,
                self.oweight, self.outlieridx,
                self.outrow, self.cnt
                )
        else:
            matshape = (self.infeatures, self.outfeatures)
            y = self.matmul(x, self.oweight, self.dequant, 
                self.qweight, self.scales, 
                self.zeros, matshape,
                self.outlierfeatures, self.outlieridx,
                self.bias)
        return y
    
    def forward_normal_outlier(self, x):
        if x.shape[-1] == x.numel():
            dtype = x.dtype
            y = self.bias.float()
            x = x.float()
            self.outmatvec(
                x, self.qweight, y,
                self.scales, self.zeros,
                self.oweight, self.outlieridx,
                self.outrow, self.cnt
                )
            y = y.to(dtype)
        else:
            matshape = (self.infeatures, self.outfeatures)
            y = self.matmul(x, self.oweight, self.dequant, 
                self.qweight, self.scales, 
                self.zeros, matshape,
                self.outlierfeatures, self.outlieridx,
                self.bias)
        return y
    
    def forward_faster(self, x):
        if x.shape[-1] == x.numel():
            y = self.bias.clone()
            self.matvec(
                x, self.qweight, y,
                self.scales, self.zeros
                )
            y = y.to(x.dtype)
        else:
            out = torch.empty((self.infeatures, self.outfeatures), dtype=x.dtype, device=x.device)
            self.dequant(self.qweight, out, self.scales, self.zeros)
            y = torch.nn.functional.linear(x, out.t(), self.bias)
        return y
    
    def forward_normal(self, x):
        if x.shape[-1] == x.numel():
            dtype = x.dtype
            y = self.bias.float()
            x = x.float()
            self.matvec(
                x, self.qweight, y,
                self.scales, self.zeros
                )
            y = y.to(dtype)
        else:
            out = torch.empty((self.infeatures, self.outfeatures), dtype=torch.float, device=x.device)
            self.dequant(self.qweight, out, self.scales, self.zeros)
            y = torch.nn.functional.linear(x, out.t().to(x.dtype), self.bias)
        return y
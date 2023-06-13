import numpy as np
import torch
import torch.nn as nn


def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

def quantize_efficient(x_round, scale, zero, maxq):
    q = torch.clamp(x_round + zero, 0, maxq)
    return scale * (q - zero)

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
            self,
            bits, perchannel=False, sym=True, 
            mse=False, norm=2.4, 
        ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.n_levels = self.maxq + 1
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
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
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
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

            xrange = xmax - xmin
            best_score = torch.zeros_like(xmin) + (1e+10)
            best_min = xmin.clone()
            best_max = xmax.clone()
            tmp_min = torch.zeros_like(xmin)
            
            for i in range(1, self.num + 1):
                tmp_max = xrange / self.num * i
                scale = torch.max((tmp_max - tmp_min) / self.maxq, self.eps)
                delta = scale.clone()
                if self.perchannel:
                    scale = scale.reshape(new_shape)
                x_round = torch.round(x / scale)
                for zp in range(0, self.n_levels):
                    new_min = tmp_min - zp * delta
                    new_max = tmp_max - zp * delta
                    zero = torch.clamp(-torch.round(new_min / delta), 0, self.maxq)
                    if self.perchannel:
                        zero = zero.reshape(new_shape)
                    x_q = quantize_efficient(x_round, scale, zero, self.maxq)
                    score = self.lp_loss(x, x_q, 2.4)
                    best_min = torch.where(score < best_score, new_min, best_min)
                    best_max = torch.where(score < best_score, new_max, best_max)
                    best_score = torch.min(best_score, score)
            
            min_val_neg = torch.min(best_min, torch.zeros_like(best_min))
            max_val_pos = torch.max(best_max, torch.zeros_like(best_max))

            self.scale = torch.max((max_val_pos - min_val_neg) / self.maxq, self.eps)
            self.zero = torch.clamp(-torch.round(min_val_neg / self.scale), 0, self.maxq)
        else:
            if self.sym:
                xmax = torch.maximum(torch.abs(xmin), xmax)
                tmp = xmin < 0
                if torch.any(tmp):
                    xmin[tmp] = -xmax[tmp]

            tmp = (xmin == 0) & (xmax == 0) 
            xmin[tmp] = -1
            xmax[tmp] = +1

            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
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
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

class n_out_saver:
    def __init__(self, n_out):
        self.n_out = n_out

try:
    import quant_cuda
except:
    print('CUDA kernel extension is not installed.')

# Assumes layer is perfectly divisible into 1024 * 1024 blocks
class Quant3Linear(nn.Module):

    def __init__(self, infeatures, outfeatures, outlierfeatures, bias, faster=False):
        super().__init__()
        self.register_buffer('zeros', torch.zeros((outfeatures, 1)))
        self.register_buffer('scales', torch.zeros((outfeatures, 1)))
        if bias:
            self.register_buffer('bias', torch.zeros(outfeatures))
        else:
            self.bias = None
        self.register_buffer(
            'qweight', torch.zeros((infeatures  // 32 * 3, outfeatures), dtype=torch.int)
        )
        if faster:
            self.register_buffer(
                'oweight', torch.zeros((outfeatures, outlierfeatures), dtype=torch.half)
            )
            if outlierfeatures % 2 == 1:
                self.oweight = nn.functional.pad(self.oweight, (0,1))
        else:
            self.register_buffer(
                'oweight', torch.zeros((outfeatures, outlierfeatures), dtype=torch.float)
            )
        
        self.faster = faster
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.outlierfeatures = outlierfeatures
        self.register_buffer(
            'outlieridx', torch.zeros((outlierfeatures), dtype=torch.int)
        )

    def pack(self, linear, scales, zeros, outlieridx:torch.Tensor):
        self.zeros = zeros * scales
        self.scales = scales.clone()
        if linear.bias is not None:
            self.bias = linear.bias.float().clone()
        else:
            self.bias = None
        
        self.outlieridx = outlieridx.to(torch.int)

        if self.outlierfeatures > 0:
            self.oweight = torch.index_select(linear.weight.data, 1, self.outlieridx)
            if self.faster and self.oweight.shape[1] % 2 == 1:
                self.oweight = nn.functional.pad(self.oweight, (0,1))
                
        intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        if self.outlierfeatures > 0:
            for idx in outlieridx:
                intweight[idx,:] = zeros.numpy().astype(np.uint32).squeeze()
        qweight = np.zeros(
            (self.infeatures // 32 * 3, intweight.shape[1]), dtype=np.uint32
        )

        i = 0
        row = 0
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

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

    def forward(self, x):
        outshape = list(x.shape)
        x = x.squeeze()
        if x.shape[-1] == x.numel():
            if self.bias is not None:
                y = self.bias.clone()
            else:
                y = torch.zeros(self.outfeatures, dtype=torch.float32, device=x.device)
            outshape[-1] = y.numel()
            if self.outlierfeatures > 0:
                x_main = x
                x_outlier = torch.index_select(x, -1, self.outlieridx)
                if self.faster and x_outlier.shape[-1] % 2 == 1:
                    paddim = [0,0] * len(x_outlier.shape)
                    paddim[-1] = 1
                    x_outlier = nn.functional.pad(x_outlier, paddim)
            dtype = x.dtype
            if self.outlierfeatures > 0:
                if self.faster:
                    quant_cuda.vecquant3outliermatmul_faster(
                        x_main, self.qweight, y,
                        self.scales, self.zeros,
                        x_outlier, self.oweight
                        )
                else:
                    x_main = x_main.float()
                    x_outlier = x_outlier.float()
                    quant_cuda.vecquant3outliermatmul(
                        x_main, self.qweight, y,
                        self.scales, self.zeros,
                        x_outlier, self.oweight
                        )
            else:
                if self.faster:
                    x = x.half()
                    quant_cuda.vecquant3matmul_faster(
                        x, self.qweight, y, 
                        self.scales, self.zeros
                        )
                else:
                    x = x.float()
                    quant_cuda.vecquant3matmul(
                        x, self.qweight, y,
                        self.scales, self.zeros
                        )
            
            y = y.to(dtype)
            return y.reshape(outshape)
        raise ValueError('Only supports a single token currently.')

def make_quant3(module, names, name='', faster=False):
    if isinstance(module, Quant3Linear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, Quant3Linear(tmp.in_features, tmp.out_features, names[name1].n_out, tmp.bias is not None, faster=faster)
            )
    for name1, child in module.named_children():
        make_quant3(child, names, name + '.' + name1 if name != '' else name1, faster=faster)

def lm_pack3(model, quantizers, faster=False):
    from owq.utils.misc import find_layers
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers, faster=faster)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero, quantizers[name].out_ids)
    print('Done.')
    return model
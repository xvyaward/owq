from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='owq_cuda',
    ext_modules=[cpp_extension.CUDAExtension(
        'owq_cuda', ['owq_cuda.cpp', 'gemv.cu', 'dequant.cu']
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)

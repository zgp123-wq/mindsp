#!/usr/bin/env python
import glob
import os
# import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension

from mindspore import context
import os

CUDA_HOME = os.environ.get('CUDA_HOME', None)

def is_gpu_available():
    if context.get_context("device_target") == "GPU":
        return True
    return False

# 设置运行环境为 GPU
context.set_context(device_target="GPU")

def get_extensions():
    extensions_dir = os.getcwd().replace('build_stuff', 'csrc')
    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
    sources = main_file + source_cpu

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (is_gpu_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = ["-DCUDA_HAS_FP16=1",
                                      "-D__CUDA_NO_HALF_OPERATORS__",
                                      "-D__CUDA_NO_HALF_CONVERSIONS__",
                                      "-D__CUDA_NO_HALF2_OPERATORS__"]

    ext_modules = [extension("_C",
                             sources,
                             include_dirs=[extensions_dir],
                             define_macros=define_macros,
                             extra_compile_args=extra_compile_args)]

    return ext_modules


setup(name="paa_minimal",
      version="1.0",
      author="feiyuhuahuo",
      url="https://github.com/feiyuhuahuo/paa_minimal",
      packages=find_packages(),
      ext_modules=get_extensions(),
      cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
      include_package_data=True)

import os
import torch
from torch.utils.cpp_extension import load

module_path = os.path.dirname(os.path.realpath(__file__))

# Eigen 路径
eigen_include = "/home/ok/miniconda3/envs/fdp/include/eigen3"

sources = [
    os.path.join(module_path, 'common.cu'),
    os.path.join(module_path, 'bindings.cpp')
]

print(f"Building with Eigen: {eigen_include}")
for s in sources:
    print(f"  {s}")

# 关键：使用 extra_include_paths（兼容旧版 PyTorch）
common = load(
    name='common',
    sources=sources,
    extra_cflags=['-O2', '-std=c++14'],
    extra_cuda_cflags=['-O2', '--expt-relaxed-constexpr'],
    extra_include_paths=[eigen_include],  # 兼容旧版！
    verbose=True
)

print("\nFINAL SUCCESS! 'common' module built and loaded.")
print("You can now: import common")

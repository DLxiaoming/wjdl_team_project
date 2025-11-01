#!/bin/bash
set -e
PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Building mycpp..."
cd "${PROJ_ROOT}/mycpp/"
rm -rf build; mkdir -p build; cd build; cmake ..; make -j$(nproc)

echo "Building mycuda..."
cd "${PROJ_ROOT}/bundlesdf/mycuda"
rm -rf build *.egg-info *.so dist __pycache__ 2>/dev/null || true

cat > build_mycuda.py << 'PY'
import os, torch
from torch.utils.cpp_extension import load
module_path = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.join(module_path, 'src')
sources = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(('.cu', '.cpp'))]
load(name='common', sources=sources, extra_cflags=['-O2'], extra_cuda_cflags=['-O2'], verbose=True)
print("mycuda built!")
PY

python build_mycuda.py

echo "SUCCESS! Try: python -c 'import common'"

"""
Build script for moe_fused CK extension.

Usage (on AMD GPU system with ROCm installed):
    python3 setup.py build_ext --inplace

This will:
1. Compile moe_fused_kernel.cpp with hipcc
2. Build pybind11 Python bindings
3. Install moe_fused.kernel module

Requirements:
- AMD ROCm 5.7+
- Composable Kernel (CK) headers
- pybind11
- HIP compiler (hipcc)
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Check for ROCm installation
ROCM_PATH = os.environ.get('ROCM_PATH', '/opt/rocm')
HIP_PATH = os.environ.get('HIP_PATH', '/opt/rocm/hip')

# Check for CK headers
CK_INCLUDE_PATH = os.environ.get('CK_INCLUDE_PATH', '/opt/rocm/include/ck')


class CMakeExtension(Extension):
    """Custom extension for HIP/CK code"""

    def __init__(self, name, source_dir=''):
        super().__init__(name, sources=[])
        self.source_dir = Path(source_dir).resolve()


class CustomBuildExt(build_ext):
    """Custom build extension for HIP compilation"""

    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = Path(self.get_ext_fullpath(ext.name)).parent
        extdir.mkdir(parents=True, exist_ok=True)

        # Get extension output path
        ext_suffix = '.so' if sys.platform != 'win32' else '.pyd'
        output_file = extdir / f'moe_fused_kernel{ext_suffix}'

        # Source file
        source_file = Path(__file__).parent / 'moe_fused' / 'cpp' / 'moe_fused_kernel.cpp'

        # Compile flags
        compile_flags = [
            '-shared',
            '-fPIC',
            '-O3',
            '--offload=amdgcn-amd-amdhsa--gfx942',  # MI300X
            '-march=native',
            f'-I{ROCM_PATH}/include',
            f'-I{CK_INCLUDE_PATH}',
            f'-I{sys.prefix}/include',  # pybind11 headers
            '-D_GLIBCXX_USE_CXX11_ABI=0',
            '-DPYBIND11_MODULE',
        ]

        # Link flags
        link_flags = [
            f'-L{ROCM_PATH}/lib',
            '-lhip_runtime',
            '-lck',
        ]

        # Build command
        cmd = ['hipcc'] + compile_flags + [str(source_file), '-o', str(output_file)] + link_flags

        print(f"Building {ext.name}...")
        print(f"  Command: {' '.join(cmd)}")

        try:
            subprocess.check_call(cmd)
            print(f"  Built: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"  Error building {ext.name}: {e}")
            print(f"\nNote: This build requires:")
            print(f"  - AMD ROCm installed at {ROCM_PATH}")
            print(f"  - Composable Kernel headers at {CK_INCLUDE_PATH}")
            print(f"  - HIP compiler (hipcc) in PATH")
            print(f"\nSkipping CK extension build. Pure Python fallback will be used.")


# Extension definition
moe_fused_extension = CMakeExtension(
    'moe_fused.moe_fused_kernel',
    source_dir='moe_fused/cpp'
)

setup(
    name='moe_fused',
    version='0.1.0',
    description='MoE Stage 1+2 Fused Kernel with LDS caching',
    packages=['moe_fused', 'moe_fused.tests'],
    ext_modules=[moe_fused_extension],
    cmdclass={'build_ext': CustomBuildExt},
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0',
        'numpy',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pybind11',
        ],
    },
)

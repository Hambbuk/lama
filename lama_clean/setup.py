#!/usr/bin/env python
"""
LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions
Setup script with automatic CUDA detection
"""

import os
import sys
import subprocess
from setuptools import setup, find_packages


def get_cuda_version():
    """Detect CUDA version from nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    return cuda_version
    except:
        pass
    return None


def get_torch_command():
    """Get appropriate PyTorch installation command based on CUDA version"""
    cuda_version = get_cuda_version()
    
    if cuda_version is None:
        print("CUDA not found. Installing CPU version of PyTorch.")
        return "torch==1.12.1+cpu torchvision==0.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html"
    
    print(f"Found CUDA version: {cuda_version}")
    cuda_major = int(cuda_version.split('.')[0])
    cuda_minor = int(cuda_version.split('.')[1])
    
    # PyTorch 1.12.1 supports CUDA 11.3 and 11.6
    if cuda_major == 11:
        if cuda_minor >= 6:
            return "torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html"
        elif cuda_minor >= 3:
            return "torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html"
    elif cuda_major == 10:
        return "torch==1.12.1+cu102 torchvision==0.13.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html"
    
    # Default to CPU if unsupported CUDA version
    print(f"Unsupported CUDA version {cuda_version}. Installing CPU PyTorch.")
    return "torch==1.12.1+cpu torchvision==0.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html"


# Install PyTorch first
print("Installing PyTorch...")
torch_cmd = get_torch_command()
subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + torch_cmd.split())


# Main dependencies
install_requires = [
    # Core
    'numpy>=1.19.2',
    'opencv-python>=4.5.0',
    'Pillow>=8.0.0',
    'scikit-image>=0.17.2',
    'scipy>=1.5.4',
    'tqdm>=4.60.0',
    
    # Deep Learning (PyTorch installed above)
    'pytorch-lightning>=1.6.0,<2.0.0',
    'kornia>=0.5.0',
    'einops>=0.4.1',
    
    # Configuration
    'hydra-core>=1.1.0',
    'omegaconf>=2.1.0',
    'PyYAML>=5.4.0',
    
    # Data processing
    'albumentations>=0.5.2',
    'webdataset>=0.1.40',
    
    # Training tools
    'tensorboard>=2.7.0',
    'matplotlib>=3.3.4',
    'pandas>=1.1.5',
    
    # ONNX export
    'onnx>=1.10.0',
]

# Conditional dependencies
extras_require = {
    'gpu': [
        'onnxruntime-gpu>=1.10.0' if get_cuda_version() else 'onnxruntime>=1.10.0',
    ],
    'dev': [
        'pytest>=6.2.0',
        'pytest-cov>=2.12.0',
        'black>=21.0',
        'flake8>=3.9.0',
        'isort>=5.9.0',
    ],
    'export': [
        'onnx-simplifier>=0.3.6',
        'onnxconverter-common>=1.8.0',
    ],
}

# Add all extras to a 'full' option
extras_require['full'] = sum(extras_require.values(), [])


setup(
    name='lama-inpainting',
    version='1.0.0',
    author='LaMa Contributors',
    description='Resolution-robust Large Mask Inpainting with Fourier Convolutions',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/your-org/lama',
    packages=find_packages(include=['lama', 'lama.*']),
    python_requires='>=3.7',
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'lama-train=scripts.train:main',
            'lama-predict=scripts.predict:main',
            'lama-export=scripts.export_onnx:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='inpainting, image-inpainting, deep-learning, fourier-convolutions, lama',
)
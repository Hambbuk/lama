#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
from setuptools import setup, find_packages


def get_cuda_version():
    """CUDA 버전 자동 감지"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            # nvidia-smi 출력에서 CUDA 버전 추출
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    return cuda_version
    except:
        pass
    return None


def get_torch_install_command():
    """시스템 환경에 맞는 PyTorch 설치 명령 생성"""
    cuda_version = get_cuda_version()
    
    if cuda_version is None:
        print("CUDA not found. Installing CPU version of PyTorch.")
        return "torch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0"
    
    print(f"Found CUDA version: {cuda_version}")
    
    # CUDA 버전에 따른 PyTorch 설치 명령
    cuda_major = int(cuda_version.split('.')[0])
    cuda_minor = int(cuda_version.split('.')[1])
    
    if cuda_major == 11:
        if cuda_minor >= 3:
            torch_cmd = "torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/cu113/torch_stable.html"
        elif cuda_minor >= 1:
            torch_cmd = "torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html"
        else:
            torch_cmd = "torch==1.10.0+cu110 torchvision==0.11.0+cu110 torchaudio==0.10.0 -f https://download.pytorch.org/whl/cu110/torch_stable.html"
    elif cuda_major == 10:
        torch_cmd = "torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 -f https://download.pytorch.org/whl/cu102/torch_stable.html"
    else:
        print(f"Unsupported CUDA version {cuda_version}. Installing default PyTorch.")
        torch_cmd = "torch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0"
    
    return torch_cmd


# PyTorch 먼저 설치
print("Installing PyTorch...")
torch_cmd = get_torch_install_command()
subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + torch_cmd.split())


# 기본 패키지 목록
install_requires = [
    'numpy>=1.19.0',
    'opencv-python>=4.5.0',
    'Pillow>=8.0.0',
    'scikit-image>=0.17.0',
    'scikit-learn>=0.24.0',
    'tqdm>=4.60.0',
    'PyYAML>=5.4.0',
    'hydra-core>=1.1.0',
    'omegaconf>=2.1.0',
    'pytorch-lightning>=1.5.0',
    'albumentations>=1.0.0',
    'tensorboard>=2.4.0',
    'kornia>=0.5.0',
    'einops>=0.3.0',
    'webdataset>=0.1.40',
    'matplotlib>=3.3.0',
    'pandas>=1.1.0',
    'onnx>=1.10.0',
    'onnxruntime-gpu>=1.10.0' if get_cuda_version() else 'onnxruntime>=1.10.0',
]


setup(
    name='lama-inpainting',
    version='1.0.0',
    description='LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions',
    author='Your Company',
    python_requires='>=3.7',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'lama-train=scripts.train:main',
            'lama-infer=scripts.inference:main',
            'lama-export=scripts.export_onnx:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
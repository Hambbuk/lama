#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lama-inpainting",
    version="1.0.0",
    author="LaMa Team",
    description="LaMa (Large Mask Inpainting) - Clean implementation for training, inference, and ONNX export",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-company/lama-inpainting",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "flake8>=3.8.0",
            "black>=21.0.0",
            "isort>=5.0.0",
        ],
        "onnx": [
            "onnx>=1.10.0",
            "onnxruntime>=1.8.0",
            "onnxsim>=0.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lama-train=train:main",
            "lama-inference=inference:main",
            "lama-export=export_onnx:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
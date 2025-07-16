from setuptools import setup, find_packages

setup(
    name="lama",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch==2.1.1",
        "pytorch-lightning==2.2.0",
        "hydra-core==1.3.2",
        "albumentations==0.5.2",
        "opencv-python-headless==4.9.0.80",
        "fsspec==2024.5.0",
    ],
    python_requires=">=3.8",
)
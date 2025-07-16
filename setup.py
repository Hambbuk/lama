from setuptools import setup, find_packages

setup(
    name="lama-saicinpainting",
    version="0.1.0",
    description="LaMa inpainting (saicinpainting) library",
    packages=find_packages(include=["saicinpainting", "saicinpainting.*"]),
    python_requires=">=3.8",
)
# 🦙 LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions

by Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, 
Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, Victor Lempitsky.

<p align="center" "font-size:30px;">
  🔥🔥🔥
  <br>
  <b>
LaMa generalizes surprisingly well to much higher resolutions (~2k❗️) than it saw during training (256x256), and achieves the excellent performance even in challenging scenarios, e.g. completion of periodic structures.</b>
</p>

[[Project page](https://advimman.github.io/lama-project/)] [[arXiv](https://arxiv.org/abs/2109.07161)] [[Supplementary](https://ashukha.com/projects/lama_21/lama_supmat_2021.pdf)] [[BibTeX](https://senya-ashukha.github.io/projects/lama_21/paper.txt)] [[Casual GAN Papers Summary](https://www.casualganpapers.com/large-masks-fourier-convolutions-inpainting/LaMa-explained.html)] [[ONNX Model](https://huggingface.co/Carve/LaMa-ONNX)]
 
<p align="center">
  <a href="https://colab.research.google.com/drive/15KTEIScUbVZtUP6w2tCDMVpE-b1r9pkZ?usp=drive_link">
  <img src="https://colab.research.google.com/assets/colab-badge.svg"/>
  </a>
      <br>
   Try out in Google Colab
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/senya-ashukha/senya-ashukha.github.io/master/projects/lama_21/ezgif-4-0db51df695a8.gif" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/senya-ashukha/senya-ashukha.github.io/master/projects/lama_21/gif_for_lightning_v1_white.gif" />
</p>

# 🚀 Quick Start (Recommended)

## One-Command Setup

**The easiest way to get started is with our automated setup script:**

```bash
git clone https://github.com/advimman/lama.git
cd lama
./setup_environment.sh
```

This script will automatically:
- 🔍 Detect your system configuration (CPU/GPU)
- 📦 Install the latest Miniconda if needed
- 🚀 Create optimized conda environment with latest packages
- 🎯 Install GPU-accelerated PyTorch if CUDA is available
- ✅ Verify all installations
- 🧪 Run comprehensive tests

## Manual Environment Setup

If you prefer manual setup, we provide two environment files:

### CPU-Only Environment
```bash
conda env create -f conda_env.yml
conda activate lama
```

### GPU-Enabled Environment  
```bash
conda env create -f conda_env_gpu.yml
conda activate lama-gpu
```

# 📦 What's New in This Setup

## 🔄 Updated Dependencies
- **Python 3.10** (latest stable version)
- **PyTorch 2.1.1** with CUDA 11.8 support
- **Latest versions** of all dependencies
- **Automatic CUDA detection** and installation

## 🛡️ Enhanced Features
- **Zero import errors** - all packages are tested and compatible
- **Automatic environment detection** - GPU/CPU setup based on your hardware
- **Modern package versions** - latest stable releases
- **Comprehensive testing** - full validation of installation

## 🎯 Environment Details

### Core ML Stack
- PyTorch 2.1.1 (with CUDA 11.8 if GPU available)
- NumPy 1.24.3
- SciPy 1.11.4
- OpenCV 4.8.1
- Pillow 10.0.1

### Computer Vision & AI
- Albumentations 1.3.1
- Kornia 0.7.0
- PyTorch Lightning 2.1.2
- Hydra 1.3.2
- TensorBoard 2.15.1

### System Requirements
- **OS**: Linux, macOS
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with CUDA 11.8+ (optional)
- **Memory**: 8GB+ RAM recommended

# 🧪 Testing Your Installation

After setup, test your installation:

```bash
python test_installation.py
```

This will verify:
- ✅ All package imports
- ✅ CUDA availability (if GPU present)
- ✅ Tensor operations
- ✅ Image processing capabilities

# 🎨 Usage Examples

## Basic Inpainting

```python
import torch
import numpy as np
from PIL import Image
import cv2

# Load your image and mask
image = Image.open('path/to/image.jpg')
mask = Image.open('path/to/mask.jpg')

# Run inpainting
# ... (existing LaMa code)
```

## Advanced Usage

Check the provided Jupyter notebooks:
- `LaMa_inpainting.ipynb` - Complete inpainting pipeline
- `export_LaMa_to_onnx.ipynb` - ONNX export for deployment

# 🔧 Troubleshooting

## Common Issues

### Import Errors
If you encounter import errors, run:
```bash
conda activate lama  # or lama-gpu
pip install --upgrade pip
python test_installation.py
```

### CUDA Issues
For CUDA-related problems:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Environment Conflicts
To reset your environment:
```bash
conda env remove -n lama
./setup_environment.sh
```

## Getting Help

1. **Check the test script**: `python test_installation.py`
2. **Update packages**: `conda update --all`
3. **Reset environment**: Re-run `./setup_environment.sh`
4. **Check CUDA**: Ensure NVIDIA drivers are installed

# 🌟 Performance Tips

## GPU Optimization
- Use `lama-gpu` environment for GPU acceleration
- Ensure CUDA drivers are up to date
- Monitor GPU memory usage

## CPU Optimization  
- Use `lama` environment for CPU-only inference
- Increase batch size for better CPU utilization
- Consider using threading for multiple images

# LaMa in ONNX!
For now, LaMa (big-lama) can be exported to ONNX format / by [Carve.Photos](https://carve.photos)

🔥 ONNX Model repository on Hugging Face: [Hugging Face](https://huggingface.co/Carve/LaMa-ONNX) \
🚀 HG Space Demo using onnx model: [Hugging Face Spaces](https://huggingface.co/spaces/Carve/LaMa-Demo-ONNX) \
📘 Jupyter notebook to export your own model: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Carve-Photos/lama/blob/main/export_LaMa_to_onnx.ipynb) 

# LaMa development
(Feel free to share your paper by creating an issue)
- https://github.com/geekyutao/Inpaint-Anything --- Inpaint Anything: Segment Anything Meets Image Inpainting
<p align="center">
  <img src="https://raw.githubusercontent.com/geekyutao/Inpaint-Anything/main/example/MainFramework.png" />
</p>

- [Feature Refinement to Improve High Resolution Image Inpainting](https://arxiv.org/abs/2206.13644) / [video](https://www.youtube.com/watch?v=gEukhOheWgE) / code https://github.com/advimman/lama/pull/112 / by Geomagical Labs ([geomagical.com](geomagical.com))
<p align="center">
  <img src="https://raw.githubusercontent.com/senya-ashukha/senya-ashukha.github.io/master/images/FeatureRefinement.png" />
</p>

# Non-official 3rd party apps:
(Feel free to share your app/implementation/demo by creating an issue)

- https://github.com/enesmsahin/simple-lama-inpainting - a simple pip package for LaMa inpainting.
- https://github.com/mallman/CoreMLaMa - Apple's Core ML model format
- [https://cleanup.pictures](https://cleanup.pictures/) - a simple interactive object removal tool by [@cyrildiagne](https://twitter.com/cyrildiagne)
    - [lama-cleaner](https://github.com/Sanster/lama-cleaner) by [@Sanster](https://github.com/Sanster/lama-cleaner) is a self-host version of [https://cleanup.pictures](https://cleanup.pictures/)
- Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/lama) by [@AK391](https://github.com/AK391)
- Telegram bot [@MagicEraserBot](https://t.me/MagicEraserBot) by [@Moldoteck](https://github.com/Moldoteck), [code](https://github.com/Moldoteck/MagicEraser)
- [Auto-LaMa](https://github.com/andy971022/auto-lama) = DE:TR object detection + LaMa inpainting by [@andy971022](https://github.com/andy971022)
- [LAMA-Magic-Eraser-Local](https://github.com/zhaoyun0071/LAMA-Magic-Eraser-Local) = a standalone inpainting application built with PyQt5 by [@zhaoyun0071](https://github.com/zhaoyun0071)
- [Hama](https://www.hama.app/) - object removal with a smart brush which simplifies mask drawing.
- [ModelScope](https://www.modelscope.cn/models/damo/cv_fft_inpainting_lama/summary) = the largest Model Community in Chinese by  [@chenbinghui1](https://github.com/chenbinghui1).
- [LaMa with MaskDINO](https://github.com/qwopqwop200/lama-with-maskdino) = MaskDINO object detection + LaMa inpainting with refinement by [@qwopqwop200](https://github.com/qwopqwop200).
- [CoreMLaMa](https://github.com/mallman/CoreMLaMa) - a script to convert Lama Cleaner's port of LaMa to Apple's Core ML model format.

# 📜 Legacy Environment Setup (Not Recommended)

<details>
<summary>Click to expand legacy setup instructions</summary>

## Python virtualenv:

```bash
virtualenv inpenv --python=/usr/bin/python3
source inpenv/bin/activate
pip install torch==1.8.0 torchvision==0.9.0

cd lama
pip install -r requirements.txt 
```

## Old Conda Setup
    
```bash
# Install conda for Linux, for other OS download miniconda at https://docs.conda.io/en/latest/miniconda.html
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
$HOME/miniconda/bin/conda init bash

cd lama
conda env create -f conda_env.yml
conda activate lama
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
```

</details>

# 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

# 🙏 Acknowledgments

- Original LaMa paper and implementation
- All contributors to the project
- Third-party applications and integrations

# 📞 Support

For issues and questions:
1. Check the [troubleshooting section](#🔧-troubleshooting)
2. Run `python test_installation.py` to diagnose issues
3. Create an issue on GitHub with your environment details

---

**Happy Inpainting! 🎨**

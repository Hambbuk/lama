#!/usr/bin/env python3
"""
Example usage of LaMa inpainting model
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2

def create_sample_mask(image_size=256, mask_type='rectangle'):
    """Create a sample mask for testing"""
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    
    if mask_type == 'rectangle':
        # Create rectangular mask
        h_start = image_size // 4
        h_end = 3 * image_size // 4
        w_start = image_size // 4
        w_end = 3 * image_size // 4
        mask[h_start:h_end, w_start:w_end] = 255
        
    elif mask_type == 'circle':
        # Create circular mask
        center = (image_size // 2, image_size // 2)
        radius = image_size // 4
        cv2.circle(mask, center, radius, 255, -1)
        
    elif mask_type == 'irregular':
        # Create irregular mask using random lines
        img = Image.fromarray(mask)
        draw = ImageDraw.Draw(img)
        
        # Draw random lines
        for _ in range(10):
            x1 = np.random.randint(0, image_size)
            y1 = np.random.randint(0, image_size)
            x2 = np.random.randint(0, image_size)
            y2 = np.random.randint(0, image_size)
            width = np.random.randint(10, 30)
            draw.line([(x1, y1), (x2, y2)], fill=255, width=width)
            
        mask = np.array(img)
    
    return mask

def quick_start_example():
    """Quick start example"""
    print("🚀 LaMa Inpainting Quick Start Example")
    print("=" * 50)
    
    # 0. Environment setup
    print("0. Environment setup (if you encounter WandB errors):")
    print("   python startup.py")
    print("   # or simply: pip install wandb")
    print()
    
    # 1. Training command
    print("1. Training your model:")
    print("   python train.py data_dir=/path/to/your/dataset")
    print()
    
    # 2. Inference command
    print("2. Running inference:")
    print("   python inference.py \\")
    print("     --checkpoint checkpoints/best_model.ckpt \\")
    print("     --image input_image.jpg \\")
    print("     --mask input_mask.png \\")
    print("     --output result.jpg")
    print()
    
    # 3. Configuration examples
    print("3. Configuration examples:")
    print("   # Use different model")
    print("   python train.py model=baseline_resnet")
    print()
    print("   # Adjust batch size and epochs")
    print("   python train.py data.batch_size=8 trainer.max_epochs=200")
    print()
    print("   # Use different mask types")
    print("   python train.py data.mask_config.types=['random','irregular','outpainting']")
    print()
    
    # 4. Hyperparameter sweep
    print("4. Hyperparameter sweep:")
    print("   python train.py -m data.batch_size=4,8,16 model.optimizer_config.generator_lr=1e-4,5e-4")
    print()

def create_sample_data():
    """Create sample data for testing"""
    print("📁 Creating sample data...")
    
    # Create directories
    os.makedirs('sample_data', exist_ok=True)
    
    # Create a sample image (random noise for demonstration)
    image_size = 256
    sample_image = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
    
    # Add some patterns to make it more interesting
    # Horizontal stripes
    for i in range(0, image_size, 20):
        sample_image[i:i+10, :] = [100, 150, 200]
    
    # Vertical stripes
    for i in range(0, image_size, 30):
        sample_image[:, i:i+15] = [200, 100, 150]
    
    # Save sample image
    cv2.imwrite('sample_data/sample_image.jpg', 
                cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR))
    
    # Create and save different types of masks
    for mask_type in ['rectangle', 'circle', 'irregular']:
        mask = create_sample_mask(image_size, mask_type)
        cv2.imwrite(f'sample_data/mask_{mask_type}.png', mask)
    
    print("✅ Sample data created in 'sample_data/' directory")
    print("   - sample_image.jpg: Test image")
    print("   - mask_rectangle.png: Rectangular mask")
    print("   - mask_circle.png: Circular mask") 
    print("   - mask_irregular.png: Irregular mask")
    print()

def show_project_structure():
    """Show the project structure"""
    print("📂 Project Structure:")
    print("""
├── configs/                 # Hydra configuration files
│   ├── config.yaml         # Main config
│   ├── model/              # Model configs
│   │   └── lama_ffc.yaml   # LaMa FFC model
│   ├── data/               # Data configs  
│   │   └── places365.yaml  # Places365 dataset
│   └── logger/             # Logger configs
│       └── tensorboard.yaml # TensorBoard logger
├── src/                    # Source code
│   ├── models/             # Model implementations
│   ├── losses/             # Loss functions
│   ├── data/               # Data loading
│   └── lightning_module.py # Lightning module
├── train.py                # Training script
├── inference.py            # Inference script
└── requirements.txt        # Dependencies
""")

def main():
    """Main function"""
    print("🦙 LaMa Inpainting - Modern PyTorch Lightning Implementation")
    print("=" * 60)
    print()
    
    # Show project structure
    show_project_structure()
    print()
    
    # Quick start guide
    quick_start_example()
    
    # Create sample data
    response = input("Would you like to create sample data for testing? (y/n): ")
    if response.lower() in ['y', 'yes']:
        create_sample_data()
    
    print("🎉 Ready to start! Check README_NEW.md for detailed instructions.")

if __name__ == '__main__':
    main()
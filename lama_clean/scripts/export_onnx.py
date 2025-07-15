#!/usr/bin/env python3
"""
Export LaMa model to ONNX format

Usage:
    python export_onnx.py --model path/to/checkpoint --output model.onnx
    python export_onnx.py --model path/to/checkpoint --output model.onnx --simplify --fp16
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lama.training.trainers import load_checkpoint


class JITWrapper(nn.Module):
    """Wrapper for ONNX/JIT export"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, mask):
        batch = {
            "image": image,
            "mask": mask
        }
        out = self.model(batch)
        return out["inpainted"]


def parse_args():
    parser = argparse.ArgumentParser(description='Export LaMa model to ONNX')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint directory')
    parser.add_argument('--checkpoint', type=str, default='last.ckpt',
                        help='Checkpoint filename')
    parser.add_argument('--output', type=str, required=True,
                        help='Output ONNX file path')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--opset-version', type=int, default=11,
                        help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model')
    parser.add_argument('--fp16', action='store_true',
                        help='Export in FP16')
    parser.add_argument('--dynamic-axes', action='store_true',
                        help='Enable dynamic batch size')
    parser.add_argument('--verify', action='store_true',
                        help='Verify exported model')
    return parser.parse_args()


def export_onnx(model, args):
    """Export model to ONNX"""
    # Create wrapper
    jit_model = JITWrapper(model)
    jit_model.eval()
    
    # Create dummy inputs
    dummy_image = torch.randn(1, 3, args.image_size, args.image_size)
    dummy_mask = torch.randn(1, 1, args.image_size, args.image_size)
    
    # Move to CUDA if available
    if torch.cuda.is_available() and not args.fp16:
        jit_model = jit_model.cuda()
        dummy_image = dummy_image.cuda()
        dummy_mask = dummy_mask.cuda()
    
    # Export to ONNX
    print(f"Exporting to ONNX with image size {args.image_size}x{args.image_size}")
    
    dynamic_axes = None
    if args.dynamic_axes:
        dynamic_axes = {
            'image': {0: 'batch_size'},
            'mask': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    with torch.no_grad():
        torch.onnx.export(
            jit_model,
            (dummy_image, dummy_mask),
            args.output,
            export_params=True,
            opset_version=args.opset_version,
            do_constant_folding=True,
            input_names=['image', 'mask'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
    
    print(f"Model exported to: {args.output}")
    
    # Simplify if requested
    if args.simplify:
        try:
            import onnx
            import onnxsim
            
            print("Simplifying ONNX model...")
            model_onnx = onnx.load(args.output)
            model_simp, check = onnxsim.simplify(model_onnx)
            
            if check:
                onnx.save(model_simp, args.output)
                print("Model simplified successfully")
            else:
                print("Simplification failed, keeping original model")
                
        except ImportError:
            print("Warning: onnx-simplifier not installed. Skipping simplification.")
    
    # Convert to FP16 if requested
    if args.fp16:
        try:
            import onnx
            from onnxconverter_common import float16
            
            print("Converting to FP16...")
            model_onnx = onnx.load(args.output)
            model_fp16 = float16.convert_float_to_float16(model_onnx)
            
            output_fp16 = args.output.replace('.onnx', '_fp16.onnx')
            onnx.save(model_fp16, output_fp16)
            print(f"FP16 model saved to: {output_fp16}")
            
        except ImportError:
            print("Warning: onnxconverter-common not installed. Skipping FP16 conversion.")
    
    # Verify if requested
    if args.verify:
        verify_onnx(args.output, dummy_image, dummy_mask, jit_model)


def verify_onnx(onnx_path, dummy_image, dummy_mask, pytorch_model):
    """Verify ONNX model output matches PyTorch"""
    try:
        import onnxruntime as ort
        import numpy as np
        
        print("\nVerifying ONNX model...")
        
        # Create ONNX runtime session
        ort_session = ort.InferenceSession(onnx_path)
        
        # Get PyTorch output
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_image, dummy_mask)
        
        # Get ONNX output
        ort_inputs = {
            'image': dummy_image.cpu().numpy(),
            'mask': dummy_mask.cpu().numpy()
        }
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        # Compare outputs
        pytorch_output_np = pytorch_output.cpu().numpy()
        diff = np.abs(pytorch_output_np - ort_output).mean()
        
        print(f"Mean absolute difference: {diff:.6f}")
        
        if diff < 1e-5:
            print("✓ Verification passed!")
        else:
            print("⚠ Warning: Large difference between PyTorch and ONNX outputs")
            
    except ImportError:
        print("Warning: onnxruntime not installed. Skipping verification.")


def main():
    args = parse_args()
    
    # Load config
    config_path = Path(args.model) / 'config.yaml'
    with open(config_path, 'r') as f:
        config = OmegaConf.create(yaml.safe_load(f))
    
    # Set to evaluation mode
    config.training_model.predict_only = True
    config.visualizer.kind = 'noop'
    
    # Load model
    checkpoint_path = Path(args.model) / 'checkpoints' / args.checkpoint
    print(f"Loading model from: {checkpoint_path}")
    
    model = load_checkpoint(config, str(checkpoint_path), strict=False, map_location='cpu')
    model.eval()
    model.freeze()
    
    # Export to ONNX
    export_onnx(model, args)
    
    # Also export as TorchScript for comparison
    if args.output.endswith('.onnx'):
        jit_path = args.output.replace('.onnx', '.pt')
        print(f"\nAlso exporting TorchScript to: {jit_path}")
        
        jit_model = JITWrapper(model)
        jit_model.eval()
        
        if torch.cuda.is_available():
            jit_model = jit_model.cuda()
            dummy_image = torch.randn(1, 3, args.image_size, args.image_size).cuda()
            dummy_mask = torch.randn(1, 1, args.image_size, args.image_size).cuda()
        else:
            dummy_image = torch.randn(1, 3, args.image_size, args.image_size)
            dummy_mask = torch.randn(1, 1, args.image_size, args.image_size)
        
        traced = torch.jit.trace(jit_model, (dummy_image, dummy_mask), strict=False)
        traced.save(jit_path)
        print("TorchScript model saved")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
LaMa Model ONNX Export Script

사용법:
    python export_onnx.py --checkpoint model.ckpt --output model.onnx
"""

import argparse
import sys
from pathlib import Path
import logging

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from omegaconf import OmegaConf

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.models.lama import LamaInpaintingModel
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Export LaMa model to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Output ONNX file path')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: use checkpoint config)')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Input image size (default: 512)')
    parser.add_argument('--opset-version', type=int, default=11,
                        help='ONNX opset version (default: 11)')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model')
    parser.add_argument('--dynamic-axes', action='store_true',
                        help='Enable dynamic axes for batch size')
    parser.add_argument('--fp16', action='store_true',
                        help='Export model in FP16')
    parser.add_argument('--verify', action='store_true',
                        help='Verify exported model')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    return parser.parse_args()


class ONNXExporter:
    def __init__(self, checkpoint_path: str, config_path: str = None):
        """
        ONNX 익스포터 초기화
        
        Args:
            checkpoint_path: 체크포인트 경로
            config_path: 설정 파일 경로
        """
        self.checkpoint_path = checkpoint_path
        
        # 설정 로드
        if config_path is None:
            ckpt_dir = Path(checkpoint_path).parent.parent
            config_path = ckpt_dir / 'config.yaml'
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
        
        self.config = OmegaConf.load(config_path)
        self.logger = setup_logger('onnx_export')
        
    def export(self, output_path: str, image_size: int = 512,
               opset_version: int = 11, simplify: bool = False,
               dynamic_axes: bool = False, fp16: bool = False,
               verify: bool = False, verbose: bool = False):
        """
        모델을 ONNX로 변환
        
        Args:
            output_path: 출력 ONNX 파일 경로
            image_size: 입력 이미지 크기
            opset_version: ONNX opset 버전
            simplify: 모델 단순화 여부
            dynamic_axes: 동적 축 사용 여부
            fp16: FP16 변환 여부
            verify: 검증 수행 여부
            verbose: 상세 출력
        """
        self.logger.info("Loading PyTorch model...")
        model = self._load_pytorch_model()
        
        # 더미 입력 생성
        dummy_image = torch.randn(1, 3, image_size, image_size)
        dummy_mask = torch.randn(1, 1, image_size, image_size)
        
        # 모델을 ONNX로 변환
        self.logger.info("Exporting to ONNX...")
        self._export_to_onnx(
            model, dummy_image, dummy_mask, output_path,
            opset_version, dynamic_axes, verbose
        )
        
        # 모델 단순화
        if simplify:
            self.logger.info("Simplifying ONNX model...")
            self._simplify_model(output_path)
        
        # FP16 변환
        if fp16:
            self.logger.info("Converting to FP16...")
            self._convert_to_fp16(output_path)
        
        # 검증
        if verify:
            self.logger.info("Verifying ONNX model...")
            self._verify_model(output_path, model, image_size)
        
        self.logger.info(f"ONNX model exported successfully to: {output_path}")
        
        # 모델 정보 출력
        self._print_model_info(output_path)
    
    def _load_pytorch_model(self) -> torch.nn.Module:
        """PyTorch 모델 로드"""
        # 모델 초기화
        model = LamaInpaintingModel(self.config)
        
        # 체크포인트 로드
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # ONNX 호환 래퍼 생성
        class ONNXWrapper(torch.nn.Module):
            def __init__(self, lama_model):
                super().__init__()
                self.model = lama_model.generator
            
            def forward(self, image, mask):
                # 마스크 이진화
                mask = (mask > 0.5).float()
                
                # 마스크된 이미지 생성
                masked_image = image * (1 - mask)
                
                # 모델 입력 준비
                x = torch.cat([masked_image, mask], dim=1)
                
                # 모델 실행
                output = self.model(x)
                
                # 인페인팅 결과 생성
                inpainted = masked_image + output * mask
                
                return inpainted
        
        return ONNXWrapper(model)
    
    def _export_to_onnx(self, model: torch.nn.Module, 
                        dummy_image: torch.Tensor, dummy_mask: torch.Tensor,
                        output_path: str, opset_version: int,
                        dynamic_axes: bool, verbose: bool):
        """ONNX로 변환"""
        # 동적 축 설정
        if dynamic_axes:
            dynamic_axes_dict = {
                'image': {0: 'batch_size'},
                'mask': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        else:
            dynamic_axes_dict = None
        
        # ONNX로 변환
        torch.onnx.export(
            model,
            (dummy_image, dummy_mask),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['image', 'mask'],
            output_names=['output'],
            dynamic_axes=dynamic_axes_dict,
            verbose=verbose
        )
    
    def _simplify_model(self, model_path: str):
        """ONNX 모델 단순화"""
        try:
            import onnxsim
            
            model = onnx.load(model_path)
            model_simplified, check = onnxsim.simplify(model)
            
            if check:
                onnx.save(model_simplified, model_path)
                self.logger.info("Model simplified successfully")
            else:
                self.logger.warning("Model simplification failed")
                
        except ImportError:
            self.logger.warning("onnx-simplifier not installed. Skipping simplification.")
    
    def _convert_to_fp16(self, model_path: str):
        """FP16으로 변환"""
        from onnxconverter_common import float16
        
        model = onnx.load(model_path)
        model_fp16 = float16.convert_float_to_float16(model)
        
        # FP16 모델 저장
        fp16_path = model_path.replace('.onnx', '_fp16.onnx')
        onnx.save(model_fp16, fp16_path)
        self.logger.info(f"FP16 model saved to: {fp16_path}")
    
    def _verify_model(self, model_path: str, pytorch_model: torch.nn.Module,
                      image_size: int):
        """ONNX 모델 검증"""
        # ONNX 런타임 세션 생성
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        
        # 테스트 입력 생성
        test_image = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
        test_mask = np.random.randint(0, 2, (1, 1, image_size, image_size)).astype(np.float32)
        
        # PyTorch 추론
        with torch.no_grad():
            pt_image = torch.from_numpy(test_image)
            pt_mask = torch.from_numpy(test_mask)
            pt_output = pytorch_model(pt_image, pt_mask).numpy()
        
        # ONNX 추론
        onnx_output = session.run(None, {
            'image': test_image,
            'mask': test_mask
        })[0]
        
        # 결과 비교
        diff = np.abs(pt_output - onnx_output).mean()
        self.logger.info(f"Mean absolute difference: {diff:.6f}")
        
        if diff < 1e-5:
            self.logger.info("✓ Model verification passed")
        else:
            self.logger.warning(f"⚠ Model verification failed. Difference: {diff}")
    
    def _print_model_info(self, model_path: str):
        """모델 정보 출력"""
        model = onnx.load(model_path)
        
        # 모델 크기
        model_size = Path(model_path).stat().st_size / (1024 * 1024)
        self.logger.info(f"Model size: {model_size:.2f} MB")
        
        # 입출력 정보
        self.logger.info("\nInput shapes:")
        for input in model.graph.input:
            shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
            self.logger.info(f"  {input.name}: {shape}")
        
        self.logger.info("\nOutput shapes:")
        for output in model.graph.output:
            shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            self.logger.info(f"  {output.name}: {shape}")


def main():
    args = parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 익스포터 생성
    exporter = ONNXExporter(args.checkpoint, args.config)
    
    # ONNX로 변환
    exporter.export(
        args.output,
        image_size=args.image_size,
        opset_version=args.opset_version,
        simplify=args.simplify,
        dynamic_axes=args.dynamic_axes,
        fp16=args.fp16,
        verify=args.verify,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
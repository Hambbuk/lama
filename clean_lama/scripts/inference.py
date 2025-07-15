#!/usr/bin/env python3
"""
LaMa Inpainting Model Inference Script

사용법:
    python inference.py --checkpoint model.ckpt --input image.jpg --mask mask.jpg --output result.jpg
    python inference.py --checkpoint model.ckpt --input-dir ./images --mask-dir ./masks --output-dir ./results
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Union, List, Tuple

import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.models.lama import LamaInpaintingModel
from src.utils.image import load_image, save_image, resize_image, pad_image
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='LaMa Inpainting Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: use checkpoint config)')
    
    # 단일 이미지 처리
    parser.add_argument('--input', type=str, default=None,
                        help='Input image path')
    parser.add_argument('--mask', type=str, default=None,
                        help='Mask image path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path')
    
    # 배치 처리
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Input images directory')
    parser.add_argument('--mask-dir', type=str, default=None,
                        help='Mask images directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    
    # 처리 옵션
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for processing')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Image size for processing (default: 512)')
    parser.add_argument('--refinement', action='store_true',
                        help='Use refinement for better quality')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 for faster inference')
    
    return parser.parse_args()


class LamaInference:
    def __init__(self, checkpoint_path: str, config_path: str = None, 
                 device: str = 'cuda', fp16: bool = False):
        """
        LaMa 추론 클래스
        
        Args:
            checkpoint_path: 체크포인트 경로
            config_path: 설정 파일 경로 (없으면 체크포인트에서 로드)
            device: 사용할 디바이스
            fp16: FP16 사용 여부
        """
        self.device = torch.device(device)
        self.fp16 = fp16
        
        # 설정 로드
        if config_path is None:
            # 체크포인트 디렉토리에서 config.yaml 찾기
            ckpt_dir = Path(checkpoint_path).parent.parent
            config_path = ckpt_dir / 'config.yaml'
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
        
        self.config = OmegaConf.load(config_path)
        
        # 모델 로드
        self.model = self._load_model(checkpoint_path)
        
    def _load_model(self, checkpoint_path: str) -> LamaInpaintingModel:
        """모델 로드"""
        # 모델 초기화
        model = LamaInpaintingModel(self.config)
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        if self.fp16 and self.device.type == 'cuda':
            model = model.half()
        
        return model
    
    @torch.no_grad()
    def inpaint(self, image: np.ndarray, mask: np.ndarray, 
                image_size: int = 512, refinement: bool = False) -> np.ndarray:
        """
        단일 이미지 인페인팅
        
        Args:
            image: 입력 이미지 (H, W, 3), RGB, 0-255
            mask: 마스크 이미지 (H, W), 0-255
            image_size: 처리할 이미지 크기
            refinement: 정제 사용 여부
        
        Returns:
            인페인팅된 이미지 (H, W, 3), RGB, 0-255
        """
        orig_height, orig_width = image.shape[:2]
        
        # 전처리
        image_tensor, mask_tensor = self._preprocess(image, mask, image_size)
        
        # 추론
        batch = {
            'image': image_tensor.unsqueeze(0).to(self.device),
            'mask': mask_tensor.unsqueeze(0).to(self.device)
        }
        
        if self.fp16 and self.device.type == 'cuda':
            batch['image'] = batch['image'].half()
            batch['mask'] = batch['mask'].half()
        
        output = self.model(batch)
        result = output['inpainted'][0]
        
        # 후처리
        result = self._postprocess(result, orig_height, orig_width)
        
        # Refinement
        if refinement:
            result = self._refine(image, mask, result)
        
        return result
    
    def _preprocess(self, image: np.ndarray, mask: np.ndarray, 
                    image_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """이미지 전처리"""
        # 리사이즈
        image = resize_image(image, image_size)
        mask = resize_image(mask, image_size)
        
        # 정규화 및 텐서 변환
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        mask_tensor = torch.from_numpy(mask).float() / 255.0
        if len(mask_tensor.shape) == 2:
            mask_tensor = mask_tensor.unsqueeze(0)  # HW -> 1HW
        else:
            mask_tensor = mask_tensor.permute(2, 0, 1)[0:1]  # HWC -> 1HW
        
        # 마스크 이진화
        mask_tensor = (mask_tensor > 0.5).float()
        
        return image_tensor, mask_tensor
    
    def _postprocess(self, result: torch.Tensor, 
                     orig_height: int, orig_width: int) -> np.ndarray:
        """결과 후처리"""
        # 텐서를 numpy로 변환
        result = result.cpu().float().permute(1, 2, 0).numpy()  # CHW -> HWC
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        # 원본 크기로 리사이즈
        result = cv2.resize(result, (orig_width, orig_height), 
                            interpolation=cv2.INTER_LANCZOS4)
        
        return result
    
    def _refine(self, original: np.ndarray, mask: np.ndarray, 
                inpainted: np.ndarray) -> np.ndarray:
        """결과 정제"""
        # 마스크 부드럽게
        mask_float = mask.astype(np.float32) / 255.0
        if len(mask_float.shape) == 3:
            mask_float = mask_float[:, :, 0]
        
        # 가우시안 블러로 경계 부드럽게
        mask_blur = cv2.GaussianBlur(mask_float, (21, 21), 11)
        mask_blur = np.stack([mask_blur] * 3, axis=2)
        
        # 블렌딩
        result = original * (1 - mask_blur) + inpainted * mask_blur
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def process_batch(self, image_paths: List[str], mask_paths: List[str],
                      output_dir: str, image_size: int = 512, 
                      refinement: bool = False):
        """배치 처리"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path, mask_path in tqdm(zip(image_paths, mask_paths), 
                                         total=len(image_paths),
                                         desc="Processing images"):
            # 이미지 로드
            image = load_image(img_path)
            mask = load_image(mask_path, grayscale=True)
            
            # 인페인팅
            result = self.inpaint(image, mask, image_size, refinement)
            
            # 저장
            output_path = output_dir / Path(img_path).name
            save_image(result, str(output_path))


def main():
    args = parse_args()
    
    # 로거 설정
    logger = setup_logger('inference')
    
    # 모델 로드
    logger.info(f"Loading model from: {args.checkpoint}")
    inpainter = LamaInference(
        args.checkpoint, 
        args.config,
        args.device,
        args.fp16
    )
    
    # 단일 이미지 처리
    if args.input and args.mask and args.output:
        logger.info(f"Processing single image: {args.input}")
        
        image = load_image(args.input)
        mask = load_image(args.mask, grayscale=True)
        
        result = inpainter.inpaint(
            image, mask, 
            args.image_size, 
            args.refinement
        )
        
        save_image(result, args.output)
        logger.info(f"Result saved to: {args.output}")
    
    # 배치 처리
    elif args.input_dir and args.mask_dir and args.output_dir:
        logger.info(f"Processing batch from: {args.input_dir}")
        
        # 이미지 파일 목록 가져오기
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        mask_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(args.input_dir).glob(f'*{ext}'))
            image_paths.extend(Path(args.input_dir).glob(f'*{ext.upper()}'))
        
        # 마스크 파일 매칭
        for img_path in sorted(image_paths):
            mask_name = img_path.stem + img_path.suffix
            mask_path = Path(args.mask_dir) / mask_name
            
            if mask_path.exists():
                mask_paths.append(str(mask_path))
            else:
                logger.warning(f"Mask not found for: {img_path}")
                image_paths.remove(img_path)
        
        image_paths = [str(p) for p in image_paths]
        
        logger.info(f"Found {len(image_paths)} image-mask pairs")
        
        # 배치 처리
        inpainter.process_batch(
            image_paths, mask_paths,
            args.output_dir,
            args.image_size,
            args.refinement
        )
        
        logger.info(f"Results saved to: {args.output_dir}")
    
    else:
        logger.error("Please provide either single image arguments or batch processing arguments")
        sys.exit(1)


if __name__ == '__main__':
    main()
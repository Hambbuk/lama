import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional
from pathlib import Path


def load_image(path: Union[str, Path], grayscale: bool = False) -> np.ndarray:
    """
    Load image from file
    
    Args:
        path: Image file path
        grayscale: Whether to load as grayscale
    
    Returns:
        Image array (H, W, 3) or (H, W) if grayscale, RGB order, 0-255
    """
    if grayscale:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    
    return image


def save_image(image: np.ndarray, path: Union[str, Path]):
    """
    Save image to file
    
    Args:
        image: Image array (H, W, 3) or (H, W), RGB order, 0-255
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if len(image.shape) == 3:
        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(str(path), image)


def resize_image(image: np.ndarray, size: Union[int, Tuple[int, int]],
                 interpolation: str = 'lanczos') -> np.ndarray:
    """
    Resize image
    
    Args:
        image: Input image
        size: Target size (single int for square, or (width, height))
        interpolation: Interpolation method
    
    Returns:
        Resized image
    """
    if isinstance(size, int):
        size = (size, size)
    
    interpolation_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    interp = interpolation_map.get(interpolation, cv2.INTER_LANCZOS4)
    
    return cv2.resize(image, size, interpolation=interp)


def pad_image(image: np.ndarray, pad_mod: int = 8, 
              mode: str = 'reflect') -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Pad image to be divisible by pad_mod
    
    Args:
        image: Input image
        pad_mod: Padding modulo
        mode: Padding mode
    
    Returns:
        Padded image and padding values (top, bottom, left, right)
    """
    h, w = image.shape[:2]
    
    # Calculate padding
    pad_h = (pad_mod - h % pad_mod) % pad_mod
    pad_w = (pad_mod - w % pad_mod) % pad_mod
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Apply padding
    if len(image.shape) == 3:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
    
    padded = np.pad(image, padding, mode=mode)
    
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def unpad_image(image: np.ndarray, padding: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Remove padding from image
    
    Args:
        image: Padded image
        padding: Padding values (top, bottom, left, right)
    
    Returns:
        Unpadded image
    """
    pad_top, pad_bottom, pad_left, pad_right = padding
    h, w = image.shape[:2]
    
    return image[pad_top:h-pad_bottom, pad_left:w-pad_right]


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [-1, 1] range"""
    return (image.astype(np.float32) / 127.5) - 1.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Denormalize image from [-1, 1] to [0, 255] range"""
    return ((image + 1.0) * 127.5).astype(np.uint8)
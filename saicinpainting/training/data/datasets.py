import glob
import logging
import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import webdataset
from omegaconf import open_dict, OmegaConf
from skimage.feature import canny
from skimage.transform import rescale, resize
from torch.utils.data import Dataset, IterableDataset, DataLoader, DistributedSampler, ConcatDataset

from saicinpainting.evaluation.data import InpaintingDataset as InpaintingEvaluationDataset, \
    OurInpaintingDataset as OurInpaintingEvaluationDataset, ceil_modulo, InpaintingEvalOnlineDataset
from saicinpainting.training.data.aug import IAAAffine2, IAAPerspective2
from saicinpainting.training.data.masks import get_mask_generator

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# ✋ HandInpaintingTrainDataset — skips inpainting on hand regions
# -----------------------------------------------------------------------------


class HandInpaintingTrainDataset(Dataset):
    """Training dataset that excludes hand regions specified by binary masks.

    Each sample consists of an RGB thumbnail and a *single-channel* hand mask
    (same spatial resolution).  A random inpainting mask is generated and then
    multiplied by the inverse hand-mask so that the network never attempts to
    inpaint hands.

    Parameters
    ----------
    indir : str
        Directory with RGB images (recursive glob on **/*.jpg).
    hand_mask_dir : str or None, optional
        Directory that keeps hand-mask PNGs with identical filenames. If
        omitted, the code looks for a sibling directory named
        ``<indir_parent>/train_hand_mask``.
    mask_generator : callable
        Function that produces random binary inpainting masks.
    transform : albumentations.Compose
        Data augmentation pipeline applied to both image & hand mask.
    mask_inflation : int or None, optional
        Number of pixels to dilate the hand mask before merging.  Use to add a
        safety margin so the painted area never touches hands.
    """

    def __init__(self, indir, mask_generator, transform,
                 hand_mask_dir=None, mask_inflation=None):
        self.indir = indir
        self.mask_generator = mask_generator
        self.transform = transform
        self.mask_inflation = mask_inflation

        # Resolve hand-mask directory
        if hand_mask_dir is None:
            hand_mask_dir = os.path.join(os.path.dirname(indir), 'train_hand_mask')

        assert os.path.isdir(hand_mask_dir), \
            f"Hand-mask directory not found: {hand_mask_dir}"

        # Build paired (img, mask) list
        self.image_mask_pairs = []
        for img_path in glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True):
            base = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(hand_mask_dir, f"{base}.png")
            if not os.path.exists(mask_path):
                # Skip if corresponding mask is missing
                continue
            self.image_mask_pairs.append((img_path, mask_path))

        LOGGER.info(f"HandInpaintingTrainDataset: {indir} → {len(self.image_mask_pairs)} paired samples")

        # Counter for mask generator
        self.iter_i = 0

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_path, hand_mask_path = self.image_mask_pairs[idx]

        # RGB image ------------------------------------------------------------------
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Hand mask (grayscale) -------------------------------------------------------
        hand_mask = cv2.imread(hand_mask_path, cv2.IMREAD_GRAYSCALE)

        # Optional dilation to grow the hand region
        if self.mask_inflation is not None and self.mask_inflation > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                               (self.mask_inflation, self.mask_inflation))
            hand_mask = cv2.dilate(hand_mask, kernel)

        # Apply shared augmentations
        transformed = self.transform(image=img, mask=hand_mask)
        img = transformed['image']
        hand_mask = transformed['mask']

        # CHW & float32 ---------------------------------------------------------------
        img = np.transpose(img, (2, 0, 1))           # C, H, W
        hand_mask = hand_mask.astype(np.float32) / 255.0
        hand_mask = np.expand_dims(hand_mask, axis=0)  # 1, H, W

        # Generate random inpainting mask and zero-out hand region
        rand_mask = self.mask_generator(img, iter_i=self.iter_i)
        self.iter_i += 1
        final_mask = rand_mask * (1.0 - hand_mask)

        return {
            'image': img,
            'mask': final_mask,
        }


class InpaintingTrainDataset(Dataset):
    def __init__(self, indir, mask_generator, transform):
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        self.mask_generator = mask_generator
        self.transform = transform
        self.iter_i = 0

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1))
        # TODO: maybe generate mask before augmentations? slower, but better for segmentation-based masks
        mask = self.mask_generator(img, iter_i=self.iter_i)
        self.iter_i += 1
        return dict(image=img,
                    mask=mask)


class InpaintingTrainWebDataset(IterableDataset):
    def __init__(self, indir, mask_generator, transform, shuffle_buffer=200):
        self.impl = webdataset.Dataset(indir).shuffle(shuffle_buffer).decode('rgb').to_tuple('jpg')
        self.mask_generator = mask_generator
        self.transform = transform

    def __iter__(self):
        for iter_i, (img,) in enumerate(self.impl):
            img = np.clip(img * 255, 0, 255).astype('uint8')
            img = self.transform(image=img)['image']
            img = np.transpose(img, (2, 0, 1))
            mask = self.mask_generator(img, iter_i=iter_i)
            yield dict(image=img,
                       mask=mask)


class ImgSegmentationDataset(Dataset):
    def __init__(self, indir, mask_generator, transform, out_size, segm_indir, semantic_seg_n_classes):
        self.indir = indir
        self.segm_indir = segm_indir
        self.mask_generator = mask_generator
        self.transform = transform
        self.out_size = out_size
        self.semantic_seg_n_classes = semantic_seg_n_classes
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.out_size, self.out_size))
        img = self.transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1))
        mask = self.mask_generator(img)
        segm, segm_classes= self.load_semantic_segm(path)
        result = dict(image=img,
                      mask=mask,
                      segm=segm,
                      segm_classes=segm_classes)
        return result

    def load_semantic_segm(self, img_path):
        segm_path = img_path.replace(self.indir, self.segm_indir).replace(".jpg", ".png")
        mask = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.out_size, self.out_size))
        tensor = torch.from_numpy(np.clip(mask.astype(int)-1, 0, None))
        ohe = F.one_hot(tensor.long(), num_classes=self.semantic_seg_n_classes) # w x h x n_classes
        return ohe.permute(2, 0, 1).float(), tensor.unsqueeze(0)


def get_transforms(transform_variant, out_size):
    if transform_variant == 'default':
        transform = A.Compose([
            A.RandomScale(scale_limit=0.2),  # +/- 20%
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.7, 1.3),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale05_1':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.5, 1.0),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_12':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.3, 1.2),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_07':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.3, 0.7),  # scale 512 to 256 in average
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_light':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.02)),
            IAAAffine2(scale=(0.8, 1.8),
                       rotate=(-20, 20),
                       shear=(-0.03, 0.03)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'non_space_transform':
        transform = A.Compose([
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'no_augs':
        transform = A.Compose([
            A.ToFloat()
        ])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    return transform


def make_default_train_dataloader(indir, kind='default', out_size=512, mask_gen_kwargs=None, transform_variant='default',
                                  mask_generator_kind="mixed", dataloader_kwargs=None, ddp_kwargs=None, **kwargs):
    LOGGER.info(f'Make train dataloader {kind} from {indir}. Using mask generator={mask_generator_kind}')

    mask_generator = get_mask_generator(kind=mask_generator_kind, kwargs=mask_gen_kwargs)
    transform = get_transforms(transform_variant, out_size)

    if kind == 'default':
        dataset = InpaintingTrainDataset(indir=indir,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         **kwargs)
    elif kind == 'default_web':
        dataset = InpaintingTrainWebDataset(indir=indir,
                                            mask_generator=mask_generator,
                                            transform=transform,
                                            **kwargs)
    elif kind == 'img_with_segm':
        dataset = ImgSegmentationDataset(indir=indir,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         out_size=out_size,
                                         **kwargs)
    elif kind == 'hand_mask':
        dataset = HandInpaintingTrainDataset(indir=indir,
                                             mask_generator=mask_generator,
                                             transform=transform,
                                             **kwargs)
    elif kind == 'hand_mask_multi':
        # Expect `indir` as a list/tuple and an optional matching list `hand_mask_dir` in kwargs
        hand_mask_dirs = kwargs.pop('hand_mask_dir', None)

        if hand_mask_dirs is not None:
            assert len(indir) == len(hand_mask_dirs), \
                "Length of indir and hand_mask_dir must match for hand_mask_multi"
        else:
            # Derive hand-mask dirs automatically
            hand_mask_dirs = [None] * len(indir)

        # Extract parameters relevant for HandInpaintingTrainDataset and ignore the rest
        mask_inflation = kwargs.get('mask_inflation')

        subdatasets = []
        for rgb_dir, mask_dir in zip(indir, hand_mask_dirs):
            subdatasets.append(
                HandInpaintingTrainDataset(
                    indir=rgb_dir,
                    hand_mask_dir=mask_dir,
                    mask_generator=mask_generator,
                    transform=transform,
                    mask_inflation=mask_inflation,
                )
            )

        dataset = ConcatDataset(subdatasets)
    else:
        raise ValueError(f'Unknown train dataset kind {kind}')

    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    is_dataset_only_iterable = kind in ('default_web',)

    if ddp_kwargs is not None and not is_dataset_only_iterable:
        dataloader_kwargs['shuffle'] = False
        dataloader_kwargs['sampler'] = DistributedSampler(dataset, **ddp_kwargs)

    if is_dataset_only_iterable and 'shuffle' in dataloader_kwargs:
        with open_dict(dataloader_kwargs):
            del dataloader_kwargs['shuffle']

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader


def make_default_val_dataset(indir, kind='default', out_size=512, transform_variant='default', **kwargs):
    if OmegaConf.is_list(indir) or isinstance(indir, (tuple, list)):
        return ConcatDataset([
            make_default_val_dataset(idir, kind=kind, out_size=out_size, transform_variant=transform_variant, **kwargs) for idir in indir 
        ])

    LOGGER.info(f'Make val dataloader {kind} from {indir}')
    mask_generator = get_mask_generator(kind=kwargs.get("mask_generator_kind"), kwargs=kwargs.get("mask_gen_kwargs"))

    if transform_variant is not None:
        transform = get_transforms(transform_variant, out_size)

    if kind == 'default':
        dataset = InpaintingEvaluationDataset(indir, **kwargs)
    elif kind == 'our_eval':
        dataset = OurInpaintingEvaluationDataset(indir, **kwargs)
    elif kind == 'img_with_segm':
        dataset = ImgSegmentationDataset(indir=indir,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         out_size=out_size,
                                         **kwargs)
    elif kind == 'online':
        dataset = InpaintingEvalOnlineDataset(indir=indir,
                                              mask_generator=mask_generator,
                                              transform=transform,
                                              out_size=out_size,
                                              **kwargs)
    else:
        raise ValueError(f'Unknown val dataset kind {kind}')

    return dataset


def make_default_val_dataloader(*args, dataloader_kwargs=None, **kwargs):
    dataset = make_default_val_dataset(*args, **kwargs)

    if dataloader_kwargs is None:
        dataloader_kwargs = {}
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader


def make_constant_area_crop_params(img_height, img_width, min_size=128, max_size=512, area=256*256, round_to_mod=16):
    min_size = min(img_height, img_width, min_size)
    max_size = min(img_height, img_width, max_size)
    if random.random() < 0.5:
        out_height = min(max_size, ceil_modulo(random.randint(min_size, max_size), round_to_mod))
        out_width = min(max_size, ceil_modulo(area // out_height, round_to_mod))
    else:
        out_width = min(max_size, ceil_modulo(random.randint(min_size, max_size), round_to_mod))
        out_height = min(max_size, ceil_modulo(area // out_width, round_to_mod))

    start_y = random.randint(0, img_height - out_height)
    start_x = random.randint(0, img_width - out_width)
    return (start_y, start_x, out_height, out_width)

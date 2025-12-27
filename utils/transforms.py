"""
transforms.py - OPTIMIZED HYBRID VERSION
âœ… Combines OLD simplicity with NEW safety
âœ… Tuned specifically for fire/smoke detection
âœ… No aggressive filtering - preserves small objects
âœ… Zero deprecation warnings
"""

import albumentations as A
import numpy as np
import cv2
from albumentations.pytorch import ToTensorV2

def resize(im, img_size=640, square=False):
    """Aspect ratio resize (unchanged from original)"""
    if square:
        im = cv2.resize(im, (img_size, img_size))
    else:
        h0, w0 = im.shape[:2]
        r = img_size / max(h0, w0)
        if r != 1:
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)))
    return im


def get_train_aug():
    """
    ðŸ"¥ OPTIMIZED FOR FIRE/SMOKE DETECTION
    âœ… Combines OLD pipeline's simplicity with NEW safety
    âœ… Minimal geometric transforms (fire/smoke are orientation-sensitive)
    âœ… Relaxed bbox validation (preserves small objects)
    âœ… Production-ready with proper parameter names
    
    PHILOSOPHY:
    - Keep it SIMPLE like old pipeline (better mAP)
    - Add SAFETY mechanisms from new pipeline
    - RELAX filtering thresholds to keep more data
    """
    return A.Compose([
        # ========================================
        # CORE AUGMENTATIONS (from OLD pipeline)
        # ========================================
        
        # Blur effects (50% probability - same as old)
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.MotionBlur(blur_limit=3, p=0.5),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.5),
        ], p=0.5),
        
        # Grayscale (10% - same as old)
        A.ToGray(p=0.1),
        
        # Brightness/Contrast (10% - same as old)
        A.RandomBrightnessContrast(
            brightness_limit=0.2,  # Slightly more than old for variety
            contrast_limit=0.2,
            p=0.1
        ),
        
        # Color Jitter (10% - same as old, but FIXED parameters)
        A.ColorJitter(
            brightness=0.1,      # âœ… FIXED: correct parameter names
            contrast=0.1,
            saturation=0.15,
            hue=0.05,
            p=0.1
        ),
        
        # Random Gamma (10% - same as old)
        A.RandomGamma(
            gamma_limit=(80, 120),  # Standard range
            p=0.1
        ),
        
        # ========================================
        # MINIMAL GEOMETRIC (NEW - but very light)
        # ========================================
        # âš ï¸ Only add if needed - fire/smoke are usually upright
        
        # Horizontal flip (useful for symmetry)
        A.HorizontalFlip(p=0.3),  # Reduced from 0.5
        
        # ========================================
        # OPTIONAL: Uncomment if you need more variation
        # ========================================
        # A.VerticalFlip(p=0.1),  # Very rare - smoke rises up
        # A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.1),
        
        # Convert to tensor (ALWAYS LAST)
        ToTensorV2(p=1.0),
        
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        
        # ========================================
        # âœ… RELAXED VALIDATION (preserves more data)
        # ========================================
        min_visibility=0.1,        # âœ… RELAXED from 0.3 (keep partially visible boxes)
        min_area=25.0,             # âœ… RELAXED from 100.0 (5x5 pixels minimum)
        clip=True,                 # âœ… SAFETY: auto-clip to bounds
        check_each_transform=True  # âœ… SAFETY: validate after each step
    ))


def get_train_aug_aggressive():
    """
    ðŸ"¥ AGGRESSIVE VERSION (if you want more augmentation)
    Use this if simple version underfits or you have lots of data
    
    To use: In datasets.py, change get_train_aug() to get_train_aug_aggressive()
    """
    return A.Compose([
        # Geometric (light)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.15),
        A.Rotate(
            limit=8,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.2
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=5,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.2
        ),
        
        # Color/Brightness (moderate)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
        ], p=0.4),
        
        A.OneOf([
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.15,
                hue=0.08,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.5
            ),
        ], p=0.3),
        
        # Blur
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.MotionBlur(blur_limit=3, p=0.5),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.5),
        ], p=0.3),
        
        # Noise (minimal)
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 15.0), per_channel=True, p=0.5),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.2), p=0.3),
        ], p=0.1),
        
        # Weather (very rare)
        A.OneOf([
            A.RandomFog(fog_coef_range=(0.1, 0.2), alpha_coef=0.08, p=0.3),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                p=0.2
            ),
        ], p=0.05),
        
        A.ToGray(p=0.02),
        ToTensorV2(p=1.0),
        
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.2,        # Still relaxed
        min_area=36.0,             # 6x6 pixels
        clip=True,
        check_each_transform=True
    ))


def get_train_transform():
    """Basic transform without augmentation (unchanged)"""
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        clip=True,
        check_each_transform=True
    ))


def transform_mosaic(mosaic, boxes, img_size=640):
    """
    âœ… IMPROVED: Combines OLD logic with NEW safety
    Resizes mosaic and transforms boxes with proper validation
    """
    aug = A.Compose([
        A.Resize(img_size, img_size, p=1.0)
    ])
    
    sample = aug(image=mosaic)
    resized_mosaic = sample['image']
    
    # Scale boxes (OLD logic)
    scale_x = resized_mosaic.shape[1] / mosaic.shape[1]
    scale_y = resized_mosaic.shape[0] / mosaic.shape[0]
    
    transformed_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        
        # Scale
        xmin = xmin * scale_x
        ymin = ymin * scale_y
        xmax = xmax * scale_x
        ymax = ymax * scale_y
        
        # âœ… NEW: Clip to bounds with safety margin
        xmin = max(0, min(xmin, resized_mosaic.shape[1] - 1))
        ymin = max(0, min(ymin, resized_mosaic.shape[0] - 1))
        xmax = max(xmin + 1, min(xmax, resized_mosaic.shape[1]))
        ymax = max(ymin + 1, min(ymax, resized_mosaic.shape[0]))
        
        # âœ… OLD LOGIC: Ensure minimum size (from original transform_mosaic)
        if (xmax - xmin) <= 1.0:
            xmax = xmin + 1.0
            if xmax >= float(resized_mosaic.shape[1]):
                xmax = float(resized_mosaic.shape[1])
        
        if (ymax - ymin) <= 1.0:
            ymax = ymin + 1.0
            if ymax >= float(resized_mosaic.shape[0]):
                ymax = float(resized_mosaic.shape[0])
        
        transformed_boxes.append([xmin, ymin, xmax, ymax])
    
    return resized_mosaic, np.array(transformed_boxes)


def get_valid_transform():
    """Validation transform (unchanged)"""
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        clip=True,
        check_each_transform=True
    ))


def infer_transforms(image):
    """Inference transforms (unchanged)"""
    from torchvision import transforms as transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)
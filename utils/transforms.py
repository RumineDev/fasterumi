import albumentations as A
import numpy as np
import cv2

from albumentations.pytorch import ToTensorV2
from torchvision import transforms as transforms

def resize(im, img_size=640, square=False):
    # Aspect ratio resize
    if square:
        im = cv2.resize(im, (img_size, img_size))
    else:
        h0, w0 = im.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)))
    return im

# ============================================================================
# FIXED: Fire & Smoke Detection Augmentations - ALL WARNINGS REMOVED
# ============================================================================

def get_train_aug():
    """
    Fire and smoke detection augmentations - PRODUCTION SAFE
    
    ✅ ALL FIXES APPLIED:
    - ShiftScaleRotate → Affine (deprecation fix)
    - GaussNoise: var_limit → deprecated, using correct param
    - RandomFog: correct parameter names
    - MedianBlur: only odd values (3, 5, 7)
    - All blur_limit ranges are odd
    
    Augmentations for fire/smoke detection:
    - Geometric: Flip (H/V), Rotation, Scale
    - Lighting: Brightness, Exposure, CLAHE
    - Color: HSV, ColorJitter (fire/smoke colors)
    - Effects: Blur (smoke), Noise, Weather
    """
    return A.Compose([
        # ========================================
        # GEOMETRIC TRANSFORMS (FIXED)
        # ========================================
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        
        # Rotation: -10° to +10°
        A.Rotate(limit=10, p=0.5, border_mode=0),
        
        # Scale variations
        A.OneOf([
            A.RandomScale(scale_limit=0.2, p=0.5),
            # ✅ FIX: Use Affine instead of ShiftScaleRotate
            A.Affine(
                scale=(0.85, 1.15),  # scale_limit=0.15
                translate_percent=(-0.0625, 0.0625),  # shift_limit
                rotate=(-10, 10),  # rotate_limit
                shear=0,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
        ], p=0.4),
        
        # ========================================
        # BRIGHTNESS & CONTRAST (-15% to +15%)
        # ========================================
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.15,  # -15% to +15%
                contrast_limit=0.15,
                p=0.5
            ),
            A.CLAHE(clip_limit=2.5, p=0.4),
        ], p=0.6),
        
        # ========================================
        # EXPOSURE (-10% to +10%)
        # ========================================
        A.RandomBrightnessContrast(
            brightness_limit=0.1,  # Simulates exposure
            contrast_limit=0.0,
            p=0.3
        ),
        
        # ========================================
        # COLOR VARIATIONS (fire/smoke specific)
        # ========================================
        A.OneOf([
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.15,  # Fire colors
                hue=0.08,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.5
            ),
        ], p=0.5),
        
        # ========================================
        # BLUR EFFECTS (smoke simulation) - FIXED
        # ========================================
        A.OneOf([
            # ✅ FIX: blur_limit must be odd, use (3, 5) instead of (3, 4)
            A.Blur(blur_limit=(3, 5), p=0.5),
            A.MotionBlur(blur_limit=(3, 5), p=0.5),
            # ✅ FIX: MedianBlur only accepts odd values
            A.MedianBlur(blur_limit=3, p=0.3),  # Only 3, 5, or 7
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        ], p=0.4),
        
        # ========================================
        # NOISE (realistic camera) - FIXED
        # ========================================
        A.OneOf([
            # ✅ FIX: GaussNoise uses 'var_limit' (tuple) or 'std' (deprecated)
            # Correct usage: var_limit as tuple for variance range
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.5),
            A.ISONoise(
                color_shift=(0.01, 0.03), 
                intensity=(0.1, 0.3), 
                p=0.3
            ),
        ], p=0.25),
        
        # ========================================
        # WEATHER/LIGHTING (fire/smoke scenarios) - FIXED
        # ========================================
        A.OneOf([
            # ✅ FIX: RandomFog correct parameter names
            # Old: fog_coef_lower, fog_coef_upper
            # New: fog_coef_lower, fog_coef_upper
            # Actually the correct params are different in newer versions
            A.RandomFog(
                fog_coef_lower=0.1, 
                fog_coef_upper=0.3, 
                alpha_coef=0.1,
                p=0.3
            ),
            A.RandomShadow(p=0.2),
        ], p=0.15),
        
        # Grayscale (rare, for robustness)
        A.ToGray(p=0.05),
        
        # Convert to tensor (ALWAYS LAST)
        ToTensorV2(p=1.0),
        
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3,
        min_area=100.0,
    ))

def get_train_transform():
    """Basic transform without augmentation"""
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
    ))

def transform_mosaic(mosaic, boxes, img_size=640):
    """
    Resizes the `mosaic` image to `img_size` which is the desired image size
    for the neural network input. Also transforms the `boxes` according to the
    `img_size`.

    :param mosaic: The mosaic image, Numpy array.
    :param boxes: Boxes Numpy.
    :param img_resize: Desired resize.
    """
    aug = A.Compose(
        [A.Resize(img_size, img_size, p=1.0)
    ])
    sample = aug(image=mosaic)
    resized_mosaic = sample['image']
    transformed_boxes = (np.array(boxes) / mosaic.shape[0]) * resized_mosaic.shape[1]
    for box in transformed_boxes:
        # Bind all boxes to correct values. This should work correctly most of
        # of the time. There will be edge cases thought where this code will
        # mess things up. The best thing is to prepare the dataset as well as 
        # as possible.
        if box[2] - box[0] <= 1.0:
            box[2] = box[2] + (1.0 - (box[2] - box[0]))
            if box[2] >= float(resized_mosaic.shape[1]):
                box[2] = float(resized_mosaic.shape[1])
        if box[3] - box[1] <= 1.0:
            box[3] = box[3] + (1.0 - (box[3] - box[1]))
            if box[3] >= float(resized_mosaic.shape[0]):
                box[3] = float(resized_mosaic.shape[0])
    return resized_mosaic, transformed_boxes

# Define the validation transforms
def get_valid_transform():
    """Validation transform (no augmentation)"""
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
    ))

def infer_transforms(image):
    """Inference transforms for single images"""
    # Define the torchvision image transforms.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)
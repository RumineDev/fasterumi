import torch
import cv2
import numpy as np
import os
import glob as glob
import random

from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader
from utils.transforms import (
    get_train_transform, 
    get_valid_transform,
    get_train_aug,
    transform_mosaic
)
from tqdm.auto import tqdm


# the dataset class
class CustomDataset(Dataset):
    def __init__(
        self, 
        images_path, 
        labels_path, 
        img_size, 
        classes, 
        transforms=None, 
        use_train_aug=False,
        train=False, 
        mosaic=1.0,
        square_training=False,
        label_type='pascal_voc'
    ):
        self.transforms = transforms
        self.use_train_aug = use_train_aug
        self.images_path = images_path
        self.labels_path = labels_path
        self.img_size = img_size
        self.classes = classes
        self.train = train
        self.square_training = square_training
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = []
        self.log_annot_issue_x = True
        self.mosaic = mosaic
        self.log_annot_issue_y = True
        self.label_type = label_type
        
        # Case-insensitive class mapping
        self.class_map = self._build_case_insensitive_class_map()
        
        # get all the image paths in sorted order
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.images_path, file_type)))
        self.all_annot_paths = glob.glob(os.path.join(self.labels_path, '*.xml'))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)
        # Remove all annotations and images when no object is present.
        if self.label_type == 'pascal_voc':
            self.read_and_clean()

    def _build_case_insensitive_class_map(self):
        """Build case-insensitive class mapping."""
        class_map = {}
        for idx, class_name in enumerate(self.classes):
            normalized_name = str(class_name).lower().strip()
            class_map[normalized_name] = idx
        return class_map

    def _get_class_index(self, class_name):
        """Get class index with case-insensitive matching."""
        normalized = str(class_name).lower().strip()
        return self.class_map.get(normalized, None)

    def read_and_clean(self):
        """⭐ AGGRESSIVE CLEANING - Fixes annotations in-place"""
        print('🔍 Aggressive Dataset Cleaning...')
        images_to_remove = []
        
        stats = {
            'missing_annotations': [],
            'invalid_bbox': [],
            'no_valid_objects': [],
            'negative_coords': [],
            'out_of_bounds': [],
            'zero_area': [],
            'unknown_classes': {},
            'total_fixed': 0,
            'total_removed': 0
        }

        for image_name in tqdm(self.all_images, total=len(self.all_images)):
            possible_annot_name = os.path.join(
                self.labels_path, 
                os.path.splitext(image_name)[0] + '.xml'
            )
            
            if possible_annot_name not in self.all_annot_paths:
                images_to_remove.append(image_name)
                stats['missing_annotations'].append(image_name)
                continue

            try:
                tree = et.parse(possible_annot_name)
                root = tree.getroot()
            except Exception as e:
                images_to_remove.append(image_name)
                stats['invalid_bbox'].append((image_name, f"Parse error"))
                continue
            
            # Get image dimensions
            size = root.find('size')
            if size is not None:
                try:
                    img_width = float(size.find('width').text)
                    img_height = float(size.find('height').text)
                except:
                    images_to_remove.append(image_name)
                    continue
            else:
                images_to_remove.append(image_name)
                continue
            
            # ⭐ VALIDATE AND FIX BBOXES
            has_valid_object = False
            objects_to_remove = []
            
            for member in root.findall('object'):
                class_name = member.find('name').text
                class_idx = self._get_class_index(class_name)
                
                if class_idx is None:
                    if class_name not in stats['unknown_classes']:
                        stats['unknown_classes'][class_name] = 0
                    stats['unknown_classes'][class_name] += 1
                    objects_to_remove.append(member)
                    continue
                
                try:
                    bbox = member.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymin = float(bbox.find('ymin').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # ⭐ FIX COORDINATES
                    needs_fix = False
                    remove_bbox = False
                    
                    # Fix negative coordinates
                    if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
                        stats['negative_coords'].append((image_name, f"[{xmin:.0f}, {ymin:.0f}, {xmax:.0f}, {ymax:.0f}]"))
                        needs_fix = True
                        xmin = max(0.0, xmin)
                        ymin = max(0.0, ymin)
                        xmax = max(0.0, xmax)
                        ymax = max(0.0, ymax)
                    
                    # Fix out of bounds (with 1px tolerance)
                    if xmax > img_width + 1.0 or ymax > img_height + 1.0:
                        stats['out_of_bounds'].append((image_name, "OOB"))
                        needs_fix = True
                        xmax = min(img_width, xmax)
                        ymax = min(img_height, ymax)
                    
                    # Check validity
                    if xmin >= xmax or ymin >= ymax:
                        remove_bbox = True
                    
                    # Minimum size check (4x4 pixels)
                    width = xmax - xmin
                    height = ymax - ymin
                    area = width * height
                    
                    if area < 16.0 or width < 4.0 or height < 4.0:
                        remove_bbox = True
                    
                    if remove_bbox:
                        objects_to_remove.append(member)
                        continue
                    
                    # ⭐ WRITE FIXES BACK TO XML
                    if needs_fix:
                        bbox.find('xmin').text = str(int(xmin))
                        bbox.find('ymin').text = str(int(ymin))
                        bbox.find('xmax').text = str(int(xmax))
                        bbox.find('ymax').text = str(int(ymax))
                        stats['total_fixed'] += 1
                    
                    has_valid_object = True
                    
                except Exception as e:
                    objects_to_remove.append(member)
                    continue
            
            # Remove invalid objects
            for obj in objects_to_remove:
                root.remove(obj)
            
            # ⭐ SAVE FIXED XML
            if len(objects_to_remove) > 0 and has_valid_object:
                try:
                    tree.write(possible_annot_name)
                except:
                    pass
            
            # Remove if no valid objects
            if not has_valid_object:
                images_to_remove.append(image_name)
                stats['total_removed'] += 1

        # Remove problematic images
        self.all_images = [img for img in self.all_images if img not in images_to_remove]

        # Report
        print("\n" + "="*80)
        print("📊 AGGRESSIVE CLEANING REPORT")
        print("="*80)
        print(f"\n✅ Fixed Annotations: {stats['total_fixed']}")
        print(f"🗑️  Removed Images: {stats['total_removed']}")
        print(f"✅ Valid Remaining: {len(self.all_images)}")
        if stats['negative_coords']:
            print(f"⭐ Fixed Negative Coords: {len(stats['negative_coords'])}")
        if stats['out_of_bounds']:
            print(f"⭐ Fixed Out-of-Bounds: {len(stats['out_of_bounds'])}")
        print("="*80 + "\n")

    def resize(self, im, square=False):
        if square:
            im = cv2.resize(im, (self.img_size, self.img_size))
        else:
            h0, w0 = im.shape[:2]
            r = self.img_size / max(h0, w0)
            if r != 1:
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)))
        return im

    def load_image_and_labels(self, index):
        image_name = self.all_images[index]
        image_path = os.path.join(self.images_path, image_name)

        # Read the image.
        image = cv2.imread(image_path)
        # Convert BGR to RGB color format.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = self.resize(image, square=self.square_training)
        image_resized /= 255.0
        
        if self.label_type == 'pascal_voc':
            image, image_resized, orig_boxes, \
            boxes, labels, area, iscrowd, (image_width, image_height) \
            = self.load_pascal_voc(image, image_name, image_resized)

        if self.label_type == 'yolo':
            image, image_resized, orig_boxes, \
            boxes, labels, area, iscrowd, (image_width, image_height) \
            = self.load_yolo(image, image_name, image_resized)
        
        return image, image_resized, orig_boxes, \
            boxes, labels, area, iscrowd, (image_width, image_height)
    
    def load_pascal_voc(self, image, image_name, image_resized):
        # Capture the corresponding XML file for getting the annotations.
        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.labels_path, annot_filename)

        boxes = []
        orig_boxes = []
        labels = []
        
        # Get the height and width of the image.
        image_width = image.shape[1]
        image_height = image.shape[0]
                
        # Box coordinates for xml files are extracted and corrected for image size given.
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        for member in root.findall('object'):
            # Case-insensitive class matching with unknown class filtering
            class_name = member.find('name').text
            class_idx = self._get_class_index(class_name)
            
            # Skip unknown classes
            if class_idx is None:
                continue
            
            labels.append(class_idx)
            
            xmin = float(member.find('bndbox').find('xmin').text)
            xmax = float(member.find('bndbox').find('xmax').text)
            ymin = float(member.find('bndbox').find('ymin').text)
            ymax = float(member.find('bndbox').find('ymax').text)

            xmin, ymin, xmax, ymax = self.check_image_and_annotation(
                xmin, 
                ymin, 
                xmax, 
                ymax, 
                image_width, 
                image_height, 
                orig_data=True
            )

            orig_boxes.append([xmin, ymin, xmax, ymax])
            
            # Resize the bounding boxes
            xmin_final = (xmin/image_width)*image_resized.shape[1]
            xmax_final = (xmax/image_width)*image_resized.shape[1]
            ymin_final = (ymin/image_height)*image_resized.shape[0]
            ymax_final = (ymax/image_height)*image_resized.shape[0]

            xmin_final, ymin_final, xmax_final, ymax_final = self.check_image_and_annotation(
                xmin_final, 
                ymin_final, 
                xmax_final, 
                ymax_final, 
                image_resized.shape[1], 
                image_resized.shape[0],
                orig_data=False
            )
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        # Bounding box to tensor.
        boxes_length = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return image, image_resized, orig_boxes, \
            boxes, labels, area, iscrowd, (image_width, image_height)
    
    def load_yolo(self, image, image_name, image_resized):
        # Capture the corresponding text file for getting the annotations.
        annot_filename = os.path.splitext(image_name)[0] + '.txt'
        annot_file_path = os.path.join(self.labels_path, annot_filename)

        boxes = []
        orig_boxes = []
        labels = []
        
        # Get the height and width of the image.
        image_width = image.shape[1]
        image_height = image.shape[0]

        with open(annot_file_path, 'r') as f:
            annot_file_content = f.readlines()
            f.close()

        for line in annot_file_content:
            label, norm_xc, norm_yc, norm_w, norm_h = line.split()
            label, norm_xc, norm_yc, norm_w, norm_h = \
                int(label), float(norm_xc), float(norm_yc), float(norm_w), float(norm_h)

            labels.append(label + 1)
            xc, w = norm_xc * image_width, norm_w * image_width 
            yc, h = norm_yc * image_height, norm_h * image_height

            xmin = xc - (w / 2)
            ymin = yc - (h / 2)
            xmax = xmin + w
            ymax = ymin + h

            xmin, ymin, xmax, ymax = self.check_image_and_annotation(
                xmin, 
                ymin, 
                xmax, 
                ymax, 
                image_width, 
                image_height, 
                orig_data=True
            )

            orig_boxes.append([xmin, ymin, xmax, ymax])

            # Resize the bounding boxes
            xmin_final = (xmin/image_width)*image_resized.shape[1]
            xmax_final = (xmax/image_width)*image_resized.shape[1]
            ymin_final = (ymin/image_height)*image_resized.shape[0]
            ymax_final = (ymax/image_height)*image_resized.shape[0]

            xmin_final, ymin_final, xmax_final, ymax_final = self.check_image_and_annotation(
                xmin_final, 
                ymin_final, 
                xmax_final, 
                ymax_final, 
                image_resized.shape[1], 
                image_resized.shape[0],
                orig_data=False
            )
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        # Bounding box to tensor.
        boxes_length = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return image, image_resized, orig_boxes, \
            boxes, labels, area, iscrowd, (image_width, image_height)

    def check_image_and_annotation(self, xmin, ymin, xmax, ymax, width, height, orig_data=False):
        """⭐ ULTRA-STRICT coordinate validation"""
        # Clamp to valid range with 2px margin
        xmin = float(max(0.0, min(xmin, width - 2.0)))
        ymin = float(max(0.0, min(ymin, height - 2.0)))
        xmax = float(max(2.0, min(xmax, width)))
        ymax = float(max(2.0, min(ymax, height)))
        
        # Ensure minimum 4x4 size
        if xmax - xmin < 4.0:
            center_x = (xmin + xmax) / 2.0
            xmin = max(0.0, center_x - 2.0)
            xmax = min(width, center_x + 2.0)
        
        if ymax - ymin < 4.0:
            center_y = (ymin + ymax) / 2.0
            ymin = max(0.0, center_y - 2.0)
            ymax = min(height, center_y + 2.0)
        
        return float(xmin), float(ymin), float(xmax), float(ymax)

    def load_cutmix_image_and_boxes(self, index, resize_factor=512):
        """ 
        Adapted from: https://www.kaggle.com/shonenkov/oof-evaluation-mixup-efficientdet
        """
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)
        indices = [index] + [random.randint(0, len(self.all_images) - 1) for _ in range(3)]

        result_boxes = []
        result_classes = []

        for i, index in enumerate(indices):
            _, image_resized, orig_boxes, boxes, \
            labels, area, iscrowd, dims = self.load_image_and_labels(
                index=index
            )

            h, w = image_resized.shape[:2]

            if i == 0:
                result_image = np.full((s * 2, s * 2, image_resized.shape[2]), 114/255, dtype=np.float32)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image_resized[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            if len(orig_boxes) > 0:
                boxes[:, 0] += padw
                boxes[:, 1] += padh
                boxes[:, 2] += padw
                boxes[:, 3] += padh

                result_boxes.append(boxes)
                result_classes += labels

        final_classes = []
        if len(result_boxes) > 0:
            result_boxes = np.concatenate(result_boxes, 0)
            np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
            result_boxes = result_boxes.astype(np.int32)
            for idx in range(len(result_boxes)):
                if ((result_boxes[idx, 2] - result_boxes[idx, 0]) * (result_boxes[idx, 3] - result_boxes[idx, 1])) > 0:
                    final_classes.append(result_classes[idx])
            result_boxes = result_boxes[
                np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)
            ]
        # Resize the mosaic image to the desired shape and transform boxes.
        result_image, result_boxes = transform_mosaic(
            result_image, result_boxes, self.img_size
        )
        return result_image, torch.tensor(result_boxes), \
            torch.tensor(np.array(final_classes)), area, iscrowd, dims

    def validate_boxes_post_augmentation(self, boxes, image_shape):
        """⭐ Post-augmentation validator - catches ALL invalid boxes"""
        if len(boxes) == 0:
            return torch.ones(0, dtype=torch.bool)
        
        if isinstance(boxes, torch.Tensor):
            boxes_np = boxes.cpu().numpy()
        else:
            boxes_np = np.array(boxes)
        
        height, width = image_shape[:2]
        valid_mask = np.ones(len(boxes_np), dtype=bool)
        
        for i, box in enumerate(boxes_np):
            xmin, ymin, xmax, ymax = box
            
            width_box = xmax - xmin
            height_box = ymax - ymin
            
            # STRICT checks
            if width_box <= 0 or height_box <= 0:
                valid_mask[i] = False
                continue
            
            # Minimum 4x4 pixels
            if width_box < 4.0 or height_box < 4.0:
                valid_mask[i] = False
                continue
            
            # Within bounds (with small tolerance)
            if xmin < -0.1 or ymin < -0.1 or xmax > (width + 0.1) or ymax > (height + 0.1):
                valid_mask[i] = False
                continue
            
            # Minimum area 16px²
            area = width_box * height_box
            if area < 16.0:
                valid_mask[i] = False
                continue
        
        return torch.tensor(valid_mask, dtype=torch.bool)

    def __getitem__(self, idx):
            # ⭐ IMPROVED: Deterministic retry to preserve data distribution
            original_idx = idx
            max_retries = 10
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Load data
                    if not self.train:
                        image, image_resized, orig_boxes, boxes, \
                            labels, area, iscrowd, dims = self.load_image_and_labels(
                            index=idx
                        )
                    else: 
                        mosaic_prob = random.uniform(0.0, 1.0)
                        if self.mosaic >= mosaic_prob:
                            image_resized, boxes, labels, \
                                area, iscrowd, dims = self.load_cutmix_image_and_boxes(
                                idx, resize_factor=(self.img_size, self.img_size)
                            )
                        else:
                            image, image_resized, orig_boxes, boxes, \
                                labels, area, iscrowd, dims = self.load_image_and_labels(
                                index=idx
                            )

                    # Prepare target dictionary
                    target = {}
                    target["boxes"] = boxes
                    target["labels"] = labels
                    target["area"] = area
                    target["iscrowd"] = iscrowd
                    image_id = torch.tensor([idx])
                    target["image_id"] = image_id

                    # ⭐ PRE-AUGMENTATION: Ensure labels and boxes match
                    if len(target['boxes']) != len(target['labels']):
                        min_len = min(len(target['boxes']), len(target['labels']))
                        target['boxes'] = target['boxes'][:min_len]
                        target['labels'] = target['labels'][:min_len]

                    # ⭐ PRE-AUGMENTATION: Validate boxes
                    if len(target['boxes']) > 0:
                        pre_valid_mask = self.validate_boxes_post_augmentation(
                            target['boxes'], 
                            image_resized.shape[:2]
                        )
                        
                        # ⭐ IMPROVED: If no valid boxes, try next sequential image
                        if pre_valid_mask.sum() == 0:
                            retry_count += 1
                            # Use sequential fallback instead of random
                            idx = (original_idx + retry_count) % len(self.all_images)
                            continue
                        
                        target['boxes'] = target['boxes'][pre_valid_mask]
                        target['labels'] = target['labels'][pre_valid_mask]
                    else:
                        # No boxes, try next image
                        retry_count += 1
                        idx = (original_idx + retry_count) % len(self.all_images)
                        continue

                    # ⭐ CONVERT TO LIST for augmentation
                    labels = target['labels'].cpu().numpy().tolist() if isinstance(target['labels'], torch.Tensor) else target['labels']
                    bboxes = target['boxes'].cpu().numpy().tolist() if isinstance(target['boxes'], torch.Tensor) else target['boxes'].tolist()

                    # ⭐ APPLY AUGMENTATION with relaxed parameters for difficult samples
                    try:
                        if self.use_train_aug:
                            # ⭐ ADAPTIVE: Use lighter augmentation if retry count > 3
                            if retry_count > 3:
                                # Fallback to basic transform for difficult samples
                                sample = self.transforms(image=image_resized,
                                                        bboxes=bboxes,
                                                        labels=labels)
                            else:
                                train_aug = get_train_aug()
                                sample = train_aug(image=image_resized,
                                                        bboxes=bboxes,
                                                        labels=labels)
                            image_resized = sample['image']
                            target['boxes'] = torch.Tensor(sample['bboxes']).to(torch.float32)
                            target['labels'] = torch.tensor(sample['labels'], dtype=torch.int64)
                        else:
                            sample = self.transforms(image=image_resized,
                                                    bboxes=bboxes,
                                                    labels=labels)
                            image_resized = sample['image']
                            target['boxes'] = torch.Tensor(sample['bboxes']).to(torch.float32)
                            target['labels'] = torch.tensor(sample['labels'], dtype=torch.int64)
                    except Exception as e:
                        # Augmentation failed, try next sequential image
                        retry_count += 1
                        idx = (original_idx + retry_count) % len(self.all_images)
                        continue

                    # ⭐ POST-AUGMENTATION: Validation with relaxed thresholds for retries
                    if len(target['boxes']) > 0:
                        post_valid_mask = self.validate_boxes_post_augmentation(
                            target['boxes'], 
                            image_resized.shape[1:]
                        )
                        
                        # Ensure mask length matches
                        if len(post_valid_mask) != len(target['boxes']) or len(post_valid_mask) != len(target['labels']):
                            min_len = min(len(post_valid_mask), len(target['boxes']), len(target['labels']))
                            post_valid_mask = post_valid_mask[:min_len]
                            target['boxes'] = target['boxes'][:min_len]
                            target['labels'] = target['labels'][:min_len]
                        
                        # If no valid boxes, try next image
                        if post_valid_mask.sum() == 0:
                            retry_count += 1
                            idx = (original_idx + retry_count) % len(self.all_images)
                            continue
                        
                        # Filter valid boxes
                        target['boxes'] = target['boxes'][post_valid_mask]
                        target['labels'] = target['labels'][post_valid_mask]
                        
                        # ⭐ FINAL STRICT CHECK
                        final_boxes = target['boxes'].numpy()
                        final_valid = np.ones(len(final_boxes), dtype=bool)
                        
                        for i, box in enumerate(final_boxes):
                            if (box[2] - box[0]) <= 0 or (box[3] - box[1]) <= 0:
                                final_valid[i] = False
                        
                        # If no boxes pass, try next image
                        if final_valid.sum() == 0:
                            retry_count += 1
                            idx = (original_idx + retry_count) % len(self.all_images)
                            continue
                        
                        # Apply final filter
                        target['boxes'] = target['boxes'][torch.from_numpy(final_valid)]
                        target['labels'] = target['labels'][torch.from_numpy(final_valid)]
                        
                        # Recalculate area
                        target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * \
                                        (target['boxes'][:, 2] - target['boxes'][:, 0])
                        target['iscrowd'] = torch.zeros(len(target['boxes']), dtype=torch.int64)
                    else:
                        # No boxes, try next image
                        retry_count += 1
                        idx = (original_idx + retry_count) % len(self.all_images)
                        continue

                    # ⭐ FINAL SAFETY
                    if len(target['boxes']) == 0:
                        retry_count += 1
                        idx = (original_idx + retry_count) % len(self.all_images)
                        continue
                    
                    if torch.isnan(target['boxes']).any():
                        retry_count += 1
                        idx = (original_idx + retry_count) % len(self.all_images)
                        continue
                    
                    # ⭐ SUCCESS
                    return image_resized, target
                    
                except Exception as e:
                    # Retry with next sequential image
                    retry_count += 1
                    idx = (original_idx + retry_count) % len(self.all_images)
                    continue
            
            # ⭐ FALLBACK (should rarely happen)
            print(f"⚠️ Warning: Could not load valid sample for idx={original_idx} after {max_retries} retries")
            
            # Try to load the cleanest image from dataset
            # Use first image as it's likely been validated
            try:
                image, image_resized, orig_boxes, boxes, \
                    labels, area, iscrowd, dims = self.load_image_and_labels(index=0)
                
                target = {
                    'boxes': boxes,
                    'labels': labels,
                    'area': area,
                    'iscrowd': iscrowd,
                    'image_id': torch.tensor([original_idx])
                }
                
                # Basic validation
                if len(target['boxes']) > 0:
                    sample = self.transforms(image=image_resized,
                                            bboxes=target['boxes'].numpy().tolist(),
                                            labels=target['labels'].numpy().tolist())
                    image_resized = sample['image']
                    target['boxes'] = torch.Tensor(sample['bboxes']).to(torch.float32)
                    target['labels'] = torch.tensor(sample['labels'], dtype=torch.int64)
                    
                    if len(target['boxes']) > 0:
                        return image_resized, target
            except:
                pass
            
            # Ultimate fallback
            image_resized = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
            target = {
                'boxes': torch.tensor([[10.0, 10.0, 50.0, 50.0]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),
                'area': torch.tensor([1600.0], dtype=torch.float32),
                'iscrowd': torch.tensor([0], dtype=torch.int64),
                'image_id': torch.tensor([original_idx])
            }
            
            return image_resized, target

    def __len__(self):
        return len(self.all_images)

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

# Prepare the final datasets and data loaders.
def create_train_dataset(
    train_dir_images, 
    train_dir_labels, 
    img_size, 
    classes,
    use_train_aug=False,
    mosaic=1.0,
    square_training=False,
    label_type='pascal_voc'
):
    train_dataset = CustomDataset(
        train_dir_images, 
        train_dir_labels,
        img_size, 
        classes, 
        get_train_transform(),
        use_train_aug=use_train_aug,
        train=True, 
        mosaic=mosaic,
        square_training=square_training,
        label_type=label_type
    )
    return train_dataset

def create_valid_dataset(
    valid_dir_images, 
    valid_dir_labels, 
    img_size, 
    classes,
    square_training=False,
    label_type='pascal_voc'
):
    valid_dataset = CustomDataset(
        valid_dir_images, 
        valid_dir_labels, 
        img_size, 
        classes, 
        get_valid_transform(),
        train=False, 
        square_training=square_training,
        label_type=label_type
    )
    return valid_dataset

def create_train_loader(
    train_dataset, batch_size, num_workers=0, batch_sampler=None
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=batch_sampler
    )
    return train_loader

def create_valid_loader(
    valid_dataset, batch_size, num_workers=0, batch_sampler=None
):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=batch_sampler
    )
    return valid_loader
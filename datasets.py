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
        
        # ⭐ NEW: Case-insensitive class mapping (matches Cell 4 validation logic)
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
        """
        Build case-insensitive class mapping.
        Maps normalized class names (lowercase) to their indices.
        Compatible with Cell 4's SmartClassValidator logic.
        """
        class_map = {}
        for idx, class_name in enumerate(self.classes):
            normalized_name = str(class_name).lower().strip()
            class_map[normalized_name] = idx
        return class_map

    def _get_class_index(self, class_name):
        """
        Get class index with case-insensitive matching.
        Returns None if class not found (filters unknown classes).
        """
        normalized = str(class_name).lower().strip()
        return self.class_map.get(normalized, None)

    def read_and_clean(self):
        """Enhanced validation with truncated output and strict bbox checking"""
        print('🔍 Checking Labels and Images...')
        images_to_remove = []
        problematic_images = []
        
        stats = {
            'missing_annotations': [],
            'invalid_bbox': [],
            'no_valid_objects': [],
            'negative_coords': [],
            'out_of_bounds': [],
            'zero_area': [],
            'unknown_classes': {}
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
                stats['invalid_bbox'].append((image_name, f"Parse error: {e}"))
                continue
            
            # Get image dimensions for validation
            size = root.find('size')
            if size is not None:
                try:
                    img_width = float(size.find('width').text)
                    img_height = float(size.find('height').text)
                except:
                    # If size info is corrupted, skip this image
                    images_to_remove.append(image_name)
                    stats['invalid_bbox'].append((image_name, "Corrupted size info in XML"))
                    continue
            else:
                # If no size info, skip this image
                images_to_remove.append(image_name)
                stats['invalid_bbox'].append((image_name, "No image size in XML"))
                continue
            
            invalid_bbox = False
            has_valid_object = False

            for member in root.findall('object'):
                # ⭐ Check class validity first (case-insensitive)
                class_name = member.find('name').text
                class_idx = self._get_class_index(class_name)
                
                if class_idx is None:
                    if class_name not in stats['unknown_classes']:
                        stats['unknown_classes'][class_name] = 0
                    stats['unknown_classes'][class_name] += 1
                    continue
                
                try:
                    bbox = member.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymin = float(bbox.find('ymin').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # ⭐ STRICT VALIDATION - Fix negative/invalid coords
                    
                    # 1. Check for invalid bbox structure
                    if xmin >= xmax or ymin >= ymax:
                        invalid_bbox = True
                        stats['invalid_bbox'].append((image_name, f"Invalid: xmin={xmin:.2f}, xmax={xmax:.2f}, ymin={ymin:.2f}, ymax={ymax:.2f}"))
                        break
                    
                    # 2. Check for negative coordinates (CRITICAL FIX)
                    if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
                        stats['negative_coords'].append((image_name, f"Negative: [{xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f}]"))
                        invalid_bbox = True
                        break
                    
                    # 3. Check for out of bounds (with small tolerance)
                    tolerance = 1.0  # Allow 1px tolerance
                    if xmin >= img_width or ymin >= img_height or xmax > (img_width + tolerance) or ymax > (img_height + tolerance):
                        stats['out_of_bounds'].append((image_name, f"OOB: [{xmin:.0f}, {ymin:.0f}, {xmax:.0f}, {ymax:.0f}] vs img [{img_width:.0f}, {img_height:.0f}]"))
                        invalid_bbox = True
                        break
                    
                    # 4. Check for zero/tiny area
                    area = (xmax - xmin) * (ymax - ymin)
                    if area < 1.0:
                        stats['zero_area'].append((image_name, f"Area={area:.2f}"))
                        invalid_bbox = True
                        break
                    
                    has_valid_object = True
                    
                except Exception as e:
                    invalid_bbox = True
                    stats['invalid_bbox'].append((image_name, f"BBox error: {e}"))
                    break

            if invalid_bbox:
                problematic_images.append((image_name, "invalid_bbox"))
                images_to_remove.append(image_name)
            elif not has_valid_object:
                problematic_images.append((image_name, "no_valid_objects"))
                images_to_remove.append(image_name)
                stats['no_valid_objects'].append(image_name)

        # Remove problematic images
        self.all_images = [img for img in self.all_images if img not in images_to_remove]
        self.all_annot_paths = [
            path for path in self.all_annot_paths 
            if not any(
                os.path.splitext(os.path.basename(path))[0] + ext in images_to_remove 
                for ext in self.image_file_types
            )
        ]

        # ⭐ TRUNCATED REPORTING
        print("\n" + "="*80)
        print("📊 DATASET VALIDATION REPORT (TRUNCATED)")
        print("="*80)
        
        if stats['missing_annotations']:
            count = len(stats['missing_annotations'])
            print(f"\n⚠️  Missing Annotations: {count}")
            for img in stats['missing_annotations'][:5]:
                print(f"   • {img}")
            if count > 5:
                print(f"   ... and {count-5} more")
        
        if stats['invalid_bbox']:
            count = len(stats['invalid_bbox'])
            print(f"\n⚠️  Invalid Bounding Boxes: {count}")
            for img, reason in stats['invalid_bbox'][:5]:
                print(f"   • {img}: {reason}")
            if count > 5:
                print(f"   ... and {count-5} more")
        
        if stats['negative_coords']:
            count = len(stats['negative_coords'])
            print(f"\n❌ NEGATIVE COORDINATES (CRITICAL): {count}")
            for img, coords in stats['negative_coords'][:5]:
                print(f"   • {img}: {coords}")
            if count > 5:
                print(f"   ... and {count-5} more")
            print(f"   💡 These cause augmentation errors!")
        
        if stats['out_of_bounds']:
            count = len(stats['out_of_bounds'])
            print(f"\n⚠️  Out of Bounds Boxes: {count}")
            for img, reason in stats['out_of_bounds'][:5]:
                print(f"   • {img}: {reason}")
            if count > 5:
                print(f"   ... and {count-5} more")
        
        if stats['zero_area']:
            count = len(stats['zero_area'])
            print(f"\n⚠️  Zero/Tiny Area Boxes: {count}")
            for img, reason in stats['zero_area'][:5]:
                print(f"   • {img}: {reason}")
            if count > 5:
                print(f"   ... and {count-5} more")
        
        if stats['no_valid_objects']:
            count = len(stats['no_valid_objects'])
            print(f"\n⚠️  No Valid Objects: {count}")
            for img in stats['no_valid_objects'][:5]:
                print(f"   • {img}")
            if count > 5:
                print(f"   ... and {count-5} more")
        
        if stats['unknown_classes']:
            print(f"\n⚠️  UNKNOWN CLASSES DETECTED:")
            total_unknown = sum(stats['unknown_classes'].values())
            print(f"   Total ignored: {total_unknown:,} annotations")
            print(f"   Classes:")
            for cls, count in sorted(stats['unknown_classes'].items(), key=lambda x: -x[1])[:5]:
                print(f"      • '{cls}': {count:,}")
            if len(stats['unknown_classes']) > 5:
                print(f"      ... and {len(stats['unknown_classes'])-5} more classes")
        
        print(f"\n" + "-"*80)
        print(f"📈 SUMMARY:")
        print(f"   • Original: {len(self.all_images) + len(images_to_remove):,}")
        print(f"   • Removed: {len(images_to_remove):,}")
        print(f"   • Remaining: {len(self.all_images):,}")
        
        if stats['negative_coords']:
            print(f"\n   ❌ CRITICAL: {len(stats['negative_coords'])} images with negative coords removed")
            print(f"      These would cause albumentations to crash!")
        
        print("="*80 + "\n")

    def resize(self, im, square=False):
        if square:
            im = cv2.resize(im, (self.img_size, self.img_size))
        else:
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
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
            # ⭐ FIXED: Case-insensitive class matching with unknown class filtering
            class_name = member.find('name').text
            class_idx = self._get_class_index(class_name)
            
            # Skip unknown classes (filter them out silently)
            if class_idx is None:
                continue
            
            labels.append(class_idx)
            
            # xmin = left corner x-coordinates
            xmin = float(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = float(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = float(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
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
            
            # Resize the bounding boxes according to the
            # desired `width`, `height`.
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
        # Area of the bounding boxes.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # Labels to tensor.
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

            # Resize the bounding boxes according to the
            # desired `width`, `height`.
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
        # Area of the bounding boxes.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # Labels to tensor.
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return image, image_resized, orig_boxes, \
            boxes, labels, area, iscrowd, (image_width, image_height)

    def check_image_and_annotation(
        self, 
        xmin, 
        ymin, 
        xmax, 
        ymax, 
        width, 
        height, 
        orig_data=False
    ):
        """
        Enhanced version with STRICT coordinate validation.
        Prevents negative coordinates and out-of-bounds issues.
        """
        # ⭐ CRITICAL: Clamp all coordinates to valid range [0, width/height]
        xmin = max(0.0, min(xmin, width - 1.0))
        ymin = max(0.0, min(ymin, height - 1.0))
        xmax = max(0.0, min(xmax, width))
        ymax = max(0.0, min(ymax, height))
        
        # Ensure max > min (maintain bbox validity)
        if ymax <= ymin:
            ymax = ymin + 1.0
            if ymax > height:
                ymax = height
                ymin = ymax - 1.0
        
        if xmax <= xmin:
            xmax = xmin + 1.0
            if xmax > width:
                xmax = width
                xmin = xmax - 1.0
        
        # Ensure minimum size (at least 1 pixel)
        if xmax - xmin <= 1.0:
            if orig_data and self.log_annot_issue_x:
                # Only log once to avoid spam
                self.log_annot_issue_x = False
            # Expand bbox slightly
            xmin = max(0.0, xmin - 0.5)
            xmax = min(width, xmax + 0.5)
        
        if ymax - ymin <= 1.0:
            if orig_data and self.log_annot_issue_y:
                # Only log once to avoid spam
                self.log_annot_issue_y = False
            # Expand bbox slightly
            ymin = max(0.0, ymin - 0.5)
            ymax = min(height, ymax + 0.5)
        
        # Final safety clamp (ensure no negative values escape)
        xmin = max(0.0, xmin)
        ymin = max(0.0, ymin)
        xmax = min(width, xmax)
        ymax = min(height, ymax)
        
        return xmin, ymin, xmax, ymax


    def load_cutmix_image_and_boxes(self, index, resize_factor=512):
        """ 
        Adapted from: https://www.kaggle.com/shonenkov/oof-evaluation-mixup-efficientdet
        """
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + [random.randint(0, len(self.all_images) - 1) for _ in range(3)]

        # Create empty image with the above resized image.
        # result_image = np.full((h, w, 3), 1, dtype=np.float32)
        result_boxes = []
        result_classes = []

        for i, index in enumerate(indices):
            _, image_resized, orig_boxes, boxes, \
            labels, area, iscrowd, dims = self.load_image_and_labels(
                index=index
            )

            h, w = image_resized.shape[:2]

            if i == 0:
                # Create empty image with the above resized image.
                result_image = np.full((s * 2, s * 2, image_resized.shape[2]), 114/255, dtype=np.float32)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
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

    def __getitem__(self, idx):
        if not self.train: # No mosaic during validation.
            image, image_resized, orig_boxes, boxes, \
                labels, area, iscrowd, dims = self.load_image_and_labels(
                index=idx
            )

        if self.train: 
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

        # Prepare the final `target` dictionary.
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # Before transformation
        labels = labels.cpu().numpy().tolist() if isinstance(labels, torch.Tensor) else labels
        bboxes = target['boxes'].cpu().numpy().tolist() if isinstance(target['boxes'], torch.Tensor) else target['boxes'].tolist()

        if self.use_train_aug: # Use train augmentation if argument is passed.
            train_aug = get_train_aug()
            sample = train_aug(image=image_resized,
                                     bboxes=bboxes,
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes']).to(torch.int64)
        else:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes']).to(torch.int64)

        # Fix to enable training without target bounding boxes,
        # see https://discuss.pytorch.org/t/fasterrcnn-images-with-no-objects-present-cause-an-error/117974/4
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
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
        # shuffle=True,
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
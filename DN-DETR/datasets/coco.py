# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
if __name__=="__main__":
    # for debug only
    import os, sys
    sys.path.append(os.path.dirname(sys.path[0]))

import json
from pathlib import Path
import random
import os

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T
from util.box_ops import box_cxcywh_to_xyxy, box_iou

__all__ = ['build']

import cv2
import smrc.utils


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, aux_target_hacks=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.aux_target_hacks = aux_target_hacks

    def __getitem__(self, idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data. 
        """
        try:
            img, target = super(CocoDetection, self).__getitem__(idx)
        except:
            print("Error idx: {}".format(idx))
            idx += 1
            img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # img_id = target['image_id'].cpu().item()
        # print(f'img_id = {img_id}, ============================')

        # record the area of the box in the original img, instead of the input image, which will be changed in
        # self._transforms
        # if target['image_id'].cpu().item() == 3845:
        #     print(f'Stop')

        # This is wrong, the target['area'] here means the segmentation mask area, not
        # the box area.
        # target['original_area'] = target['area'].clone()

        """ 
        https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
             bb = ann['bbox']
            x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
             ann['area'] = bb[2]*bb[3]
        """
        # coco annotation, x1, y1, w, h, after load, x1, y1, x2, y2.
        target['original_area'] = (target['boxes'][:, 2] - target['boxes'][:, 0]) * \
                                  (target['boxes'][:, 3] - target['boxes'][:, 1])  # w * h

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # self.save_img_ann(img, target)
        return img, target

    def save_img_ann(self, img, target):
        """

        Args:
            img: torch.Size([3, 512, 682])
            target:
                'boxes' = {Tensor: 7} tensor([[0.6380, 0.7271, 0.3703, 0.3887],\n        [0.3042, 0.4869, 0.0415, 0.0573],\n        [0.3713, 0.5150, 0.0577, 0.1279],\n        [0.4509, 0.9541, 0.0517, 0.0917],\n        [0.3688, 0.9701, 0.0402, 0.0561],\n        [0.4517, 0.9695, 0.0315, 0.0610],\n
                'labels' = {Tensor: 7} tensor([ 7,  7,  7,  1,  1, 27,  1])
                'image_id' = {Tensor: 1} tensor([553030])
                'area' = {Tensor: 7} tensor([30096.4785,   594.6974,  2135.0161,   970.2112,   649.2836,   435.8598,\n           95.1035])
                'iscrowd' = {Tensor: 7} tensor([0, 0, 0, 0, 0, 0, 0])
                'orig_size' = {Tensor: 2} tensor([480, 640])
                'size' = {Tensor: 2} tensor([512, 682])
        Returns:

        """

        out_dir = '/disks/cnn1/kaikai/project/DN-DETR/visualize'

        img_id = target['image_id'].cpu().item()

        mean = img.new_tensor([0.485, 0.456, 0.406])
        std = img.new_tensor([0.229, 0.224, 0.225])  #
        img = img * std[:, None, None] + mean[:, None, None]  # torch.Size([3, 512, 682])
        ori_img = img.permute(1, 2, 0).cpu().numpy() * 255

        box_img = ori_img.copy()
        img_target = target
        input_img_size = img_target['size']
        h, w = input_img_size[0].item(), input_img_size[1]
        ratio = torch.tensor([w, h, w, h], dtype=torch.float32)
        num_box = len(img_target['boxes'])
        #   box_ops.box_cxcywh_to_xyxy(src_boxes),
        if num_box > 0:  # tensor([32.0000, 31.6452], device='cuda:0')
            # aspect_ratio = padded_img_size / feat_map_size  # padded input image / padded feat map
            print(f'img_id = {img_id}, coco target["boxes"] = {img_target["boxes"]}, ratio = {ratio}')
            box_unnormalized = img_target['boxes'] * ratio.unsqueeze(dim=0).repeat(num_box, 1)
            box_unnormalized = torch.floor(torch.stack(
                [box_unnormalized[:, 0] - box_unnormalized[:, 2] / 2,
                 box_unnormalized[:, 1] - box_unnormalized[:, 3] / 2,
                 box_unnormalized[:, 0] + box_unnormalized[:, 2] / 2 + 1,
                 box_unnormalized[:, 1] + box_unnormalized[:, 3] / 2 + 1],  # + 0.5
                dim=-1)).int()  # cx, cy, w, h to x1, y1, x2, y2

            print(f'coco target["boxes"] = {box_unnormalized}')
            for box, box_area in zip(box_unnormalized, img_target['area']):
                # h, w
                x1, y1, x2, y2 = box
                box_img = cv2.rectangle(box_img, (int(x1), int(y1)), (int(x2), int(y2)), smrc.utils.YELLOW, 4)

            img_path = os.path.join(out_dir, f'{img_id}_coco.jpg')
            cv2.imwrite(img_path, box_img)


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]
    
    # # update args from config files
    # scales = getattr(args, 'data_aug_scales', scales)
    # max_size = getattr(args, 'data_aug_max_size', max_size)
    # scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    # scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    # # resize them
    # data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
    # if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
    #     data_aug_scale_overlap = float(data_aug_scale_overlap)
    #     scales = [int(i*data_aug_scale_overlap) for i in scales]
    #     max_size = int(max_size*data_aug_scale_overlap)
    #     scales2_resize = [int(i*data_aug_scale_overlap) for i in scales2_resize]
    #     scales2_crop = [int(i*data_aug_scale_overlap) for i in scales2_crop]


    # datadict_for_print = {
    #     'scales': scales,
    #     'max_size': max_size,
    #     'scales2_resize': scales2_resize,
    #     'scales2_crop': scales2_crop
    # }
    # print("data_aug_params:", json.dumps(datadict_for_print, indent=2))
        

    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                normalize,
            ])

        if strong_aug:
            import datasets.sltransform as SLT
            
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    # SLT.Rotate(10),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),              
                # # for debug only  
                # SLT.RandomCrop(),
                # SLT.LightingNoise(),
                # SLT.AdjustBrightness(2),
                # SLT.AdjustContrast(2),
                # SLT.Rotate(10),
                normalize,
            ])
        
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set in ['val', 'test', 'train_val']:  # ['val', 'test']

        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ])   

        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])



    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    # assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "train_reg": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "eval_debug": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test": (root / "test2017", root / "annotations" / 'image_info_test-dev2017.json'),

        "train_val": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
    }

    # add some hooks to datasets
    aux_target_hacks_list = None
    img_folder, ann_file = PATHS[image_set]

    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False

    try:
        fix_size = args.fix_size
    except:
        fix_size = False
    # dataset = CocoDetection(img_folder, ann_file,
    #         transforms=make_coco_transforms(image_set, fix_size=fix_size, strong_aug=strong_aug, args=args),
    #         return_masks=args.masks,
    #         aux_target_hacks=aux_target_hacks_list,
    #     )
    dataset = CocoDetection(img_folder, ann_file,
            transforms=make_coco_transforms(image_set, fix_size=fix_size, strong_aug=strong_aug, args=args),
            return_masks=args.masks or args.load_masks,
            aux_target_hacks=aux_target_hacks_list,
        )
    return dataset


# leaf-only SAM
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import json

# sam1 libs
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# sam2 libs
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

from torchvision.ops import masks_to_boxes
from pycocotools import mask as maskUtils

from PIL import Image
from typing import Annotated, Literal, List, Dict, Optional, Tuple, Any

def checkcolour(masks, hsv):
    colours = np.zeros((0,3))

    for i in range(len(masks)):
        color = hsv[masks[i]['segmentation']].mean(axis=(0))
        colours = np.append(colours,color[None,:], axis=0)
        
    idx_green = (colours[:,0]<75) & (colours[:,0]>35) & (colours[:,1]>35)
    if idx_green.sum()==0:
        # grow lights on adjust
        idx_green = (colours[:,0]<100) & (colours[:,0]>35) & (colours[:,1]>35)
    
    return(idx_green)

def checkfullplant(masks):
    mask_all = np.zeros(masks[0]['segmentation'].shape[:2])

    for mask in masks:
        mask_all +=mask['segmentation']*1
        
    iou_withall = []
    for mask in masks:
        iou_withall.append(iou(mask['segmentation'], mask_all>0))
        
    idx_notall = np.array(iou_withall)<0.9
    return idx_notall

def getbiggestcontour(contours):
    nopoints = [len(cnt) for cnt in contours]
    return(np.argmax(nopoints))

def checkshape(masks):
    cratio = []

    for i in range(len(masks)):
        test_mask = masks[i]['segmentation']
        
        if not test_mask.max():
            cratio.append(0)
        else:

            contours,hierarchy = cv2.findContours((test_mask*255).astype('uint8'), 1, 2)

            # multiple objects possibly detected. Find contour with most points on it and just use that as object
            cnt = contours[getbiggestcontour(contours)]
            M = cv2.moments(cnt)

            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt,True)

            (x,y),radius = cv2.minEnclosingCircle(cnt)

            carea = np.pi*radius**2

            cratio.append(area/carea)
    idx_shape = np.array(cratio)>0.1
    return(idx_shape)

def iou(gtmask, test_mask):
    intersection = np.logical_and(gtmask, test_mask)
    union = np.logical_or(gtmask, test_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return (iou_score)

def issubset(mask1, mask2):
    # is mask2 subpart of mask1
    intersection = np.logical_and(mask1, mask2)
    return(np.sum(intersection)/mask2.sum()>0.9)

def istoobig(masks):
    idx_toobig = []
    
    mask_all = np.zeros(masks[0]['segmentation'].shape[:2])

    for mask in masks:
        mask_all +=mask['segmentation']*1 

    for idx in range(len(masks)):
        if idx in idx_toobig:
            continue
        for idx2 in range(len(masks)):
            if idx==idx2:
                continue
            if idx2 in idx_toobig:
                continue
            if issubset(masks[idx2]['segmentation'], masks[idx]['segmentation']):
                # check if actually got both big and small copy delete if do
                if mask_all[masks[idx2]['segmentation']].mean() > 1.5:
                
                    idx_toobig.append(idx2)
    
    idx_toobig.sort(reverse=True)        
    return(idx_toobig)

def remove_toobig(masks, idx_toobig):
    masks_ntb = masks.copy()

    idx_del = []
    for idxbig in idx_toobig[1:]:
        maskbig = masks_ntb[idxbig]['segmentation'].copy()
        submasks = np.zeros(maskbig.shape)

        for idx in range(len(masks_ntb)):
            if idx==idxbig:
                continue
            if issubset(masks_ntb[idxbig]['segmentation'], masks_ntb[idx]['segmentation']):
                submasks +=masks_ntb[idx]['segmentation']

        if np.logical_and(maskbig, submasks>0).sum()/maskbig.sum()>0.9:
            # can safely remove maskbig
            idx_del.append(idxbig)
            del(masks_ntb[idxbig])
            
    return(masks_ntb)

def create_coco_json(outputs,
                     raw_images,
                     image_urls,
                     label2id):
    # binary masks to coco format
    coco_images = []
    coco_preds = []
    label_id = 0
    for i, output in enumerate(outputs):
        # img info
        coco_images.append({
                #  "license": -1, 
                #  "flickr_url": "", 
                #  "coco_url": "", 
                #  "date_captured": ""
                 "id": i,
                 "width": raw_images[i].size[0], 
                 "height": raw_images[i].size[1],
                 "file_name": image_urls[i], 
                 })
        # masks = output['segmentation'].cpu().to(torch.uint8)
        # masks_info = output['segments_info']
        if len(output) > 0:
            for j, single_mask in enumerate(output):
                seg_mask = torch.from_numpy(single_mask['segmentation']).to(torch.uint8)
                bbox = masks_to_boxes(seg_mask.unsqueeze(0))
                bbox = bbox.tolist()[0] # [x,y,x,y]
                bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

                # mask
                area = float(seg_mask.count_nonzero().item())
                coco_mask = maskUtils.encode(np.asarray(seg_mask, order='F')) # rle
                coco_mask['counts'] = coco_mask['counts'].decode('utf-8') # to store in json

                coco_preds.append({
                    "id": label_id,
                    "image_id": i,
                    "category_id": 0,
                    "segmentation": coco_mask,
                    "area": area,
                    # "bbox": bbox_xywh,
                    "score": single_mask['stability_score'],
                    "iscrowd": 0,
                })
                label_id += 1
    coco_json = {
        # "info": {},
        # "licenses": [],
        "images": coco_images,
        "annotations": coco_preds,
        "categories": [{"id": int(v), "name": k} for k, v in label2id.items()],
    }

    return coco_json

def infer_leaf_only_sam(
        image_urls: Annotated[Optional[List[str]], "List of image paths"] = None,
        file_path: Annotated[Optional[str], "Path to CSV/JSON file with a 'file_name' column/key containing image URLs"] = None,
        device: Annotated[str, "Device to use for inference ('cuda' or 'cpu')."] = "cuda",
        output_dir: Annotated[str, "Directory path to save results."] = "./results",
        sam_version: Annotated[str, "SAM version."] = "sam2", # sam or sam2
        ) -> str:
    """
    Perform leaf instance segmentation on a list of images using Leaf_Only_SAM.
    Please provide either image_urls or file_path as input.

    Returns:
    - str: Path to a JSON file containing instance segmentation results in COCO format.
    """
    if (image_urls and file_path) or (not image_urls and not file_path):
        return "Error: Provide either 'image_urls' or 'file_path', but not both."

    if file_path:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            if 'file_name' not in df.columns:
                return "Error: CSV file must contain a 'file_name' column."
            image_urls = df['file_name'].tolist()
        elif file_path.endswith(".json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
            if 'file_name' not in data:
                return "Error: JSON file must contain a 'file_name' key."
            image_urls = data['file_name']
        else:
            return "Error: Unsupported file format. Use CSV or JSON."

    if not image_urls:
        return "Error: No image URLs found."

    # build sam
    if sam_version == "sam":
        sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=200,  
        )
    elif sam_version == "sam2":
        sam_checkpoint = "./models//sam2.1_hiera_large.pt"
        model_cfg = ".models/sam2.1_hiera_l.yaml"
        sam = build_sam2(model_cfg, sam_checkpoint, device=device, apply_postprocessing=False)
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=200,  
        )
    else:
        raise ValueError(f"Invalid sam_version: {sam_version}")
    
    masks_list = []
    masks_g_list = []
    masks_na_list = []
    masks_s_list = []

    masks_ntb_list = []
    raw_images = []

    # downsize image to fit on gpu easier
    for imname in image_urls:
        print(imname)
        image = (Image.open(imname)).convert('RGB')
        raw_images.append(image)
        image = np.array(image)
        # image = cv2.imread(imname)
        image_w, image_h = image.shape[0], image.shape[1]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fx, fy = 0.5, 0.5
        image = cv2.resize(image,None,fx=fx,fy=fy)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # get masks
        masks = mask_generator.generate(image) # binary masks
        # remove things that aren't green enough to be leaves
        idx_green = checkcolour(masks,hsv)
        masks_g = []
        for idx, use in enumerate(idx_green):
            if use:
                masks_g.append(masks[idx])
        if len(masks_g) > 2:
            # check to see if full plant detected and remove
            idx_notall = checkfullplant(masks_g)
            masks_na = []
            for idx, use in enumerate(idx_notall):
                if use:
                    masks_na.append(masks_g[idx])
        else:
            masks_na = masks_g # remove whole plant
        idx_shape = checkshape(masks_na)

        masks_s = []
        for idx, use in enumerate(idx_shape):
            if use:
                masks_s.append(masks_na[idx])

        idx_toobig = istoobig(masks_s)
        masks_ntb = remove_toobig(masks_s, idx_toobig) # remove too big leaves 
        
        # save results at each step as npz file 
        # np.savez(folder_out + imname.replace('.JPG','leafonly_allmasks.npz'),
        #           masks, masks_g, masks_na, masks_s, masks_ntb)
        
        if fx * fy != 1:
            for mask in masks_ntb:
                mask['segmentation'] = cv2.resize(
                    mask['segmentation'].astype(np.uint8), 
                    (image_h, image_w), 
                    interpolation=cv2.INTER_NEAREST
                )

        # masks_list.append(masks)
        # masks_g_list.append(masks_g)
        # masks_na_list.append(masks_na)
        # masks_s_list.append(masks_s)
        masks_ntb_list.append(masks_ntb)
    
    # turn to coco format
    coco_json = create_coco_json(masks_ntb_list, raw_images, image_urls, label2id={"leaf": "0"})
    save_path = os.path.join(output_dir, f"leaf_only_{sam_version}_results.json")
    with open(save_path, 'w') as f:
        json.dump(coco_json, f)

    # return masks_list, masks_g_list, masks_na_list, masks_s_list, masks_ntb_list
    # return masks_ntb_list
    # return coco_json
    return f"The instance segmentation results are saved at {save_path} using COCO format."
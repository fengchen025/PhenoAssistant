import numpy as np
import cv2
from pycocotools import mask as maskUtils
from scipy.spatial import ConvexHull, distance
from typing import Annotated
import json
import pandas as pd

def compute_phenotypes(image, annotations, pixel_to_cm):
    '''compute phenotypes for one image'''
    leaf_count = len(annotations)
    leaf_areas = [ann['area'] * (pixel_to_cm**2) for ann in annotations]
    average_leaf_area = np.mean(leaf_areas) if leaf_areas else 0
    
    # get foreground mask
    mask_shape = annotations[0]['segmentation']['size'] # [530, 500]
    combined_mask = np.zeros(mask_shape, dtype=np.uint8)
    for ann in annotations:
        binary_mask = maskUtils.decode(ann['segmentation'])
        combined_mask = np.logical_xor(combined_mask, binary_mask).astype(np.uint8)
        # combined_mask = np.maximum(combined_mask, binary_mask)
    
    pla = np.count_nonzero(combined_mask) * (pixel_to_cm**2) # projected leaf area

    # contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # n x 1 x 2
    contour_points = np.vstack([contour.squeeze() for contour in contours])  # n x 2
    contour_points = np.unique(contour_points, axis=0)  # Remove duplicates

    perimeter = len(contour_points) * pixel_to_cm
    diameter = np.max(distance.pdist(contour_points)) * pixel_to_cm

    points = contour_points
    if len(points) < 3:
        hull_area = 0
        hull_perimeter = 0
    else:
        hull = ConvexHull(points)
        hull_area = hull.volume * (pixel_to_cm**2)
        hull_perimeter = hull.area * pixel_to_cm

    compactness = pla / hull_area if hull_area > 0 else 0
    stockiness = 4 * np.pi * pla / (perimeter**2) if perimeter > 0 else 0

    return {
        'file_name': image['file_name'],
        'leaf_count': leaf_count,
        'average_leaf_area': average_leaf_area,
        'projected_leaf_area': pla,
        'diameter': diameter,
        'perimeter': perimeter,
        'compactness': compactness,
        'stockiness': stockiness,
    }

def compute_phenotypes_from_ins_seg(ins_seg_result_path: Annotated[str, "Path to a COCO format JSON file containing instance segmentation results."],
                                    save_path: Annotated[str, "Path to save the computed phenotypes."],
                                    pixel_to_cm: Annotated[float, "The scale to map pixel values to cm. i.e. number of pixels * pixel_to_cm = cm."] = 1.) -> str:
    """
    Computes phenotypes from an instance segmentation result file (COCO JSON format).
    Returns:
        str: A message indicating the path where the computed phenotypes are saved.  
        The phenotypes are saved as a list of dictionaries containing the computed phenotypes, where each dictionary inlcudes the following keys:
            - file_name (str): The file name of the image.
            - leaf_count (int): The number of leaves in the image.
            - average_leaf_area (float): The average area of the leaves in the image.
            - projected_leaf_area (float): The projected leaf area of the plant.
            - diameter (float): The diameter of the plant.
            - perimeter (float): The perimeter of the plant.
            - compactness (float): The compactness of the plant.
            - stockiness (float): The stockiness of the plant.
    """
            # - average_hue (float): The average hue of the plant.
    with open(ins_seg_result_path, "r") as f:
        results = json.load(f)
    phenotypes = []
    for image in results['images']:
        image_id = image['id']
        annotations = [ann for ann in results['annotations'] if ann['image_id'] == image_id]
        if annotations:
            phenotypes.append(compute_phenotypes(image, annotations, pixel_to_cm))
        else:
            phenotypes.append({
                'file_name': image['file_name'],
                'leaf_count': 0,
                'average_leaf_area': 0.,
                'projected_leaf_area': 0.,
                'diameter': 0.,
                'perimeter': 0.,
                'compactness': 0.,
                'stockiness': 0.,
                })
    # return phenotypes
    
    if save_path.endswith('.json'):
        with open(save_path, "w") as f:
            json.dump(phenotypes, f)
    elif save_path.endswith('.csv'):
        df = pd.DataFrame(phenotypes)
        df.to_csv(save_path, index=False)  # index=False prevents adding an extra index column
    else:
        return "Error: The save_path must end with '.json' or '.csv'."

    # return f'''The computed phenotypes are saved to {save_path} as a list of dictionaries, 
    # each containing the following keys: 'file_name', 'leaf_count', 'average_leaf_area', 'projected_leaf_area', 'diameter', 'perimeter', 'compactness', 'stockiness', 'average_hue'.'''
    return f'''The computed phenotypes are saved to {save_path} as a list of dictionaries, 
    each containing the following keys: 'file_name', 'leaf_count', 'average_leaf_area', 'projected_leaf_area', 'diameter', 'perimeter', 'compactness', 'stockiness'.'''
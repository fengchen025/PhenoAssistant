import os
import sys
import glob
import json
import copy
import random
import requests
from typing import Annotated, Literal, List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging

from functions.instance_segmentation import infer_instance_segmentation
from functions.image_classification import finetune_image_classification, infer_image_classification
from functions.create_hf_dataset import prepare_dataset, get_dataset_format
from functions.search import search_and_scrape
from functions.compute_phenotypes import compute_phenotypes_from_ins_seg
from functions.generic_tools import (
    set_env_vars, 
    set_random_seed, 
    print_trainable_parameters, 
    handle_grayscale_image, 
    download_hffile, 
    load_images, 
    get_model_zoo,
    calculator,
    make_dir,
    extract_column_name_from_csv,
)


def ara_crop_pipeline(metadata_path: Annotated[str, 'Path to the metadata JSON file'], 
                      output_dir: Annotated[str, 'Directory to save the results'], 
                      pixel_to_cm: Annotated[float, 'Pixel to cm mapping scale'] = 0.03) -> Dict[str, Any]:
    """
    Pipeline to compute phenotypes for Arabidopsis plant images, merge with metadata, and save the results.
    Args:
        metadata_path (str): Path to the metadata JSON file.
        output_dir (str): Directory to save the results.
        pixel_to_cm (float): Pixel to cm mapping scale. Default is 0.03.
    Returns:
        Dict[str, Any]: Status and path to the saved results.
    """

    import json
    import pandas as pd

    try:
        logging.info("Starting ara_crop_pipeline execution.")

        # Step 1: Check available checkpoints for instance segmentation
        logging.info("Checking available checkpoints for instance segmentation.")
        checkpoints = get_model_zoo()
        selected_checkpoint = "fengchen025/arabidopsis_leaf-instance-segmentation_cvppp2017-a1a4_m2fb_fullft"

        # Step 2: Perform instance segmentation
        logging.info("Performing instance segmentation.")
        infer_instance_segmentation(file_path=metadata_path, checkpoint=selected_checkpoint, output_dir=output_dir)

        # Step 3: Compute phenotypes from instance segmentation results
        logging.info("Computing phenotypes from instance segmentation results.")
        ins_seg_result_path = f"{output_dir}/ins_seg_results.json"
        phenotypes_save_path = f"{output_dir}/phenotypes.csv"
        compute_phenotypes_from_ins_seg(ins_seg_result_path=ins_seg_result_path, save_path=phenotypes_save_path, pixel_to_cm=pixel_to_cm)

        # Step 4: Merge computed phenotypes with metadata
        logging.info("Merging computed phenotypes with metadata.")
        phenotypes_df = pd.read_csv(phenotypes_save_path)
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
        metadata_df = pd.DataFrame(metadata)
        merged_df = pd.merge(phenotypes_df, metadata_df, on='file_name')
        final_save_path = f"{output_dir}/aracrop_phenotypes.csv"
        merged_df.to_csv(final_save_path, index=False)

        logging.info("Pipeline executed successfully.")
        return {"status": "success", "result_path": final_save_path}

    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

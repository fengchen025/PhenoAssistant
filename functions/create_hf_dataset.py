import os
import json
from datasets import Dataset, Features, ClassLabel, Value, Image, DatasetDict, DatasetInfo
from PIL import Image as PILImage
from huggingface_hub import HfApi, HfFolder, login, model_info, dataset_info, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
import glob
import pandas as pd
from typing import Annotated, List, Optional, Union

'''
dataset structure:
root/
│
├── train/ # folder containing training images and labels (in metadata.csv)
│   ├── metadata.csv # for classification, must have "file_name" and "label" fields, for instance segmentation, must have "file_name" and "annotation" fields, for regression, must have "file_name" field
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
│
├── test/ # (Optional) folder containing test (not validation) images and labels (in metadata.csv)
│   ├── metadata.csv # must have "file_name" field
│   ├── 1.jpg
│   ├── 2.jpg   
│   └── ...
│
├── label2id.json # must have for instance segmentation and classification
'''

DATASET_FORMAT_CLASS='''
dataset_name/
│
├── train/ # folder containing training images and labels
│   ├── metadata.csv # must include columns of "file_name" (image file names) and "label" (corresponding label names for each image)
│   ├── 1.jpg # training image 1
│   ├── 2.jpg # training image 2
│   └── ...
│
├── test/ # (Optional) folder containing test images and labels
│   ├── metadata.csv # must include "file_name" column
│   ├── 1.jpg # test image 1
│   ├── 2.jpg # test image 2  
│   └── ...
│
├── label2id.json # dictionary mapping label names to numerical IDs, e.g., {"Healthy": 0, "Unhealthy": 1}
'''

DATASET_FORMAT_REG='''
dataset_name/
│
├── train/ # folder containing training images and corresponding regression values
│   ├── metadata.csv # must include columns of "file_name" (image file names) and "{regression_property}" (corresponding regression values for each image)
│   ├── 1.jpg # training image 1
│   ├── 2.jpg # training image 2
│   └── ...
│
├── test/ # (Optional) folder containing test images and labels
│   ├── metadata.csv # must include "file_name" column
│   ├── 1.jpg # test image 1
│   ├── 2.jpg # test image 2  
│   └── ...
'''

DATASET_FORMAT_INSSEG='''
dataset_name/
│
├── train/ # folder containing training images and segmentation masks
│   ├── metadata.csv # must include columns of "file_name" (image file names) and "annotation" (corresponding segmentation mask file names for each image)
│   ├── 1.jpg # training image 1
│   ├── 2.jpg # training image 2
│   └── ...
│
├── test/ # (Optional) folder containing test images and labels
│   ├── metadata.csv # must include "file_name" column
│   ├── 1.jpg # test image 1
│   ├── 2.jpg # test image 2  
│   └── ...
│
├── label2id.json # dictionary mapping segmentation categories to numerical IDs, e.g., {"Leaf": 0, "Stem": 1}
'''

def get_dataset_format(task: Annotated[str, "Type of supported task: 'image-classification', 'instance-segmentation', or 'image-regression'."]) -> str:
    '''
    Return the dataset structure requirement for preparing training datasets for different tasks.
    '''
    if task == 'image-classification':
        return DATASET_FORMAT_CLASS
    elif task == 'instance-segmentation':
        return DATASET_FORMAT_INSSEG
    elif task == 'image-regression':
        return DATASET_FORMAT_REG
    else:
        raise ValueError(f"task {task} not supported")

def parse_metadata(image_folder, 
                   metadata_path,):
    metadata = pd.read_csv(metadata_path)
    if 'file_name' not in metadata.columns:
        raise ValueError("file_name not found in metadata")
    metadata_dict = (metadata.rename(columns={"file_name": "image"})).to_dict('records')
    # check absolute path
    for entry in metadata_dict:
        if len(entry['image'].split('/')) == 1:
            entry['image'] = os.path.join(image_folder, entry['image'])
        if "annotation" in entry.keys(): # instance segmentation
            if len(entry['annotation'].split('/')) == 1:
                entry['annotation'] = os.path.join(image_folder, entry['annotation'])

    return Dataset.from_list(metadata_dict)

def prepare_dataset(dataset_name: Annotated[str, "Name of dataset to be uploaded to HuggingFace."], 
                    root: Annotated[str, "Root directory where the dataset is stored."], 
                    task: Annotated[str, "Type of supported task: 'classification', 'instance segmentation', or 'regression'."]) -> str:
    '''
    Prepares a dataset for training vision models. The dataset will be uploaded to HuggingFace.
    Currently only support classification, instance segmentation, and regression tasks.
    Returns: str: a message indicating the dataset has been successfully uploaded, with the dataset name.
    '''

    if task not in ['classification', 'instance segmentation', 'regression']:
        raise ValueError(f"task {task} not supported")
    
    user_name = os.getenv('HF_USER')
    try:
        dataset_info(f'{user_name}/{dataset_name}')
        return f"Dataset '{dataset_name}' exists. You can proceed to train models with it."
    except HfHubHTTPError:
        print(f"Dataset '{dataset_name}' does not exist. Proceed to create the dataset.")

    # load files
    train_image_folder = os.path.join(root, "train")
    train_metadata_path = os.path.join(train_image_folder, "metadata.csv")
    # check all the above must exist, otherwise raise error
    for data_path in [train_image_folder, train_metadata_path]:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found")
    if task == 'classification' or task == 'instance segmentation':
        label2id_path = os.path.join(root, "label2id.json")
        if not os.path.exists(label2id_path):
            raise FileNotFoundError(f"{data_path} not found")
        with open(label2id_path, "r") as f:
            label2id = json.load(f)
    
    # optional test set
    test_image_folder = os.path.join(root, "test")
    test_metadata_path = os.path.join(test_image_folder, "metadata.csv")
    if not os.path.exists(test_image_folder):
        test_image_folder = None
        test_metadata_path = None
    else:
        if not os.path.exists(test_metadata_path):
            raise FileNotFoundError(f"{test_metadata_path} not found")

    # parsing metadata
    train_dataset = parse_metadata(image_folder = train_image_folder,
                                   metadata_path = train_metadata_path)
    if test_image_folder:
        test_dataset = parse_metadata(image_folder = test_image_folder,
                                      metadata_path = test_metadata_path)
    else:
        test_dataset = None
    
    # Define features for the datasets
    features = Features({})
    if task == 'classification':
        features = Features({
            "image": Image(decode=True),  # Image feature
            "label": ClassLabel(names=list(label2id.keys()))  # Class labels
            })
    elif task == 'instance segmentation':
        features = Features({
            "image": Image(decode=True),  # Image feature
            "annotation": Image(decode=True)  # Annotation
            })
    elif task == 'regression':
        features = Features({
            "image": Image(decode=True)  # Image feature
            })
        for feature in train_dataset.features.keys():
            if feature != 'image':
                features[feature] = Value('float32')
                # features[feature] = Value(train_dataset.features[feature].dtype)

    # Cast the datasets with the appropriate features
    train_dataset = train_dataset.cast(features)
    train_dataset.info.description = f"{task} on {dataset_name}."
    # the following are already in dataset.features['label'].names for classification
    if task == 'instance segmentation':
        train_dataset = train_dataset.map(lambda x: {'label2id': label2id}, remove_columns=[])

    if test_dataset:
        # add missing values as -1 (features of train/test should be the same for push_to_hub)
        for feature in features.keys():
            if feature != 'image' and feature not in test_dataset.column_names:
                test_dataset = test_dataset.map(lambda x: {feature: -1}, 
                                                remove_columns=[])
        test_dataset = test_dataset.cast(features)
        if task == 'instance segmentation':
            test_dataset = test_dataset.map(lambda x: {'label2id': label2id}, remove_columns=[])
        # Combine into a DatasetDict
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })
    else:
        dataset_dict = DatasetDict({
            "train": train_dataset
        })

    # Push the dataset to Hugging Face Hub
    dataset_dict.push_to_hub(dataset_name, private=True)
    
    # we return a dataset_dict that always have a 'train' split
    # return dataset_dict
    return f"The dataset has been uploaded to HuggingFace as '{dataset_name}'."

# example usage
if __name__ == "__main__":
    # dataset_name should be {plant species}_{task}_{personal/dataset identifier}
    dataset_root = "/home/fchen2/RDS/codes/PhenoAssistantX_datasets/"
    # ins seg
    dataset_name = "cvppp2017-a1a4"
    dataset_dict = prepare_dataset(dataset_name=dataset_name, root=os.path.join(dataset_root, dataset_name), task="instance segmentation")
    
    # # regression
    # dataset_name = "arabidopsis_leaf-counting_cvppp2017-a1"
    # dataset_dict = prepare_dataset(dataset_name=dataset_name, root=os.path.join(dataset_root, dataset_name), task="regression")
    
    # # classification
    # dataset_name = "winter-wheat_fertiliser-identification_dnd-ww2020"
    # dataset_dict = prepare_dataset(dataset_name=dataset_name, root=os.path.join(dataset_root, dataset_name), task="classification")


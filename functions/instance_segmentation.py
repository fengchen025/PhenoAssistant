import os
import glob
import json
import copy
import random
import requests
from typing import Annotated, Literal, List, Dict, Optional, Tuple, Any, Union
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

import datasets
from datasets import load_dataset
import accelerate
import evaluate
import albumentations as A

import transformers
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    AutoConfig,
    AutoModelForImageSegmentation, 
    AutoModelForObjectDetection,
    Mask2FormerImageProcessor, 
    Mask2FormerModel, 
    Mask2FormerForUniversalSegmentation, 
    Mask2FormerConfig,
    PretrainedConfig,
    TrainingArguments, 
    Trainer,
)

import peft
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model
from huggingface_hub import login, HfApi, hf_hub_download
import io

from torchvision.ops import masks_to_boxes
from pycocotools import mask as maskUtils

# My functions
from .metrics import InsSegEvaluator
from .generic_tools import (
    set_env_vars, 
    set_random_seed, 
    print_trainable_parameters, 
    handle_grayscale_image, 
    download_hffile, 
    load_images, 
    update_model_zoo,
)
from .leaf_only_sam import infer_leaf_only_sam

class InsSegDataset(Dataset):
    def __init__(self, dataset, processor, transform=None):
        # Initialize the dataset, processor, and transform variables
        self.dataset = dataset
        self.processor = processor
        self.transform = transform
        
    def __len__(self):
        # Return the number of datapoints
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # image = np.array(handle_grayscale_image(self.dataset[idx]["image"]).convert("RGB"))
        image = np.array(self.dataset[idx]["image"].convert("RGB"))
        annot = np.array(self.dataset[idx]["annotation"])
        if annot.ndim == 2: # one class only
            annot = np.stack((np.uint8(annot > 0), annot), axis=-1)
        instance_seg = annot[..., 1]
        class_id_map = annot[..., 0] # e.g. 0 is background, 1 is leaf
        class_labels = np.unique(class_id_map)
        # Build the instance to class dictionary
        inst2class = {}
        for label in class_labels:
            instance_ids = np.unique(instance_seg[class_id_map == label])
            inst2class.update({i: label for i in instance_ids}) # e.g. {0: 0, 1: 1, 2: 1, ...}
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, 
                                         mask=instance_seg)
            (image, instance_seg) = (transformed["image"], transformed["mask"])
            
            # Convert from channels last to channels first
            image = image.transpose(2,0,1)
        
        if class_labels.shape[0] < 2: # i.e. only background
            # If the image has no objects then it is skipped
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k:v.squeeze() for k,v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros(
                (0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1])
            )
        else:
            # Else use process the image with the segmentation maps
            inputs = self.processor(
                [image],
                [instance_seg],
                instance_id_to_semantic_id=inst2class,
                return_tensors="pt"
            )
            inputs = {
                k:v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()
            }
        # Return the inputs
        return inputs
    
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_mask = torch.stack([example["pixel_mask"] for example in examples])
    mask_labels = [example["mask_labels"] for example in examples]
    class_labels = [example["class_labels"] for example in examples]
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels
    }

def finetune_instance_segmentation(
        dataset_name: Annotated[str, "Dataset name for finetuning."],
        ft_method: Annotated[Literal['lora', 'fullft'], "Finetuning method ('lora' or 'fullft')."] = 'lora',
        train_bs: Annotated[int, "Training batch size."] = 32,
        val_bs: Annotated[int, "Validation batch size."] = 1,
        img_w: Annotated[int, "Image width for training."] = 448,
        img_h: Annotated[int, "Image height for training."] = 448,
        lr: Annotated[float, "Learning rate for training."] = 1e-4,
        epochs: Annotated[int, "Number of training epochs."] = 1,
        ) -> str:
    '''
    Finetune a vision model for instance segmentation on a custom dataset.
    Returns:
    - str: Best validation results achieved during training.
    '''

    # pretrained_model = "facebook/mask2former-swin-base-coco-instance"
    pretrained_model = "facebook/mask2former-swin-base-IN21k-coco-instance"
    auto_find_batch_size = False
    img_size = (img_w, img_h)
    
    # environment variables
    # set_env_vars('../.env.yaml')
    # set_random_seed()
    # login(token=os.getenv('HF_TOKEN')) # login to Hugging Face
    user_name=os.getenv('HF_USER')
    repo_id = f"{user_name}/{dataset_name}"
    # model_name = '_'.join(('instance-segmentation', ft_method, pretrained_model, dataset_name)) # output_dir
    model_name = '_'.join(('instance-segmentation', ft_method, dataset_name)) # output_dir
    if 'mask2former' in pretrained_model:
        lora_modules_to_save = [
            "model.pixel_level_module.decoder",
            "model.transformer_module",
            "class_predictor",
            # "criterion",
            ]
    else:
        raise ValueError("Only Mask2Former models are supported for now.")
    dataset = load_dataset(repo_id,)['train']
    do_reduce_labels = True
    ignore_index = 255

    raw_label2id = dataset[0]['label2id'] # by default we ignore background, and start labels from 0
    assert int(list(raw_label2id.values())[0]) == 0, "id must start from 0."
    if 'background' in raw_label2id.keys():
        assert int(raw_label2id['background']) == 0, "background must be 0"
        label2id = {k: int(v) - 1 for k, v in raw_label2id.items() if k != 'background'} # remove background class
    else:
        label2id = raw_label2id
    id2label = {v: k for k, v in label2id.items()}

    # load image preprocessor and Mask2FormerModel trained on COCO instance segmentation dataset
    image_processor = AutoImageProcessor.from_pretrained(pretrained_model,
                                                         do_resize = False,
                                                         do_rescale = False,
                                                         do_normalize = False,
                                                         num_labels = len(label2id),
                                                         size = img_size,
                                                         ignore_index = ignore_index,
                                                         do_reduce_labels = do_reduce_labels,
                                                        )
    # saving this for inference
    image_processor.label2id = label2id
    image_processor.id2label = id2label

    # data augmentation
    train_transform = A.Compose([
        A.Resize(width=img_size[0], height=img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=image_processor.image_mean, 
                    std=image_processor.image_std),
    ])
    test_transform = A.Compose([
        A.Resize(width=img_size[0], height=img_size[1]),
        A.Normalize(mean=image_processor.image_mean, 
                    std=image_processor.image_std),
    ])
    splits = dataset.train_test_split(test_size=0.1)
    train_dataset = InsSegDataset(
        splits['train'],
        processor=image_processor,
        transform=train_transform
    )
    val_dataset = InsSegDataset(
        splits['test'],
        processor=image_processor,
        transform=test_transform
    )

    # # Building the training and validation dataloader
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size = train_bs,
    #     shuffle=True,
    #     collate_fn=collate_fn
    # )
    # val_dataloader = DataLoader(
    #     val_dataset,
    #     batch_size = val_bs,
    #     shuffle=False,
    #     collate_fn=collate_fn
    # )

    # build mask2former model and lora
    # model_config = Mask2FormerConfig.from_pretrained(pretrained_model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_model,
        label2id=label2id, # need this to tell the model how many classes to predict
        id2label=id2label,
        ignore_mismatched_sizes=True, # warning on mismatched weights
    )
    print_trainable_parameters(model)

    # model to lora_model
    if ft_method == 'lora': # else full finetune
        config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=["query", "value"],
            # layers_to_transform=["model.pixel_level_module.encoder"],
            lora_dropout=0.1,
            bias="lora_only",
            use_rslora=True,
            modules_to_save=lora_modules_to_save,
        )
        model = get_peft_model(model, config)
        print_trainable_parameters(model)
    
    # metrics
    compute_metrics = InsSegEvaluator(image_processor, id2label, threshold = 0.85, metric_name = "cvppp")
    # compute_metrics = InsSegEvaluator(image_processor, id2label, threshold = 0.85, metric_name = "map")
    training_args = TrainingArguments(
        learning_rate = lr,
        num_train_epochs = epochs,
        auto_find_batch_size = auto_find_batch_size,
        per_device_train_batch_size = train_bs,
        per_device_eval_batch_size = val_bs,
        save_total_limit = 2, # total number of models saved (not incl. best model)
        batch_eval_metrics = True,
        eval_do_concat_batches = False, # check this, default is True
        label_names = ["mask_labels",
                       "class_labels"],
        metric_for_best_model = "sbd", # 'loss' by default
        greater_is_better = True,
        lr_scheduler_type = "constant",

        hub_token = os.getenv('HF_TOKEN'),
        push_to_hub = True,
        hub_private_repo = True,
        output_dir = model_name,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_steps = 5,
        fp16 = True,
        dataloader_num_workers = 0,
        load_best_model_at_end = True,
        disable_tqdm=True,

        ## other useful args
        # remove_unused_columns defult True
        # eval_on_start = True, # make sure the evaluation works before training
        # eval_steps # defualt to logging_steps, but only when xxx = "steps" works
        # fp16_full_eval=True, # for evaluation use fp16 as well
        # dataloader_prefetch_factor = 2,
        # save_safetensors # default is True
        # seed # default is 42
        # include_inputs_for_metrics
        # optim # defaults to "adamw_torch"
        # resume_from_checkpoint # local path to checkpoint
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        tokenizer=image_processor,
    )

    trainer.train()
    # evaluate_results = trainer.evaluate() # this is the best model if load_best_model_at_end = True
    # print(evaluate_results)
    # best_model_checkpoint = trainer.state.best_model_checkpoint
    metric_for_best_model = trainer.args.metric_for_best_model
    best_metric = trainer.state.best_metric

    # Write model card and push to hub
    kwargs = {
        "finetuned_from": pretrained_model,
        "dataset": dataset_name,
        "tags": ["image-segmentation", "instance-segmentation", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    # upload label2id and id2label to hub because mask2former image_processor cannot access it (for inference)
    label2id_json = io.BytesIO(json.dumps(label2id).encode('utf-8'))
    id2label_json = io.BytesIO(json.dumps(id2label).encode('utf-8'))
    api = HfApi()
    api.upload_file(
        path_or_fileobj=label2id_json,
        path_in_repo="label2id.json",
        repo_id=f'{user_name}/{model_name}',
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=id2label_json,
        path_in_repo="id2label.json",
        repo_id=f'{user_name}/{model_name}',
        repo_type="model",
    )

    # update model_zoo
    update_model_zoo('instance-segmentation',
                     f'{user_name}/{model_name}')
    
    # read model card
    model_card = hf_hub_download(f'{user_name}/{model_name}', 'README.md')
    with open(model_card, "r") as f:
        detailed_model_info = f.read()
    
    # return trainer
    # return evaluate_results
    # print (f"Best validation {metric_for_best_model} is {best_metric}.")
    return f"Best validation {metric_for_best_model} is {best_metric}.\n The trained model is saved as {user_name}/{model_name} in model zoo.\nFull details of the trained model can be found in the model card:\n{detailed_model_info}."
    # return trainer

def infer_instance_segmentation(
        image_urls: Annotated[Optional[List[str]], "List of image paths"] = None,
        file_path: Annotated[Optional[str], "Path to CSV/JSON file with a 'file_name' column/key containing image URLs"] = None,
        # checkpoint: Annotated[str, "Segmentation model checkpoint identifier."] = 'fengchen025/arabidopsis_leaf-instance-segmentation_cvppp2017-a1a4_m2fb_fullft',
        checkpoint: Annotated[str, "Segmentation model checkpoint identifier."] = None,
        batch_size: Annotated[int, "Number of images to process per batch."] = 1,
        device: Annotated[str, "Device to use for inference ('cuda' or 'cpu')."] = "cuda",
        # for save files
        output_dir: Annotated[str, "Directory path to save results."] = "./results"
        ) -> str:
    """
    Perform instance segmentation on a list of images using a specified deep learning model.
    Please provide either image_urls or file_path as input.

    Returns:
    - str: Path to a JSON file containing instance segmentation results in COCO format.
    """
    if checkpoint == 'potato_leaf-instance-segmentation_leaf-only-sam':
        return infer_leaf_only_sam(image_urls, file_path, device, output_dir, sam_version='sam')
    elif checkpoint == 'potato_leaf-instance-segmentation_leaf-only-sam2':
        return infer_leaf_only_sam(image_urls, file_path, device, output_dir, sam_version='sam2')

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
    
    dataset_name = image_urls[0].split('/')[-2]
    # save_dir = os.path.join(output_dir, dataset_name)
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ins_seg_results.json")

    # Load model and image processor
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    image_processor.do_normalize = True
    image_processor.do_resize = True # depends
    image_processor.do_rescale = True
    # image_processor.do_reduce_labels = True
    # image_processor.ignore_index = 255
    
    # label2id = image_processor.label2id
    # id2label = image_processor.id2label
    api = HfApi()
    with open(api.hf_hub_download(repo_id=checkpoint, filename="label2id.json", repo_type="model"), 'r') as f:
        label2id = json.load(f)
    with open(api.hf_hub_download(repo_id=checkpoint, filename="id2label.json", repo_type="model"), 'r') as f:
        id2label = json.load(f)

    # model
    if 'lora' in checkpoint:
        config = PeftConfig.from_pretrained(checkpoint)
        pretrained_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            config.base_model_name_or_path,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(pretrained_model, checkpoint, 
                                        #   device_map=device
                                          )
    else:
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=False,
            # device_map=device,
        )

    model.to(device)
    model.eval()
    # batch inference in case of large number of images
    all_images = []
    all_preds = []
    for i in range(0, len(image_urls), batch_size):
        batch_urls = image_urls[i:i+batch_size]
        raw_images = load_images(batch_urls) # return a list of PIL.RGB images
        inputs = image_processor(images=raw_images, return_tensors="pt").to(device)
        with torch.no_grad():
            batch_outputs = model(**inputs)
        batch_outputs = image_processor.post_process_instance_segmentation(batch_outputs, 
                                                                           threshold=.8,
                                                                           target_sizes=[raw_image.size[::-1] for raw_image in raw_images],
                                                                           return_binary_maps=True,
                                                                           )
        all_images.extend(raw_images)
        all_preds.extend(batch_outputs)
    # return all_images, all_preds

    coco_json = create_coco_json(all_preds, all_images, image_urls, label2id)
    
    # save coco_json
    with open(save_path, 'w') as f:
        json.dump(coco_json, f)

    # return coco_json
    # return f"The instance segmentation results of {image_urls} are saved at {save_path} using coco format."
    return f"The instance segmentation results are saved at {save_path} using COCO format."

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

        masks = output['segmentation'].cpu().to(torch.uint8)
        masks_info = output['segments_info']
        if masks_info:
            for j, single_mask in enumerate(masks):
                # # get bounding boxes
                bbox = masks_to_boxes(single_mask.unsqueeze(0))
                bbox = bbox.tolist()[0] # [x,y,x,y]
                bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

                # mask
                area = float(single_mask.count_nonzero().item())
                coco_mask = maskUtils.encode(np.asarray(single_mask, order='F')) # rle
                coco_mask['counts'] = coco_mask['counts'].decode('utf-8') # to store in json

                coco_preds.append({
                    "id": label_id,
                    "image_id": i,
                    "category_id": masks_info[j]['label_id'],
                    "segmentation": coco_mask,
                    "area": area,
                    # "bbox": bbox_xywh,
                    "score": masks_info[j]['score'],
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

def plot_ins_seg_output(output,
                        save_path):
    segmentation_mask = np.zeros((output['segmentation'].shape[-2], 
                                  output['segmentation'].shape[-1], 
                                  3), dtype=np.uint8)
    for segment in output['segments_info']:
        mask = output['segmentation'] == segment['id']
        color = np.random.randint(0, 255, 3)  # Random color for the segment
        segmentation_mask[mask] = color

    pil_mask = Image.fromarray(segmentation_mask)
    pil_mask.save(save_path)

    return save_path
    
def ins_seg_output_to_csv(result_dict,
                          save_path):
    # Flatten the data to handle multiple labels
    csv_dict = []
    keys = result_dict.keys()
    for i, file_name in enumerate(result_dict['file_name']):
        stats_dict = result_dict['output_stats'][i]
        for label_name, stats in stats_dict.items():
            row = {
                'file_name': file_name,
                'label_name': label_name,
                'instance_count': stats['instance_count'],
                'min_area': stats['min_area'],
                'max_area': stats['max_area'],
                'avg_area': stats['avg_area']
            }
            if 'segmentation_result' in keys:
                row['segmentation_result'] = result_dict['segmentation_result'][i]
            csv_dict.append(row)

    # Convert the flattened data to a pandas DataFrame
    df = pd.DataFrame(csv_dict)
    # Save the DataFrame to a CSV file
    df.to_csv(save_path, index=False)

    return save_path

if __name__ == "__main__":
    set_env_vars(env_file='../.env.yaml')
    finetune_instance_segmentation(dataset_name = "cvppp2017-a1a4",
                                   ft_method = 'fullft',
                                   pretrained_model = "facebook/mask2former-swin-base-IN21k-coco-instance",
                                   img_size = (448, 448),
                                   auto_find_batch_size = False,
                                   train_bs = 16,
                                   val_bs = 1,
                                   lr = 1e-4,
                                   epochs = 500,
                                   )
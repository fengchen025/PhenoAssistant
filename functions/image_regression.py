import os
import glob
import json
import copy
import random
import requests
from typing import Annotated, Literal, List, Dict, Optional, Tuple, Any
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
from huggingface_hub import login, HfApi, model_info, dataset_info, hf_hub_download

# My functions
from .generic_tools import (
    set_env_vars, 
    set_random_seed, 
    print_trainable_parameters, 
    handle_grayscale_image, 
    download_hffile, 
    load_images, 
    update_model_zoo,
)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    label_keys = [k for k in examples[0].keys() if k not in ["pixel_values", "image"]]
    labels = torch.stack([torch.tensor([example[k] for example in examples]) for k in label_keys], dim=1)
    return {"pixel_values": pixel_values, "labels": labels}

def finetune_image_regression(
        # dataset_name: Annotated[str, "Dataset name for finetuning."] = "arabidopsis_leaf-counting_cvppp2017-a1",
        dataset_name: Annotated[str, "Dataset name for finetuning."],
        ft_method: Annotated[Literal['lora', 'fullft'], "Finetuning method ('lora' or 'fullft')."] = 'lora',
        train_bs: Annotated[int, "Training batch size."] = 32,
        val_bs: Annotated[int, "Validation batch size."] = 1,
        img_w: Annotated[int, "Image width for training."] = 224,
        img_h: Annotated[int, "Image height for training."] = 224,
        lr: Annotated[float, "Learning rate for training."] = 1e-4,
        epochs: Annotated[int, "Number of training epochs."] = 1,
        ) -> str:
    '''
    Finetune a vision model for image regression on a custom dataset.
    Returns:
    - str: Best validation results achieved during training.
    '''

    pretrained_model = "facebook/dinov2-base"
    auto_find_batch_size = False
    img_size = (img_w, img_h)
    # environment variables
    # set_env_vars('../.env.yaml')
    # set_random_seed()
    # login(token=os.getenv('HF_TOKEN')) # login to Hugging Face
    user_name=os.getenv('HF_USER')
    repo_id = f"{user_name}/{dataset_name}"
    model_name = '_'.join(('image-regression', ft_method, pretrained_model, dataset_name)) # output_dir
    # model_name = '_'.join(('image-regression', ft_method, pretrained_model.split('/')[-1], dataset_name)) # output_dir
    lora_modules_to_save = ["classifier"]

    dataset = load_dataset(repo_id,)['train']
    # get labels
    # alternatively let user create the label2id
    label_dict = {k: v for k, v in dataset.features.items() if k != 'image'}
    label2id = {label: i for i, label in enumerate(label_dict.keys())}
    id2label = {i: label for label, i in label2id.items()}

    image_processor = AutoImageProcessor.from_pretrained(pretrained_model,
                                                        do_resize = False,
                                                        do_rescale = False,
                                                        do_normalize = False,
                                                        do_center_crop = False,
                                                        do_convert_rgb = False,
                                                        size = img_size,
                                                        )
    image_processor.label2id = label2id
    image_processor.id2label = id2label
    
    # if input is RGB/RGBA/P/L, ToTensor will do scaling
    normalize = Normalize(mean=image_processor.image_mean, 
                          std=image_processor.image_std)
    train_transforms = Compose([
        Resize((img_size[0], img_size[1])),
        # RandomResizedCrop
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])
    val_transforms = Compose([
        Resize((img_size[0], img_size[1])),
        ToTensor(),
        normalize,
    ])    

    def preprocess_train(examples):
        """Apply transforms across a batch."""
        examples["pixel_values"] = [train_transforms(image.convert("RGB")) for image in examples["image"]]
        return examples
    
    def preprocess_val(examples):
        """Apply transforms across a batch."""
        examples["pixel_values"] = [val_transforms(image.convert("RGB")) for image in examples["image"]]
        return examples

    splits = dataset.train_test_split(test_size=0.1) # in this dataset the dataset["test"] does not contaion labels
    train_dataset = splits["train"]
    val_dataset = splits["test"]
    train_dataset.set_transform(preprocess_train)
    val_dataset.set_transform(preprocess_val)

    # # Building the training and validation dataloader (for debugging)
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

    model_config = AutoConfig.from_pretrained(pretrained_model,
                                              label2id=label2id,
                                              id2label=id2label,
                                              problem_type='regression',)
    model = AutoModelForImageClassification.from_pretrained(
        pretrained_model,
        config=model_config,
        ignore_mismatched_sizes=True, # warning on mismatched weights
    )
    print_trainable_parameters(model)

    # model to lora_model
    if ft_method == 'lora': # else full finetune
        config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=["query", "value"],
            # layers_to_transform=["xxx"], # where to apply lora
            lora_dropout=0.1,
            bias="lora_only",
            use_rslora=True,
            modules_to_save=lora_modules_to_save,
        )
        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    # using MSE metrics for evaluation (same as loss)
    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        # metric = evaluate.load("accuracy")
        # metric = evaluate.load("r_squared")
        if len(label2id.keys()) == 1:
            metric = evaluate.load("mse")
            return metric.compute(predictions=eval_pred.predictions, references=eval_pred.label_ids)
        else:
            metric = evaluate.load("mse", "multilist")
            return metric.compute(predictions=eval_pred.predictions, references=eval_pred.label_ids, 
                                  multioutput='uniform_average')

    # training
    training_args = TrainingArguments(
        learning_rate = lr,
        num_train_epochs = epochs,
        auto_find_batch_size = auto_find_batch_size,
        per_device_train_batch_size = train_bs,
        per_device_eval_batch_size = val_bs,
        save_total_limit = 1, # total number of models saved (not incl. best model)
        label_names = ["labels"],
        metric_for_best_model = "mse", # 'loss' by default
        greater_is_better = False,
        lr_scheduler_type = "constant",
        # batch_eval_metrics = True, # default is False
        # eval_do_concat_batches = False, # default is True
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
        disable_tqdm = True,

        ## other useful args
        remove_unused_columns = False, # default is True
        # gradient_accumulation_steps=4,
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
        "tags": ["image-regression", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
    
    # update model_zoo
    update_model_zoo('image-regression',
                     f'{user_name}/{model_name}')

    # read model card
    model_card = hf_hub_download(f'{user_name}/{model_name}', 'README.md')
    with open(model_card, "r") as f:
        detailed_model_info = f.read()
    
    # print(f"Best validation {metric_for_best_model} is {best_metric}.")
        # return trainer
    return f"Best validation {metric_for_best_model} is {best_metric}.\n The trained model is saved as {user_name}/{model_name} in model zoo.\nFull details of the trained model can be found in the model card:\n{detailed_model_info}."

def infer_image_regression(
        image_urls: Annotated[Optional[List[str]], "List of image paths"] = None,
        file_path: Annotated[Optional[str], "Path to CSV/JSON file with a 'file_name' column/key containing image URLs"] = None,
        # checkpoint: Annotated[str, "Regression model checkpoint identifier."] = "fengchen025/dinov2-base-lora-arabidopsis_leaf-counting_cvppp2017-a1",
        checkpoint: Annotated[str, "Regression model checkpoint identifier."] = None,
        batch_size: Annotated[int, "Number of images to process per batch."] = 1,
        device: Annotated[str, "Device to use for inference ('cuda' or 'cpu')."] = "cuda",
        output_dir: Annotated[str, "Directory path to save results."] = "./results") -> str:
    """
    Perform image regression on a list of images using a specified deep learning model.
    Please provide either image_urls or file_path as input.

    Returns:
    - str: Path to a CSV file containing image regression results.
    """
    
    # raise errors
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
    save_path = os.path.join(save_dir, "img_regression_results.csv")

    # image processor
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    image_processor.do_resize = True
    image_processor.do_normalize = True
    image_processor.do_rescale = True # 255 -> 1

    # model
    if 'lora' in checkpoint:
        config = PeftConfig.from_pretrained(checkpoint)
        pretrained_model = AutoModelForImageClassification.from_pretrained(
            config.base_model_name_or_path,
            label2id=image_processor.label2id,
            id2label=image_processor.id2label,
            ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(pretrained_model, 
                                          checkpoint, 
                                        #   device_map=device,
                                          )
    else:
        model = AutoModelForImageClassification.from_pretrained(
            checkpoint,
            label2id=image_processor.label2id,
            id2label=image_processor.id2label,
            ignore_mismatched_sizes=False,
            # device_map=device,
        )

    # inference
    model.to(device)
    model.eval()
    # batch inference in case of large number of images
    all_images = []
    result_dict = {}
    result_dict['file_name'] = image_urls
    for i in range(0, len(image_urls), batch_size):
        batch_urls = image_urls[i:i+batch_size]
        raw_images = load_images(batch_urls) # return a list of PIL.RGB images # raw_images are [pil.rgb, ...]
        inputs = image_processor(images=raw_images, return_tensors="pt").to(device) # inputs['pixel_values'].shape = batch_size x 3 x H x W
        with torch.no_grad():
            batch_outputs = model(**inputs)
        batch_logits = batch_outputs.logits
        # some post-processing if needed
        for i, k in enumerate(image_processor.label2id.keys()):
            if k in result_dict.keys():
                result_dict[k].extend(batch_logits[:,i].tolist())
            else:
                result_dict[k] = batch_logits[:,i].tolist()
        # for k, v in image_processor.label2id.items():
        #     result_dict[k] = logits[:,int(v)].tolist()
        all_images.extend(raw_images)

    df = pd.DataFrame(result_dict)
    df.to_csv(save_path, index=False)

    return f"The image regression results are saved at {save_path}."

# example usage
if __name__ == "__main__":
    set_env_vars(env_file='../.env.yaml')
    finetune_image_regression(dataset_name = "arabidopsis_leaf-counting_cvppp2017-a1", 
                            ft_method = 'lora',
                            pretrained_model = "facebook/dinov2-base", 
                            img_size = (224, 224),
                            auto_find_batch_size = False,
                            train_bs = 16,
                            val_bs = 1,
                            lr = 5e-5,
                            epochs = 50,
                            )
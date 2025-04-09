import numpy as np
from PIL import Image
import os
import json
from typing import Any, Dict, List, Mapping, Optional
import torch
from torchvision.io import read_image, ImageReadMode
import datasets
from transformers import AutoImageProcessor
from transformers.trainer import EvalPrediction
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchmetrics
from dataclasses import dataclass, field
import evaluate

# cvppp evaluation functions
def Dice(inLabel, gtLabel, i, j):
# calculate Dice score for the given labels i and j
    
    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return 0

    one = np.ones(inLabel.shape)
    inMask = (inLabel==i*one) # find region of label i in inLabel
    gtMask = (gtLabel==j*one) # find region of label j in gtLabel
    inSize = np.sum(inMask*one) # cardinality of set i in inLabel
    gtSize = np.sum(gtMask*one) # cardinality of set j in gtLabel
    overlap= np.sum(inMask*gtMask*one) # cardinality of overlap of the two regions
    if ((inSize + gtSize)>1e-8):
        out = 2*overlap/(inSize + gtSize) # Dice score
    else:
        out = 0

    return out

def BestDice(inLabel,gtLabel):
# input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
#        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
# output: score: Dice score
#
# We assume that the lowest label in inLabel is background, same for gtLabel
# and do not use it. This is necessary to avoid that the trivial solution, 
# i.e. finding only background, gives excellent results.
#
# For the original Dice score, labels corresponding to each other need to
# be known in advance. Here we simply take the best matching label from 
# gtLabel in each comparison. We do not make sure that a label from gtLabel
# is used only once. Better measures may exist. Please enlighten me if I do
# something stupid here...

    score = 0 # initialize output
    
    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return score
    
    maxInLabel = np.max(inLabel) # maximum label value in inLabel
    minInLabel = np.min(inLabel) # minimum label value in inLabel
    maxGtLabel = np.max(gtLabel) # maximum label value in gtLabel
    minGtLabel = np.min(gtLabel) # minimum label value in gtLabel
    
    if(maxInLabel==minInLabel): # trivial solution
        return score

    for i in range(minInLabel+1,maxInLabel+1): # loop all labels of inLabel, but background
        sMax = 0; # maximum Dice value found for label i so far
        for j in range(minGtLabel+1,maxGtLabel+1): # loop all labels of gtLabel, but background
            s = Dice(inLabel, gtLabel, i, j) # compare labelled regions            
            # keep max Dice value for label i
            if(sMax < s):
                sMax = s
        score = score + sMax; # sum up best found values
    
    score = score/(maxInLabel-minInLabel)

    return score

def FGBGDice(inLabel,gtLabel):
# input: inLabel: label image to be evaluated. Background label is assumed to be the lowest one.
#        gtLabel: ground truth label image. Background label is assumed to be the lowest one.
# output: Dice score for foreground/background segmentation, only.

    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return 0

    minInLabel = np.min(inLabel) # minimum label value in inLabel
    minGtLabel = np.min(gtLabel) # minimum label value in gtLabel

    one = np.ones(inLabel.shape)    
    inFgLabel = (inLabel != minInLabel*one)*one
    gtFgLabel = (gtLabel != minGtLabel*one)*one
    
    return Dice(inFgLabel,gtFgLabel,1,1) # Dice score for the foreground

def DiffFGLabels(inLabel,gtLabel):
# input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
#        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
# output: difference of the number of foreground labels

    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return -1

    maxInLabel = int(np.max(inLabel)) # maximum label value in inLabel
    minInLabel = int(np.min(inLabel)) # minimum label value in inLabel
    maxGtLabel = int(np.max(gtLabel)) # maximum label value in gtLabel
    minGtLabel = int(np.min(gtLabel)) # minimum label value in gtLabel

    return (maxInLabel-minInLabel) - (maxGtLabel-minGtLabel)

def filter_gt_mask(gt, size_threshold=[0,1e8]):
    # mask size filtering
    gt_masks = gt["masks"]
    gt_labels = gt["labels"] # not useful for cvppp as we only have 1 class
    if isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.numpy()
    size_filtered_indices = [False] * gt_masks.shape[0]
    for i in range(gt_masks.shape[0]):
        mask_size = np.count_nonzero(gt_masks[i])
        if mask_size > size_threshold[0] and mask_size <= size_threshold[1]:
            size_filtered_indices[i] = True
    size_filtered_masks = gt_masks[size_filtered_indices]

    gt_single_channel_mask = np.zeros((size_filtered_masks.shape[1], size_filtered_masks.shape[2]), dtype=np.uint8) # np (w, h)
    for i in range(size_filtered_masks.shape[0]):
        gt_single_channel_mask[size_filtered_masks[i] > 0] = i+1
    # gt_binary_mask = np.where(gt_single_channel_mask > 0, 255, 0).astype(np.uint8) # np (w, h)

    return gt_single_channel_mask

# w/ small objects
def filter_pred_mask(pred, score_threshold=0.0, size_threshold=[0,1e8]):
    pred_scores = pred["scores"]
    pred_masks = pred["masks"]
    pred_labels = pred["labels"] # not useful for cvppp as we only have 1 class

    # confidence score filtering
    filtered_indices = pred_scores >= score_threshold
    filtered_scores = pred_scores[filtered_indices]
    filtered_masks = pred_masks[filtered_indices] # np (w, h)

    # mask size filtering
    size_filtered_indices = [False] * filtered_masks.shape[0]
    for i in range(filtered_masks.shape[0]):
        mask_size = filtered_masks[i].count_nonzero().item()
        if mask_size > size_threshold[0] and mask_size <= size_threshold[1]:
            size_filtered_indices[i] = True

    size_filtered_scores = filtered_scores[size_filtered_indices]
    size_filtered_masks = filtered_masks[size_filtered_indices]

    # combine multi-channel masks to single channel
    single_channel_mask = np.zeros((size_filtered_masks.shape[1], size_filtered_masks.shape[2]), dtype=np.uint8)
    for i in range(size_filtered_masks.shape[0]):
        single_channel_mask[size_filtered_masks[i].numpy() > 0] = i+1
    # binary_mask = np.where(single_channel_mask > 0, 255, 0).astype(np.uint8)

    return single_channel_mask

# huggingface version
class LeafInsSegMetric(evaluate.Metric):
    def __init__(self, score_threshold=0.85, 
                evaluate_sizes=['all'], 
                # evaluate_sizes=['all', 'large', 'medium', 'small'],
                large=[64**2, 1e8], 
                medium=[32**2, 64**2], 
                small=[0, 32**2], 
                **kwargs):
        super().__init__(**kwargs) # Call the parent class's __init__ to avoid overwriting important initialization
        self.score_threshold = score_threshold
        self.evaluate_sizes = evaluate_sizes
        self.size_dict = {
            'all': [0, 1e8],
            'large': large, 
            'medium': medium, 
            'small': small,
        }
        self.reset()

    def reset(self):
        self.diff_fg = {size: [] for size in self.evaluate_sizes}
        self.dice = {size: [] for size in self.evaluate_sizes}
        self.sbd = {size: [] for size in self.evaluate_sizes}
        self.fgbg_dice = {size: [] for size in self.evaluate_sizes}
    
    def _info(self):
        return datasets.MetricInfo(
            description="CVPPP Leaf Instance Segmentation Metric",
            citation="Insert appropriate citation or leave blank",
            inputs_description="This metric evaluates instance segmentation of leaf images. "
                                "Predictions and references should be dictionaries containing "
                                "segmentation masks, class labels, and optional scores.",
            features=datasets.Features({
                "predictions": {
                    "masks": datasets.Sequence(datasets.Array2D(shape=(None, None), dtype="bool")),  # Binary masks for each instance
                    "labels": datasets.Sequence(datasets.Value("int64")),  # Class labels, though in CVPPP only one class exists
                    "scores": datasets.Sequence(datasets.Value("float32")),  # Confidence scores for each mask in predictions
                },
                "references": {
                    "masks": datasets.Sequence(datasets.Array2D(shape=(None, None), dtype="bool")),  # Ground truth binary masks
                    "labels": datasets.Sequence(datasets.Value("int64")),  # Ground truth class labels
                },
            }),
            reference_urls=[],
        )


    def _compute(self, predictions, references):
        # assuming these predictions and references are post processed (filtered etc.)
        # 'segmentation' is a single mask starting from -1 (background)
        # 'mask_labels' (gt_labels) are binary masks (0 for background and 1 for instance)

        for pred, ref in zip(predictions, references):
            for size in self.evaluate_sizes:
                gt_label = filter_gt_mask(ref, size_threshold=self.size_dict[size]) # done
                pred_label = filter_pred_mask(pred, self.score_threshold, size_threshold=self.size_dict[size]) # done
                
                if np.max(gt_label) > 0:  # filter out images without leaves
                    self.diff_fg[size].append(DiffFGLabels(pred_label, gt_label))
                    self.fgbg_dice[size].append(FGBGDice(pred_label, gt_label))
                    bd = BestDice(pred_label, gt_label)
                    bd2 = BestDice(gt_label, pred_label)
                    self.dice[size].append(bd)  # like precision
                    self.sbd[size].append(min(bd, bd2))

    def accumulate(self):
        # Compute final metrics
        results = {size: {} for size in self.evaluate_sizes}
        for size in self.evaluate_sizes:
            if len(self.dice[size]) > 0:
                results[size] = {
                    'SBD_mean': np.mean(self.sbd[size]),
                    'SBD_std': np.std(self.sbd[size]),
                    "absdiffFG_mean": np.mean(np.abs(self.diff_fg[size])),
                    "absdiffFG_std": np.std(np.abs(self.diff_fg[size])),
                    # "diffFG_mean": np.mean(self.diff_fg[size]),
                    # "diffFG_std": np.std(self.diff_fg[size]),
                    # "bestDice_mean": np.mean(self.dice[size]),
                    # "bestDice_std": np.std(self.dice[size]),
                    # "FgBgDice_mean": np.mean(self.fgbg_dice[size]),
                    # "FgBgDice_std": np.std(self.fgbg_dice[size]),
                }
            else:
                results[size] = {
                    'SBD_mean': 0.,
                    'SBD_std': 0.,
                    "absdiffFG_mean": 0.,
                    "absdiffFG_std": 0.,
                    # "diffFG_mean": 0.,
                    # "diffFG_std": 0.,
                    # "bestDice_mean": 0.,
                    # "bestDice_std": 0.,
                    # "FgBgDice_mean": 0.,
                    # "FgBgDice_std": 0.,
                }
        
        return results

@dataclass
class ModelOutput:
    class_queries_logits: torch.Tensor
    masks_queries_logits: torch.Tensor

def nested_cpu(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_cpu(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_cpu(t) for k, t in tensors.items()})
    elif isinstance(tensors, torch.Tensor):
        return tensors.cpu().detach()
    else:
        return tensors

class InsSegEvaluator:
    def __init__(
        self,
        image_processor: AutoImageProcessor,
        id2label: Mapping[int, str],
        threshold: float = 0.85,
        metric_name: str = "cvppp", # or "map"
    ):
        """
        Initialize evaluator with image processor, id2label mapping and threshold for filtering predictions.
        Args:
            image_processor (AutoImageProcessor): Image processor for
                `post_process_instance_segmentation` method.
            id2label (Mapping[int, str]): Mapping from class id to class name.
            threshold (float): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        """
        self.image_processor = image_processor
        self.id2label = id2label
        self.threshold = threshold
        self.metric_name = metric_name
        self.metric = self.get_metric(self.metric_name)

    def get_metric(self, metric_name: str):
        if metric_name == "cvppp":
            metric = LeafInsSegMetric(score_threshold=0.85, evaluate_sizes=['all'], 
                                      large=[64**2, 1e8], medium=[32**2, 64**2], small=[0, 32**2])
        elif metric_name == "map":
            metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)
        else:
            raise ValueError(f"Unsupported metric name: {metric}")
        return metric

    def reset_metric(self):
        self.metric.reset()

    def postprocess_target_batch(self, target_batch) -> List[Dict[str, torch.Tensor]]:
        """Collect targets in a form of list of dictionaries with keys "masks", "labels"."""
        batch_masks = target_batch[0] # mask labels
        batch_labels = target_batch[1] # class labels
        post_processed_targets = []
        for masks, labels in zip(batch_masks, batch_labels):
            post_processed_targets.append(
                {
                    "masks": masks.to(dtype=torch.bool), # gt can be bool
                    # "masks": masks.astype(dtype=bool), # gt can be bool
                    "labels": labels, # semantic labels
                }
            )
        return post_processed_targets

    def get_target_sizes(self, post_processed_targets) -> List[List[int]]:
        target_sizes = []
        for target in post_processed_targets:
            target_sizes.append(target["masks"].shape[-2:])
        return target_sizes

    def postprocess_prediction_batch(self, prediction_batch, target_sizes) -> List[Dict[str, torch.Tensor]]:
        """Collect predictions in a form of list of dictionaries with keys "masks", "labels", "scores"."""

        model_output = ModelOutput(class_queries_logits=prediction_batch[0], # class logits before post processing
                                   masks_queries_logits=prediction_batch[1], # mask logits before post processing
                                   )
        post_processed_output = self.image_processor.post_process_instance_segmentation(
            model_output,
            threshold=self.threshold,
            target_sizes=target_sizes,
            return_binary_maps=True,
        )

        post_processed_predictions = []
        for image_predictions, target_size in zip(post_processed_output, target_sizes):
            if image_predictions["segments_info"]:
                post_processed_image_prediction = {
                    "masks": image_predictions["segmentation"].to(dtype=torch.bool), # if return_binary_maps=True above, return binary masks
                    "labels": torch.tensor([x["label_id"] for x in image_predictions["segments_info"]]), # semantic labels
                    "scores": torch.tensor([x["score"] for x in image_predictions["segments_info"]]), # confidence
                }
            else:
                # for void predictions, we need to provide empty tensors
                post_processed_image_prediction = {
                    "masks": torch.zeros([0, *target_size], dtype=torch.bool),
                    "labels": torch.tensor([]),
                    "scores": torch.tensor([]),
                }
            post_processed_predictions.append(post_processed_image_prediction)
        return post_processed_predictions

    @torch.no_grad()
    def __call__(self, evaluation_results: EvalPrediction, 
                 compute_result: bool = False) -> Mapping[str, float]:
        """
        Update metrics with current evaluation results and return metrics if `compute_result` is True.

        Args:
            evaluation_results (EvalPrediction): Predictions and targets from evaluation.
            compute_result (bool): Whether to compute and return metrics.

        Returns:
            Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
        """
        prediction_batch = nested_cpu(evaluation_results.predictions)
        target_batch = nested_cpu(evaluation_results.label_ids)

        # For metric computation we need to provide:
        #  - targets in a form of list of dictionaries with keys "masks", "labels"
        #  - predictions in a form of list of dictionaries with keys "masks", "labels", "scores"
        post_processed_targets = self.postprocess_target_batch(target_batch)
        target_sizes = self.get_target_sizes(post_processed_targets)
        post_processed_predictions = self.postprocess_prediction_batch(prediction_batch, target_sizes)

        # # Compute metrics
        if self.metric_name == "map":
            self.metric.update(post_processed_predictions, 
                               post_processed_targets)
            if not compute_result:
                return
            metrics = self.metric.compute()

            # this is for the mean IoU
            # Replace list of per class metrics with separate metric for each class
            classes = metrics.pop("classes")
            map_per_class = metrics.pop("map_per_class")
            mar_100_per_class = metrics.pop("mar_100_per_class")
            for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
                class_name = self.id2label[class_id.item()] if self.id2label is not None else class_id.item()
                metrics[f"map_{class_name}"] = class_map
                metrics[f"mar_100_{class_name}"] = class_mar

            metrics = {k: round(v.item(), 4) for k, v in metrics.items()}
            self.reset_metric()
        elif self.metric_name == "cvppp":
            self.metric._compute(post_processed_predictions, post_processed_targets)
            if not compute_result:
                return
            metrics = self.metric.accumulate()
            metrics['sbd'] =  metrics['all']['SBD_mean'] # main metrics
            metrics['absdiffFG'] = metrics['all']['absdiffFG_mean']
            self.reset_metric()
        else:
            raise ValueError(f"Unsupported metric name: {self.metric_name}")
        
        # # Reset metric for next evaluation
        # self.reset_metric()

        return metrics
    
# # torchmetrics version
# class LeafInsSegMetric(torchmetrics.Metric):
#     def __init__(self, score_threshold=0.85, 
#                  evaluate_sizes=['all'], 
#                  # evaluate_sizes=['all', 'large', 'medium', 'small'],
#                  large=[64**2, 1e8], 
#                  medium=[32**2, 64**2], 
#                  small=[0, 32**2], 
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.score_threshold = score_threshold
#         self.evaluate_sizes = evaluate_sizes
#         self.size_dict = {
#             'all': [0, 1e8],
#             'large': large, 
#             'medium': medium, 
#             'small': small
#         }

#         # Initialize states for accumulating results
#         self.add_state("diff_fg", default=[], dist_reduce_fx="cat") # dist_reduce_fx for distributed computing
#         self.add_state("dice", default=[], dist_reduce_fx="cat")
#         self.add_state("sbd", default=[], dist_reduce_fx="cat")
#         self.add_state("fgbg_dice", default=[], dist_reduce_fx="cat")

#     def update(self, preds: torch.Tensor, refs: torch.Tensor):
#         diff_fg = {size: [] for size in self.evaluate_sizes}
#         dice = {size: [] for size in self.evaluate_sizes}
#         sbd = {size: [] for size in self.evaluate_sizes}
#         fgbg_dice = {size: [] for size in self.evaluate_sizes}

#         for gt_masks, output in zip(refs, preds):
#             for size in self.evaluate_sizes:
#                 gt_label = filter_gt_mask(gt_masks, size_threshold=self.size_dict[size])
#                 pred_label = filter_pred_mask([output], self.score_threshold, size_threshold=self.size_dict[size])

#                 if torch.max(gt_label) > 0:  # filter out images without leaves
#                     diff_fg[size].append(DiffFGLabels(pred_label, gt_label))
#                     fgbg_dice[size].append(FGBGDice(pred_label, gt_label))
#                     bd = BestDice(pred_label, gt_label)
#                     bd2 = BestDice(gt_label, pred_label)
#                     dice[size].append(bd)  # like precision
#                     sbd[size].append(min(bd, bd2))

#         # Update the states with new batch results
#         for size in self.evaluate_sizes:
#             self.diff_fg.extend(diff_fg[size])
#             self.dice.extend(dice[size])
#             self.sbd.extend(sbd[size])
#             self.fgbg_dice.extend(fgbg_dice[size])

#     def compute(self):
#         results = {size: {} for size in self.evaluate_sizes}
#         for size in self.evaluate_sizes:
#             if len(self.dice) > 0:
#                 results[size] = {
#                     # "diffFG_mean": torch.mean(torch.tensor(self.diff_fg)),
#                     # "diffFG_std": torch.std(torch.tensor(self.diff_fg)),
#                     "absdiffFG_mean": torch.mean(torch.abs(torch.tensor(self.diff_fg))),
#                     "absdiffFG_std": torch.std(torch.abs(torch.tensor(self.diff_fg))),
#                     # "bestDice_mean": torch.mean(torch.tensor(self.dice)),
#                     # "bestDice_std": torch.std(torch.tensor(self.dice)),
#                     # "FgBgDice_mean": torch.mean(torch.tensor(self.fgbg_dice)),
#                     # "FgBgDice_std": torch.std(torch.tensor(self.fgbg_dice)),
#                     'SBD_mean': torch.mean(torch.tensor(self.sbd)),
#                     'SBD_std': torch.std(torch.tensor(self.sbd)),
#                 }
#             else:
#                 results[size] = {
#                     # "diffFG_mean": torch.tensor(0.0),
#                     # "diffFG_std": torch.tensor(0.0),
#                     "absdiffFG_mean": torch.tensor(0.0),
#                     "absdiffFG_std": torch.tensor(0.0),
#                     # "bestDice_mean": torch.tensor(0.0),
#                     # "bestDice_std": torch.tensor(0.0),
#                     # "FgBgDice_mean": torch.tensor(0.0),
#                     # "FgBgDice_std": torch.tensor(0.0),
#                     'SBD_mean': torch.tensor(0.0),
#                     'SBD_std': torch.tensor(0.0),
#                 }
        
#         return results
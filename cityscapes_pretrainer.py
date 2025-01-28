import os
import json
import argparse
import logging
import numpy as np
from PIL import Image
from typing import Tuple, Dict

# Logging / progress
from tqdm import tqdm

# Hugging Face Datasets
from datasets import load_from_disk, Dataset

# Label files
from huggingface_hub import cached_download, hf_hub_url

# Image transformations
from transformers import SegformerImageProcessor

# Model & Training
from transformers import (
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer
)
import transformers

# PyTorch
import torch
import torch.nn as nn

# Evaluation
import evaluate


##################################
# Setup logging
##################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


##################################
# 1) Helper functions
##################################
def load_dataset_with_images(
    data_dir: str,
    file_list_path: str,
    base_path: str,
    extension_input: str,
    extension_gt: str
) -> Dataset:
    """
    Creates a Hugging Face Dataset from local image paths + annotation paths.

    Args:
        data_dir: Directory containing ground-truth annotations.
        file_list_path: Path to .txt with base filenames.
        base_path: Directory containing the RGB images.
        extension_input: e.g., '_leftImg8bit.png'.
        extension_gt: e.g., '_gtFine_labelTrainIds.png'.

    Returns:
        A Hugging Face Dataset with columns ['image', 'annotation'],
        each containing in-memory PIL images.
    """
    if not os.path.isfile(file_list_path):
        raise FileNotFoundError(f"File list not found: {file_list_path}")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Base path for images not found: {base_path}")

    with open(file_list_path, "r") as f:
        file_paths = f.read().splitlines()

    if len(file_paths) == 0:
        raise ValueError(f"No filenames found in: {file_list_path}")

    images = []
    annotations = []

    for file_path in tqdm(file_paths, desc="Creating dataset"):
        img_path = os.path.join(base_path, f"{file_path}{extension_input}")
        ann_path = os.path.join(data_dir, f"{file_path}{extension_gt}")

        if not (os.path.exists(img_path) and os.path.exists(ann_path)):
            logger.warning(f"Skipping {file_path}; missing {img_path} or {ann_path}")
            continue

        with Image.open(img_path).convert("RGB") as img:
            images.append(img)
        with Image.open(ann_path).convert("L") as ann:
            annotations.append(ann)

    if len(images) == 0:
        raise ValueError("No valid image/annotation pairs found. Dataset is empty.")

    data_dict = {
        "image": images,
        "annotation": annotations,
    }

    dataset = Dataset.from_dict(data_dict)
    logger.info(f"Dataset created successfully with {len(dataset)} examples.")
    return dataset


def train_transforms(example_batch: Dict, num_labels: int, processor: SegformerImageProcessor) -> Dict:
    """
    Transform function for the training dataset.
    Each 'image'/'annotation' is a PIL image; optionally apply data augmentations,
    then encode with SegformerImageProcessor.
    """

    images = [Image.fromarray(np.array(x, dtype=np.uint8)) for x in example_batch["image"]]
    labels = [Image.fromarray(np.array(x, dtype=np.uint8)) for x in example_batch["annotation"]]

    # Clamp labels to [0..(num_labels - 1)]
    labels_clamped = [
        Image.fromarray(np.minimum(np.array(lbl), num_labels - 1), mode="L")
        for lbl in labels
    ]

    encoded = processor(images=images, segmentation_maps=labels_clamped, return_tensors="pt")
    return encoded


def val_transforms(example_batch: Dict, processor: SegformerImageProcessor) -> Dict:
    """
    Transform function for val/test dataset (no augmentations).
    """

    images = [Image.fromarray(np.array(x, dtype=np.uint8)) for x in example_batch["image"]]
    labels = [Image.fromarray(np.array(x, dtype=np.uint8)) for x in example_batch["annotation"]]

    encoded = processor(images=images, segmentation_maps=labels, return_tensors="pt")
    return encoded


def compute_metrics(
    eval_pred: Tuple[np.ndarray, np.ndarray],
    metric,
    num_labels: int,
    processor: SegformerImageProcessor,
    id2label: Dict[int, str]
) -> Dict:
    """
    Compute mean_iou + per-category accuracies/IoUs for model predictions.
    """
    with torch.no_grad():
        logits, labels = eval_pred  # shape: (batch, num_labels, H, W), (batch, H, W)
        logits_tensor = torch.from_numpy(logits)

        # Upsample to label size & take argmax
        upsampled = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],  # (H, W)
            mode="bilinear",
            align_corners=False
        )
        preds = upsampled.argmax(dim=1).cpu().numpy()

        results = metric.compute(
            predictions=preds,
            references=labels,
            num_labels=num_labels,
            ignore_index=19,  # 'ignore'
            reduce_labels=processor.do_reduce_labels
        )

        # Extract & rename per-category stats
        per_cat_acc = results.pop("per_category_accuracy", None)
        per_cat_iou = results.pop("per_category_iou", None)

        if per_cat_acc is not None:
            for i, acc in enumerate(per_cat_acc.tolist()):
                class_name = id2label[i]
                results[f"accuracy_{class_name}"] = acc

        if per_cat_iou is not None:
            for i, iou_val in enumerate(per_cat_iou.tolist()):
                class_name = id2label[i]
                results[f"iou_{class_name}"] = iou_val

        return results


def parse_arguments() -> argparse.Namespace:
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train SegFormer on Cityscapes with optional dataset creation.")

    # Dataset creation arguments
    parser.add_argument("--create_dataset", action="store_true",
                        help="If set, create dataset from image/annotation paths and save to disk, then exit.")

    parser.add_argument("--data_dir_train", type=str, default="",
                        help="Directory with ground-truth annotations for train.")
    parser.add_argument("--file_list_path_train", type=str, default="",
                        help="Text file listing base filenames for train.")
    parser.add_argument("--base_path_train", type=str, default="",
                        help="Directory with the RGB images for train.")
    parser.add_argument("--extension_input_train", type=str, default="_leftImg8bit.png",
                        help="Extension for RGB train images.")
    parser.add_argument("--extension_gt_train", type=str, default="_gtFine_labelTrainIds.png",
                        help="Extension for GT train images.")
    parser.add_argument("--output_train_dataset", type=str, default="./cs_dataset/cityscapes_train_dataset",
                        help="Where to save the train dataset after creation.")

    parser.add_argument("--data_dir_val", type=str, default="",
                        help="Directory with ground-truth annotations for val.")
    parser.add_argument("--file_list_path_val", type=str, default="",
                        help="Text file listing base filenames for val.")
    parser.add_argument("--base_path_val", type=str, default="",
                        help="Directory with the RGB images for val.")
    parser.add_argument("--extension_input_val", type=str, default="_leftImg8bit.png",
                        help="Extension for RGB val images.")
    parser.add_argument("--extension_gt_val", type=str, default="_gtFine_labelTrainIds.png",
                        help="Extension for GT val images.")
    parser.add_argument("--output_val_dataset", type=str, default="./cs_dataset/cityscapes_val_dataset",
                        help="Where to save the val dataset after creation.")

    # Dataset load arguments
    parser.add_argument("--train_dataset_path", type=str, default="./data/cityscapes/cityscapes_train_dataset",
                        help="Path to load the pre-made train dataset from disk.")
    parser.add_argument("--val_dataset_path", type=str, default="./data/cityscapes/cityscapes_val_dataset",
                        help="Path to load the pre-made val dataset from disk.")

    parser.add_argument("--test_size", type=float, default=0.1,
                        help="Fraction for splitting train into train/test.")

    # Training params
    parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=6e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training & eval.")
    parser.add_argument("--name", type=str, default="segformerb5-cs", help="Name for experiment.")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Whether to overwrite existing output dir.")

    # HPC / performance
    parser.add_argument("--dataloader_num_workers", type=int, default=15,
                        help="Number of workers for data loading.")

    return parser.parse_args()


##################################
# 2) Main
##################################
def main():
    args = parse_arguments()

    # If the user wants to create dataset(s), do so and then exit.
    if args.create_dataset:
        # Train dataset creation
        logger.info("Creating train dataset...")
        train_ds = load_dataset_with_images(
            data_dir=args.data_dir_train,
            file_list_path=args.file_list_path_train,
            base_path=args.base_path_train,
            extension_input=args.extension_input_train,
            extension_gt=args.extension_gt_train
        )
        train_ds.save_to_disk(args.output_train_dataset)
        logger.info(f"Train dataset saved to: {args.output_train_dataset}")

        # Val dataset creation
        logger.info("Creating val dataset...")
        val_ds = load_dataset_with_images(
            data_dir=args.data_dir_val,
            file_list_path=args.file_list_path_val,
            base_path=args.base_path_val,
            extension_input=args.extension_input_val,
            extension_gt=args.extension_gt_val
        )
        val_ds.save_to_disk(args.output_val_dataset)
        logger.info(f"Val dataset saved to: {args.output_val_dataset}")

        logger.info("Finished dataset creation. Exiting.")
        return

    # Otherwise, load the pre-made datasets from disk
    assert os.path.isdir(args.train_dataset_path), f"Train dataset path not found: {args.train_dataset_path}"
    assert os.path.isdir(args.val_dataset_path), f"Val dataset path not found: {args.val_dataset_path}"

    logger.info(f"Loading train dataset from: {args.train_dataset_path}")
    cs_dataset = load_from_disk(args.train_dataset_path)

    logger.info(f"Loading val dataset from: {args.val_dataset_path}")
    cs_dataset_val = load_from_disk(args.val_dataset_path)

    # We do a train/test split from the loaded train dataset
    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test_size must be between 0 and 1.")

    split_ds = cs_dataset.train_test_split(test_size=args.test_size)
    train_ds = split_ds["train"]
    test_ds = split_ds["test"]
    val_ds = cs_dataset_val

    logger.info(f"Train set size: {len(train_ds)}, test set size: {len(test_ds)}, val set size: {len(val_ds)}")

    # 3) Label mappings
    repo_id = "huggingface/label-files"
    filename = "cityscapes-id2label.json"
    logger.info("Loading label mappings from Hugging Face Hub...")
    label_json_path = cached_download(hf_hub_url(repo_id, filename, repo_type="dataset"))

    with open(label_json_path, "r") as f:
        id2label_raw = json.load(f)

    id2label = {int(k): v for k, v in id2label_raw.items()}
    label2id = {v: k for k, v in id2label.items()}

    # Add 'ignore' class as index 19
    id2label[19] = "ignore"
    label2id["ignore"] = 19
    num_labels = len(id2label)
    logger.info(f"Total labels (including ignore): {num_labels}")

    # 4) Processor
    processor = SegformerImageProcessor()

    # 5) Set transforms with lambdas
    train_ds.set_transform(lambda batch: train_transforms(batch, num_labels, processor))
    test_ds.set_transform(lambda batch: val_transforms(batch, processor))
    val_ds.set_transform(lambda batch: val_transforms(batch, processor))

    # 6) Load Model
    logger.info("Loading the SegFormer model (nvidia/mit-b5)...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b5",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # 7) TrainingArguments
    output_dir = f"./models/{args.name}_hf"
    total_examples = len(train_ds)
    total_steps = (total_examples // args.batch_size) * args.epochs
    logger.info(f"Total steps: {total_steps}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        logging_steps=1,
        eval_accumulation_steps=10,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    # 8) Metric
    metric = evaluate.load("mean_iou")

    # 9) HF Trainer
    def hf_compute_metrics(eval_pred):
        return compute_metrics(eval_pred, metric, num_labels, processor, id2label)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,  # or val_ds if you prefer
        compute_metrics=hf_compute_metrics,
        tokenizer=None  # not needed for segmentation
    )

    transformers.logging.set_verbosity_info()

    # 10) Train
    logger.info("Starting training...")
    trainer.train()

    logger.info("Training completed. If desired, save final model via trainer.save_model(...).")


##################################
# 3) Entry point & Example Usage
##################################
if __name__ == "__main__":
    """
    Example usage:

    1) Creating the datasets from local image/annotation files, then exit:
       python script.py --create_dataset \
         --data_dir_train /data/datasets/cityscapes/gtFine/train \
         --file_list_path_train ./data/cityscapes/cityscapes_train.txt \
         --base_path_train /data/datasets/cityscapes/leftImg8bit/train \
         --extension_input_train _leftImg8bit.png \
         --extension_gt_train _gtFine_labelTrainIds.png \
         --output_train_dataset ./cs_dataset/cityscapes_train_dataset \
         --data_dir_val /data/datasets/cityscapes/gtFine/val \
         --file_list_path_val ./data/cityscapes/cityscapes_val.txt \
         --base_path_val /data/datasets/cityscapes/leftImg8bit/val \
         --extension_input_val _leftImg8bit.png \
         --extension_gt_val _gtFine_labelTrainIds.png \
         --output_val_dataset ./cs_dataset/cityscapes_val_dataset

    2) Training on pre-existing arrow datasets:
       python script.py \
         --train_dataset_path ./data/cityscapes/cityscapes_train_dataset \
         --val_dataset_path ./data/cityscapes/cityscapes_val_dataset \
         --test_size 0.1 \
         --epochs 120 \
         --learning_rate 6e-5 \
         --batch_size 2 \
         --name segformerb5-cs \
         --overwrite_output_dir \
         --dataloader_num_workers 8
    """
    main()

# Cityscapes Segmentation & ATTA Training

This repository contains code for training semantic segmentation models on the Cityscapes dataset using Hugging Face’s [SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer) model and for performing Active Test-Time Adaptation (ATTA) on segmentation models. The project is split into two main components:

1. **SegFormer Training on Cityscapes:**  
   - Create Hugging Face datasets from local Cityscapes image/annotation files.
   - Train and evaluate a SegFormer model using the Hugging Face Trainer.
   - Compute metrics such as mean Intersection over Union (mean IoU) and per-category accuracies.

2. **Active Test-Time Adaptation (ATTA):**  
   - Adapt a pretrained segmentation model at test time under different conditions (e.g., fog, night, rain, snow).
   - Customize adaptation via region-based annotations and entropy loss weighting.
   - Train, evaluate, and save adapted models.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Creating the Cityscapes Dataset](#1-creating-the-cityscapes-dataset)
  - [2. Training SegFormer on Cityscapes](#2-training-segformer-on-cityscapes)
  - [3. Active Test-Time Adaptation (ATTA) Training](#3-active-test-time-adaptation-atta-training)
- [Configuration and Customization](#configuration-and-customization)
- [Logging and Output](#logging-and-output)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Features

- **Dataset Creation:**  
  Create Hugging Face `Dataset` objects from local image and annotation files (e.g., from the Cityscapes dataset).

- **SegFormer Training:**  
  Fine-tune the SegFormer model (using NVIDIA’s `mit-b5` weights) with custom data transforms, and evaluate using mean IoU and per-category metrics.

- **Active Test-Time Adaptation (ATTA):**  
  Apply various test-time adaptation strategies (e.g., FTTA, CTTA) to improve model performance under adverse conditions. Customize adaptation parameters like entropy loss weighting and region selection.

- **Modular Code:**  
  Organized helper functions for dataset loading, image processing, training, evaluation, and ATTA.

- **Logging & Progress:**  
  Integrated logging and progress tracking via Python’s `logging` module and `tqdm`.

---

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [Transformers](https://github.com/huggingface/transformers)
- [Datasets](https://github.com/huggingface/datasets)
- [Evaluate](https://github.com/huggingface/evaluate)
- [Pillow](https://python-pillow.org/)
- [NumPy](https://numpy.org/)
- [tqdm](https://github.com/tqdm/tqdm)

---

## Installation

### Using Conda Environment (Recommended)

This project includes an `environment.yml` file that lists all required packages. To create and activate the Conda environment, run:

```bash
conda env create -f environment.yml
conda activate your_env_name  # Replace 'your_env_name' with the name specified in environment.yml


## Usage

### 1. Creating the Cityscapes Dataset

Before training, you may need to convert your local Cityscapes images and annotations into a Hugging Face dataset. To create and save the datasets, run:

```bash
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

Once the datasets are created (or if you already have pre-saved datasets), you can train the segmentation model with:

```bash
python cityscapes_pretrainer.py \
  --train_dataset_path ./data/cityscapes/cityscapes_train_dataset \
  --val_dataset_path ./data/cityscapes/cityscapes_val_dataset \
  --test_size 0.1 \
  --epochs 120 \
  --learning_rate 6e-5 \
  --batch_size 2 \
  --name segformerb5-cs \
  --overwrite_output_dir \
  --dataloader_num_workers 8

The repository also includes support for ATTA training. To run ATTA training, use the dedicated script (e.g., atta_script.py):

```bash
python atta_script.py \
  --gpu 0 \
  --tta_type FTTA \
  --lr 0.0001 \
  --lambda_entropy 0.01 \
  --train_images_list_path ./data/train_list.txt \
  --val_images_list_path ./data/val_list.txt \
  --base_path_images /path/to/images \
  --base_path_annotations /path/to/annotations \
  --path_to_save_model ./saved_models \
  --model_name segmodel_adapted \
  --region_annotation_extension _rgb_anon.npz \
  --images_extension _rgb_anon.png \
  --annotations_extension _gt_labelTrainIds.png \
  --epochs 10 \
  --evaluate_every 100 \
  --annotations_script sam_region_annotate.py \
  --dataset acdc_train_annotations \
  --path-masks /path/to/sam/masks \
  --min-area 1000 \
  --budget 16 \
  --sort-by BvsSB \
  --seed 123 \
  --ignore-label-annotations 255 \
  --path-output ./output_annotations \
  --delete_output_annotations True


Acknowledgments

    Hugging Face Transformers and Datasets
    NVIDIA for the SegFormer model
    Community contributions and open-source libraries